"""
video_localizer.py

녹화된 테스트 동영상에서 ArUco 기반 위치를 프레임별로 계산하고,
모델/방법 비교에 사용할 수 있도록 CSV + 결과 영상 + summary JSON을 저장한다.

기본 비교 출력
---------------
1) best_single
   - 현재 프레임에서 weight가 가장 높은 단일 마커 결과

2) weighted_fusion
   - 여러 마커의 이상치를 제거하고
     marker area / reprojection_error^2 가중 평균

3) temporal_ema
   - weighted_fusion 결과에 시간축 EMA 적용

출력 예
-------
results/test_001/
    annotated.mp4
    positions.csv
    summary.json

CSV에는 각 프레임별:
- detected marker IDs
- registered marker IDs
- best_single 위치
- weighted_fusion 위치
- temporal_ema 위치
- yaw
- spread
- reprojection error
- nearest landmark
등이 기록된다.

미리보기 조작
-------------
SPACE : pause/play
N 또는 . : 일시정지 상태에서 다음 프레임 1장
[ / - : 재생속도 감소
] / + : 재생속도 증가
R : 1배속
S : 현재 화면 snapshot 저장
Q / ESC : 종료

실행 예
-------
python video_localizer.py test.mp4

python video_localizer.py test.mp4 --dict DICT_4X4_50

# 다른 카메라로 촬영한 영상, 기존 캘리브레이션 무시
python video_localizer.py test.mp4 --no-calibration

# 수평 화각을 70도로 가정
python video_localizer.py test.mp4 --no-calibration --hfov 70

python video_localizer.py test.mp4 --output-dir results/test_001

캘리브레이션:
- 기본적으로 이 스크립트와 같은 폴더의 camera_calibration.npz를 사용한다.
- 테스트 영상은 가능하면 캘리브레이션에 사용한 동일 외장 웹캠,
  동일 해상도로 촬영하는 것이 좋다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from aruco_world_map import get_marker_by_id
from camera_calibration import (
    load_calibration,
    make_approximate_calibration,
)
from coordinate_system import (
    nearest_door_landmark,
)
from pose_localizer import (
    CameraPoseEstimate,
    camera_yaw_deg,
    estimate_camera_pose_from_marker,
    fuse_camera_poses,
)
from live_map_view import (
    LiveMapRenderer,
)


DEFAULT_DICT = "DICT_4X4_50"


@dataclass
class EmaState:
    value: Optional[np.ndarray] = None

    def update(
        self,
        current: np.ndarray,
        alpha: float,
    ) -> np.ndarray:

        current = np.asarray(
            current,
            dtype=np.float64,
        ).reshape(3)

        if self.value is None:
            self.value = current.copy()
        else:
            self.value = (
                alpha * current
                +
                (1.0 - alpha)
                *
                self.value
            )

        return self.value.copy()

    def reset(self) -> None:
        self.value = None


def create_detector(
    dictionary_name: str,
):
    if not hasattr(
        cv2,
        "aruco",
    ):
        raise RuntimeError(
            "cv2.aruco가 없습니다. "
            "opencv-contrib-python을 설치하세요."
        )

    if not hasattr(
        cv2.aruco,
        dictionary_name,
    ):
        raise ValueError(
            f"잘못된 ArUco dictionary: "
            f"{dictionary_name}"
        )

    dictionary = (
        cv2.aruco
        .getPredefinedDictionary(
            getattr(
                cv2.aruco,
                dictionary_name,
            )
        )
    )

    params = (
        cv2.aruco
        .DetectorParameters()
    )

    if hasattr(
        cv2.aruco,
        "CORNER_REFINE_SUBPIX",
    ):
        params.cornerRefinementMethod = (
            cv2.aruco
            .CORNER_REFINE_SUBPIX
        )

    if hasattr(
        cv2.aruco,
        "ArucoDetector",
    ):
        detector = (
            cv2.aruco
            .ArucoDetector(
                dictionary,
                params,
            )
        )

        return detector.detectMarkers

    def detect(gray):
        return (
            cv2.aruco
            .detectMarkers(
                gray,
                dictionary,
                parameters=params,
            )
        )

    return detect


def put_text(
    frame,
    text: str,
    x: int,
    y: int,
    scale: float = 0.55,
    color=(255, 255, 255),
    thickness: int = 1,
):
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def safe_float(
    value: Optional[float],
):
    if value is None:
        return ""

    try:
        if not math.isfinite(
            float(value)
        ):
            return ""
    except Exception:
        return ""

    return float(value)


def xyz_fields(
    prefix: str,
    position: Optional[np.ndarray],
):
    if position is None:
        return {
            f"{prefix}_x_mm": "",
            f"{prefix}_y_mm": "",
            f"{prefix}_z_mm": "",
        }

    p = np.asarray(
        position,
        dtype=np.float64,
    ).reshape(3)

    return {
        f"{prefix}_x_mm": float(p[0]),
        f"{prefix}_y_mm": float(p[1]),
        f"{prefix}_z_mm": float(p[2]),
    }


def build_side_by_side(
    camera_frame: np.ndarray,
    map_frame: np.ndarray,
) -> np.ndarray:

    camera_h, camera_w = (
        camera_frame.shape[:2]
    )

    map_h, map_w = (
        map_frame.shape[:2]
    )

    # 지도 폭을 카메라 폭에 맞추고 aspect ratio 유지.
    target_map_w = camera_w

    scale = (
        target_map_w
        /
        map_w
    )

    target_map_h = max(
        1,
        int(
            round(
                map_h * scale
            )
        ),
    )

    map_resized = cv2.resize(
        map_frame,
        (
            target_map_w,
            target_map_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return np.vstack(
        [
            camera_frame,
            map_resized,
        ]
    )



def resize_for_preview(
    frame: np.ndarray,
    max_width: int,
) -> np.ndarray:
    """
    원본 결과영상은 그대로 유지하고,
    화면에 보여줄 preview만 축소한다.
    """
    if max_width <= 0:
        return frame

    h, w = frame.shape[:2]

    if w <= max_width:
        return frame

    scale = (
        float(max_width)
        /
        float(w)
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    return cv2.resize(
        frame,
        (
            int(max_width),
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )


def draw_preview_status(
    frame: np.ndarray,
    frame_index: int,
    total_frames_meta: int,
    time_sec: float,
    paused: bool,
    playback_speed: float,
    preview_mode: str = "all",
    skipped_frames: int = 0,
) -> None:
    """
    큰 프레임 번호와 조작 상태를 preview 화면에 표시한다.
    """
    h, w = frame.shape[:2]

    # 오른쪽 위 반투명 패널
    panel_w = min(
        430,
        max(300, w // 3),
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            w - panel_w,
            0,
        ),
        (
            w - 1,
            154,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.62,
        frame,
        0.38,
        0,
        frame,
    )

    total_text = (
        str(total_frames_meta)
        if total_frames_meta > 0
        else "?"
    )

    cv2.putText(
        frame,
        f"FRAME {frame_index} / {total_text}",
        (
            w - panel_w + 12,
            38,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"time={time_sec:.2f}s   speed={playback_speed:.2f}x",
        (
            w - panel_w + 12,
            70,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    state_text = (
        "PAUSED"
        if paused
        else "PLAYING"
    )

    state_color = (
        (0, 165, 255)
        if paused
        else (0, 255, 0)
    )

    cv2.putText(
        frame,
        state_text,
        (
            w - panel_w + 12,
            101,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        state_color,
        2,
        cv2.LINE_AA,
    )

    mode_text = (
        f"mode={preview_mode.upper()}  "
        f"skipped={skipped_frames}"
    )

    cv2.putText(
        frame,
        mode_text,
        (
            w - panel_w + 12,
            130,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 하단 조작 도움말
    help_text = (
        "SPACE pause/play | N next frame | [ ] speed | R 1x | S snapshot | Q quit"
    )

    text_y = h - 14

    cv2.rectangle(
        frame,
        (
            0,
            max(0, h - 42),
        ),
        (
            min(w - 1, 820),
            h - 1,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        help_text,
        (
            10,
            text_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def save_preview_snapshot(
    frame: np.ndarray,
    output_dir: Path,
    frame_index: int,
) -> Path:
    snapshot_dir = (
        output_dir
        /
        "snapshots"
    )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        snapshot_dir
        /
        f"frame_{frame_index:06d}.png"
    )

    if not cv2.imwrite(
        str(path),
        frame,
    ):
        raise RuntimeError(
            f"스냅샷 저장 실패: {path}"
        )

    return path

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        help="입력 테스트 동영상 경로",
    )

    parser.add_argument(
        "--calibration",
        default=str(
            Path(__file__).resolve().parent
            /
            "camera_calibration.npz"
        ),
    )

    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help=(
            "camera_calibration.npz를 사용하지 않고 "
            "HFOV 기반 근사 카메라 모델을 사용"
        ),
    )

    parser.add_argument(
        "--hfov",
        type=float,
        default=60.0,
        help=(
            "캘리브레이션 미사용 시 가정할 "
            "수평 화각(deg). 기본 60"
        ),
    )

    parser.add_argument(
        "--dict",
        dest="dictionary_name",
        default=DEFAULT_DICT,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "결과 폴더. 생략하면 "
            "video파일명_results"
        ),
    )

    parser.add_argument(
        "--outlier-mm",
        type=float,
        default=1500.0,
    )

    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.25,
        help=(
            "시간축 EMA 현재 프레임 반영 비율. "
            "0~1, 기본 0.25"
        ),
    )

    parser.add_argument(
        "--reset-after-missing",
        type=int,
        default=30,
        help=(
            "등록 마커가 이 프레임 수만큼 연속 미검출되면 "
            "EMA 상태를 초기화"
        ),
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="annotated.mp4 저장 안 함",
    )

    parser.add_argument(
        "--no-map",
        action="store_true",
        help="결과 영상에 지도 합성 안 함",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="처리 중 화면 표시. Q/ESC로 중단",
    )

    parser.add_argument(
        "--preview-width",
        type=int,
        default=1100,
        help=(
            "미리보기 창에 표시할 최대 폭(px). "
            "기본 1100. 원본 영상이 커도 자동 축소됨"
        ),
    )

    parser.add_argument(
        "--start-speed",
        type=float,
        default=1.0,
        help=(
            "미리보기 시작 재생속도. "
            "기본 1.0배"
        ),
    )

    parser.add_argument(
        "--preview-fast",
        action="store_true",
        help=(
            "ALL 모드에서 원본 FPS 대기 없이 "
            "분석 가능한 최대 속도로 preview"
        ),
    )

    parser.add_argument(
        "--preview-mode",
        choices=[
            "all",
            "realtime",
        ],
        default="all",
        help=(
            "all: 모든 프레임 분석, "
            "realtime: 재생시간을 맞추기 위해 필요 시 프레임 건너뜀"
        ),
    )

    args = parser.parse_args()

    if not (
        0.0
        <
        args.ema_alpha
        <=
        1.0
    ):
        raise ValueError(
            "--ema-alpha는 0보다 크고 1 이하여야 합니다."
        )

    video_path = Path(
        args.video
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"동영상을 찾을 수 없습니다: {video_path}"
        )

    calibration_path = Path(
        args.calibration
    )

    if args.output_dir is None:
        output_dir = (
            video_path.parent
            /
            f"{video_path.stem}_results"
        )
    else:
        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        /
        "positions.csv"
    )

    summary_path = (
        output_dir
        /
        "summary.json"
    )

    output_video_path = (
        output_dir
        /
        "annotated.mp4"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"동영상을 열 수 없습니다: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    frame_count_meta = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # --------------------------------------------------------
    # 카메라 모델 선택
    # --------------------------------------------------------
    if args.no_calibration:
        calibration = make_approximate_calibration(
            image_width=width,
            image_height=height,
            horizontal_fov_deg=args.hfov,
        )

        calibration_mode = (
            f"APPROX_HFOV_{args.hfov:.1f}"
        )

        print(
            "[CALIBRATION] 미사용"
        )
        print(
            f"[CALIBRATION] 근사 HFOV = "
            f"{args.hfov:.1f} deg"
        )
        print(
            "[CALIBRATION] 렌즈 왜곡 = 0으로 가정"
        )

    else:
        if not calibration_path.exists():
            raise FileNotFoundError(
                "캘리브레이션 파일이 없습니다: "
                f"{calibration_path}\n"
                "다른 카메라 영상이라면 "
                "--no-calibration 옵션을 사용하세요."
            )

        calibration = load_calibration(
            calibration_path
        )

        calibration_mode = (
            f"CALIBRATED_{calibration_path.name}"
        )

        print(
            f"[CALIBRATION] 사용: "
            f"{calibration_path}"
        )

    # 캘리브레이션 해상도 검증
    resolution_warning = None

    if (
        not args.no_calibration
        and calibration.image_width > 0
        and calibration.image_height > 0
        and (
            calibration.image_width != width
            or calibration.image_height != height
        )
    ):
        resolution_warning = (
            f"Calibration resolution "
            f"{calibration.image_width}x{calibration.image_height} "
            f"!= video resolution {width}x{height}"
        )

        print(
            "[WARNING] "
            +
            resolution_warning
        )

    detect = create_detector(
        args.dictionary_name
    )

    map_renderer = LiveMapRenderer(
        width=1100,
        height=420,
    )

    # side-by-side output size는 첫 프레임에서 결정.
    writer = None

    ema = EmaState()
    missing_streak = 0

    # 미리보기 조작 상태
    paused = False
    playback_speed = max(
        0.125,
        float(args.start_speed),
    )

    # 지원할 속도 단계
    speed_steps = [
        0.125,
        0.25,
        0.5,
        1.0,
        1.5,
        2.0,
        4.0,
        8.0,
    ]

    if args.preview:
        cv2.namedWindow(
            "Video Localization Test",
            cv2.WINDOW_NORMAL,
        )

        # 창 자체도 사용자가 드래그해 자유롭게 리사이즈 가능.
        cv2.resizeWindow(
            "Video Localization Test",
            int(args.preview_width),
            max(
                500,
                int(args.preview_width * 0.75),
            ),
        )

    total_frames = 0
    skipped_frames_realtime = 0
    frames_with_any_aruco = 0
    frames_with_registered = 0
    frames_with_fused_pose = 0

    used_marker_frequency = {}
    detected_marker_frequency = {}

    reprojection_errors = []
    spreads = []

    csv_columns = [
        "frame",
        "time_sec",
        "detected_ids",
        "registered_ids",
        "used_ids",
        "rejected_ids",
        "best_marker_point",
        "best_marker_id",
        "best_weight",
        "best_reprojection_error_px",
        "best_marker_area_px2",
        "best_single_x_mm",
        "best_single_y_mm",
        "best_single_z_mm",
        "fused_x_mm",
        "fused_y_mm",
        "fused_z_mm",
        "ema_x_mm",
        "ema_y_mm",
        "ema_z_mm",
        "yaw_deg",
        "spread_mm",
        "nearest_landmark",
        "nearest_landmark_distance_mm",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer_csv = csv.DictWriter(
            csv_file,
            fieldnames=csv_columns,
        )

        writer_csv.writeheader()

        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()

                if not ok or frame is None:
                    break

                # 이 프레임의 실제 처리 시간을 측정해서
                # preview 대기시간에서 빼준다.
                frame_process_started = time.perf_counter()

                total_frames += 1

                time_sec = (
                    frame_index
                    /
                    fps
                )

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )

                (
                    corners,
                    ids,
                    rejected,
                ) = detect(
                    gray
                )

                detected_ids = []
                registered_ids = []
                estimates: List[
                    CameraPoseEstimate
                ] = []

                display = frame.copy()

                if ids is not None:
                    frames_with_any_aruco += 1

                    cv2.aruco.drawDetectedMarkers(
                        display,
                        corners,
                        ids,
                    )

                    ids_flat = (
                        np.asarray(ids)
                        .reshape(-1)
                        .astype(int)
                    )

                    for (
                        marker_corners,
                        marker_id,
                    ) in zip(
                        corners,
                        ids_flat,
                    ):
                        marker_id = int(
                            marker_id
                        )

                        detected_ids.append(
                            marker_id
                        )

                        detected_marker_frequency[
                            str(marker_id)
                        ] = (
                            detected_marker_frequency
                            .get(
                                str(marker_id),
                                0,
                            )
                            +
                            1
                        )

                        marker = get_marker_by_id(
                            marker_id
                        )

                        pts = (
                            np.asarray(
                                marker_corners,
                                dtype=np.float64,
                            )
                            .reshape(4, 2)
                        )

                        center = np.mean(
                            pts,
                            axis=0,
                        )

                        cx = int(
                            round(
                                float(
                                    center[0]
                                )
                            )
                        )

                        cy = int(
                            round(
                                float(
                                    center[1]
                                )
                            )
                        )

                        if marker is None:
                            put_text(
                                display,
                                f"ID {marker_id} UNREGISTERED",
                                cx + 5,
                                cy - 8,
                                color=(
                                    0,
                                    165,
                                    255,
                                ),
                                thickness=2,
                            )
                            continue

                        registered_ids.append(
                            marker_id
                        )

                        estimate = (
                            estimate_camera_pose_from_marker(
                                marker,
                                marker_corners,
                                calibration.camera_matrix,
                                calibration.dist_coeffs,
                            )
                        )

                        if estimate is None:
                            continue

                        estimates.append(
                            estimate
                        )

                        p = (
                            estimate
                            .position_world_mm
                        )

                        put_text(
                            display,
                            (
                                f"{marker.point_name} "
                                f"ID={marker_id} "
                                f"({p[0]:.0f},"
                                f"{p[1]:.0f},"
                                f"{p[2]:.0f})"
                            ),
                            cx + 5,
                            cy - 10,
                            color=(
                                0,
                                255,
                                255,
                            ),
                            thickness=2,
                        )

                if registered_ids:
                    frames_with_registered += 1

                # --------------------------------------------
                # method 1: best_single
                # --------------------------------------------
                best = (
                    max(
                        estimates,
                        key=lambda e: e.weight,
                    )
                    if estimates
                    else None
                )

                best_position = (
                    best.position_world_mm
                    if best is not None
                    else None
                )

                # --------------------------------------------
                # method 2: weighted fusion
                # --------------------------------------------
                fused = fuse_camera_poses(
                    estimates,
                    args.outlier_mm,
                )

                if fused is not None:
                    frames_with_fused_pose += 1
                    missing_streak = 0

                    for estimate in (
                        fused.estimates_used
                    ):
                        key = (
                            estimate.marker.point_name
                        )

                        used_marker_frequency[
                            key
                        ] = (
                            used_marker_frequency
                            .get(
                                key,
                                0,
                            )
                            +
                            1
                        )

                    p_fused = (
                        fused
                        .position_world_mm
                    )

                    # ----------------------------------------
                    # method 3: temporal EMA
                    # ----------------------------------------
                    p_ema = ema.update(
                        p_fused,
                        args.ema_alpha,
                    )

                    yaw = camera_yaw_deg(
                        fused
                        .rotation_world_camera
                    )

                    nearest_landmark, nearest_distance = (
                        nearest_door_landmark(
                            float(
                                p_fused[0]
                            ),
                            float(
                                p_fused[1]
                            ),
                        )
                    )

                    if fused.estimates_used:
                        for e in (
                            fused.estimates_used
                        ):
                            reprojection_errors.append(
                                float(
                                    e.reprojection_error_px
                                )
                            )

                    spreads.append(
                        float(
                            fused.spread_mm
                        )
                    )

                    used_ids = [
                        e.marker.marker_id
                        for e
                        in fused.estimates_used
                    ]

                    rejected_ids = [
                        e.marker.marker_id
                        for e
                        in fused.estimates_rejected
                    ]

                else:
                    missing_streak += 1

                    p_fused = None
                    p_ema = None
                    yaw = None
                    nearest_landmark = None
                    nearest_distance = None
                    used_ids = []
                    rejected_ids = []

                    if (
                        missing_streak
                        >=
                        args.reset_after_missing
                    ):
                        ema.reset()

                # --------------------------------------------
                # CSV row
                # --------------------------------------------
                row = {
                    "frame": frame_index,
                    "time_sec": time_sec,
                    "detected_ids": ",".join(
                        str(x)
                        for x
                        in detected_ids
                    ),
                    "registered_ids": ",".join(
                        str(x)
                        for x
                        in registered_ids
                    ),
                    "used_ids": ",".join(
                        str(x)
                        for x
                        in used_ids
                    ),
                    "rejected_ids": ",".join(
                        str(x)
                        for x
                        in rejected_ids
                    ),
                    "best_marker_point": (
                        best.marker.point_name
                        if best is not None
                        else ""
                    ),
                    "best_marker_id": (
                        best.marker.marker_id
                        if best is not None
                        else ""
                    ),
                    "best_weight": (
                        float(best.weight)
                        if best is not None
                        else ""
                    ),
                    "best_reprojection_error_px": (
                        float(
                            best
                            .reprojection_error_px
                        )
                        if best is not None
                        else ""
                    ),
                    "best_marker_area_px2": (
                        float(
                            best
                            .marker_area_px2
                        )
                        if best is not None
                        else ""
                    ),
                    **xyz_fields(
                        "best_single",
                        best_position,
                    ),
                    **xyz_fields(
                        "fused",
                        p_fused,
                    ),
                    **xyz_fields(
                        "ema",
                        p_ema,
                    ),
                    "yaw_deg": safe_float(
                        yaw
                    ),
                    "spread_mm": (
                        float(
                            fused.spread_mm
                        )
                        if fused is not None
                        else ""
                    ),
                    "nearest_landmark": (
                        nearest_landmark.name
                        if nearest_landmark
                        is not None
                        else ""
                    ),
                    "nearest_landmark_distance_mm": (
                        float(
                            nearest_distance
                        )
                        if nearest_distance
                        is not None
                        else ""
                    ),
                }

                writer_csv.writerow(
                    row
                )

                # --------------------------------------------
                # Annotated display
                # --------------------------------------------
                overlay = display.copy()

                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (
                        min(
                            display.shape[1],
                            1000,
                        ),
                        154,
                    ),
                    (0, 0, 0),
                    -1,
                )

                cv2.addWeighted(
                    overlay,
                    0.58,
                    display,
                    0.42,
                    0,
                    display,
                )

                put_text(
                    display,
                    (
                        f"frame={frame_index} "
                        f"time={time_sec:.2f}s "
                        f"detected={detected_ids}"
                    ),
                    12,
                    27,
                    0.58,
                )

                if best_position is not None:
                    put_text(
                        display,
                        (
                            f"best_single: "
                            f"({best_position[0]:.0f}, "
                            f"{best_position[1]:.0f}, "
                            f"{best_position[2]:.0f})"
                        ),
                        12,
                        55,
                        0.52,
                        (
                            0,
                            255,
                            255,
                        ),
                        2,
                    )

                if p_fused is not None:
                    put_text(
                        display,
                        (
                            f"weighted_fusion: "
                            f"({p_fused[0]:.0f}, "
                            f"{p_fused[1]:.0f}, "
                            f"{p_fused[2]:.0f}) "
                            f"spread={fused.spread_mm:.0f}mm"
                        ),
                        12,
                        82,
                        0.52,
                        (
                            0,
                            255,
                            0,
                        ),
                        2,
                    )

                if p_ema is not None:
                    put_text(
                        display,
                        (
                            f"temporal_ema: "
                            f"({p_ema[0]:.0f}, "
                            f"{p_ema[1]:.0f}, "
                            f"{p_ema[2]:.0f})"
                        ),
                        12,
                        109,
                        0.52,
                        (
                            255,
                            255,
                            255,
                        ),
                        2,
                    )

                if p_fused is not None:
                    put_text(
                        display,
                        (
                            f"nearest={nearest_landmark.name} "
                            f"distance={nearest_distance:.0f}mm "
                            f"({nearest_distance/1000.0:.2f}m) "
                            f"Z={p_fused[2]:.0f}mm"
                        ),
                        12,
                        136,
                        0.54,
                        (0, 255, 255),
                        2,
                    )

                # Map uses EMA for visually stable point.
                if p_ema is not None:
                    map_frame = map_renderer.render(
                        position_world_mm=p_ema,
                        yaw_deg=yaw,
                        spread_mm=(
                            fused.spread_mm
                            if fused is not None
                            else None
                        ),
                        used_marker_names=[
                            e.marker.point_name
                            for e
                            in (
                                fused.estimates_used
                                if fused is not None
                                else []
                            )
                        ],
                        z_mm=float(
                            p_ema[2]
                        ),
                    )
                else:
                    map_frame = (
                        map_renderer
                        .render()
                    )

                if args.no_map:
                    final_frame = display
                else:
                    final_frame = (
                        build_side_by_side(
                            display,
                            map_frame,
                        )
                    )

                if (
                    not args.no_video
                    and writer is None
                ):
                    out_h, out_w = (
                        final_frame.shape[:2]
                    )

                    fourcc = (
                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )
                    )

                    writer = cv2.VideoWriter(
                        str(
                            output_video_path
                        ),
                        fourcc,
                        fps,
                        (
                            out_w,
                            out_h,
                        ),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            "결과 동영상 writer를 "
                            "열 수 없습니다."
                        )

                if writer is not None:
                    writer.write(
                        final_frame
                    )

                if args.preview:
                    preview_frame = (
                        final_frame.copy()
                    )

                    draw_preview_status(
                        preview_frame,
                        frame_index=frame_index,
                        total_frames_meta=frame_count_meta,
                        time_sec=time_sec,
                        paused=paused,
                        playback_speed=playback_speed,
                        preview_mode=args.preview_mode,
                        skipped_frames=skipped_frames_realtime,
                    )

                    preview_frame = (
                        resize_for_preview(
                            preview_frame,
                            args.preview_width,
                        )
                    )

                    cv2.imshow(
                        "Video Localization Test",
                        preview_frame,
                    )

                    # 원본 영상의 프레임 간격에서 실제 분석 시간을 빼고
                    # 남은 시간만 기다린다.
                    # 예: 30fps = 33.3ms/frame, 분석 20ms이면 약 13ms만 대기.
                    target_frame_ms = (
                        1000.0
                        /
                        max(fps, 1.0)
                        /
                        max(
                            playback_speed,
                            0.125,
                        )
                    )

                    proceed_to_next_frame = (
                        not paused
                    )
                    manual_step_requested = False

                    while True:
                        if paused:
                            wait_ms = 0
                        else:
                            # 재생속도가 조작키로 바뀔 수 있으므로
                            # 매 반복마다 목표 프레임 간격을 다시 계산한다.
                            target_frame_ms = (
                                1000.0
                                /
                                max(fps, 1.0)
                                /
                                max(
                                    playback_speed,
                                    0.125,
                                )
                            )

                            processing_ms = (
                                time.perf_counter()
                                -
                                frame_process_started
                            ) * 1000.0

                            remaining_ms = (
                                target_frame_ms
                                -
                                processing_ms
                            )

                            # preview-fast는 ALL 모드에서만 적용.
                            if (
                                args.preview_mode == "all"
                                and args.preview_fast
                            ):
                                wait_ms = 1
                            else:
                                wait_ms = max(
                                    1,
                                    int(
                                        round(
                                            remaining_ms
                                        )
                                    ),
                                )

                        key_full = cv2.waitKeyEx(
                            wait_ms
                        )

                        if key_full < 0:
                            # 재생 중 timeout이면 다음 프레임.
                            if not paused:
                                # 일시정지 시간은 realtime skip 계산에서 제외.
                                frame_process_started = (
                                    time.perf_counter()
                                )
                                proceed_to_next_frame = True
                                break

                            continue

                        key = (
                            key_full
                            &
                            0xFF
                        )

                        # 종료
                        if key in (
                            ord("q"),
                            ord("Q"),
                            27,
                        ):
                            raise KeyboardInterrupt

                        # SPACE: pause/play
                        if key == 32:
                            paused = not paused

                            # 현재 프레임의 상태 표시를 즉시 갱신
                            preview_state = (
                                final_frame.copy()
                            )

                            draw_preview_status(
                                preview_state,
                                frame_index=frame_index,
                                total_frames_meta=frame_count_meta,
                                time_sec=time_sec,
                                paused=paused,
                                playback_speed=playback_speed,
                            )

                            preview_state = (
                                resize_for_preview(
                                    preview_state,
                                    args.preview_width,
                                )
                            )

                            cv2.imshow(
                                "Video Localization Test",
                                preview_state,
                            )

                            if not paused:
                                proceed_to_next_frame = True
                                break

                            continue

                        # N: paused 상태에서 정확히 다음 프레임 1장
                        if key in (
                            ord("n"),
                            ord("N"),
                            ord("."),
                        ):
                            paused = True
                            manual_step_requested = True
                            proceed_to_next_frame = True
                            break

                        # ] 또는 + : 속도 증가
                        if key in (
                            ord("]"),
                            ord("+"),
                            ord("="),
                        ):
                            higher = [
                                value
                                for value
                                in speed_steps
                                if value
                                >
                                playback_speed
                                + 1e-9
                            ]

                            if higher:
                                playback_speed = min(
                                    higher
                                )

                            print(
                                f"[PLAYBACK] "
                                f"{playback_speed:.3g}x"
                            )

                            continue

                        # [ 또는 - : 속도 감소
                        if key in (
                            ord("["),
                            ord("-"),
                            ord("_"),
                        ):
                            lower = [
                                value
                                for value
                                in speed_steps
                                if value
                                <
                                playback_speed
                                - 1e-9
                            ]

                            if lower:
                                playback_speed = max(
                                    lower
                                )

                            print(
                                f"[PLAYBACK] "
                                f"{playback_speed:.3g}x"
                            )

                            continue

                        # R: 1배속
                        if key in (
                            ord("r"),
                            ord("R"),
                        ):
                            playback_speed = 1.0

                            print(
                                "[PLAYBACK] 1.0x"
                            )

                            continue

                        # S: 현재 프레임 저장
                        if key in (
                            ord("s"),
                            ord("S"),
                        ):
                            snapshot_path = (
                                save_preview_snapshot(
                                    preview_frame,
                                    output_dir,
                                    frame_index,
                                )
                            )

                            print(
                                f"[SNAPSHOT] "
                                f"{snapshot_path}"
                            )

                            continue

                        # 재생 중 다른 키는 무시하고 계속 재생.
                        if not paused:
                            continue

                    if not proceed_to_next_frame:
                        continue

                    # ------------------------------------------------
                    # REALTIME 모드:
                    # 분석 시간이 원본 프레임 간격보다 길면
                    # source frame을 grab()으로 건너뛰어
                    # 실제 재생시간에 최대한 맞춘다.
                    #
                    # 건너뛴 프레임은 분석/CSV 기록하지 않는다.
                    # ------------------------------------------------
                    frames_to_skip = 0

                    if (
                        args.preview_mode == "realtime"
                        and not paused
                        and not manual_step_requested
                    ):
                        target_frame_ms = (
                            1000.0
                            /
                            max(fps, 1.0)
                            /
                            max(
                                playback_speed,
                                0.125,
                            )
                        )

                        elapsed_ms = (
                            time.perf_counter()
                            -
                            frame_process_started
                        ) * 1000.0

                        # 현재 처리한 프레임 자체가 한 구간을 차지하므로
                        # 추가로 지나간 구간만큼만 skip.
                        frames_to_skip = max(
                            0,
                            int(
                                elapsed_ms
                                //
                                max(
                                    target_frame_ms,
                                    1.0,
                                )
                            )
                            -
                            1,
                        )

                        # 비정상적으로 큰 점프 방지
                        frames_to_skip = min(
                            frames_to_skip,
                            300,
                        )

                    actual_skipped = 0

                    for _ in range(
                        frames_to_skip
                    ):
                        if not cap.grab():
                            break

                        actual_skipped += 1

                    skipped_frames_realtime += (
                        actual_skipped
                    )

                    if actual_skipped > 0:
                        print(
                            f"[REALTIME] "
                            f"frame {frame_index}: "
                            f"skip {actual_skipped} frame(s)"
                        )

                    frame_index += (
                        1
                        +
                        actual_skipped
                    )

                else:
                    frame_index += 1

        except KeyboardInterrupt:
            print(
                "\n[STOP] 사용자 입력으로 동영상 처리를 종료합니다."
            )

        finally:
            cap.release()

            if writer is not None:
                writer.release()

            cv2.destroyAllWindows()

    # ========================================================
    # Summary
    # ========================================================
    def mean_or_none(values):
        if not values:
            return None

        return float(
            np.mean(
                np.asarray(
                    values,
                    dtype=np.float64,
                )
            )
        )

    def median_or_none(values):
        if not values:
            return None

        return float(
            np.median(
                np.asarray(
                    values,
                    dtype=np.float64,
                )
            )
        )

    summary = {
        "method": "aruco_pnp",
        "input_video": str(
            video_path.resolve()
        ),
        "calibration": (
            None
            if args.no_calibration
            else str(
                calibration_path.resolve()
            )
        ),
        "camera_model_mode": calibration_mode,
        "approx_hfov_deg": (
            float(args.hfov)
            if args.no_calibration
            else None
        ),
        "dictionary": (
            args.dictionary_name
        ),
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "metadata_frame_count": (
                frame_count_meta
            ),
            "processed_frames": (
                total_frames
            ),
            "skipped_frames_realtime": (
                skipped_frames_realtime
            ),
            "source_frames_advanced": (
                total_frames
                +
                skipped_frames_realtime
            ),
            "analysis_fraction_of_advanced_frames": (
                total_frames
                /
                (
                    total_frames
                    +
                    skipped_frames_realtime
                )
                if (
                    total_frames
                    +
                    skipped_frames_realtime
                )
                else 0.0
            ),
        },
        "settings": {
            "outlier_threshold_mm": (
                args.outlier_mm
            ),
            "ema_alpha": (
                args.ema_alpha
            ),
            "reset_after_missing_frames": (
                args.reset_after_missing
            ),
            "preview_mode": (
                args.preview_mode
            ),
            "preview_fast": (
                bool(args.preview_fast)
            ),
        },
        "coverage": {
            "frames_with_any_aruco": (
                frames_with_any_aruco
            ),
            "frames_with_registered_aruco": (
                frames_with_registered
            ),
            "frames_with_pose": (
                frames_with_fused_pose
            ),
            "aruco_detection_rate": (
                frames_with_any_aruco
                /
                total_frames
                if total_frames
                else 0.0
            ),
            "registered_detection_rate": (
                frames_with_registered
                /
                total_frames
                if total_frames
                else 0.0
            ),
            "pose_coverage_rate": (
                frames_with_fused_pose
                /
                total_frames
                if total_frames
                else 0.0
            ),
        },
        "quality": {
            "mean_reprojection_error_px": (
                mean_or_none(
                    reprojection_errors
                )
            ),
            "median_reprojection_error_px": (
                median_or_none(
                    reprojection_errors
                )
            ),
            "mean_spread_mm": (
                mean_or_none(
                    spreads
                )
            ),
            "median_spread_mm": (
                median_or_none(
                    spreads
                )
            ),
        },
        "marker_frequency": {
            "detected_id_frame_count": (
                detected_marker_frequency
            ),
            "used_point_frame_count": (
                used_marker_frequency
            ),
        },
        "resolution_warning": (
            resolution_warning
        ),
        "outputs": {
            "positions_csv": str(
                csv_path.resolve()
            ),
            "annotated_video": (
                str(
                    output_video_path.resolve()
                )
                if not args.no_video
                else None
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("VIDEO LOCALIZATION COMPLETE")
    print("=" * 72)
    print(f"Input       : {video_path}")
    print(f"Frames      : {total_frames}")
    print(
        f"Preview mode: {args.preview_mode}"
    )
    print(
        f"Skipped     : {skipped_frames_realtime}"
    )
    print(
        f"Pose frames : "
        f"{frames_with_fused_pose} "
        f"({summary['coverage']['pose_coverage_rate'] * 100:.1f}%)"
    )
    print(f"CSV         : {csv_path}")
    print(f"Summary     : {summary_path}")

    if not args.no_video:
        print(
            f"Video       : "
            f"{output_video_path}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
