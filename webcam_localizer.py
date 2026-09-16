"""
webcam_localizer.py

새 웹캠용 실시간 ArUco 위치추정 + 현장 검증 화면.

기능
----
- 연결된 카메라 검색 / 선택
- ArUco ID 검출
- 최신 P01~P12 메타데이터 표시
- FACE / TOP / RIGHT 월드 방향 표시
- 마커별 카메라 월드 위치 표시
- 여러 마커 위치 융합
- 문 랜드마크 3m 판정
- camera_calibration.npz 없으면 HFOV 근사모드

키
--
S : 화면 캡처 저장
I : 마커별 상세정보 콘솔 출력
Q / ESC : 종료
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from aruco_world_map import (
    ARUCO_MARKERS_BY_ID,
    get_marker_by_id,
)
from camera_calibration import (
    load_calibration,
    make_approximate_calibration,
)
from coordinate_system import (
    describe_position,
    nearest_door_landmark,
    NEAR_LANDMARK_THRESHOLD_MM,
)
from pose_localizer import (
    estimate_camera_pose_from_marker,
    fuse_camera_poses,
    camera_yaw_deg,
)
from live_map_view import (
    LiveMapRenderer,
)


DEFAULT_DICT = "DICT_4X4_50"


def preferred_backend() -> Optional[int]:
    if platform.system().lower() == "windows":
        return cv2.CAP_DSHOW
    return None


def open_camera(
    index: int,
    width: int = 1280,
    height: int = 720,
):
    backend = preferred_backend()

    if backend is None:
        cap = cv2.VideoCapture(
            index
        )
    else:
        cap = cv2.VideoCapture(
            index,
            backend,
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        width,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        height,
    )

    return cap


def probe_camera(index: int):
    cap = open_camera(
        index,
        640,
        480,
    )

    if not cap.isOpened():
        cap.release()
        return None

    frame = None

    for _ in range(8):
        ok, frame = cap.read()

        if ok and frame is not None:
            break

        time.sleep(0.05)

    if frame is None:
        cap.release()
        return None

    h, w = frame.shape[:2]

    cap.release()

    return (
        index,
        w,
        h,
    )


def list_cameras(
    max_index: int,
):
    cameras = []

    print(
        "웹캠 검색 중..."
    )

    for index in range(
        max_index + 1
    ):
        result = probe_camera(
            index
        )

        if result is not None:
            cameras.append(
                result
            )

            print(
                f"[FOUND] camera {result[0]} "
                f"{result[1]}x{result[2]}"
            )

    return cameras


def choose_camera(
    cameras,
) -> int:

    if not cameras:
        raise RuntimeError(
            "사용 가능한 웹캠이 없습니다."
        )

    if len(cameras) == 1:
        return int(
            cameras[0][0]
        )

    valid = {
        int(item[0])
        for item in cameras
    }

    while True:
        value = input(
            f"사용할 camera 번호 {sorted(valid)}: "
        ).strip()

        try:
            index = int(value)
        except ValueError:
            continue

        if index in valid:
            return index


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


def put_line(
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
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )




def draw_location_banner(
    frame,
    text: str,
):
    """
    현재 위치 설명을 라이브 화면에서 크게 강조한다.
    전체 프레임 폭의 어두운 배경 + 밝은 테두리 + 굵은 글씨를 사용한다.
    """
    h, w = frame.shape[:2]

    top = 100
    bottom = min(
        h - 1,
        178,
    )

    if bottom <= top:
        return

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, top),
        (w - 1, bottom),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (2, top + 2),
        (w - 3, bottom - 2),
        (0, 255, 255),
        3,
    )

    # 그림자를 먼저 그려서 작은 노트북 화면에서도 눈에 띄게 한다.
    cv2.putText(
        frame,
        text,
        (17, top + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (14, top + 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )


def save_frame(
    frame,
    output_dir: Path,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        datetime.now()
        .strftime(
            "localizer_%Y%m%d_%H%M%S_%f.png"
        )
    )

    path = (
        output_dir
        /
        filename
    )

    cv2.imwrite(
        str(path),
        frame,
    )

    return path



def resize_for_preview(
    frame: np.ndarray,
    max_width: int,
) -> np.ndarray:
    """
    위치 계산은 원본 프레임으로 유지하고,
    화면 표시용 영상만 max_width에 맞춰 축소한다.
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--list-cameras",
        action="store_true",
    )

    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--dict",
        dest="dictionary_name",
        default=DEFAULT_DICT,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
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
        "--hfov",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--force-approx",
        action="store_true",
    )

    parser.add_argument(
        "--outlier-mm",
        type=float,
        default=1500.0,
    )

    parser.add_argument(
        "--captures",
        default="localizer_captures",
    )

    parser.add_argument(
        "--no-map",
        action="store_true",
        help="실시간 XY 지도 창을 끕니다.",
    )

    parser.add_argument(
        "--map-width",
        type=int,
        default=1100,
    )

    parser.add_argument(
        "--map-height",
        type=int,
        default=420,
    )

    parser.add_argument(
        "--preview-width",
        type=int,
        default=960,
        help=(
            "라이브 카메라 미리보기 최대 폭(px). "
            "기본 960. 위치 계산 해상도에는 영향 없음."
        ),
    )

    parser.add_argument(
        "--map-preview-width",
        type=int,
        default=900,
        help=(
            "분리 창 모드에서 사용할 미니맵 최대 폭(px)."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------
    if (
        args.camera is None
        or
        args.list_cameras
    ):
        cameras = list_cameras(
            args.max_camera_index
        )

        if args.list_cameras:
            return

        camera_index = choose_camera(
            cameras
        )
    else:
        camera_index = int(
            args.camera
        )

    cap = open_camera(
        camera_index,
        args.width,
        args.height,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"camera {camera_index}를 "
            "열 수 없습니다."
        )

    ok, first_frame = cap.read()

    if not ok:
        cap.release()

        raise RuntimeError(
            "웹캠 첫 프레임을 "
            "읽을 수 없습니다."
        )

    frame_h, frame_w = (
        first_frame.shape[:2]
    )

    calibration_path = Path(
        args.calibration
    )

    if (
        calibration_path.exists()
        and
        not args.force_approx
    ):
        calibration = (
            load_calibration(
                calibration_path
            )
        )

        calibration_text = (
            f"CALIBRATED: "
            f"{calibration_path.name}"
        )
    else:
        calibration = (
            make_approximate_calibration(
                frame_w,
                frame_h,
                args.hfov,
            )
        )

        calibration_text = (
            f"APPROX HFOV="
            f"{args.hfov:.1f}deg"
        )

    detect = create_detector(
        args.dictionary_name
    )

    output_dir = Path(
        args.captures
    )

    map_renderer = (
        None
        if args.no_map
        else LiveMapRenderer(
            width=args.map_width,
            height=args.map_height,
        )
    )

    print(
        "=" * 72
    )
    print(
        f"Camera     : {camera_index}"
    )
    print(
        f"Dictionary : "
        f"{args.dictionary_name}"
    )
    print(
        f"Calibration: "
        f"{calibration_text}"
    )
    print(
        "S=capture / I=marker info / Q=quit"
    )

    if not args.no_map:
        print(
            "Indoor XY Map 창에서 현재 카메라 위치(빨간 점)와 "
            "방향(빨간 화살표)을 표시합니다."
        )
    print(
        "=" * 72
    )
    print(
        f"Display mode: separate windows "
        f"(camera={args.preview_width}px, "
        f"map={args.map_preview_width}px)"
    )
    print(
        "카메라 화면과 미니맵은 서로 독립된 창으로 표시됩니다."
    )
    print(
        "표시 화면만 축소되며 ArUco/PnP 계산은 원본 프레임으로 수행됩니다."
    )

    # --------------------------------------------------------
    # 라이브 표시 창
    # 카메라 화면과 미니맵을 서로 독립된 창으로 표시한다.
    # 각 창은 표시용으로만 축소되며 위치 계산은 원본 프레임 사용.
    # --------------------------------------------------------
    cv2.namedWindow(
        "ArUco Indoor Localizer V2",
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        "ArUco Indoor Localizer V2",
        int(args.preview_width),
        max(
            420,
            int(args.preview_width * 0.58),
        ),
    )

    if map_renderer is not None:
        cv2.namedWindow(
            "Indoor XY Map",
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            "Indoor XY Map",
            int(args.map_preview_width),
            max(
                260,
                int(args.map_preview_width * 0.34),
            ),
        )

    last_estimates = []

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            corners, ids, rejected = (
                detect(gray)
            )

            estimates = []

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    frame,
                    corners,
                    ids,
                )

                ids_flat = (
                    np.asarray(ids)
                    .reshape(-1)
                    .astype(int)
                )

                for marker_corners, marker_id in zip(
                    corners,
                    ids_flat,
                ):
                    marker = get_marker_by_id(
                        int(marker_id)
                    )

                    pts = (
                        np.asarray(
                            marker_corners
                        )
                        .reshape(4, 2)
                    )

                    center = np.mean(
                        pts,
                        axis=0,
                    )

                    cx = int(
                        round(
                            float(center[0])
                        )
                    )

                    cy = int(
                        round(
                            float(center[1])
                        )
                    )

                    if marker is None:
                        put_line(
                            frame,
                            f"ID {marker_id} / UNREGISTERED",
                            cx + 5,
                            cy - 12,
                            color=(0, 165, 255),
                            thickness=2,
                        )
                        continue

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

                    # 등록된 현장 메타데이터 표시
                    put_line(
                        frame,
                        (
                            f"{marker.point_name} "
                            f"ID={marker.marker_id} "
                            f"{marker.size_mm:.0f}mm"
                        ),
                        cx + 5,
                        cy - 34,
                        color=(0, 255, 255),
                        thickness=2,
                    )

                    put_line(
                        frame,
                        (
                            f"FACE {marker.face_world} | "
                            f"TOP {marker.top_world} | "
                            f"RIGHT {marker.right_world}"
                        ),
                        cx + 5,
                        cy - 12,
                        color=(0, 255, 0),
                        thickness=2,
                    )

                    p = (
                        estimate
                        .position_world_mm
                    )

                    put_line(
                        frame,
                        (
                            f"cam=({p[0]:.0f},"
                            f"{p[1]:.0f},"
                            f"{p[2]:.0f})"
                        ),
                        cx + 5,
                        cy + 14,
                        color=(255, 255, 255),
                        thickness=1,
                    )

                    # local marker axis 표시.
                    try:
                        cv2.drawFrameAxes(
                            frame,
                            calibration.camera_matrix,
                            calibration.dist_coeffs,
                            estimate.rvec_marker_to_camera,
                            estimate.tvec_marker_to_camera,
                            marker.size_mm * 0.6,
                            2,
                        )
                    except Exception:
                        pass

            fused = fuse_camera_poses(
                estimates,
                args.outlier_mm,
            )

            # ------------------------------------------------
            # top overlay
            # ------------------------------------------------
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (
                    min(
                        frame.shape[1],
                        950,
                    ),
                    170,
                ),
                (0, 0, 0),
                -1,
            )

            cv2.addWeighted(
                overlay,
                0.58,
                frame,
                0.42,
                0,
                frame,
            )

            put_line(
                frame,
                (
                    f"Camera {camera_index} | "
                    f"{args.dictionary_name} | "
                    f"{calibration_text}"
                ),
                12,
                27,
                scale=0.62,
            )

            if fused is None:
                put_line(
                    frame,
                    "REGISTERED ArUco marker not detected",
                    12,
                    58,
                    scale=0.65,
                    color=(0, 165, 255),
                    thickness=2,
                )
            else:
                p = (
                    fused
                    .position_world_mm
                )

                put_line(
                    frame,
                    (
                        f"WORLD camera: "
                        f"x={p[0]:.0f} "
                        f"y={p[1]:.0f} "
                        f"z={p[2]:.0f} mm"
                    ),
                    12,
                    58,
                    scale=0.68,
                    color=(0, 255, 0),
                    thickness=2,
                )

                yaw = camera_yaw_deg(
                    fused
                    .rotation_world_camera
                )

                yaw_text = (
                    "n/a"
                    if yaw is None
                    else f"{yaw:+.1f} deg"
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

                put_line(
                    frame,
                    (
                        f"used={used_ids} "
                        f"rejected={rejected_ids} "
                        f"spread={fused.spread_mm:.0f}mm "
                        f"yaw={yaw_text}"
                    ),
                    12,
                    88,
                    scale=0.52,
                )

                description = (
                    describe_position(
                        float(p[0]),
                        float(p[1]),
                        float(p[2]),
                    )
                )

                landmark, landmark_distance = (
                    nearest_door_landmark(
                        float(p[0]),
                        float(p[1]),
                    )
                )

                if (
                    landmark_distance
                    <=
                    NEAR_LANDMARK_THRESHOLD_MM
                ):
                    location_banner_text = (
                        f"{landmark.name} 주변입니다."
                    )
                else:
                    location_banner_text = (
                        "등록된 문 랜드마크에서 3m보다 멉니다."
                    )

                # 위치 판정 문구를 큰 배너로 표시.
                draw_location_banner(
                    frame,
                    location_banner_text,
                )

                # 상세 좌표/최근접 랜드마크 정보는 작은 글씨로 별도 유지.
                put_line(
                    frame,
                    description[:120],
                    12,
                    202,
                    scale=0.46,
                    color=(230, 230, 230),
                    thickness=1,
                )

                if calibration.approximate:
                    put_line(
                        frame,
                        (
                            "WARNING: approximate camera model; "
                            "position is for field checking, not final accuracy."
                        ),
                        12,
                        228,
                        scale=0.48,
                        color=(0, 165, 255),
                        thickness=2,
                    )

            if map_renderer is not None:
                if fused is None:
                    map_image = map_renderer.render()
                else:
                    map_yaw = camera_yaw_deg(
                        fused.rotation_world_camera
                    )

                    used_marker_names = [
                        estimate.marker.point_name
                        for estimate
                        in fused.estimates_used
                    ]

                    map_image = map_renderer.render(
                        position_world_mm=(
                            fused.position_world_mm
                        ),
                        yaw_deg=map_yaw,
                        spread_mm=fused.spread_mm,
                        used_marker_names=used_marker_names,
                        z_mm=fused.z,
                    )

                map_preview = resize_for_preview(
                    map_image,
                    args.map_preview_width,
                )

                cv2.imshow(
                    "Indoor XY Map",
                    map_preview,
                )

            frame_preview = resize_for_preview(
                frame,
                args.preview_width,
            )

            cv2.imshow(
                "ArUco Indoor Localizer V2",
                frame_preview,
            )

            last_estimates = estimates

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key in (
                ord("s"),
                ord("S"),
            ):
                path = save_frame(
                    frame,
                    output_dir,
                )
                print(
                    f"[SAVED] {path}"
                )

            if key in (
                ord("i"),
                ord("I"),
            ):
                print()
                print(
                    "현재 검출 마커별 결과"
                )
                print(
                    "-" * 72
                )

                if not last_estimates:
                    print(
                        "등록된 마커 없음"
                    )

                for estimate in last_estimates:
                    m = estimate.marker
                    p = (
                        estimate
                        .position_world_mm
                    )

                    print(
                        f"{m.point_name} "
                        f"ID={m.marker_id}, "
                        f"size={m.size_mm:.0f}mm, "
                        f"FACE={m.face_world}, "
                        f"TOP={m.top_world}, "
                        f"RIGHT={m.right_world}, "
                        f"cam=({p[0]:.1f}, "
                        f"{p[1]:.1f}, "
                        f"{p[2]:.1f}), "
                        f"reproj="
                        f"{estimate.reprojection_error_px:.2f}px"
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
