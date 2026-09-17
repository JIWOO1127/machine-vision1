"""
navigation_live.py

실시간 ArUco 위치추정 + 실내 네비게이션 V7.

창 구성
-------
1. ArUco Indoor Navigation
   - 실제 웹캠 + 시선 기준 방향 카드

2. Indoor Navigation Map
   - 미니맵 + 이동경로 + 현재위치
   - 버튼 없음

3. Navigation Control
   - 시작/종료 버튼
   - 현재 안내
   - 현재 좌표
   - 최근접 문
   - Frame / FPS / 처리속도
   - 화살표 범례

키보드
------
N = 시작/재시작
X = 네비게이션 종료
S = 현재 카메라 화면 저장
Q / ESC = 프로그램 전체 종료
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

from aruco_world_map import get_marker_by_id
from camera_calibration import (
    load_calibration,
    make_approximate_calibration,
)
from coordinate_system import nearest_door_landmark
from live_map_view import LiveMapRenderer
from navigation_engine import (
    IndoorNavigator,
    PositionStabilizer,
    draw_camera_navigation_overlay,
    draw_navigation_dashboard,
    draw_navigation_map,
)
from pose_localizer import (
    camera_yaw_deg,
    estimate_camera_pose_from_marker,
    fuse_camera_poses,
)


DEFAULT_DICT = "DICT_4X4_50"


def preferred_backend() -> Optional[int]:
    if platform.system().lower() == "windows":
        return cv2.CAP_DSHOW

    return None


def open_camera(
    index: int,
    width: int,
    height: int,
):
    backend = preferred_backend()

    cap = (
        cv2.VideoCapture(index)
        if backend is None
        else cv2.VideoCapture(index, backend)
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        int(width),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        int(height),
    )

    return cap


def list_cameras(
    max_index: int = 8,
):
    found = []

    for index in range(
        int(max_index) + 1
    ):
        cap = open_camera(
            index,
            640,
            480,
        )

        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()

        if (
            ok
            and frame is not None
        ):
            h, w = frame.shape[:2]

            found.append(
                (index, w, h)
            )

            print(
                f"[FOUND] camera {index}: "
                f"{w}x{h}"
            )

        cap.release()

    return found


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
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            params,
        )

        return detector.detectMarkers

    def detect(gray):
        return cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=params,
        )

    return detect


def resize_for_preview(
    frame: np.ndarray,
    max_width: int,
) -> np.ndarray:
    if (
        max_width <= 0
        or frame.shape[1] <= max_width
    ):
        return frame

    scale = (
        float(max_width)
        /
        float(frame.shape[1])
    )

    height = max(
        1,
        int(
            round(
                frame.shape[0]
                * scale
            )
        ),
    )

    return cv2.resize(
        frame,
        (
            int(max_width),
            height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def save_frame(
    frame: np.ndarray,
    output_dir: Path,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        /
        datetime.now().strftime(
            "navigation_%Y%m%d_%H%M%S_%f.png"
        )
    )

    cv2.imwrite(
        str(path),
        frame,
    )

    return path


def point_in_rect(
    x: int,
    y: int,
    rect,
) -> bool:
    if rect is None:
        return False

    x1, y1, x2, y2 = rect

    return (
        x1 <= x <= x2
        and y1 <= y <= y2
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ArUco 기반 실시간 실내 네비게이션 V4"
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--list-cameras",
        action="store_true",
    )

    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=8,
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
            / "camera_calibration.npz"
        ),
    )

    parser.add_argument(
        "--force-approx",
        action="store_true",
    )

    parser.add_argument(
        "--hfov",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--outlier-mm",
        type=float,
        default=1500.0,
    )

    parser.add_argument(
        "--preview-width",
        type=int,
        default=1050,
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
        "--map-preview-width",
        type=int,
        default=900,
    )

    parser.add_argument(
        "--dashboard-width",
        type=int,
        default=760,
    )

    parser.add_argument(
        "--dashboard-height",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--captures",
        default="navigation_captures",
    )

    args = parser.parse_args()

    if args.list_cameras:
        list_cameras(
            args.max_camera_index
        )
        return

    cap = open_camera(
        args.camera,
        args.width,
        args.height,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"camera {args.camera}를 열 수 없습니다."
        )

    ok, first_frame = cap.read()

    if (
        not ok
        or first_frame is None
    ):
        cap.release()

        raise RuntimeError(
            "카메라 첫 프레임을 읽을 수 없습니다."
        )

    frame_h, frame_w = first_frame.shape[:2]
    calibration_path = Path(
        args.calibration
    )

    if (
        calibration_path.exists()
        and not args.force_approx
    ):
        calibration = load_calibration(
            calibration_path
        )

        calibration_text = (
            f"CALIBRATED "
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
            f"{args.hfov:.1f}"
        )

    detector = create_detector(
        args.dictionary_name
    )

    renderer = LiveMapRenderer(
        width=args.map_width,
        height=args.map_height,
    )

    stabilizer = PositionStabilizer()
    navigator = IndoorNavigator()
    captures_dir = Path(
        args.captures
    )

    camera_window = (
        "ArUco Indoor Navigation"
    )

    map_window = (
        "Indoor Navigation Map"
    )

    dashboard_window = (
        "Navigation Control"
    )

    cv2.namedWindow(
        camera_window,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        camera_window,
        int(args.preview_width),
        max(
            620,
            int(
                args.preview_width * 0.72
            ),
        ),
    )

    cv2.namedWindow(
        map_window,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        map_window,
        int(args.map_preview_width),
        max(
            300,
            int(
                args.map_preview_width * 0.40
            ),
        ),
    )

    cv2.namedWindow(
        dashboard_window,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        dashboard_window,
        int(args.dashboard_width),
        int(args.dashboard_height),
    )

    ui_state = {
        "button_rects": {},
        "latest_position": None,
        "mouse_down_button": None,
        "pressed_button": None,
        "pressed_until": 0.0,
    }

    def trigger_start():
        now = time.monotonic()

        navigator.start(
            ui_state.get(
                "latest_position"
            ),
            now=now,
        )

        print(
            "[NAV] 네비게이션 시작/재시작"
        )

    def trigger_stop():
        now = time.monotonic()

        navigator.stop(
            now=now,
        )

        print(
            "[NAV] 네비게이션 종료"
        )

    def on_dashboard_mouse(
        event,
        x,
        y,
        flags,
        param,
    ):
        rects = ui_state.get(
            "button_rects",
            {},
        )

        hit = None

        for name, rect in rects.items():
            if point_in_rect(
                x,
                y,
                rect,
            ):
                hit = name
                break

        if event == cv2.EVENT_LBUTTONDOWN:
            if hit is None:
                return

            if (
                hit == "stop"
                and not (
                    navigator.active
                    or navigator.completed
                )
            ):
                return

            ui_state[
                "mouse_down_button"
            ] = hit

            ui_state[
                "pressed_button"
            ] = hit

            ui_state[
                "pressed_until"
            ] = (
                time.monotonic()
                + 0.18
            )

            return

        if event == cv2.EVENT_LBUTTONUP:
            down = ui_state.get(
                "mouse_down_button"
            )

            ui_state[
                "mouse_down_button"
            ] = None

            if (
                down is None
                or hit != down
            ):
                return

            ui_state[
                "pressed_button"
            ] = down

            ui_state[
                "pressed_until"
            ] = (
                time.monotonic()
                + 0.18
            )

            if down == "start":
                trigger_start()

            elif down == "stop":
                trigger_stop()

    cv2.setMouseCallback(
        dashboard_window,
        on_dashboard_mouse,
    )

    print("=" * 72)
    print("ArUco Indoor Navigation V7")
    print("=" * 72)
    print(f"Camera      : {args.camera}")
    print(f"Calibration : {calibration_text}")
    print(
        "Windows     : Camera / Map / Navigation Control"
    )
    print(
        "Route       : 앞문 -> 4번 강의실 -> "
        "2번 강의실 -> 뒷문"
    )
    print(
        "Checkpoint  : 2.5 m 이내 진입 시 통과/도착"
    )
    print(
        "Keys        : N=start/restart, "
        "X=stop nav, S=capture, Q=quit"
    )
    print("=" * 72)

    last_yaw = None
    last_nearest_name = ""
    last_nearest_distance = None

    pending_frame = first_frame

    frame_number = 0
    last_loop_time = None
    display_fps_ema = 0.0

    try:
        while True:
            frame_start = (
                time.perf_counter()
            )

            loop_now = (
                time.monotonic()
            )

            if last_loop_time is not None:
                dt = max(
                    1e-6,
                    loop_now - last_loop_time,
                )

                instant_fps = (
                    1.0 / dt
                )

                if display_fps_ema <= 0.0:
                    display_fps_ema = (
                        instant_fps
                    )
                else:
                    display_fps_ema = (
                        0.88 * display_fps_ema
                        + 0.12 * instant_fps
                    )

            last_loop_time = loop_now

            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ok, frame = cap.read()

                if (
                    not ok
                    or frame is None
                ):
                    break

            frame_number += 1

            now = time.monotonic()

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            corners, ids, rejected = detector(
                gray
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

                for (
                    marker_corners,
                    marker_id,
                ) in zip(
                    corners,
                    ids_flat,
                ):
                    marker = get_marker_by_id(
                        int(marker_id)
                    )

                    if marker is None:
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

                    pts = np.asarray(
                        marker_corners,
                        dtype=np.float64,
                    ).reshape(4, 2)

                    center = np.mean(
                        pts,
                        axis=0,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"{marker.point_name} "
                            f"ID={marker.marker_id}"
                        ),
                        (
                            int(center[0]) + 6,
                            int(center[1]) - 8,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            fused = fuse_camera_poses(
                estimates,
                args.outlier_mm,
            )

            raw_pose_available = (
                fused is not None
            )

            if fused is None:
                filter_result = (
                    stabilizer.update(
                        None,
                        now,
                    )
                )

                yaw = last_yaw
            else:
                filter_result = (
                    stabilizer.update(
                        fused.position_world_mm,
                        now,
                    )
                )

                yaw = camera_yaw_deg(
                    fused.rotation_world_camera
                )

                if yaw is not None:
                    last_yaw = yaw

            display_position = (
                filter_result.position
            )

            ui_state[
                "latest_position"
            ] = (
                None
                if display_position is None
                else display_position.copy()
            )

            fresh_position = (
                raw_pose_available
                and not filter_result.stale
                and not filter_result.rejected
            )

            if fresh_position:
                guidance = navigator.update(
                    display_position,
                    yaw,
                    now=now,
                )
            else:
                guidance = navigator.update(
                    None,
                    last_yaw,
                    now=now,
                )

            if (
                fresh_position
                and display_position is not None
            ):
                landmark, distance = (
                    nearest_door_landmark(
                        float(display_position[0]),
                        float(display_position[1]),
                    )
                )

                last_nearest_name = (
                    landmark.name
                )

                last_nearest_distance = (
                    float(distance)
                )

            used_marker_names = (
                [
                    estimate.marker.point_name
                    for estimate in fused.estimates_used
                ]
                if fused is not None
                else []
            )

            map_image = renderer.render(
                position_world_mm=display_position,
                yaw_deg=last_yaw,
                spread_mm=(
                    fused.spread_mm
                    if fused is not None
                    else None
                ),
                used_marker_names=used_marker_names,
                z_mm=(
                    float(display_position[2])
                    if display_position is not None
                    else None
                ),
            )

            draw_navigation_map(
                map_image,
                renderer,
                navigator,
                guidance,
                display_position,
            )

            processing_ms = (
                (
                    time.perf_counter()
                    - frame_start
                )
                * 1000.0
            )

            processing_fps = (
                1000.0 / processing_ms
                if processing_ms > 1e-6
                else 0.0
            )

            draw_camera_navigation_overlay(
                frame,
                navigator,
                guidance,
                display_position,
                last_nearest_name,
                last_nearest_distance,
                filter_result,
                last_yaw,
                frame_number=frame_number,
                display_fps=display_fps_ema,
                processing_ms=processing_ms,
                processing_fps=processing_fps,
            )

            current_pressed = (
                ui_state["pressed_button"]
                if (
                    time.monotonic()
                    < ui_state["pressed_until"]
                )
                else None
            )

            if current_pressed is None:
                ui_state[
                    "pressed_button"
                ] = None

            dashboard_image, button_rects = (
                draw_navigation_dashboard(
                    width=args.dashboard_width,
                    height=args.dashboard_height,
                    navigator=navigator,
                    guidance=guidance,
                    position_world_mm=display_position,
                    nearest_door_name=last_nearest_name,
                    nearest_door_distance_mm=(
                        last_nearest_distance
                    ),
                    camera_yaw_deg=last_yaw,
                    frame_number=frame_number,
                    display_fps=display_fps_ema,
                    processing_ms=processing_ms,
                    processing_fps=processing_fps,
                    pressed_button=current_pressed,
                )
            )

            ui_state[
                "button_rects"
            ] = button_rects

            frame_preview = resize_for_preview(
                frame,
                args.preview_width,
            )

            map_preview = resize_for_preview(
                map_image,
                args.map_preview_width,
            )

            cv2.imshow(
                camera_window,
                frame_preview,
            )

            cv2.imshow(
                map_window,
                map_preview,
            )

            cv2.imshow(
                dashboard_window,
                dashboard_image,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key in (
                ord("n"),
                ord("N"),
            ):
                ui_state[
                    "pressed_button"
                ] = "start"

                ui_state[
                    "pressed_until"
                ] = (
                    time.monotonic()
                    + 0.18
                )

                trigger_start()

            if key in (
                ord("x"),
                ord("X"),
            ):
                ui_state[
                    "pressed_button"
                ] = "stop"

                ui_state[
                    "pressed_until"
                ] = (
                    time.monotonic()
                    + 0.18
                )

                trigger_stop()

            if key in (
                ord("s"),
                ord("S"),
            ):
                saved = save_frame(
                    frame,
                    captures_dir,
                )

                print(
                    f"[SAVED] {saved}"
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
