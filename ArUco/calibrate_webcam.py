"""
calibrate_webcam.py

체스보드 패턴을 이용해 현재 사용할 웹캠의 내부 파라미터를 계산하고
camera_calibration.npz로 저장한다.

기본값:
- checkerboard inner corners: 9 x 6
- square size: 25 mm
- camera: 자동 검색/선택
- requested resolution: 1280 x 720

키:
    SPACE : 현재 체스보드 프레임을 샘플로 저장
    C     : 8장 이상 모였을 때 캘리브레이션 실행/저장
    Q/ESC : 종료

예:
    python calibrate_webcam.py
    python calibrate_webcam.py --camera 1 --cols 9 --rows 6 --square-mm 25
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
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, backend)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    return cap


def probe_camera(index: int):
    cap = open_camera(index, 640, 480)

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

    return index, w, h


def list_cameras(max_index: int = 10):
    cameras = []

    print("웹캠 검색 중...")

    for index in range(max_index + 1):
        result = probe_camera(index)

        if result is not None:
            cameras.append(result)
            print(
                f"[FOUND] camera {result[0]} "
                f"{result[1]}x{result[2]}"
            )

    return cameras


def choose_camera(cameras) -> int:
    if not cameras:
        raise RuntimeError("사용 가능한 웹캠이 없습니다.")

    if len(cameras) == 1:
        index = int(cameras[0][0])
        print(f"camera {index}를 사용합니다.")
        return index

    valid = {
        int(camera[0])
        for camera in cameras
    }

    while True:
        raw = input(
            f"사용할 camera 번호 {sorted(valid)}: "
        ).strip()

        try:
            index = int(raw)
        except ValueError:
            continue

        if index in valid:
            return index


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="생략하면 연결된 카메라를 검색 후 선택",
    )

    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="체스보드 내부 코너 열 개수",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="체스보드 내부 코너 행 개수",
    )

    parser.add_argument(
        "--square-mm",
        type=float,
        default=25.0,
        help="체스보드 한 칸의 실제 크기(mm)",
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
        "--output",
        default=None,
        help=(
            "저장 경로. 생략하면 이 스크립트와 같은 폴더의 "
            "camera_calibration.npz"
        ),
    )

    parser.add_argument(
        "--samples-dir",
        default=None,
        help=(
            "SPACE로 채택한 체스보드 샘플 이미지를 저장할 폴더. "
            "생략하면 스크립트 폴더/calibration_samples"
        ),
    )

    args = parser.parse_args()

    if args.camera is None:
        cameras = list_cameras(
            args.max_camera_index
        )

        camera_index = choose_camera(
            cameras
        )
    else:
        camera_index = int(
            args.camera
        )

    if args.output is None:
        output_path = (
            Path(__file__).resolve().parent
            /
            "camera_calibration.npz"
        )
    else:
        output_path = Path(
            args.output
        )

    if args.samples_dir is None:
        samples_dir = (
            Path(__file__).resolve().parent
            /
            "calibration_samples"
        )
    else:
        samples_dir = Path(
            args.samples_dir
        )

    samples_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    board_size = (
        int(args.cols),
        int(args.rows),
    )

    square_mm = float(
        args.square_mm
    )

    # 3D 체스보드 기준점.
    object_template = np.zeros(
        (
            args.rows
            *
            args.cols,
            3,
        ),
        dtype=np.float32,
    )

    object_template[:, :2] = (
        np.mgrid[
            0:args.cols,
            0:args.rows
        ]
        .T
        .reshape(-1, 2)
        *
        square_mm
    )

    object_points = []
    image_points = []

    cap = open_camera(
        camera_index,
        args.width,
        args.height,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"camera {camera_index}를 열 수 없습니다."
        )

    print()
    print("=" * 72)
    print("Webcam Calibration")
    print("=" * 72)
    print(f"Camera      : {camera_index}")
    print(
        f"Checkerboard: "
        f"{args.cols} x {args.rows} INNER CORNERS"
    )
    print(
        f"Square size : {square_mm:.2f} mm"
    )
    print(f"Output      : {output_path}")
    print(f"Samples dir : {samples_dir}")
    print()
    print("SPACE = 샘플 저장")
    print("C     = 캘리브레이션 계산/저장 (최소 8장)")
    print("Q/ESC = 종료")
    print()
    print(
        "15~30장 정도를 권장합니다. "
        "중앙/모서리, 가까이/멀리, 기울어진 각도를 섞으세요."
    )
    print("=" * 72)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    last_gray = None
    last_corners = None
    last_found = False
    image_size = None

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("웹캠 프레임 읽기 실패")
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            image_size = (
                gray.shape[1],
                gray.shape[0],
            )

            found, corners = (
                cv2.findChessboardCorners(
                    gray,
                    board_size,
                    flags=(
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        |
                        cv2.CALIB_CB_NORMALIZE_IMAGE
                    ),
                )
            )

            display = frame.copy()

            if found:
                refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria,
                )

                cv2.drawChessboardCorners(
                    display,
                    board_size,
                    refined,
                    True,
                )

                last_corners = refined
                last_found = True
                last_gray = gray
            else:
                last_corners = None
                last_found = False
                last_gray = gray

            cv2.putText(
                display,
                (
                    f"camera {camera_index} | "
                    f"samples={len(image_points)} | "
                    f"board={'FOUND' if found else 'NOT FOUND'}"
                ),
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if found else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "SPACE=save sample   C=calibrate   Q=quit",
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Webcam Calibration",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key == 32:  # SPACE
                if (
                    last_found
                    and
                    last_corners is not None
                ):
                    object_points.append(
                        object_template.copy()
                    )

                    image_points.append(
                        last_corners.copy()
                    )

                    sample_no = len(image_points)
                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )

                    sample_path = (
                        samples_dir
                        /
                        f"sample_{sample_no:02d}_{timestamp}.png"
                    )

                    sample_image = frame.copy()

                    cv2.drawChessboardCorners(
                        sample_image,
                        board_size,
                        last_corners,
                        True,
                    )

                    cv2.imwrite(
                        str(sample_path),
                        sample_image,
                    )

                    print(
                        f"[SAMPLE] {sample_no}장 저장 | "
                        f"{sample_path}"
                    )
                else:
                    print(
                        "[SKIP] 체스보드가 검출되지 않았습니다."
                    )

            if key in (
                ord("c"),
                ord("C"),
            ):
                if len(image_points) < 8:
                    print(
                        "[WAIT] 최소 8장의 샘플이 필요합니다. "
                        f"현재 {len(image_points)}장"
                    )
                    continue

                if image_size is None:
                    print(
                        "[ERROR] 영상 크기를 알 수 없습니다."
                    )
                    continue

                (
                    rms,
                    camera_matrix,
                    dist_coeffs,
                    rvecs,
                    tvecs,
                ) = cv2.calibrateCamera(
                    object_points,
                    image_points,
                    image_size,
                    None,
                    None,
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                np.savez(
                    output_path,
                    camera_matrix=np.asarray(
                        camera_matrix,
                        dtype=np.float64,
                    ),
                    dist_coeffs=np.asarray(
                        dist_coeffs,
                        dtype=np.float64,
                    ),
                    image_width=int(
                        image_size[0]
                    ),
                    image_height=int(
                        image_size[1]
                    ),
                    rms=float(rms),
                    camera_index=int(
                        camera_index
                    ),
                    checkerboard_cols=int(
                        args.cols
                    ),
                    checkerboard_rows=int(
                        args.rows
                    ),
                    square_mm=float(
                        square_mm
                    ),
                )

                print()
                print("=" * 72)
                print("CALIBRATION COMPLETE")
                print("=" * 72)
                print(f"RMS error : {rms:.6f}")
                print(f"Resolution: {image_size[0]}x{image_size[1]}")
                print("Camera matrix:")
                print(camera_matrix)
                print("Distortion:")
                print(dist_coeffs.ravel())
                print(f"Saved: {output_path}")
                print(
                    "파일 존재 확인: "
                    f"{output_path.exists()}"
                )
                print("=" * 72)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
