"""
image_localizer.py

정지 이미지 한 장에서 ArUco 마커를 검출하고 카메라의 월드 위치를 추정한다.

주요 기능
---------
- ArUco 검출
- 등록된 마커별 solvePnP 위치 계산
- 다중 마커 이상치 제거 + 가중 평균
- Best Single 결과 계산
- 최종 XYZ / Yaw / landmark 주변 판정
- annotated image / mini map / JSON 저장
- calibration 사용 또는 HFOV 기반 근사 카메라 모델 선택

실행 예
-------
# 정식 캘리브레이션 사용
python image_localizer.py test.jpg --preview

# 촬영 장비가 불명확한 이미지
python image_localizer.py test.jpg --no-calibration --preview

# 근사 수평 화각을 70도로 지정
python image_localizer.py test.jpg --no-calibration --hfov 70 --preview

# 출력 폴더 지정
python image_localizer.py test.jpg --no-calibration --output-dir results/image_001
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from aruco_world_map import get_marker_by_id
from camera_calibration import (
    load_calibration,
    make_approximate_calibration,
)
from coordinate_system import (
    NEAR_LANDMARK_THRESHOLD_MM,
    describe_position,
    nearest_door_landmark,
)
from live_map_view import LiveMapRenderer
from pose_localizer import (
    CameraPoseEstimate,
    camera_yaw_deg,
    estimate_camera_pose_from_marker,
    fuse_camera_poses,
)


DEFAULT_DICT = "DICT_4X4_50"


def read_image_unicode(path: Path) -> Optional[np.ndarray]:
    """
    Windows 한글/공백 경로에서도 비교적 안전하게 이미지를 읽는다.
    """
    try:
        data = np.fromfile(
            str(path),
            dtype=np.uint8,
        )
    except OSError:
        return None

    if data.size == 0:
        return None

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )


def write_image_unicode(
    path: Path,
    image: np.ndarray,
) -> None:
    """
    Windows 한글/공백 경로에서도 비교적 안전하게 이미지를 저장한다.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        path.suffix.lower()
        if path.suffix
        else ".png"
    )

    if suffix not in (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ):
        suffix = ".png"

    ok, encoded = cv2.imencode(
        suffix,
        image,
    )

    if not ok:
        raise RuntimeError(
            f"이미지 인코딩 실패: {path}"
        )

    encoded.tofile(
        str(path)
    )


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


def put_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.58,
    color=(255, 255, 255),
    thickness: int = 1,
) -> None:
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


def draw_info_panel(
    frame: np.ndarray,
    lines,
) -> None:
    """
    OpenCV 기본 글꼴은 한글을 제대로 그리지 못하므로
    이미지 overlay에는 ASCII 중심 정보를 표시하고,
    한국어 판정은 콘솔/JSON에도 저장한다.
    """
    if not lines:
        return

    panel_h = min(
        frame.shape[0],
        32 + 29 * len(lines),
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (
            min(
                frame.shape[1] - 1,
                1050,
            ),
            panel_h,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.67,
        frame,
        0.33,
        0,
        frame,
    )

    y = 27

    for text, color, thickness in lines:
        put_text(
            frame,
            text,
            12,
            y,
            scale=0.58,
            color=color,
            thickness=thickness,
        )
        y += 29


def fit_for_preview(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    h, w = image.shape[:2]

    scale = min(
        1.0,
        float(max_width) / float(w),
        float(max_height) / float(h),
    )

    if scale >= 0.999:
        return image

    return cv2.resize(
        image,
        (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


def vec3_or_none(
    value,
):
    if value is None:
        return None

    value = np.asarray(
        value,
        dtype=np.float64,
    ).reshape(3)

    return [
        float(value[0]),
        float(value[1]),
        float(value[2]),
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "정지 이미지 한 장에서 ArUco 기반 "
            "카메라 위치를 판정합니다."
        )
    )

    parser.add_argument(
        "image",
        help="입력 이미지 경로",
    )

    parser.add_argument(
        "--calibration",
        default=str(
            Path(__file__).resolve().parent
            /
            "camera_calibration.npz"
        ),
        help="camera_calibration.npz 경로",
    )

    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help=(
            "캘리브레이션 파일을 사용하지 않고 "
            "HFOV 기반 근사 카메라 모델 사용"
        ),
    )

    parser.add_argument(
        "--hfov",
        type=float,
        default=60.0,
        help=(
            "근사 카메라 모델의 수평 화각(deg). "
            "기본 60"
        ),
    )

    parser.add_argument(
        "--dict",
        dest="dictionary_name",
        default=DEFAULT_DICT,
        help="ArUco dictionary. 기본 DICT_4X4_50",
    )

    parser.add_argument(
        "--outlier-mm",
        type=float,
        default=1500.0,
        help="다중 마커 이상치 제거 기준(mm)",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "결과 폴더. 생략하면 "
            "<이미지명>_image_result"
        ),
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="판정 결과 이미지와 미니맵을 화면에 표시",
    )

    parser.add_argument(
        "--preview-width",
        type=int,
        default=900,
        help="결과 이미지 미리보기 최대 폭",
    )

    parser.add_argument(
        "--preview-height",
        type=int,
        default=650,
        help="결과 이미지 미리보기 최대 높이",
    )

    parser.add_argument(
        "--no-map",
        action="store_true",
        help="미니맵 생성/표시 안 함",
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다: "
            f"{image_path}"
        )

    frame = read_image_unicode(
        image_path
    )

    if frame is None:
        raise RuntimeError(
            f"이미지를 읽을 수 없습니다: "
            f"{image_path}"
        )

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Camera model
    # --------------------------------------------------------
    calibration_path = Path(
        args.calibration
    )

    resolution_warning = None

    if args.no_calibration:
        calibration = (
            make_approximate_calibration(
                image_width=width,
                image_height=height,
                horizontal_fov_deg=args.hfov,
            )
        )

        camera_model_mode = (
            f"APPROX_HFOV_{args.hfov:.1f}"
        )

    else:
        if not calibration_path.exists():
            raise FileNotFoundError(
                "캘리브레이션 파일이 없습니다: "
                f"{calibration_path}\n"
                "촬영 장비가 다르거나 불명확하다면 "
                "--no-calibration 옵션을 사용하세요."
            )

        calibration = load_calibration(
            calibration_path
        )

        camera_model_mode = (
            f"CALIBRATED_{calibration_path.name}"
        )

        if (
            calibration.image_width > 0
            and calibration.image_height > 0
            and (
                calibration.image_width != width
                or calibration.image_height != height
            )
        ):
            resolution_warning = (
                "Calibration resolution "
                f"{calibration.image_width}x"
                f"{calibration.image_height} "
                "!= image resolution "
                f"{width}x{height}"
            )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    if args.output_dir:
        output_dir = Path(
            args.output_dir
        )
    else:
        output_dir = (
            image_path.parent
            /
            f"{image_path.stem}_image_result"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotated_path = (
        output_dir
        /
        "annotated.png"
    )

    map_path = (
        output_dir
        /
        "map.png"
    )

    json_path = (
        output_dir
        /
        "result.json"
    )

    # --------------------------------------------------------
    # ArUco detection
    # --------------------------------------------------------
    detect = create_detector(
        args.dictionary_name
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    corners, ids, rejected = detect(
        gray
    )

    annotated = frame.copy()

    detected_ids = []
    registered_ids = []
    estimates: list[
        CameraPoseEstimate
    ] = []

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(
            annotated,
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
            marker_id = int(
                marker_id
            )

            detected_ids.append(
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
                    float(center[0])
                )
            )

            cy = int(
                round(
                    float(center[1])
                )
            )

            marker = get_marker_by_id(
                marker_id
            )

            if marker is None:
                put_text(
                    annotated,
                    f"ID {marker_id} UNREGISTERED",
                    cx + 6,
                    max(24, cy - 8),
                    scale=0.5,
                    color=(0, 165, 255),
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
                put_text(
                    annotated,
                    f"{marker.point_name} PnP FAIL",
                    cx + 6,
                    max(24, cy - 8),
                    scale=0.5,
                    color=(0, 0, 255),
                    thickness=2,
                )
                continue

            estimates.append(
                estimate
            )

            p = (
                estimate
                .position_world_mm
            )

            put_text(
                annotated,
                (
                    f"{marker.point_name} ID={marker_id} "
                    f"XYZ=({p[0]:.0f},"
                    f"{p[1]:.0f},"
                    f"{p[2]:.0f})"
                ),
                cx + 6,
                max(24, cy - 8),
                scale=0.47,
                color=(0, 255, 255),
                thickness=2,
            )

            try:
                cv2.drawFrameAxes(
                    annotated,
                    calibration.camera_matrix,
                    calibration.dist_coeffs,
                    estimate.rvec_marker_to_camera,
                    estimate.tvec_marker_to_camera,
                    float(marker.size_mm) * 0.5,
                    2,
                )
            except Exception:
                # OpenCV 버전 차이 등으로 축 그리기가 실패해도
                # 위치 판정 자체는 계속한다.
                pass

    # --------------------------------------------------------
    # Best single + weighted fusion
    # --------------------------------------------------------
    best = (
        max(
            estimates,
            key=lambda e: e.weight,
        )
        if estimates
        else None
    )

    fused = fuse_camera_poses(
        estimates,
        args.outlier_mm,
    )

    final_position = None
    yaw = None
    location_text_ko = (
        "유효한 등록 ArUco Pose가 없습니다."
    )
    location_text_ascii = (
        "NO VALID REGISTERED ARUCO POSE"
    )
    nearest_name = None
    nearest_distance = None
    description = None
    used_ids = []
    rejected_ids = []

    if fused is not None:
        final_position = (
            fused.position_world_mm
        )

        yaw = camera_yaw_deg(
            fused.rotation_world_camera
        )

        x, y, z = (
            float(final_position[0]),
            float(final_position[1]),
            float(final_position[2]),
        )

        landmark, nearest_distance = (
            nearest_door_landmark(
                x,
                y,
            )
        )

        nearest_name = (
            landmark.name
        )

        if (
            nearest_distance
            <=
            NEAR_LANDMARK_THRESHOLD_MM
        ):
            location_text_ko = (
                f"{landmark.name} 주변입니다."
            )

            ascii_alias = {
                "4번강의실": "C4",
                "3번강의실": "C3",
                "2번강의실": "C2",
                "쪽문": "SIDE DOOR",
                "앞문": "FRONT DOOR",
            }.get(
                landmark.name,
                "LANDMARK",
            )

            location_text_ascii = (
                f"NEAR {ascii_alias}"
            )
        else:
            location_text_ko = (
                f"가장 가까운 문은 {landmark.name}이며 "
                f"약 {nearest_distance / 1000.0:.2f} m 떨어져 있습니다."
            )

            location_text_ascii = (
                f"NEAREST DOOR: {landmark.name}"
            )

        description = describe_position(
            x,
            y,
            z,
        )

        used_ids = [
            int(e.marker.marker_id)
            for e in fused.estimates_used
        ]

        rejected_ids = [
            int(e.marker.marker_id)
            for e in fused.estimates_rejected
        ]

    # --------------------------------------------------------
    # Annotated info panel
    # --------------------------------------------------------
    lines = [
        (
            (
                f"Image: {image_path.name}  "
                f"{width}x{height}"
            ),
            (255, 255, 255),
            1,
        ),
        (
            (
                f"Detected IDs: {detected_ids}  "
                f"Registered: {registered_ids}"
            ),
            (255, 255, 255),
            1,
        ),
    ]

    if best is not None:
        bp = best.position_world_mm
        lines.append(
            (
                (
                    f"Best Single: {best.marker.point_name} "
                    f"XYZ=({bp[0]:.0f}, {bp[1]:.0f}, {bp[2]:.0f}) "
                    f"err={best.reprojection_error_px:.2f}px"
                ),
                (0, 255, 255),
                2,
            )
        )

    if fused is not None:
        fp = fused.position_world_mm
        yaw_text = (
            f"{yaw:.1f}"
            if yaw is not None
            else "N/A"
        )

        lines.append(
            (
                (
                    f"FUSED XYZ=({fp[0]:.0f}, "
                    f"{fp[1]:.0f}, {fp[2]:.0f}) "
                    f"yaw={yaw_text}deg "
                    f"spread={fused.spread_mm:.0f}mm"
                ),
                (0, 255, 0),
                2,
            )
        )

        lines.append(
            (
                location_text_ascii,
                (0, 255, 255),
                3,
            )
        )
        lines.append(
            (
                (
                    f"distance={nearest_distance:.0f} mm "
                    f"({nearest_distance/1000.0:.2f} m) | "
                    f"Z={fp[2]:.0f} mm"
                ),
                (150, 255, 150),
                2,
            )
        )
    else:
        lines.append(
            (
                location_text_ascii,
                (0, 0, 255),
                3,
            )
        )

    if resolution_warning:
        lines.append(
            (
                "WARNING: calibration/image resolution mismatch",
                (0, 165, 255),
                2,
            )
        )

    if args.no_calibration:
        lines.append(
            (
                (
                    f"APPROX CAMERA MODEL: "
                    f"HFOV={args.hfov:.1f}deg"
                ),
                (0, 165, 255),
                2,
            )
        )

    draw_info_panel(
        annotated,
        lines,
    )

    # --------------------------------------------------------
    # Mini map
    # --------------------------------------------------------
    map_image = None

    if not args.no_map:
        renderer = LiveMapRenderer(
            width=1100,
            height=420,
        )

        map_image = renderer.render(
            position_world_mm=final_position,
            yaw_deg=yaw,
            spread_mm=(
                fused.spread_mm
                if fused is not None
                else None
            ),
            used_marker_names=(
                [
                    e.marker.point_name
                    for e in fused.estimates_used
                ]
                if fused is not None
                else []
            ),
            z_mm=(
                float(final_position[2])
                if final_position is not None
                else None
            ),
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    write_image_unicode(
        annotated_path,
        annotated,
    )

    if map_image is not None:
        write_image_unicode(
            map_path,
            map_image,
        )

    individual_results = []

    for e in estimates:
        individual_results.append(
            {
                "point_name": (
                    e.marker.point_name
                ),
                "marker_id": int(
                    e.marker.marker_id
                ),
                "position_world_mm": (
                    vec3_or_none(
                        e.position_world_mm
                    )
                ),
                "reprojection_error_px": float(
                    e.reprojection_error_px
                ),
                "marker_area_px2": float(
                    e.marker_area_px2
                ),
                "weight": float(
                    e.weight
                ),
            }
        )

    result = {
        "input_image": str(
            image_path.resolve()
        ),
        "image_size": {
            "width": width,
            "height": height,
        },
        "dictionary": (
            args.dictionary_name
        ),
        "camera_model": {
            "mode": camera_model_mode,
            "calibration_file": (
                None
                if args.no_calibration
                else str(
                    calibration_path.resolve()
                )
            ),
            "approx_hfov_deg": (
                float(args.hfov)
                if args.no_calibration
                else None
            ),
            "resolution_warning": (
                resolution_warning
            ),
        },
        "detected_ids": (
            detected_ids
        ),
        "registered_ids": (
            registered_ids
        ),
        "individual_estimates": (
            individual_results
        ),
        "best_single": (
            {
                "point_name": (
                    best.marker.point_name
                ),
                "marker_id": int(
                    best.marker.marker_id
                ),
                "position_world_mm": (
                    vec3_or_none(
                        best.position_world_mm
                    )
                ),
                "reprojection_error_px": float(
                    best.reprojection_error_px
                ),
                "weight": float(
                    best.weight
                ),
            }
            if best is not None
            else None
        ),
        "fused": (
            {
                "position_world_mm": (
                    vec3_or_none(
                        fused.position_world_mm
                    )
                ),
                "yaw_deg": (
                    float(yaw)
                    if yaw is not None
                    else None
                ),
                "spread_mm": float(
                    fused.spread_mm
                ),
                "used_marker_ids": (
                    used_ids
                ),
                "rejected_marker_ids": (
                    rejected_ids
                ),
            }
            if fused is not None
            else None
        ),
        "location": {
            "text_ko": (
                location_text_ko
            ),
            "nearest_landmark": (
                nearest_name
            ),
            "nearest_landmark_distance_mm": (
                float(nearest_distance)
                if nearest_distance is not None
                else None
            ),
            "description": (
                description
            ),
        },
        "outputs": {
            "annotated_image": str(
                annotated_path.resolve()
            ),
            "map_image": (
                str(map_path.resolve())
                if map_image is not None
                else None
            ),
            "result_json": str(
                json_path.resolve()
            ),
        },
    }

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("IMAGE LOCALIZATION RESULT")
    print("=" * 72)
    print(f"Image       : {image_path}")
    print(f"Resolution  : {width}x{height}")
    print(f"Camera model: {camera_model_mode}")
    print(f"Detected IDs: {detected_ids}")
    print(f"Registered  : {registered_ids}")

    if resolution_warning:
        print(
            f"[WARNING] {resolution_warning}"
        )

    if best is not None:
        p = best.position_world_mm
        print(
            "Best Single : "
            f"{best.marker.point_name} "
            f"ID={best.marker.marker_id} "
            f"XYZ=({p[0]:.0f}, "
            f"{p[1]:.0f}, "
            f"{p[2]:.0f}) mm"
        )
        print(
            "Best error  : "
            f"{best.reprojection_error_px:.3f} px"
        )

    if fused is not None:
        p = fused.position_world_mm

        print(
            "Fused XYZ   : "
            f"({p[0]:.0f}, "
            f"{p[1]:.0f}, "
            f"{p[2]:.0f}) mm"
        )

        print(
            "Yaw         : "
            + (
                f"{yaw:.1f} deg"
                if yaw is not None
                else "N/A"
            )
        )

        print(
            "Spread      : "
            f"{fused.spread_mm:.1f} mm"
        )

        print(
            f"Used IDs    : {used_ids}"
        )

        print(
            f"Rejected IDs: {rejected_ids}"
        )

        print(
            f"판정        : {location_text_ko}"
        )

        if description:
            print(
                f"상세        : {description}"
            )
    else:
        print(
            "판정        : 유효한 등록 ArUco Pose가 없습니다."
        )

    print(
        f"Annotated   : {annotated_path}"
    )

    if map_image is not None:
        print(
            f"Map         : {map_path}"
        )

    print(
        f"JSON        : {json_path}"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------
    if args.preview:
        cv2.namedWindow(
            "Image Localization Result",
            cv2.WINDOW_NORMAL,
        )

        image_preview = fit_for_preview(
            annotated,
            args.preview_width,
            args.preview_height,
        )

        cv2.imshow(
            "Image Localization Result",
            image_preview,
        )

        if map_image is not None:
            cv2.namedWindow(
                "Image Localization Map",
                cv2.WINDOW_NORMAL,
            )

            map_preview = fit_for_preview(
                map_image,
                args.preview_width,
                max(
                    300,
                    args.preview_height // 2,
                ),
            )

            cv2.imshow(
                "Image Localization Map",
                map_preview,
            )

        print(
            "미리보기 창에서 아무 키나 누르면 종료합니다."
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
