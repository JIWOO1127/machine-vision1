"""
pose_localizer.py

ArUco 한 장에서 카메라의 월드 위치를 계산하고,
여러 마커 결과를 융합한다.

중요 변경점:
- 벽 방향을 추론하지 않는다.
- aruco_world_map.ArucoMarker.face_world/top_world를 직접 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from aruco_world_map import ArucoMarker


@dataclass
class CameraPoseEstimate:
    marker: ArucoMarker
    position_world_mm: np.ndarray
    rotation_world_camera: np.ndarray
    rvec_marker_to_camera: np.ndarray
    tvec_marker_to_camera: np.ndarray
    reprojection_error_px: float
    marker_area_px2: float
    weight: float


@dataclass
class FusedCameraPose:
    position_world_mm: np.ndarray
    rotation_world_camera: np.ndarray
    estimates_used: List[CameraPoseEstimate]
    estimates_rejected: List[CameraPoseEstimate]
    spread_mm: float

    @property
    def x(self) -> float:
        return float(
            self.position_world_mm[0]
        )

    @property
    def y(self) -> float:
        return float(
            self.position_world_mm[1]
        )

    @property
    def z(self) -> float:
        return float(
            self.position_world_mm[2]
        )


def marker_object_points(
    size_mm: float,
) -> np.ndarray:
    """
    marker local frame:
      +X = RIGHT
      +Y = TOP
      +Z = FRONT/FACE

    corner order:
      TL, TR, BR, BL
    """
    h = float(size_mm) / 2.0

    return np.array(
        [
            [-h, +h, 0.0],
            [+h, +h, 0.0],
            [+h, -h, 0.0],
            [-h, -h, 0.0],
        ],
        dtype=np.float64,
    )


def polygon_area(
    corners: np.ndarray,
) -> float:

    pts = np.asarray(
        corners,
        dtype=np.float64,
    ).reshape(4, 2)

    x = pts[:, 0]
    y = pts[:, 1]

    return abs(
        0.5
        *
        float(
            np.dot(
                x,
                np.roll(y, 1),
            )
            -
            np.dot(
                y,
                np.roll(x, 1),
            )
        )
    )


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    projected = (
        projected
        .reshape(-1, 2)
    )

    image_points = (
        np.asarray(
            image_points,
            dtype=np.float64,
        )
        .reshape(-1, 2)
    )

    errors = np.linalg.norm(
        projected - image_points,
        axis=1,
    )

    return float(
        np.mean(errors)
    )


def estimate_camera_pose_from_marker(
    marker: ArucoMarker,
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[CameraPoseEstimate]:

    image_points = (
        np.asarray(
            corners,
            dtype=np.float64,
        )
        .reshape(4, 2)
    )

    object_points = (
        marker_object_points(
            marker.size_mm
        )
    )

    success, rvec, tvec = (
        cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    )

    if not success:
        return None

    # marker -> camera
    R_cm, _ = cv2.Rodrigues(
        rvec
    )

    t_cm = (
        np.asarray(
            tvec,
            dtype=np.float64,
        )
        .reshape(3)
    )

    # camera origin expressed in marker coordinates.
    # X_c = R_cm X_m + t_cm
    # C_m = -R_cm^T t_cm
    camera_in_marker = (
        -R_cm.T @ t_cm
    )

    # marker local -> world
    R_wm = (
        marker
        .rotation_marker_to_world()
    )

    # camera position in world
    camera_world = (
        marker.position_mm
        +
        R_wm
        @ camera_in_marker
    )

    # camera local -> world rotation
    R_mc = R_cm.T
    R_wc = R_wm @ R_mc

    error_px = reprojection_error(
        object_points,
        image_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    area_px2 = polygon_area(
        image_points
    )

    # 큰 마커 영상 + 작은 reprojection error에 높은 가중치.
    weight = (
        max(area_px2, 1.0)
        /
        max(error_px, 0.25) ** 2
    )

    return CameraPoseEstimate(
        marker=marker,
        position_world_mm=camera_world,
        rotation_world_camera=R_wc,
        rvec_marker_to_camera=rvec,
        tvec_marker_to_camera=tvec,
        reprojection_error_px=error_px,
        marker_area_px2=area_px2,
        weight=float(weight),
    )


def fuse_camera_poses(
    estimates: Sequence[
        CameraPoseEstimate
    ],
    outlier_threshold_mm: float = 1500.0,
) -> Optional[FusedCameraPose]:

    estimates = list(estimates)

    if not estimates:
        return None

    if len(estimates) == 1:
        estimate = estimates[0]

        return FusedCameraPose(
            position_world_mm=(
                estimate
                .position_world_mm
                .copy()
            ),
            rotation_world_camera=(
                estimate
                .rotation_world_camera
                .copy()
            ),
            estimates_used=[estimate],
            estimates_rejected=[],
            spread_mm=0.0,
        )

    positions = np.vstack(
        [
            e.position_world_mm
            for e in estimates
        ]
    )

    # 평균보다 이상치에 강한 median을 기준점으로 사용.
    center = np.median(
        positions,
        axis=0,
    )

    distances = np.linalg.norm(
        positions - center,
        axis=1,
    )

    used = [
        estimate
        for estimate, distance
        in zip(
            estimates,
            distances,
        )
        if distance <= outlier_threshold_mm
    ]

    rejected = [
        estimate
        for estimate, distance
        in zip(
            estimates,
            distances,
        )
        if distance > outlier_threshold_mm
    ]

    # 너무 엄격해서 전부 잘리면 가장 center에 가까운 1개 사용.
    if not used:
        index = int(
            np.argmin(distances)
        )

        used = [
            estimates[index]
        ]

        rejected = [
            estimate
            for i, estimate
            in enumerate(estimates)
            if i != index
        ]

    weights = np.array(
        [
            max(
                estimate.weight,
                1e-9,
            )
            for estimate in used
        ],
        dtype=np.float64,
    )

    weights /= np.sum(
        weights
    )

    fused_position = np.sum(
        np.vstack(
            [
                e.position_world_mm
                for e in used
            ]
        )
        *
        weights[:, None],
        axis=0,
    )

    # rotation matrix는 단순 평균하지 않고
    # 가장 신뢰도가 높은 마커의 rotation 사용.
    best = max(
        used,
        key=lambda e: e.weight,
    )

    used_positions = np.vstack(
        [
            e.position_world_mm
            for e in used
        ]
    )

    spread = float(
        np.max(
            np.linalg.norm(
                used_positions
                -
                fused_position,
                axis=1,
            )
        )
    )

    return FusedCameraPose(
        position_world_mm=fused_position,
        rotation_world_camera=(
            best
            .rotation_world_camera
            .copy()
        ),
        estimates_used=used,
        estimates_rejected=rejected,
        spread_mm=spread,
    )


def camera_yaw_deg(
    R_wc: np.ndarray,
) -> Optional[float]:
    """
    OpenCV camera의 optical forward는 camera +Z.
    이를 world XY에 투영해 yaw 계산.
    +X = 0 deg, +Y = +90 deg.
    """
    forward_world = (
        np.asarray(
            R_wc,
            dtype=np.float64,
        )
        @
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
    )

    xy = forward_world[:2]

    norm = float(
        np.linalg.norm(xy)
    )

    if norm < 1e-8:
        return None

    return float(
        np.degrees(
            np.arctan2(
                xy[1],
                xy[0],
            )
        )
    )
