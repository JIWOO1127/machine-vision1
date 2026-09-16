"""
camera_calibration.py

정식 캘리브레이션 파일을 읽거나,
없는 경우 지정된 수평 화각(HFOV)을 이용해 근사 카메라 행렬 생성.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_width: int
    image_height: int
    approximate: bool = False


def load_calibration(
    path: str | Path,
) -> CameraCalibration:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(
        path,
        allow_pickle=False,
    )

    camera_matrix = np.asarray(
        data["camera_matrix"],
        dtype=np.float64,
    )

    dist_coeffs = np.asarray(
        data["dist_coeffs"],
        dtype=np.float64,
    )

    width = int(
        data["image_width"]
    ) if "image_width" in data else 0

    height = int(
        data["image_height"]
    ) if "image_height" in data else 0

    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=width,
        image_height=height,
        approximate=False,
    )


def make_approximate_calibration(
    image_width: int,
    image_height: int,
    horizontal_fov_deg: float = 60.0,
) -> CameraCalibration:

    if not (
        10.0
        <
        horizontal_fov_deg
        <
        170.0
    ):
        raise ValueError(
            "HFOV는 10~170도 사이여야 합니다."
        )

    fov_rad = math.radians(
        horizontal_fov_deg
    )

    fx = (
        image_width
        /
        (
            2.0
            *
            math.tan(
                fov_rad / 2.0
            )
        )
    )

    fy = fx
    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dist = np.zeros(
        (5, 1),
        dtype=np.float64,
    )

    return CameraCalibration(
        camera_matrix=K,
        dist_coeffs=dist,
        image_width=image_width,
        image_height=image_height,
        approximate=True,
    )
