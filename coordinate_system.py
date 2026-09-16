"""
coordinate_system.py

문 중앙 랜드마크와 현재 위치 설명.
거리 판정은 XY 평면상의 유클리드 거리(mm)를 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math


@dataclass(frozen=True)
class DoorLandmark:
    key: str
    name: str
    x: float
    y: float


DOOR_LANDMARKS: Dict[str, DoorLandmark] = {
    "classroom_4": DoorLandmark(
        "classroom_4",
        "4번강의실",
        500.0,
        0.0,
    ),
    "classroom_3": DoorLandmark(
        "classroom_3",
        "3번강의실",
        10180.0,
        0.0,
    ),
    "classroom_2": DoorLandmark(
        "classroom_2",
        "2번강의실",
        19640.0,
        0.0,
    ),
    "side_door": DoorLandmark(
        "side_door",
        "쪽문",
        20306.0,
        6792.0,
    ),
}

NEAR_LANDMARK_THRESHOLD_MM = 3000.0


def distance_xy_mm(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    return math.hypot(
        x1 - x2,
        y1 - y2,
    )


def nearest_door_landmark(
    x: float,
    y: float,
) -> Tuple[DoorLandmark, float]:

    results = [
        (
            landmark,
            distance_xy_mm(
                x,
                y,
                landmark.x,
                landmark.y,
            ),
        )
        for landmark in DOOR_LANDMARKS.values()
    ]

    return min(
        results,
        key=lambda item: item[1],
    )


def describe_position(
    x: float,
    y: float,
    z: float,
) -> str:

    landmark, distance = (
        nearest_door_landmark(
            x,
            y,
        )
    )

    dx = x - landmark.x
    dy = y - landmark.y

    if distance <= NEAR_LANDMARK_THRESHOLD_MM:
        area_text = (
            f"{landmark.name} 주변입니다."
        )
    else:
        area_text = (
            "등록된 문 랜드마크에서 "
            "3 m보다 멀리 있습니다."
        )

    return (
        f"WORLD: "
        f"x={x:.0f} mm, "
        f"y={y:.0f} mm, "
        f"z={z:.0f} mm | "
        f"{area_text} "
        f"nearest={landmark.name}, "
        f"distance={distance:.0f} mm, "
        f"dx={dx:+.0f}, dy={dy:+.0f}"
    )
