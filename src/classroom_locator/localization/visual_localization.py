"""핀홀 카메라 모델 + 격자 탐색(grid search) 기반 2D 위치 추정.

machine-vision1 프로젝트(github.com/JIWOO1127/machine-vision1)의
`VisualLocationEstimator` 방식을 이식했습니다. 다만 그쪽은 표지판/문의
실측 물리 크기(mm)와 카메라 화각을 알아야 하는데, 저희는 `distance_data`
사진으로 직접 찍어서 얻은 "크기<->거리" 경험적 캘리브레이션
(`data/distance_calibration.json`, `scripts/calibrate_distance.py`가 생성)을
그대로 씁니다 — 실측 없이도 동일한 원리로 동작합니다.

원리:
1. 탐지된 bbox 크기(프레임 대비 비율)로 "이 물체까지의 거리"를 추정
   (크기는 거리에 반비례한다는 핀홀 카메라 근사)
2. bbox의 화면 좌우 위치로 "이 물체가 내 정면 기준 몇 도 방향에 있는지" 추정
3. 지도를 촘촘한 격자로 나누고, 각 격자점에서 "여기서 보이는 거리/각도가
   실측값과 얼마나 잘 맞는지" 비용을 계산해서 가장 비용이 낮은 격자점을
   현재 위치로 선택 (여러 물체가 동시에 보이면 삼각측량처럼 더 정확해짐)
4. 이전 프레임 위치와 너무 동떨어지지 않도록 완만한 안정화 보정을 더함

한계(v1, machine-vision1 대비 단순화한 부분):
- 표지판/문이 "정면에서만 보인다"는 각도 제약(facing/surface visibility)은
  아직 반영 안 함 — 뒤에서 봐도 앞에서 본 것처럼 계산될 수 있음
- 명시적 구역(zone) 정의 대신, 가장 가까운 랜드마크 이름으로 대체
- optical flow(보조 UI용 이동방향 텍스트)는 이식 안 함
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..detection.base import Detection
from .location_map import Location


def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _weighted_circular_mean_deg(values: list[float], weights: list[float]) -> float:
    x = sum(w * math.cos(math.radians(v)) for v, w in zip(values, weights))
    y = sum(w * math.sin(math.radians(v)) for v, w in zip(values, weights))
    return math.degrees(math.atan2(y, x))


def _calibration_group(class_name: str) -> str:
    """room2/room3/room4는 표지판 실제 물리 크기가 같으므로 "sign" 하나로 묶음."""
    return "sign" if class_name.startswith("room") else class_name


def estimate_distance_m(
    det: Detection,
    calibration: dict[str, dict[str, float]],
    frame_width: int,
    frame_height: int,
) -> float | None:
    """bbox 크기(프레임 대비 비율)로 물체까지의 거리를 추정합니다.

    calibration에 이 클래스에 대한 보정 상수가 없으면 None을 반환합니다.
    GridPositionTracker와 VisualLocationEstimator가 공용으로 씁니다.
    """
    cal = calibration.get(_calibration_group(det.class_name))
    if cal is None:
        return None

    x1, y1, x2, y2 = det.bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    height_ratio = bbox_h / frame_height
    width_ratio = bbox_w / frame_width
    height_distance = cal["k_height"] / height_ratio
    width_distance = cal["k_width"] / width_ratio
    # 높이 기준이 원근 왜곡에 덜 민감해서 더 신뢰함 (machine-vision1과 동일 가중치)
    return height_distance * 0.7 + width_distance * 0.3


@dataclass
class LocalizationResult:
    status: str  # "unknown" | "candidates" | "localized"
    x_m: float | None
    y_m: float | None
    confidence: float
    label: str
    anchors: list[str]
    measurements: list[dict[str, Any]] = field(default_factory=list)


class VisualLocationEstimator:
    def __init__(
        self,
        locations: list[Location],
        calibration: dict[str, dict[str, float]],
        camera_horizontal_fov_deg: float = 70.0,
        grid_step_m: float = 0.5,
        range_tolerance_ratio: float = 0.4,
        min_range_tolerance_m: float = 1.0,
        map_padding_m: float = 2.0,
        stable_frames_required: int = 3,
    ) -> None:
        self.locations_by_name: dict[str, Location] = {
            loc.name: loc for loc in locations if loc.coords and "x" in loc.coords and "y" in loc.coords
        }
        self.calibration = calibration
        self.fov_deg = camera_horizontal_fov_deg
        self.grid_step_m = grid_step_m
        self.range_tolerance_ratio = range_tolerance_ratio
        self.min_range_tolerance_m = min_range_tolerance_m
        self.stable_frames_required = stable_frames_required

        xs = [loc.coords["x"] for loc in self.locations_by_name.values()]
        ys = [loc.coords["y"] for loc in self.locations_by_name.values()]
        self.bounds = (
            min(xs) - map_padding_m,
            max(xs) + map_padding_m,
            min(ys) - map_padding_m,
            max(ys) + map_padding_m,
        )
        self.grid = self._build_grid()

        self.previous_xy: tuple[float, float] | None = None
        self.stable_label: str | None = None
        self.stable_frames = 0

    def _build_grid(self) -> list[tuple[float, float]]:
        x_min, x_max, y_min, y_max = self.bounds
        points = []
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                points.append((x, y))
                y += self.grid_step_m
            x += self.grid_step_m
        return points

    def _measurements(
        self, detections: list[Detection], frame_width: int, frame_height: int
    ) -> list[dict[str, Any]]:
        focal_length_px = frame_width / (2.0 * math.tan(math.radians(self.fov_deg / 2.0)))
        measured = []
        for det in detections:
            location = self.locations_by_name.get(det.class_name)
            if location is None:
                continue
            distance = estimate_distance_m(det, self.calibration, frame_width, frame_height)
            if distance is None:
                continue

            x1, _y1, x2, _y2 = det.bbox
            center_x = (x1 + x2) / 2.0
            bearing = math.degrees(math.atan2(center_x - frame_width / 2.0, focal_length_px))

            measured.append(
                {
                    "class_name": det.class_name,
                    "location": location,
                    "distance_m": distance,
                    "bearing_deg": bearing,
                    "weight": max(0.05, det.confidence),
                }
            )
        return measured

    def _score_grid(self, measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored = []
        for x, y in self.grid:
            range_cost = 0.0
            yaw_values = []
            yaw_weights = []
            for m in measurements:
                lx, ly = m["location"].coords["x"], m["location"].coords["y"]
                dx, dy = lx - x, ly - y
                predicted_range = math.hypot(dx, dy)
                sigma = max(self.min_range_tolerance_m, m["distance_m"] * self.range_tolerance_ratio)
                residual = (predicted_range - m["distance_m"]) / sigma
                range_cost += m["weight"] * residual * residual

                world_bearing = math.degrees(math.atan2(dy, dx))
                yaw_values.append(world_bearing - m["bearing_deg"])
                yaw_weights.append(m["weight"])

            yaw = _weighted_circular_mean_deg(yaw_values, yaw_weights) if yaw_values else 0.0
            bearing_cost = 0.0
            if len(measurements) >= 2:
                for v, w in zip(yaw_values, yaw_weights):
                    residual = _angle_diff_deg(v, yaw) / 12.0
                    bearing_cost += w * residual * residual

            prior_cost = 0.0
            if self.previous_xy is not None:
                prior_dist = math.hypot(x - self.previous_xy[0], y - self.previous_xy[1])
                prior_cost = 0.12 * (prior_dist / 2.5) ** 2

            scored.append({"x": x, "y": y, "cost": range_cost + bearing_cost + prior_cost})
        return sorted(scored, key=lambda item: item["cost"])

    def _nearest_landmark_label(self, x: float, y: float) -> str:
        nearest = min(
            self.locations_by_name.values(),
            key=lambda loc: math.hypot(x - loc.coords["x"], y - loc.coords["y"]),
        )
        name = nearest.display_name or nearest.name
        return f"{name} 부근"

    def estimate(self, detections: list[Detection], frame_width: int, frame_height: int) -> LocalizationResult:
        measurements = self._measurements(detections, frame_width, frame_height)
        if not measurements:
            return LocalizationResult("unknown", None, None, 0.0, "위치 불확실", [])

        scored = self._score_grid(measurements)
        best = scored[0]
        unique_anchors = sorted({m["class_name"] for m in measurements})

        anchor_factor = 0.9 if len(unique_anchors) >= 2 else 0.5
        confidence = min(0.95, anchor_factor * math.exp(-0.3 * best["cost"]))

        label = self._nearest_landmark_label(best["x"], best["y"])
        if label == self.stable_label:
            self.stable_frames += 1
        else:
            self.stable_label = label
            self.stable_frames = 1

        constrained = len(unique_anchors) >= 2 or self.stable_frames >= self.stable_frames_required
        status = "localized" if constrained else "candidates"
        if constrained:
            self.previous_xy = (best["x"], best["y"])

        return LocalizationResult(
            status=status,
            x_m=round(best["x"], 2),
            y_m=round(best["y"], 2),
            confidence=round(confidence, 2),
            label=label,
            anchors=unique_anchors,
            measurements=[
                {
                    "class_name": m["class_name"],
                    "distance_m": round(m["distance_m"], 2),
                    "bearing_deg": round(m["bearing_deg"], 1),
                }
                for m in measurements
            ],
        )
