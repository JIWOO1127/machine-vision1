from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _angle_difference_deg(left: float, right: float) -> float:
    return (left - right + 180.0) % 360.0 - 180.0


def _weighted_circular_mean_deg(values, weights) -> float:
    x = sum(weight * math.cos(math.radians(value)) for value, weight in zip(values, weights))
    y = sum(weight * math.sin(math.radians(value)) for value, weight in zip(values, weights))
    return math.degrees(math.atan2(y, x))


class VisualLocationEstimator:
    """Coarse camera localization from mapped detector/OCR landmarks.

    A single landmark only defines a range ring, not a unique camera pose.
    Therefore a single landmark returns ranked areas; multiple landmarks or a
    stable temporal prior can narrow the result to an approximate XY pose.
    """

    def __init__(self, map_path: Path, frame_width: int):
        map_data = json.loads(map_path.read_text(encoding="utf-8"))
        self.map_data = map_data
        self.landmarks = {
            item["key"]: item for item in map_data.get("door_landmarks", [])
        }
        localization = map_data.get("visual_localization", {})
        self.visual_landmarks = {
            item["label"]: item for item in localization.get("landmarks", [])
        }
        self.frame_width = frame_width
        horizontal_fov = float(localization.get("camera_horizontal_fov_deg", 70.0))
        self.focal_length_px = frame_width / (
            2.0 * math.tan(math.radians(horizontal_fov / 2.0))
        )
        self.grid_step_mm = float(localization.get("grid_step_mm", 250.0))
        self.range_tolerance_ratio = float(
            localization.get("range_tolerance_ratio", 0.38)
        )
        self.min_range_tolerance_mm = float(
            localization.get("min_range_tolerance_mm", 1200.0)
        )
        self.max_surface_view_angle_deg = float(
            localization.get("max_surface_view_angle_deg", 82.0)
        )
        self.zones = localization.get("zones", [])
        self.previous_gray = None
        self.previous_xy = None
        self.previous_distance_by_anchor = {}
        self.stable_zone = None
        self.stable_frames = 0
        self.grid = self._build_walkable_grid(map_data, localization)

    def _build_walkable_grid(self, map_data, localization):
        bounds = localization.get("walkable_bounds") or map_data["map_bounds"]
        x_min = float(bounds["x_min"])
        x_max = float(bounds["x_max"])
        y_min = float(bounds["y_min"])
        y_max = float(bounds["y_max"])
        excluded = map_data.get("rectangles", [])
        points = []
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                inside_obstacle = any(
                    float(item["x1"]) < x < float(item["x2"])
                    and float(item["y1"]) < y < float(item["y2"])
                    for item in excluded
                )
                if not inside_obstacle:
                    points.append((x, y))
                y += self.grid_step_mm
            x += self.grid_step_mm
        return points

    def _measurements(self, observations):
        measured = []
        for observation in observations:
            config = self.visual_landmarks.get(observation["label"])
            if config is None:
                continue
            landmark = self.landmarks.get(config["map_key"])
            if landmark is None:
                continue
            box = observation["box"]
            box_width = max(1.0, float(box[2] - box[0]))
            box_height = max(1.0, float(box[3] - box[1]))
            center_x = (float(box[0]) + float(box[2])) / 2.0
            physical_height = float(config["physical_height_mm"])
            physical_width = float(config["physical_width_mm"])
            distance_scale = float(config.get("distance_scale", 1.0))
            height_distance = self.focal_length_px * physical_height / box_height
            width_distance = self.focal_length_px * physical_width / box_width
            # Height is less sensitive to horizontal foreshortening, while
            # width still adds useful object-size evidence.
            distance = (height_distance * 0.7 + width_distance * 0.3) * distance_scale
            score = max(0.05, float(observation.get("score", 0.5)))
            measured.append(
                {
                    "observation": observation,
                    "config": config,
                    "landmark": landmark,
                    "distance_mm": distance,
                    "height_distance_mm": height_distance * distance_scale,
                    "width_distance_mm": width_distance * distance_scale,
                    "bearing_deg": math.degrees(
                        math.atan2(center_x - self.frame_width / 2.0, self.focal_length_px)
                    ),
                    "weight": score,
                    "calibrated": bool(config.get("calibrated", False)),
                }
            )
        return measured

    def _surface_visible(self, x, y, measurement):
        facing = measurement["config"].get("facing_deg")
        if facing is None:
            return True
        landmark = measurement["landmark"]
        camera_angle = math.degrees(
            math.atan2(y - float(landmark["y"]), x - float(landmark["x"]))
        )
        return (
            abs(_angle_difference_deg(camera_angle, float(facing)))
            <= self.max_surface_view_angle_deg
        )

    def _score_grid(self, measurements):
        scored = []
        for x, y in self.grid:
            yaw_values = []
            yaw_weights = []
            range_cost = 0.0
            visible = True
            for measurement in measurements:
                if not self._surface_visible(x, y, measurement):
                    visible = False
                    break
                landmark = measurement["landmark"]
                dx = float(landmark["x"]) - x
                dy = float(landmark["y"]) - y
                predicted_range = math.hypot(dx, dy)
                sigma = max(
                    self.min_range_tolerance_mm,
                    measurement["distance_mm"] * self.range_tolerance_ratio,
                )
                residual = (predicted_range - measurement["distance_mm"]) / sigma
                range_cost += measurement["weight"] * residual * residual
                world_bearing = math.degrees(math.atan2(dy, dx))
                yaw_values.append(world_bearing - measurement["bearing_deg"])
                yaw_weights.append(measurement["weight"])
            if not visible:
                continue

            yaw = _weighted_circular_mean_deg(yaw_values, yaw_weights)
            bearing_cost = 0.0
            if len(measurements) >= 2:
                for value, weight in zip(yaw_values, yaw_weights):
                    residual = _angle_difference_deg(value, yaw) / 12.0
                    bearing_cost += weight * residual * residual

            prior_cost = 0.0
            if self.previous_xy is not None:
                prior_distance = math.hypot(x - self.previous_xy[0], y - self.previous_xy[1])
                prior_cost = 0.12 * (prior_distance / 2500.0) ** 2
            scored.append(
                {
                    "x": x,
                    "y": y,
                    "yaw_deg": yaw,
                    "cost": range_cost + bearing_cost + prior_cost,
                }
            )
        return sorted(scored, key=lambda item: item["cost"])

    def _zone_for_point(self, x, y):
        for zone in self.zones:
            if (
                float(zone["x_min"]) <= x <= float(zone["x_max"])
                and float(zone["y_min"]) <= y <= float(zone["y_max"])
            ):
                return zone["name"]
        nearest = min(
            self.landmarks.values(),
            key=lambda item: math.hypot(x - float(item["x"]), y - float(item["y"])),
        )
        return f"{nearest['name']} 부근"

    def _rank_zones(self, scored):
        if not scored:
            return []
        best_cost = scored[0]["cost"]
        weights_by_zone = defaultdict(float)
        for item in scored:
            if item["cost"] > best_cost + 5.0:
                break
            weight = math.exp(-0.5 * (item["cost"] - best_cost))
            weights_by_zone[self._zone_for_point(item["x"], item["y"])] += weight
        total = sum(weights_by_zone.values()) or 1.0
        return sorted(
            (
                {"name": name, "probability": weight / total}
                for name, weight in weights_by_zone.items()
            ),
            key=lambda item: item["probability"],
            reverse=True,
        )

    def update(self, frame, observations):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_x = flow_y = 0.0
        if self.previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_x, flow_y = flow.reshape(-1, 2).mean(axis=0)
        self.previous_gray = gray

        measurements = self._measurements(observations)
        if not measurements:
            return {
                "status": "unknown",
                "label": "위치 불확실",
                "detail": "지도 랜드마크가 보이지 않음",
                "confidence": 0.0,
                "motion": self._motion_label(flow_x, flow_y),
                "candidates": [],
            }

        scored = self._score_grid(measurements)
        if not scored:
            return {
                "status": "unknown",
                "label": "위치 불확실",
                "detail": "지도와 맞는 위치 후보가 없음",
                "confidence": 0.0,
                "motion": self._motion_label(flow_x, flow_y),
                "candidates": [],
            }

        zones = self._rank_zones(scored)
        best = scored[0]
        top_zone = zones[0]["name"] if zones else self._zone_for_point(best["x"], best["y"])
        if top_zone == self.stable_zone:
            self.stable_frames += 1
        else:
            self.stable_zone = top_zone
            self.stable_frames = 1

        top_probability = zones[0]["probability"] if zones else 0.0
        unique_anchor_count = len(
            {item["config"]["map_key"] for item in measurements}
        )
        constrained = unique_anchor_count >= 2 or (
            top_probability >= 0.58 and self.stable_frames >= 3
        )
        if constrained:
            self.previous_xy = (best["x"], best["y"])

        anchors = []
        distance_details = []
        for measurement in measurements:
            key = measurement["config"]["map_key"]
            anchors.append(key)
            previous = self.previous_distance_by_anchor.get(key)
            trend = ""
            if previous is not None:
                ratio = measurement["distance_mm"] / max(previous, 1.0)
                trend = "접근" if ratio < 0.92 else "멀어짐" if ratio > 1.08 else "유지"
            self.previous_distance_by_anchor[key] = measurement["distance_mm"]
            calibrated_mark = "" if measurement["calibrated"] else "(미보정)"
            distance_details.append(
                f"{self._landmark_name(key)} {measurement['distance_mm'] / 1000:.1f}m{calibrated_mark}{trend}"
            )

        candidate_names = [item["name"] for item in zones[:3]]
        if constrained:
            status = "localized"
            label = top_zone
            coordinate = f"x={best['x'] / 1000:.1f}m, y={best['y'] / 1000:.1f}m"
        else:
            status = "candidates"
            label = "후보: " + " / ".join(candidate_names)
            coordinate = "단일 랜드마크라 좌표 확정 불가"

        calibration_factor = 1.0 if all(item["calibrated"] for item in measurements) else 0.55
        evidence_factor = min(1.0, 0.45 + 0.25 * unique_anchor_count)
        confidence = min(0.95, top_probability * calibration_factor * evidence_factor)
        return {
            "status": status,
            "label": label,
            "detail": f"{coordinate} | {'; '.join(distance_details)}",
            "x_mm": round(best["x"]),
            "y_mm": round(best["y"]),
            "yaw_deg": round(best["yaw_deg"], 1),
            "anchors": anchors,
            "confidence": round(confidence, 2),
            "motion": self._motion_label(flow_x, flow_y),
            "candidates": [
                {"name": item["name"], "score": round(item["probability"], 3)}
                for item in zones[:5]
            ],
            "measurements": [
                {
                    "anchor": item["config"]["map_key"],
                    "distance_mm": round(item["distance_mm"]),
                    "height_distance_mm": round(item["height_distance_mm"]),
                    "width_distance_mm": round(item["width_distance_mm"]),
                    "bearing_deg": round(item["bearing_deg"], 1),
                    "calibrated": item["calibrated"],
                }
                for item in measurements
            ],
        }

    @staticmethod
    def _landmark_name(anchor):
        return {
            "classroom_2": "강의실2",
            "classroom_3": "강의실3",
            "classroom_4": "강의실4",
            "front_door": "앞문",
            "rear_door": "쪽문",
        }.get(anchor, anchor)

    @staticmethod
    def _motion_label(flow_x, flow_y):
        if abs(flow_x) < 0.15 and abs(flow_y) < 0.15:
            return "화면 이동 거의 없음"
        horizontal = "오른쪽" if flow_x < 0 else "왼쪽"
        vertical = "아래쪽" if flow_y < 0 else "위쪽"
        if abs(flow_x) >= abs(flow_y) * 1.4:
            return f"{horizontal}으로 이동"
        if abs(flow_y) >= abs(flow_x) * 1.4:
            return f"{vertical}으로 이동"
        return f"{horizontal}/{vertical}으로 이동"


def draw_location_overlay(image, location):
    """Draw a readable Korean location panel on a BGR OpenCV image."""
    height, width = image.shape[:2]
    panel_height = max(92, round(height * 0.10))
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (12, height - panel_height - 12),
        (width - 12, height - 12),
        (0, 0, 0),
        -1,
    )
    image = cv2.addWeighted(overlay, 0.82, image, 0.18, 0)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    title_size = max(22, min(36, round(width / 55)))
    detail_size = max(17, min(26, round(width / 75)))
    title_font = (
        ImageFont.truetype(str(font_path), title_size)
        if font_path.exists()
        else ImageFont.load_default()
    )
    detail_font = (
        ImageFont.truetype(str(font_path), detail_size)
        if font_path.exists()
        else ImageFont.load_default()
    )
    left = 28
    top = height - panel_height
    draw.text(
        (left, top),
        f"현재 위치: {location.get('label', '위치 불확실')}",
        font=title_font,
        fill=(255, 235, 40),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    detail = f"{location.get('detail', '')}  {location.get('motion', '')}".strip()
    draw.text(
        (left, top + title_size + 10),
        detail,
        font=detail_font,
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
