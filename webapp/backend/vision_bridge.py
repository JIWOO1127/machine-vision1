"""Flask 백엔드 <-> classroom_locator 파이프라인 연결부.

frontend/(machine-vision1 저장소의 React 앱을 그대로 재사용)가 기대하는
JSON 응답 모양(App.jsx가 읽는 result/cue 필드들)에 맞춰 결과를 만들어
내지만, 실제 판정 로직은 전부 우리 classroom_locator.pipeline의
SeojiwooGridPositionTracker(= external/seojiwoo_core Locator 이식 +
3강의실 OCR 정정 + 격자 스냅)를 그대로 사용한다. 새 탐지/거리추정/위치
판정 로직은 여기 만들지 않는다 - 이미 있는 걸 웹 응답 모양으로 옮겨 담을
뿐이다.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from classroom_locator.localization.location_map import (
    GridConfig,
    Location,
    load_grid_config,
    load_locations,
)
from classroom_locator.pipeline.grid_tracker import SeojiwooGridPositionTracker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "final_best.pt"
DEFAULT_LOGO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "final_best_v2_with_logo.pt"
LOCATIONS_FILE = PROJECT_ROOT / "configs" / "locations.yaml"


def decode_image(payload: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return frame


def _direction_from_box(box: tuple[float, float, float, float], frame_width: int, left: float = 0.35, right: float = 0.65) -> str:
    cx = (box[0] + box[2]) / 2 / max(1, frame_width)
    return "왼쪽" if cx < left else "오른쪽" if cx > right else "앞쪽"


def draw_annotated(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Locator가 내는 dict 형태 탐지 결과를 그대로 그림 (Detection 변환 불필요)."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        color = (0, 200, 90) if det["distance_m"] is not None and det["distance_m"] <= 3.0 else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['name']} {det['conf']:.2f} {det['distance_m']:.1f}m"
        cv2.putText(annotated, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


class VisionBridge:
    """사진 1장 분석(stateless)과 라이브 스트림 분석(프레임 버퍼 유지) 둘 다 제공."""

    def __init__(self) -> None:
        self.load_error: str | None = None
        self.grid: GridConfig | None = None
        self.locations: list[Location] = []
        self.by_name: dict[str, Location] = {}
        self.live_tracker: SeojiwooGridPositionTracker | None = None
        self.photo_tracker: SeojiwooGridPositionTracker | None = None
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - /api/health가 이유를 보여줘야 함
            self.load_error = str(exc)

    def _load(self) -> None:
        self.grid = load_grid_config(LOCATIONS_FILE)
        self.locations = load_locations(LOCATIONS_FILE)
        self.by_name = {loc.name: loc for loc in self.locations}
        weights = str(DEFAULT_WEIGHTS)
        logo_weights = str(DEFAULT_LOGO_WEIGHTS) if DEFAULT_LOGO_WEIGHTS.exists() else None

        # 라이브 스트림(/api/analyze-frame): 최근 5프레임 다수결로 안정화 (기본값 그대로)
        self.live_tracker = SeojiwooGridPositionTracker(
            self.grid, self.locations, weights=weights, logo_weights=logo_weights
        )
        # 사진 1장(/api/analyze): 프레임 투표 없이 그 한 장으로 바로 판정
        self.photo_tracker = SeojiwooGridPositionTracker(
            self.grid,
            self.locations,
            weights=weights,
            logo_weights=logo_weights,
            window=1,
            min_votes=1,
            approach_only=False,
        )

    @property
    def model_ready(self) -> bool:
        return self.live_tracker is not None and self.live_tracker._locator.model is not None

    @property
    def device(self) -> str:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def reset_live_tracking(self) -> None:
        if self.live_tracker is None:
            return
        self.live_tracker._locator.reset()
        self.live_tracker.current_cell = None
        self.live_tracker.current_label = None

    # ---- 공용 빌더 ----

    def _landmarks_and_labels(self, tracker: SeojiwooGridPositionTracker, out: dict, frame: np.ndarray, frame_width: int):
        landmarks, labels, detections = [], [], []
        for det in out["detections"]:
            location_name = tracker.resolve_location_name(det, frame)
            loc = self.by_name.get(location_name)
            display = (loc.display_name or loc.name) if loc else location_name
            snap = loc.snap_distance_m if (loc and loc.snap_distance_m) else 3.0
            distance_m = det["distance_m"]
            state = "near" if distance_m <= snap else "far"
            landmarks.append(
                {
                    "name": display,
                    "distance_m": round(distance_m, 2),
                    "direction": _direction_from_box(det["box"], frame_width),
                    "state": state,
                    "state_text": "가까움" if state == "near" else "거리가 있음",
                }
            )
            labels.append(display)
            detections.append({"label": display, "confidence": round(float(det["conf"]), 4)})
        order = sorted(range(len(landmarks)), key=lambda i: landmarks[i]["distance_m"])
        landmarks = [landmarks[i] for i in order]
        detections = [detections[i] for i in order]
        return landmarks, labels, detections

    def _location_summary(self, tracker: SeojiwooGridPositionTracker) -> dict:
        if tracker.current_cell is None:
            return {
                "status": "unknown",
                "label": "위치 불확실",
                "detail": "아직 위치를 확정하지 못했습니다.",
                "candidates": [],
            }
        return {
            "status": "localized",
            "label": f"{tracker.current_label} 근처" if tracker.current_label else str(tracker.current_cell),
            "detail": f"격자 좌표 {tracker.current_cell}",
            "candidates": [],
        }

    def _build_cue(self, tracker: SeojiwooGridPositionTracker, frame: np.ndarray, frame_width: int, time_seconds: float) -> dict:
        out = tracker.last_result or {"detections": [], "landmark": None, "reason": ""}
        landmarks, labels, _detections = self._landmarks_and_labels(tracker, out, frame, frame_width)
        location = self._location_summary(tracker)

        nearest = landmarks[0] if landmarks else None
        if nearest:
            guidance = (
                f"{nearest['direction']} 약 {nearest['distance_m']:.1f}m에서 {nearest['name']}이 보입니다. "
                f"현재 위치: {location['label']}."
            )
        else:
            guidance = "이 화면에서는 위치를 판단할 기준 객체가 보이지 않습니다."

        locator = tracker._locator
        votes = Counter(name for name, _, _ in locator.buf if name)
        vote_count = votes.most_common(1)[0][1] if votes else 0
        reason = out.get("reason") or ""
        confirmed = out.get("landmark") is not None and not reason.startswith("votes")

        return {
            "time_seconds": round(time_seconds, 3),
            "texts": [],
            "detections": labels,
            "landmarks": landmarks,
            "location": location,
            "guidance": guidance,
            "stability": {
                "confirmed": confirmed,
                "count": vote_count,
                "required": locator.min_votes,
                "collected": len(locator.buf),
                "window": locator.window,
            },
        }

    # ---- 엔드포인트에서 부르는 함수들 ----

    def analyze_photo(self, frame: np.ndarray) -> dict:
        tracker = self.photo_tracker
        tracker.update_from_frame(frame, 0.0)
        out = tracker.last_result or {"detections": []}
        landmarks, _labels, detections = self._landmarks_and_labels(tracker, out, frame, frame.shape[1])
        location = self._location_summary(tracker)
        location["candidates"] = [
            {"name": item["name"], "score": max(0.0, 1.0 - item["distance_m"] / 3.0)} for item in landmarks
        ]
        annotated = draw_annotated(frame, out["detections"])
        return {"detections": detections, "location": location, "annotated_image": annotated}

    def analyze_live_frame(self, frame: np.ndarray, time_seconds: float) -> dict:
        tracker = self.live_tracker
        tracker.update_from_frame(frame, time_seconds)
        return self._build_cue(tracker, frame, frame.shape[1], time_seconds)
