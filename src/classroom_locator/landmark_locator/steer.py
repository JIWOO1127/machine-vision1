"""
steer.py — 로고/뒷문 좌우 방향 안내 (격자 트래커 탐지 결과를 재사용, 별도 YOLO 추론 없음)

grid_tracker.LandmarkGridPositionTracker가 이미 계산해둔 detections
(Locator.process() 결과: name/box/distance_m, REAL_SIZE 실측 물리 크기 기반
핀홀 거리추정 완료)를 그대로 받아서, 로고(logo)와 뒷문(rear_door) 두 랜드마크에
대해서만 "화면 좌/우/정면 중 어디 있는지" 방향 안내 문장을 만든다. 위치 판정
(격자 지도, room2/3/4 진행)은 건드리지 않는 독립된 보조 안내이며, 프레임마다
새로 추론하지 않고 같은 프레임의 detections를 넘겨받아 쓴다(webapp/backend/
services/vision_bridge.py._build_cue 참고).

문구는 navigator.py(Navigator._steer/ON_REACH)와 같은 DISPLAY 라벨・좌우 임계값
(0.35/0.65)을 쓰되, Navigator의 상태 머신(순서 강제)과 달리 로고/뒷문 각각
독립적으로 판단한다 - 두 랜드마크가 동시에 보여도 무방하며, 우선순위만
STEER_TARGETS 등록 순서(logo -> rear_door)를 따른다.

사용:
    guide = SteerGuide()
    result = guide.update(out["detections"], frame.shape[1], t)
    # result: steer_text / steer / steer_target / steer_distance_m / steer_announce
    guide.reset()   # /api/nav/reset
"""
from __future__ import annotations

from collections import deque
from typing import Any

from .locator import DISPLAY

LEFT, RIGHT = 0.35, 0.65
REPEAT_S = 2.5          # 방향 안내 반복 간격(같은 문장이면 이 시간 지나야 다시 발화)
WINDOW = 5              # 최근 몇 프레임을 볼지
MIN_VOTES = 3           # WINDOW 중 몇 프레임 이상 보여야 판단할지 (깜빡임 방지)

# name -> (도달 거리(m), 도달 시 안내 문구)
STEER_TARGETS: dict[str, dict[str, Any]] = {
    "logo": {"reach_m": 1.2, "reach_text": "벽입니다. 왼쪽으로 돌아 벽을 따라 이동하세요"},
    "rear_door": {"reach_m": 3.0, "reach_text": "뒷문입니다. 목적지에 도착했습니다"},
}


def _direction(box: tuple[float, float, float, float], frame_width: int) -> str:
    cx = (box[0] + box[2]) / 2 / max(1, frame_width)
    return "left" if cx < LEFT else "right" if cx > RIGHT else "center"


def _steer_text(name: str, direction: str, distance_m: float | None) -> str:
    if name == "logo" and direction == "center":
        distance_text = f"{distance_m:.0f}" if distance_m is not None else "?"
        return f"로고가 정면 {distance_text}미터. 직진하세요"
    if name == "rear_door" and direction == "center":
        distance_text = f"{distance_m:.0f}" if distance_m is not None else "?"
        return f"뒷문이 정면 {distance_text}미터. 직진하세요"
    label = DISPLAY[name]
    return {
        "left": f"{label}이 왼쪽에 있습니다. 조금 왼쪽으로",
        "right": f"{label}이 오른쪽에 있습니다. 조금 오른쪽으로",
    }[direction]


class SteerGuide:
    """세션(카메라 한 번 시작 ~ /api/nav/reset)당 하나. 도달/도착 플래그와
    마지막 발화 문장·시각을 들고 있다."""

    def __init__(self) -> None:
        self._buffers: dict[str, deque] = {name: deque(maxlen=WINDOW) for name in STEER_TARGETS}
        self._reached: dict[str, bool] = {name: False for name in STEER_TARGETS}
        self._last_text: str | None = None
        self._last_t: float = -1e9

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
        for name in self._reached:
            self._reached[name] = False
        self._last_text = None
        self._last_t = -1e9

    def _say(self, text: str, t: float, min_gap: float) -> bool:
        if text != self._last_text or t - self._last_t >= min_gap:
            self._last_text, self._last_t = text, t
            return True
        return False

    def update(self, detections: list[dict], frame_width: int, t: float) -> dict[str, Any]:
        empty = {
            "steer_text": None,
            "steer": None,
            "steer_target": None,
            "steer_distance_m": None,
            "steer_announce": False,
        }
        for name in STEER_TARGETS:
            if self._reached[name]:
                continue  # 이 타겟은 도달 안내를 이미 한 번 했으므로 더 이상 안내하지 않음

            det = next((d for d in detections if d["name"] == name), None)
            self._buffers[name].append(det is not None)
            if det is None or sum(self._buffers[name]) < MIN_VOTES:
                continue

            cfg = STEER_TARGETS[name]
            distance_m = det.get("distance_m")
            direction = _direction(det["box"], frame_width)

            if distance_m is not None and distance_m <= cfg["reach_m"]:
                text = cfg["reach_text"]
                announce = self._say(text, t, 1e9)  # 도달 안내는 한 번만
                if announce:
                    self._reached[name] = True
                return {
                    "steer_text": text,
                    "steer": None,
                    "steer_target": name,
                    "steer_distance_m": round(distance_m, 2),
                    "steer_announce": announce,
                }

            text = _steer_text(name, direction, distance_m)
            announce = self._say(text, t, REPEAT_S)
            return {
                "steer_text": text,
                "steer": direction,
                "steer_target": name,
                "steer_distance_m": round(distance_m, 2) if distance_m is not None else None,
                "steer_announce": announce,
            }

        return empty
