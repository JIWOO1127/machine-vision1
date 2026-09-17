"""실시간 안내를 단계별로 관리하는 경량 상태 머신.

첨부 navigator.py의 흐름(목표 탐색 → 좌/우/직진 유도 → 근접 시 다음 단계 →
직전 표지판이 화면에서 사라질 때까지 대기)을 웹 API의 거리 측정 결과에 맞춘 버전이다.
final_best.pt의 logo 클래스를 위치 시작 기준점으로 사용할 수 있다. 3강의실은
YOLO 전용 클래스가 없지만, 2/4 표지판 탐지 뒤 EasyOCR가 숫자 3을 읽으면
경로 단계에 포함된다.
"""

from __future__ import annotations

from typing import Any


class RouteNavigator:
    STEPS = (
        {
            "key": "room4",
            "anchor": "classroom_4",
            "near_m": 3.0,
            "search": "왼쪽 벽을 따라 이동하세요. 4강의실을 찾습니다.",
            "reached": "4강의실 앞입니다. 계속 직진하세요.",
        },
        {
            "key": "room3",
            "anchor": "classroom_3",
            "near_m": 3.0,
            "search": "계속 벽을 따라 이동하세요. 3강의실을 지납니다.",
            "reached": "3강의실 앞입니다. 계속 직진하세요.",
        },
        {
            "key": "room2",
            "anchor": "classroom_2",
            "near_m": 3.0,
            "search": "계속 벽을 따라 이동하세요. 2강의실을 찾습니다.",
            "reached": "2강의실 앞입니다. 왼쪽으로 돌아 뒷문을 찾으세요.",
        },
        {
            "key": "rear_door",
            "anchor": "rear_door",
            "near_m": 3.0,
            "search": "왼쪽으로 돌아 뒷문을 찾으세요.",
            "reached": "뒷문입니다. 목적지에 도착했습니다.",
        },
    )

    DISPLAY = {
        "classroom_4": "4강의실",
        "classroom_3": "3강의실",
        "classroom_2": "2강의실",
        "rear_door": "뒷문",
    }

    def __init__(self, left: float = -8.0, right: float = 8.0):
        self.left = left
        self.right = right
        self.reset()

    def reset(self) -> None:
        self.step_index = 0
        self.done = False
        self.clear_needed = False
        self.clear_count = 0

    def update(self, measurements: list[dict[str, Any]]) -> dict[str, Any]:
        step = self.STEPS[self.step_index]
        matching = next((item for item in measurements if item.get("anchor") == step["anchor"]), None)

        # 다음 표지판으로 넘어간 뒤에도 직전 표지판이 계속 보이는 동안은 재인식하지 않는다.
        if self.clear_needed:
            if matching is None:
                self.clear_count += 1
            else:
                self.clear_count = 0
            if self.clear_count < 3:
                return self._result(step, step["search"], None, False)
            self.clear_needed = False

        if self.done:
            return self._result(step, "목적지에 도착했습니다.", None, True)
        if matching is None:
            return self._result(step, step["search"], None, False)

        distance_m = float(matching.get("distance_mm", 0.0)) / 1000.0
        bearing = float(matching.get("bearing_deg", 0.0))
        steer = "left" if bearing < self.left else "right" if bearing > self.right else "center"
        if distance_m <= step["near_m"]:
            text = step["reached"]
            is_last = self.step_index == len(self.STEPS) - 1
            if is_last:
                self.done = True
            else:
                self.step_index += 1
                self.clear_needed = True
                self.clear_count = 0
            return self._result(step, text, steer, self.done, distance_m)

        target = self.DISPLAY[step["anchor"]]
        direction_text = {
            "left": f"{target}이 왼쪽에 있습니다. 조금 왼쪽으로 가세요.",
            "right": f"{target}이 오른쪽에 있습니다. 조금 오른쪽으로 가세요.",
            "center": f"{target}이 정면 약 {distance_m:.1f}미터에 있습니다. 직진하세요.",
        }[steer]
        return self._result(step, direction_text, steer, False, distance_m)

    def _result(self, step, text, steer, done, distance_m=None):
        return {
            "step": step["key"],
            "target": self.DISPLAY[step["anchor"]],
            "text": text,
            "steer": steer,
            "done": done,
            "distance_m": round(distance_m, 2) if distance_m is not None else None,
        }
