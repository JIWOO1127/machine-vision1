"""
navigation_engine.py

ArUco 기반 실내 네비게이션 엔진.

경로
----
앞문 -> 4번 강의실 -> 2번 강의실 -> 뒷문

현재 뒷문은 기존 지도에서 "쪽문"으로 등록된
(20306, 6792) 좌표를 사용한다.

이번 버전의 주요 동작
--------------------
- 출발/경유/도착 판정 반경: 2500 mm
- 갑작스러운 위치 튐 제거
- 위치 인식이 잠시 끊기면 마지막 안내 상태 그대로 유지
- 웹캠 화면 상단에 전체 경로/진행률 표시
- 이미 지나온 경로는 초록색으로 변경
- 사용자가 뒤로 돌아가면 진행 색도 현재 위치에 맞춰 뒤로 줄어듦
- 시선 방향, 안내 방향, 실제 이동 방향을 각각 표시
- 시작/경유/도착 이벤트를 2초간 크게 표시
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import atan2, degrees, hypot, radians, sin, cos
from pathlib import Path
import platform
import time
from typing import Optional, Sequence

import cv2
import numpy as np

from coordinate_system import DOOR_LANDMARKS
from live_map_view import (
    BOTTOM_DOORS,
    EAST_WALL_X,
    FRONT_DOOR_X1,
    FRONT_DOOR_X2,
    MAP_X_MAX,
    MAP_X_MIN,
    MAP_Y_MAX,
    MAP_Y_MIN,
    RECTANGLES,
    SIDE_DOOR_X1,
    SIDE_DOOR_X2,
    SIDE_WALL_Y,
)

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


# ============================================================
# Route
# ============================================================

FRONT = DOOR_LANDMARKS["front_door"]
CLASSROOM_4 = DOOR_LANDMARKS["classroom_4"]
CLASSROOM_2 = DOOR_LANDMARKS["classroom_2"]
BACK = DOOR_LANDMARKS["side_door"]

ROUTE_TITLE = "앞문 -> 4번 강의실 -> 2번 강의실 -> 뒷문 순서"

ROUTE_POINTS = np.array(
    [
        [FRONT.x, FRONT.y],
        [FRONT.x, 6100.0],
        [-1800.0, 6100.0],
        [-1800.0, 1200.0],
        [-300.0, 1200.0],
        [CLASSROOM_4.x, CLASSROOM_4.y],
        [1300.0, 1000.0],
        [CLASSROOM_2.x, 1000.0],
        [CLASSROOM_2.x, CLASSROOM_2.y],
        [20500.0, 1000.0],
        [20500.0, 5900.0],
        [BACK.x, BACK.y],
    ],
    dtype=np.float64,
)

CHECKPOINTS = (
    ("앞문", 0),
    ("4번 강의실", 5),
    ("2번 강의실", 8),
    ("뒷문", 11),
)

# 요청사항: 해당 공간 중심 2.5m 이내면 출발/경유/도착으로 판정
CHECKPOINT_RADIUS_MM = 2500.0
START_RADIUS_MM = CHECKPOINT_RADIUS_MM

LOOKAHEAD_MM = 1200.0
ROUTE_DEVIATION_WARNING_MM = 1500.0
WALL_WARNING_MM = 650.0

# 실제 이동방향 계산용 최소 이동량
MOVEMENT_MIN_MM = 140.0
# 안내방향과 실제 이동방향 차이가 이 값 이상이면 반대방향 후보
# 노이즈로 한 번 튄 것을 막기 위해 연속 2번 확인

EVENT_DISPLAY_SECONDS = 2.0


def _segment_lengths() -> np.ndarray:
    return np.linalg.norm(
        ROUTE_POINTS[1:] - ROUTE_POINTS[:-1],
        axis=1,
    )


SEGMENT_LENGTHS = _segment_lengths()
ROUTE_CUMULATIVE = np.concatenate(
    ([0.0], np.cumsum(SEGMENT_LENGTHS))
)
ROUTE_TOTAL_MM = float(ROUTE_CUMULATIVE[-1])
CHECKPOINT_S = tuple(
    float(ROUTE_CUMULATIVE[index])
    for _, index in CHECKPOINTS
)


# ============================================================
# Korean text
# ============================================================

_FONT_CACHE = {}


def _candidate_fonts(bold: bool = False) -> list[str]:
    result = []
    if platform.system().lower() == "windows":
        if bold:
            result += [
                r"C:\Windows\Fonts\malgunbd.ttf",
                r"C:\Windows\Fonts\NanumGothicBold.ttf",
            ]
        result += [
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\gulim.ttc",
            r"C:\Windows\Fonts\batang.ttc",
        ]

    result += [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    return result


def _get_font(size: int, bold: bool = False):
    if ImageFont is None:
        return None

    key = (int(size), bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    for candidate in _candidate_fonts(bold):
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            font = ImageFont.truetype(str(path), int(size))
            _FONT_CACHE[key] = font
            return font
        except Exception:
            pass

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    _FONT_CACHE[key] = font
    return font


def draw_korean_texts(image_bgr: np.ndarray, items: Sequence[tuple]) -> np.ndarray:
    """
    item = (text, (x, y), size, BGR, bold)
    """
    if Image is None or ImageDraw is None or ImageFont is None:
        for text, xy, size, color, bold in items:
            fallback = text.encode("ascii", errors="ignore").decode("ascii").strip()
            if not fallback:
                continue
            cv2.putText(
                image_bgr,
                fallback,
                (int(xy[0]), int(xy[1] + size)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, size / 34.0),
                color,
                2 if bold else 1,
                cv2.LINE_AA,
            )
        return image_bgr

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)

    for text, xy, size, color_bgr, bold in items:
        font = _get_font(size, bold)
        color_rgb = (
            int(color_bgr[2]),
            int(color_bgr[1]),
            int(color_bgr[0]),
        )
        draw.text(
            (int(xy[0]), int(xy[1])),
            str(text),
            font=font,
            fill=color_rgb,
        )

    image_bgr[:] = cv2.cvtColor(
        np.asarray(pil_image),
        cv2.COLOR_RGB2BGR,
    )
    return image_bgr


# ============================================================
# Position stabilizer
# ============================================================

@dataclass
class FilterResult:
    position: Optional[np.ndarray]
    rejected: bool
    stale: bool
    message: str = ""


class PositionStabilizer:
    """
    큰 단발성 좌표 튐을 바로 반영하지 않는다.

    위치 인식이 끊기면 마지막 정상 위치를 삭제하지 않고
    다음 정상 위치가 들어올 때까지 계속 보존한다.
    """

    def __init__(
        self,
        history_size: int = 5,
        ema_alpha: float = 0.45,
        base_xy_jump_mm: float = 850.0,
        max_speed_mm_s: float = 2200.0,
        base_z_jump_mm: float = 600.0,
        max_z_speed_mm_s: float = 800.0,
        reacquire_radius_mm: float = 900.0,
        reacquire_count: int = 3,
    ):
        self.history = deque(maxlen=max(3, int(history_size)))
        self.ema_alpha = float(ema_alpha)
        self.base_xy_jump_mm = float(base_xy_jump_mm)
        self.max_speed_mm_s = float(max_speed_mm_s)
        self.base_z_jump_mm = float(base_z_jump_mm)
        self.max_z_speed_mm_s = float(max_z_speed_mm_s)
        self.reacquire_radius_mm = float(reacquire_radius_mm)
        self.reacquire_count = int(reacquire_count)

        self.output = None
        self.last_accept_time = None
        self.pending_center = None
        self.pending_count = 0
        self.total_rejected = 0

    def reset(self) -> None:
        self.history.clear()
        self.output = None
        self.last_accept_time = None
        self.pending_center = None
        self.pending_count = 0

    def _accept(self, raw: np.ndarray, now: float, hard_reset: bool = False) -> FilterResult:
        raw = np.asarray(raw, dtype=np.float64).reshape(3)

        if hard_reset:
            self.history.clear()
            self.output = None

        self.history.append(raw.copy())
        median = np.median(np.stack(list(self.history), axis=0), axis=0)

        if self.output is None:
            self.output = median
        else:
            self.output = (
                self.ema_alpha * median
                + (1.0 - self.ema_alpha) * self.output
            )

        self.last_accept_time = float(now)
        self.pending_center = None
        self.pending_count = 0

        return FilterResult(
            position=self.output.copy(),
            rejected=False,
            stale=False,
        )

    def update(
        self,
        position_world_mm: Optional[np.ndarray],
        now: Optional[float] = None,
    ) -> FilterResult:
        if now is None:
            now = time.monotonic()
        now = float(now)

        # 요청사항: 위치가 잠깐 사라져도 마지막 상태를 계속 유지
        if position_world_mm is None:
            if self.output is None:
                return FilterResult(
                    position=None,
                    rejected=False,
                    stale=True,
                    message="아직 유효한 위치가 없습니다.",
                )

            return FilterResult(
                position=self.output.copy(),
                rejected=False,
                stale=True,
                message="위치 인식 대기 중 - 마지막 위치와 안내를 유지합니다.",
            )

        raw = np.asarray(position_world_mm, dtype=np.float64).reshape(3)

        if self.output is None:
            return self._accept(raw, now, hard_reset=True)

        dt = (
            now - self.last_accept_time
            if self.last_accept_time is not None
            else 0.0
        )
        dt_gate = min(max(dt, 0.03), 3.0)

        dxy = hypot(
            float(raw[0] - self.output[0]),
            float(raw[1] - self.output[1]),
        )
        dz = abs(float(raw[2] - self.output[2]))

        allowed_xy = self.base_xy_jump_mm + self.max_speed_mm_s * dt_gate
        allowed_z = self.base_z_jump_mm + self.max_z_speed_mm_s * dt_gate

        if dxy <= allowed_xy and dz <= allowed_z:
            return self._accept(raw, now)

        if self.pending_center is None:
            self.pending_center = raw.copy()
            self.pending_count = 1
        else:
            pending_distance = float(
                np.linalg.norm(raw - self.pending_center)
            )
            if pending_distance <= self.reacquire_radius_mm:
                self.pending_count += 1
                self.pending_center = (
                    0.65 * self.pending_center
                    + 0.35 * raw
                )
            else:
                self.pending_center = raw.copy()
                self.pending_count = 1

        if self.pending_count >= self.reacquire_count:
            return self._accept(
                self.pending_center,
                now,
                hard_reset=True,
            )

        self.total_rejected += 1
        return FilterResult(
            position=self.output.copy(),
            rejected=True,
            stale=False,
            message=(
                "급격한 위치 튐 제거 "
                f"(ΔXY={dxy:.0f} mm, ΔZ={dz:.0f} mm)"
            ),
        )


# ============================================================
# Geometry
# ============================================================

def normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def angular_difference_deg(a_deg: float, b_deg: float) -> float:
    return abs(normalize_angle_deg(float(a_deg) - float(b_deg)))


def bearing_deg(start_xy: np.ndarray, end_xy: np.ndarray) -> float:
    delta = np.asarray(end_xy, dtype=np.float64) - np.asarray(
        start_xy,
        dtype=np.float64,
    )
    return degrees(atan2(float(delta[1]), float(delta[0])))


def bearing_label(angle_deg: Optional[float]) -> str:
    if angle_deg is None:
        return "-"

    a = float(angle_deg) % 360.0
    directions = (
        (22.5, "+X"),
        (67.5, "+X/+Y"),
        (112.5, "+Y"),
        (157.5, "-X/+Y"),
        (202.5, "-X"),
        (247.5, "-X/-Y"),
        (292.5, "-Y"),
        (337.5, "+X/-Y"),
        (360.0, "+X"),
    )
    for limit, label in directions:
        if a < limit:
            return f"{label} ({normalize_angle_deg(angle_deg):.0f}°)"
    return f"+X ({normalize_angle_deg(angle_deg):.0f}°)"


def project_to_route(point_xy: np.ndarray):
    p = np.asarray(point_xy, dtype=np.float64).reshape(2)
    best = None

    for index in range(len(ROUTE_POINTS) - 1):
        a = ROUTE_POINTS[index]
        b = ROUTE_POINTS[index + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))

        if denom <= 1e-12:
            t = 0.0
        else:
            t = float(np.dot(p - a, ab) / denom)
            t = min(1.0, max(0.0, t))

        projection = a + t * ab
        distance = float(np.linalg.norm(p - projection))
        s = float(ROUTE_CUMULATIVE[index]) + t * float(
            SEGMENT_LENGTHS[index]
        )

        candidate = (
            distance,
            index,
            t,
            projection,
            s,
        )
        if best is None or distance < best[0]:
            best = candidate

    assert best is not None
    return (
        int(best[1]),
        float(best[2]),
        np.asarray(best[3], dtype=np.float64),
        float(best[4]),
        float(best[0]),
    )


def point_at_route_s(s_mm: float) -> np.ndarray:
    s = min(ROUTE_TOTAL_MM, max(0.0, float(s_mm)))

    for index, length in enumerate(SEGMENT_LENGTHS):
        start_s = float(ROUTE_CUMULATIVE[index])
        end_s = float(ROUTE_CUMULATIVE[index + 1])

        if s <= end_s or index == len(SEGMENT_LENGTHS) - 1:
            if length <= 1e-9:
                return ROUTE_POINTS[index].copy()

            t = (s - start_s) / float(length)
            return ROUTE_POINTS[index] + t * (
                ROUTE_POINTS[index + 1] - ROUTE_POINTS[index]
            )

    return ROUTE_POINTS[-1].copy()


def route_points_to_s(s_mm: float) -> list[np.ndarray]:
    s = min(ROUTE_TOTAL_MM, max(0.0, float(s_mm)))
    result = [ROUTE_POINTS[0].copy()]

    for index in range(len(SEGMENT_LENGTHS)):
        end_s = float(ROUTE_CUMULATIVE[index + 1])
        if s >= end_s:
            result.append(ROUTE_POINTS[index + 1].copy())
        else:
            result.append(point_at_route_s(s))
            break

    return result


def _distance_point_to_segment(point_xy, a_xy, b_xy) -> float:
    p = np.asarray(point_xy, dtype=np.float64)
    a = np.asarray(a_xy, dtype=np.float64)
    b = np.asarray(b_xy, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))

    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _distance_to_rectangle(point_xy, rectangle) -> float:
    _, x1, y1, x2, y2 = rectangle
    x, y = map(float, point_xy)

    left, right = sorted((float(x1), float(x2)))
    bottom, top = sorted((float(y1), float(y2)))

    dx = max(left - x, 0.0, x - right)
    dy = max(bottom - y, 0.0, y - top)

    if left <= x <= right and bottom <= y <= top:
        return 0.0

    return hypot(dx, dy)


def _horizontal_wall_segments(y, x_min, x_max, openings):
    normalized = sorted(
        [
            (
                min(float(a), float(b)),
                max(float(a), float(b)),
            )
            for a, b in openings
        ],
        key=lambda item: item[0],
    )

    cursor = float(x_min)
    result = []

    for start, end in normalized:
        if start > cursor:
            result.append(
                (
                    np.array([cursor, float(y)], dtype=np.float64),
                    np.array([start, float(y)], dtype=np.float64),
                )
            )
        cursor = max(cursor, end)

    if cursor < float(x_max):
        result.append(
            (
                np.array([cursor, float(y)], dtype=np.float64),
                np.array([float(x_max), float(y)], dtype=np.float64),
            )
        )

    return result


BOTTOM_WALL_SEGMENTS = _horizontal_wall_segments(
    0.0,
    MAP_X_MIN,
    MAP_X_MAX,
    [(x1, x2) for x1, x2, _ in BOTTOM_DOORS],
)

UPPER_WALL_SEGMENTS = _horizontal_wall_segments(
    SIDE_WALL_Y,
    MAP_X_MIN,
    MAP_X_MAX,
    [
        (FRONT_DOOR_X1, FRONT_DOOR_X2),
        (SIDE_DOOR_X1, SIDE_DOOR_X2),
    ],
)


def nearest_hazard(position_xy: np.ndarray) -> tuple[str, float]:
    p = np.asarray(position_xy, dtype=np.float64).reshape(2)
    candidates = []

    names = {
        "table_1": "TABLE1",
        "water_1": "정수기1",
        "meeting_1": "회의실1",
        "meeting_2": "회의실2",
        "table_2": "TABLE2",
        "water_2": "정수기2",
        "wall_block": "벽 구조물",
        "table_3": "TABLE3",
    }

    for rectangle in RECTANGLES:
        candidates.append(
            (
                names.get(str(rectangle[0]), str(rectangle[0])),
                _distance_to_rectangle(p, rectangle),
            )
        )

    for a, b in BOTTOM_WALL_SEGMENTS:
        candidates.append(
            ("하단 벽", _distance_point_to_segment(p, a, b))
        )

    for a, b in UPPER_WALL_SEGMENTS:
        candidates.append(
            ("상단 벽", _distance_point_to_segment(p, a, b))
        )

    candidates.append(
        (
            "동쪽 벽",
            _distance_point_to_segment(
                p,
                np.array([EAST_WALL_X, MAP_Y_MIN]),
                np.array([EAST_WALL_X, MAP_Y_MAX]),
            ),
        )
    )

    return min(candidates, key=lambda item: item[1])



# ============================================================
# Navigation state
# ============================================================

@dataclass
class NavigationGuidance:
    active: bool
    completed: bool
    route_started: bool

    headline: str
    target_name: str
    instruction: str

    target_distance_mm: float
    route_remaining_mm: float
    route_progress_s: float
    route_distance_mm: float

    desired_point_xy: Optional[np.ndarray]
    desired_bearing_deg: Optional[float]
    relative_turn_deg: Optional[float]

    # 카메라 시선과 별개로 좌표 변화에서 계산한 실제 이동 방향
    movement_bearing_deg: Optional[float]
    movement_relative_to_view_deg: Optional[float]

    safety_warning: str
    route_warning: str

    # 위치 인식이 끊겼을 때 마지막 안내 상태 유지 여부
    localization_waiting: bool

    # 2초간 크게 표시할 이벤트
    event_message: str


class IndoorNavigator:
    def __init__(self):
        self.active = False
        self.completed = False
        self.route_started = False

        # 0=앞문, 1=4번, 2=2번, 3=뒷문
        self.stage_index = 0

        self.last_position = None

        self.event_message = ""
        self.event_until = 0.0

        self.last_guidance = None

    def _set_event(
        self,
        message: str,
        now: Optional[float] = None,
    ) -> None:
        if now is None:
            now = time.monotonic()

        self.event_message = str(message)
        self.event_until = (
            float(now)
            +
            EVENT_DISPLAY_SECONDS
        )

    def _current_event(
        self,
        now: float,
    ) -> str:
        if float(now) <= self.event_until:
            return self.event_message

        return ""

    @property
    def current_checkpoint(self):
        return CHECKPOINTS[
            self.stage_index
        ]

    def passed_checkpoint_count(
        self,
    ) -> int:
        if self.completed:
            return len(
                CHECKPOINTS
            )

        return max(
            0,
            min(
                self.stage_index,
                len(
                    CHECKPOINTS
                ),
            ),
        )

    def stop(
        self,
        now: Optional[float] = None,
    ) -> None:
        if now is None:
            now = time.monotonic()

        self.active = False
        self.completed = False
        self.route_started = False
        self.stage_index = 0
        self.last_position = None
        self.last_guidance = None

        self._set_event(
            "네비게이션 종료",
            now,
        )

    def start(
        self,
        position_world_mm: Optional[np.ndarray],
        now: Optional[float] = None,
    ) -> None:
        if now is None:
            now = time.monotonic()

        self.active = True
        self.completed = False
        self.route_started = False
        self.stage_index = 0
        self.last_position = None
        self.last_guidance = None

        if position_world_mm is None:
            self._set_event(
                "네비게이션 시작\n"
                "위치를 인식하면 앞문부터 안내합니다.",
                now,
            )
            return

        p = np.asarray(
            position_world_mm,
            dtype=np.float64,
        ).reshape(3)

        distance_to_front = hypot(
            float(
                p[0]
                -
                FRONT.x
            ),
            float(
                p[1]
                -
                FRONT.y
            ),
        )

        if (
            distance_to_front
            <=
            START_RADIUS_MM
        ):
            self.route_started = True
            self.stage_index = 1

            self._set_event(
                "네비게이션 시작\n앞문 출발",
                now,
            )
        else:
            self._set_event(
                "네비게이션 시작\n"
                "먼저 앞문으로 이동하세요.",
                now,
            )

    def _advance_if_arrived(
        self,
        position_xy: np.ndarray,
        now: float,
    ) -> None:
        if not self.active:
            return

        if self.stage_index == 0:
            distance = float(
                np.linalg.norm(
                    position_xy
                    -
                    ROUTE_POINTS[0]
                )
            )

            if (
                distance
                <=
                START_RADIUS_MM
            ):
                self.route_started = True
                self.stage_index = 1

                self._set_event(
                    "앞문 도착\n네비게이션 출발",
                    now,
                )

            return

        (
            checkpoint_name,
            route_index,
        ) = CHECKPOINTS[
            self.stage_index
        ]

        target_xy = ROUTE_POINTS[
            route_index
        ]

        distance = float(
            np.linalg.norm(
                position_xy
                -
                target_xy
            )
        )

        if (
            distance
            >
            CHECKPOINT_RADIUS_MM
        ):
            return

        if self.stage_index == 1:
            self.stage_index = 2

            self._set_event(
                "4번 강의실 경유",
                now,
            )
            return

        if self.stage_index == 2:
            self.stage_index = 3

            self._set_event(
                "2번 강의실 경유",
                now,
            )
            return

        self.completed = True
        self.active = False
        self.route_started = True

        self._set_event(
            "뒷문 도착\n안내 완료",
            now,
        )

    def _waiting_guidance(
        self,
        now: float,
    ) -> NavigationGuidance:
        event = self._current_event(
            now
        )

        if self.last_guidance is not None:
            # 위치가 잠깐 끊겨도 마지막 경로/목적지/방향 상태를 유지.
            waiting = replace(
                self.last_guidance,
                localization_waiting=True,
                event_message=event,
            )

            self.last_guidance = waiting
            return waiting

        guidance = NavigationGuidance(
            active=self.active,
            completed=self.completed,
            route_started=self.route_started,
            headline=(
                "위치 인식 대기 중"
                if self.active
                else
                "네비게이션 시작 버튼을 누르세요."
            ),
            target_name=(
                "앞문"
                if self.active
                else ""
            ),
            instruction=(
                "마커가 다시 인식될 때까지 "
                "마지막 경로 상태를 유지합니다."
                if self.active
                else
                "출발점은 앞문입니다."
            ),
            target_distance_mm=0.0,
            route_remaining_mm=0.0,
            route_progress_s=0.0,
            route_distance_mm=0.0,
            desired_point_xy=None,
            desired_bearing_deg=None,
            relative_turn_deg=None,
            movement_bearing_deg=None,
            movement_relative_to_view_deg=None,
            safety_warning="",
            route_warning="",
            localization_waiting=True,
            event_message=event,
        )

        self.last_guidance = guidance
        return guidance

    def update(
        self,
        position_world_mm: Optional[np.ndarray],
        camera_yaw_deg: Optional[float],
        now: Optional[float] = None,
    ) -> NavigationGuidance:
        if now is None:
            now = time.monotonic()

        now = float(
            now
        )

        if position_world_mm is None:
            return self._waiting_guidance(
                now
            )

        position = np.asarray(
            position_world_mm,
            dtype=np.float64,
        ).reshape(3)

        xy = position[
            :2
        ]

        (
            _,
            _,
            projection,
            route_s,
            route_distance,
        ) = project_to_route(
            xy
        )

        if self.active:
            self._advance_if_arrived(
                xy,
                now,
            )

        event = self._current_event(
            now
        )

        if self.completed:
            guidance = NavigationGuidance(
                active=False,
                completed=True,
                route_started=True,
                headline="뒷문에 도착했습니다.",
                target_name="뒷문",
                instruction="안내를 완료했습니다.",
                target_distance_mm=0.0,
                route_remaining_mm=0.0,
                route_progress_s=ROUTE_TOTAL_MM,
                route_distance_mm=route_distance,
                desired_point_xy=None,
                desired_bearing_deg=None,
                relative_turn_deg=None,
                movement_bearing_deg=None,
                movement_relative_to_view_deg=None,
                safety_warning="",
                route_warning="",
                localization_waiting=False,
                event_message=event,
            )

            self.last_guidance = guidance
            return guidance

        if not self.active:
            guidance = NavigationGuidance(
                active=False,
                completed=False,
                route_started=False,
                headline="네비게이션 시작 버튼을 누르세요.",
                target_name="",
                instruction="출발점은 앞문입니다.",
                target_distance_mm=0.0,
                route_remaining_mm=0.0,
                route_progress_s=route_s,
                route_distance_mm=route_distance,
                desired_point_xy=None,
                desired_bearing_deg=None,
                relative_turn_deg=None,
                movement_bearing_deg=None,
                movement_relative_to_view_deg=None,
                safety_warning="",
                route_warning="",
                localization_waiting=False,
                event_message=event,
            )

            self.last_guidance = guidance
            return guidance

        (
            checkpoint_name,
            route_index,
        ) = self.current_checkpoint

        target_xy = ROUTE_POINTS[
            route_index
        ]

        target_s = float(
            ROUTE_CUMULATIVE[
                route_index
            ]
        )

        target_distance = float(
            np.linalg.norm(
                xy
                -
                target_xy
            )
        )

        route_warning = ""

        if (
            route_distance
            >
            ROUTE_DEVIATION_WARNING_MM
        ):
            desired_point = projection

            route_warning = (
                "경로에서 벗어났습니다. "
                "경로 선 쪽으로 복귀하세요."
            )

        elif self.stage_index == 0:
            # 앞문이 아닌 곳에서 시작하면 경로를 역으로 따라 앞문 복귀.
            desired_s = max(
                0.0,
                route_s
                -
                LOOKAHEAD_MM,
            )

            desired_point = point_at_route_s(
                desired_s
            )

            if route_s < LOOKAHEAD_MM:
                desired_point = (
                    ROUTE_POINTS[0]
                    .copy()
                )

        else:
            if route_s <= target_s:
                desired_s = min(
                    target_s,
                    route_s
                    +
                    LOOKAHEAD_MM,
                )

                desired_point = point_at_route_s(
                    desired_s
                )
            else:
                desired_point = (
                    target_xy.copy()
                )

        desired_bearing = bearing_deg(
            xy,
            desired_point,
        )

        # 카메라 시선 대비 어느 방향으로 화면을 돌려야 하는지.
        if camera_yaw_deg is None:
            relative_turn = None

            instruction = (
                f"{checkpoint_name} 방향으로 "
                "방향 카드를 따라 이동하세요."
            )
        else:
            relative_turn = normalize_angle_deg(
                desired_bearing
                -
                float(
                    camera_yaw_deg
                )
            )

            abs_turn = abs(
                relative_turn
            )

            if abs_turn <= 15.0:
                instruction = "현재 보고 있는 방향으로 직진하세요."
            else:
                side = (
                    "왼쪽"
                    if relative_turn > 0.0
                    else
                    "오른쪽"
                )

                instruction = (
                    f"현재 시선 기준 {side} "
                    f"{abs_turn:.0f}° 방향으로 가세요."
                )

        # 실제 이동 방향은 표시용으로만 계산.
        # V3에서는 역방향 판정/경고를 하지 않는다.
        movement_bearing = None
        movement_relative_to_view = None

        if self.last_position is not None:
            movement = (
                xy
                -
                self.last_position
            )

            movement_norm = float(
                np.linalg.norm(
                    movement
                )
            )

            if (
                movement_norm
                >=
                MOVEMENT_MIN_MM
            ):
                movement_bearing = bearing_deg(
                    np.array(
                        [0.0, 0.0]
                    ),
                    movement,
                )

                if camera_yaw_deg is not None:
                    movement_relative_to_view = (
                        normalize_angle_deg(
                            movement_bearing
                            -
                            float(
                                camera_yaw_deg
                            )
                        )
                    )

        self.last_position = (
            xy.copy()
        )

        (
            hazard_name,
            hazard_distance,
        ) = nearest_hazard(
            xy
        )

        safety_warning = ""

        if (
            hazard_distance
            <
            WALL_WARNING_MM
        ):
            safety_warning = (
                f"{hazard_name}과 너무 가깝습니다. "
                f"거리 약 {hazard_distance:.0f} mm"
            )

        if self.stage_index == 0:
            headline = (
                "먼저 앞문으로 이동하세요."
            )

            route_remaining = (
                route_s
                +
                route_distance
            )
        else:
            headline = (
                f"다음 경유/목적지: "
                f"{checkpoint_name}"
            )

            route_remaining = (
                max(
                    0.0,
                    target_s
                    -
                    route_s
                )
                +
                route_distance
            )

        guidance = NavigationGuidance(
            active=True,
            completed=False,
            route_started=self.route_started,
            headline=headline,
            target_name=checkpoint_name,
            instruction=instruction,
            target_distance_mm=target_distance,
            route_remaining_mm=float(
                route_remaining
            ),
            route_progress_s=float(
                route_s
            ),
            route_distance_mm=float(
                route_distance
            ),
            desired_point_xy=np.asarray(
                desired_point,
                dtype=np.float64,
            ),
            desired_bearing_deg=float(
                desired_bearing
            ),
            relative_turn_deg=(
                None
                if relative_turn is None
                else float(
                    relative_turn
                )
            ),
            movement_bearing_deg=(
                None
                if movement_bearing is None
                else float(
                    movement_bearing
                )
            ),
            movement_relative_to_view_deg=(
                None
                if movement_relative_to_view is None
                else float(
                    movement_relative_to_view
                )
            ),
            safety_warning=safety_warning,
            route_warning=route_warning,
            localization_waiting=False,
            event_message=event,
        )

        self.last_guidance = guidance
        return guidance



# ============================================================
# Drawing helpers
# ============================================================

def _blend_panel(
    image,
    p1,
    p2,
    color,
    alpha,
):
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        p1,
        p2,
        color,
        -1,
    )
    cv2.addWeighted(
        overlay,
        float(alpha),
        image,
        1.0 - float(alpha),
        0,
        image,
    )


def _text_width(
    text: str,
    size: int,
    bold: bool = False,
) -> int:
    font = _get_font(size, bold)

    if (
        Image is not None
        and ImageDraw is not None
        and font is not None
    ):
        try:
            canvas = Image.new("RGB", (4, 4))
            draw = ImageDraw.Draw(canvas)
            box = draw.textbbox(
                (0, 0),
                str(text),
                font=font,
            )
            return int(box[2] - box[0])
        except Exception:
            pass

    return int(len(str(text)) * int(size) * 0.58)


def centered_text_item(
    image_width: int,
    text: str,
    y: int,
    size: int,
    color,
    bold: bool = False,
):
    width = _text_width(
        text,
        size,
        bold,
    )

    x = max(
        8,
        int(
            (
                int(image_width)
                - width
            )
            / 2
        ),
    )

    return (
        text,
        (x, int(y)),
        int(size),
        color,
        bool(bold),
    )


def _rounded_rect(
    image: np.ndarray,
    p1,
    p2,
    radius: int,
    color,
    thickness: int = -1,
):
    x1, y1 = p1
    x2, y2 = p2
    r = max(
        0,
        min(
            int(radius),
            (x2 - x1) // 2,
            (y2 - y1) // 2,
        ),
    )

    if thickness < 0:
        cv2.rectangle(
            image,
            (x1 + r, y1),
            (x2 - r, y2),
            color,
            -1,
        )
        cv2.rectangle(
            image,
            (x1, y1 + r),
            (x2, y2 - r),
            color,
            -1,
        )
        cv2.circle(
            image,
            (x1 + r, y1 + r),
            r,
            color,
            -1,
        )
        cv2.circle(
            image,
            (x2 - r, y1 + r),
            r,
            color,
            -1,
        )
        cv2.circle(
            image,
            (x1 + r, y2 - r),
            r,
            color,
            -1,
        )
        cv2.circle(
            image,
            (x2 - r, y2 - r),
            r,
            color,
            -1,
        )
        return

    cv2.line(
        image,
        (x1 + r, y1),
        (x2 - r, y1),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (x1 + r, y2),
        (x2 - r, y2),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (x1, y1 + r),
        (x1, y2 - r),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (x2, y1 + r),
        (x2, y2 - r),
        color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.ellipse(
        image,
        (x1 + r, y1 + r),
        (r, r),
        180,
        0,
        90,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (x2 - r, y1 + r),
        (r, r),
        270,
        0,
        90,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (x2 - r, y2 - r),
        (r, r),
        0,
        0,
        90,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (x1 + r, y2 - r),
        (r, r),
        90,
        0,
        90,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _arrow_endpoint(
    center,
    angle_deg,
    radius,
):
    angle = radians(float(angle_deg))
    dx = sin(angle)
    dy = -cos(angle)

    return (
        int(round(center[0] + dx * radius)),
        int(round(center[1] + dy * radius)),
    )



# ============================================================
# Camera direction guidance
# ============================================================

def _turn_instruction_from_relative(
    relative_turn_deg: Optional[float],
) -> tuple[str, str, str]:
    """
    현재 카메라 시선을 0도로 두고
    실제로 어느 방향으로 가야 하는지를 텍스트로 만든다.

    return:
        title       : 큰 제목
        explanation : 기존 설명형 지시
        command     : 아래에 추가하는 짧고 직접적인 이동 지시
    """
    if relative_turn_deg is None:
        return (
            "방향 계산 중",
            "카메라 시선 방향을 계산하고 있습니다.",
            "잠시 방향을 확인하세요.",
        )

    value = float(relative_turn_deg)
    magnitude = abs(value)

    if magnitude <= 12.0:
        return (
            "직진",
            "현재 보고 있는 방향이 이동 방향과 거의 같습니다.",
            "직진하세요.",
        )

    side = (
        "왼쪽"
        if value > 0.0
        else
        "오른쪽"
    )

    if magnitude >= 150.0:
        explanation = (
            f"현재 시선의 {side} 뒤쪽 방향으로 가야 합니다."
        )
    else:
        explanation = (
            f"현재 시선 기준 {side} 방향으로 가세요."
        )

    return (
        f"{side} {magnitude:.0f}°",
        explanation,
        f"{side}으로 {magnitude:.0f}° 가세요.",
    )


def _draw_turn_icon(
    image: np.ndarray,
    center: tuple[int, int],
    relative_turn_deg: Optional[float],
    radius: int,
):
    """
    큰 방향 아이콘.

    V7에서는 둥근 화살표/U턴 화살표를 전혀 사용하지 않는다.
    모든 방향을 직선 화살표 하나로 표시한다.

    화면 기준:
        0°      = 위/직진
        +90°    = 왼쪽
        -90°    = 오른쪽
        ±180°   = 뒤쪽
    """
    cx, cy = center

    radius = max(
        28,
        int(radius),
    )

    cv2.circle(
        image,
        (cx, cy),
        radius,
        (35, 42, 52),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        image,
        (cx, cy),
        radius,
        (68, 78, 92),
        1,
        cv2.LINE_AA,
    )

    if relative_turn_deg is None:
        cv2.circle(
            image,
            (cx, cy),
            max(
                5,
                radius // 10,
            ),
            (0, 210, 240),
            -1,
            cv2.LINE_AA,
        )
        return

    value = float(
        relative_turn_deg
    )

    # navigation relative convention:
    # + = left, - = right.
    # _arrow_endpoint() uses + = screen-right,
    # so sign is inverted here.
    screen_angle = -value

    start = (
        cx,
        cy,
    )

    end = _arrow_endpoint(
        start,
        screen_angle,
        int(
            radius
            *
            0.72
        ),
    )

    cv2.arrowedLine(
        image,
        start,
        end,
        (0, 220, 245),
        8,
        cv2.LINE_AA,
        tipLength=0.30,
    )


def _direction_marker_x(
    center_x: int,
    half_width: int,
    relative_angle_deg: float,
) -> int:
    """
    화면 중앙 = 현재 카메라 정면.
    + angle은 왼쪽이므로 x가 감소한다.
    """
    clamped = max(
        -100.0,
        min(
            100.0,
            float(
                relative_angle_deg
            ),
        ),
    )

    return int(
        round(
            center_x
            -
            (
                clamped
                /
                100.0
            )
            *
            half_width
        )
    )


def _draw_heading_gauge(
    image: np.ndarray,
    rect,
    guidance: NavigationGuidance,
):
    """
    시선과 목표 방향을 한눈에 비교하는 수평 게이지.

    가운데 흰 선 = 현재 카메라 정면
    노란 삼각형 = 가야 할 방향
    초록 점 = 실제 이동 방향
    """
    x1, y1, x2, y2 = rect

    center_x = (
        x1 + x2
    ) // 2

    line_y = (
        y1 + y2
    ) // 2

    half = max(
        20,
        (
            x2
            -
            x1
        )
        //
        2
        -
        16,
    )

    cv2.line(
        image,
        (
            x1 + 12,
            line_y,
        ),
        (
            x2 - 12,
            line_y,
        ),
        (86, 96, 110),
        2,
        cv2.LINE_AA,
    )

    # tick marks
    for ratio in (
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    ):
        tx = int(
            round(
                center_x
                +
                ratio
                *
                half
            )
        )

        tick_h = (
            12
            if ratio == 0.0
            else 7
        )

        color = (
            (255, 255, 255)
            if ratio == 0.0
            else
            (105, 115, 128)
        )

        cv2.line(
            image,
            (
                tx,
                line_y - tick_h,
            ),
            (
                tx,
                line_y + tick_h,
            ),
            color,
            2,
            cv2.LINE_AA,
        )

    # current camera forward label
    draw_korean_texts(
        image,
        [
            (
                "정면",
                (
                    center_x - 17,
                    y2 - 22,
                ),
                13,
                (235, 238, 242),
                True,
            ),
            (
                "왼쪽",
                (
                    x1 + 4,
                    y2 - 22,
                ),
                12,
                (125, 135, 150),
                False,
            ),
            (
                "오른쪽",
                (
                    x2 - 48,
                    y2 - 22,
                ),
                12,
                (125, 135, 150),
                False,
            ),
        ],
    )

    if guidance.relative_turn_deg is not None:
        target_x = _direction_marker_x(
            center_x,
            half,
            guidance.relative_turn_deg,
        )

        triangle = np.array(
            [
                [
                    target_x,
                    line_y - 18,
                ],
                [
                    target_x - 8,
                    line_y - 31,
                ],
                [
                    target_x + 8,
                    line_y - 31,
                ],
            ],
            dtype=np.int32,
        )

        cv2.fillConvexPoly(
            image,
            triangle,
            (0, 220, 245),
            cv2.LINE_AA,
        )

    if (
        guidance.movement_relative_to_view_deg
        is not None
    ):
        move_x = _direction_marker_x(
            center_x,
            half,
            guidance
            .movement_relative_to_view_deg,
        )

        cv2.circle(
            image,
            (
                move_x,
                line_y + 18,
            ),
            6,
            (70, 220, 95),
            -1,
            cv2.LINE_AA,
        )


def draw_direction_guidance_card(
    frame: np.ndarray,
    guidance: NavigationGuidance,
    camera_yaw_deg: Optional[float],
):
    """
    화면을 가리지 않는 우측 상단 길안내 카드.

    도로를 영상 위에 그리지 않고,
    카메라 정면과 가야 할 방향의 차이를 표지판처럼 보여준다.
    """
    h, w = frame.shape[:2]

    card_w = min(
        430,
        max(
            350,
            int(
                w
                *
                0.34
            ),
        ),
    )

    card_h = min(
        235,
        max(
            220,
            int(
                h
                *
                0.31
            ),
        ),
    )

    x2 = w - 18
    x1 = x2 - card_w
    y1 = 58
    y2 = y1 + card_h

    # dark translucent card
    overlay = frame.copy()

    _rounded_rect(
        overlay,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        20,
        (18, 23, 31),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.88,
        frame,
        0.12,
        0,
        frame,
    )

    _rounded_rect(
        frame,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        20,
        (68, 76, 88),
        1,
    )

    title, subtitle, command_line = (
        _turn_instruction_from_relative(
            guidance.relative_turn_deg
        )
    )

    icon_center = (
        x1 + 72,
        y1 + 73,
    )

    _draw_turn_icon(
        frame,
        icon_center,
        guidance.relative_turn_deg,
        radius=48,
    )

    # target info
    target_line = (
        guidance.target_name
        if guidance.target_name
        else "목표 대기"
    )

    if (
        guidance.target_name
        and
        guidance.target_distance_mm > 0.0
    ):
        target_line = (
            f"{guidance.target_name} · "
            f"{guidance.target_distance_mm / 1000.0:.1f} m"
        )

    text_x = x1 + 136

    draw_korean_texts(
        frame,
        [
            (
                "시선 기준 길안내",
                (
                    text_x,
                    y1 + 18,
                ),
                14,
                (125, 140, 160),
                True,
            ),
            (
                title,
                (
                    text_x,
                    y1 + 43,
                ),
                27,
                (0, 225, 250),
                True,
            ),
            (
                target_line,
                (
                    text_x,
                    y1 + 80,
                ),
                17,
                (238, 241, 245),
                True,
            ),
            (
                subtitle,
                (
                    x1 + 22,
                    y1 + 111,
                ),
                15,
                (200, 207, 216),
                False,
            ),
            (
                command_line,
                (
                    x1 + 22,
                    y1 + 139,
                ),
                18,
                (0, 225, 250),
                True,
            ),
        ],
    )

    gauge_rect = (
        x1 + 18,
        y1 + 168,
        x2 - 18,
        y2 - 8,
    )

    _draw_heading_gauge(
        frame,
        gauge_rect,
        guidance,
    )


def draw_camera_navigation_overlay(
    frame: np.ndarray,
    navigator: IndoorNavigator,
    guidance: NavigationGuidance,
    position_world_mm: Optional[np.ndarray],
    nearest_door_name: str,
    nearest_door_distance_mm: Optional[float],
    filter_result: FilterResult,
    camera_yaw_deg: Optional[float],
    frame_number: int = 0,
    display_fps: float = 0.0,
    processing_ms: float = 0.0,
    processing_fps: float = 0.0,
):
    """
    V5 webcam UI:
    - 중앙 영상에는 길을 덮어 그리지 않음
    - 우측 상단에 compact 방향 카드
    - 현재 카메라 정면과 목표 방향 차이를 게이지로 표현
    - 텍스트 + 방향 아이콘을 함께 사용
    """
    h, w = frame.shape[:2]

    # slim route bar
    route_bar_h = 46

    _blend_panel(
        frame,
        (0, 0),
        (
            w - 1,
            route_bar_h,
        ),
        (10, 14, 20),
        0.68,
    )

    route_line = (
        "앞문  →  4번 강의실  →  "
        "2번 강의실  →  뒷문"
    )

    draw_korean_texts(
        frame,
        [
            centered_text_item(
                w,
                route_line,
                11,
                19,
                (242, 245, 248),
                True,
            )
        ],
    )

    if guidance.active:
        draw_direction_guidance_card(
            frame,
            guidance,
            camera_yaw_deg,
        )
    else:
        # Small idle message only.
        idle_w = min(
            380,
            max(
                300,
                int(
                    w
                    *
                    0.30
                ),
            ),
        )

        idle_x2 = w - 18
        idle_x1 = idle_x2 - idle_w
        idle_y1 = 60
        idle_y2 = 112

        overlay = frame.copy()

        _rounded_rect(
            overlay,
            (
                idle_x1,
                idle_y1,
            ),
            (
                idle_x2,
                idle_y2,
            ),
            16,
            (18, 23, 31),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.86,
            frame,
            0.14,
            0,
            frame,
        )

        draw_korean_texts(
            frame,
            [
                centered_text_item(
                    w,
                    "Navigation Control에서 시작을 눌러주세요.",
                    idle_y1 + 14,
                    16,
                    (225, 230, 236),
                    True,
                )
            ],
        )

    # compact legend in lower-left, away from central view
    legend_x1 = 16
    legend_y1 = h - 68
    legend_x2 = min(
        w - 16,
        390,
    )
    legend_y2 = h - 16

    overlay = frame.copy()

    _rounded_rect(
        overlay,
        (
            legend_x1,
            legend_y1,
        ),
        (
            legend_x2,
            legend_y2,
        ),
        14,
        (12, 17, 24),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.80,
        frame,
        0.20,
        0,
        frame,
    )

    # marker legend
    center_line_x = legend_x1 + 28
    mid_y = legend_y1 + 18

    cv2.line(
        frame,
        (
            center_line_x,
            mid_y - 9,
        ),
        (
            center_line_x,
            mid_y + 9,
        ),
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    tri = np.array(
        [
            [
                center_line_x + 94,
                mid_y - 7,
            ],
            [
                center_line_x + 86,
                mid_y - 18,
            ],
            [
                center_line_x + 102,
                mid_y - 18,
            ],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        frame,
        tri,
        (0, 220, 245),
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        (
            center_line_x + 206,
            mid_y - 11,
        ),
        6,
        (70, 220, 95),
        -1,
        cv2.LINE_AA,
    )

    draw_korean_texts(
        frame,
        [
            (
                "현재 시선",
                (
                    center_line_x + 12,
                    legend_y1 + 8,
                ),
                13,
                (230, 233, 238),
                False,
            ),
            (
                "목표",
                (
                    center_line_x + 108,
                    legend_y1 + 8,
                ),
                13,
                (0, 220, 245),
                True,
            ),
            (
                "실제 이동",
                (
                    center_line_x + 220,
                    legend_y1 + 8,
                ),
                13,
                (70, 220, 95),
                True,
            ),
            (
                "카드의 가운데 흰 선이 지금 카메라가 보고 있는 정면입니다.",
                (
                    legend_x1 + 14,
                    legend_y1 + 30,
                ),
                12,
                (165, 175, 188),
                False,
            ),
        ],
    )

    # status chips, kept at upper-left
    chip_y = 60

    if guidance.localization_waiting:
        chip_w = 340

        overlay = frame.copy()

        _rounded_rect(
            overlay,
            (
                16,
                chip_y,
            ),
            (
                16 + chip_w,
                chip_y + 40,
            ),
            14,
            (20, 28, 38),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.84,
            frame,
            0.16,
            0,
            frame,
        )

        draw_korean_texts(
            frame,
            [
                (
                    "위치 인식 대기 · 마지막 안내 상태 유지",
                    (
                        30,
                        chip_y + 9,
                    ),
                    15,
                    (0, 190, 255),
                    True,
                )
            ],
        )

        chip_y += 48

    if filter_result.rejected:
        overlay = frame.copy()

        _rounded_rect(
            overlay,
            (
                16,
                chip_y,
            ),
            (
                250,
                chip_y + 40,
            ),
            14,
            (20, 28, 38),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.84,
            frame,
            0.16,
            0,
            frame,
        )

        draw_korean_texts(
            frame,
            [
                (
                    "급격한 위치 튐 제거",
                    (
                        30,
                        chip_y + 9,
                    ),
                    15,
                    (0, 180, 255),
                    True,
                )
            ],
        )

        chip_y += 48

    if guidance.route_warning:
        overlay = frame.copy()

        _rounded_rect(
            overlay,
            (
                16,
                chip_y,
            ),
            (
                min(
                    w - 18,
                    480,
                ),
                chip_y + 42,
            ),
            14,
            (20, 28, 38),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.86,
            frame,
            0.14,
            0,
            frame,
        )

        draw_korean_texts(
            frame,
            [
                (
                    guidance.route_warning,
                    (
                        30,
                        chip_y + 9,
                    ),
                    15,
                    (0, 180, 255),
                    True,
                )
            ],
        )

        chip_y += 50

    if guidance.safety_warning:
        overlay = frame.copy()

        _rounded_rect(
            overlay,
            (
                16,
                chip_y,
            ),
            (
                min(
                    w - 18,
                    500,
                ),
                chip_y + 42,
            ),
            14,
            (20, 28, 38),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.86,
            frame,
            0.14,
            0,
            frame,
        )

        draw_korean_texts(
            frame,
            [
                (
                    "주의 · "
                    + guidance.safety_warning,
                    (
                        30,
                        chip_y + 9,
                    ),
                    15,
                    (0, 110, 255),
                    True,
                )
            ],
        )

    draw_event_banner(
        frame,
        guidance.event_message,
    )


def draw_event_banner(
    frame: np.ndarray,
    message: str,
):
    if not message:
        return

    h, w = frame.shape[:2]

    card_w = min(
        int(w * 0.64),
        760,
    )

    card_h = min(
        160,
        max(118, h // 5),
    )

    x1 = (w - card_w) // 2
    x2 = x1 + card_w
    y1 = max(
        120,
        h // 2 - card_h // 2,
    )
    y2 = y1 + card_h

    overlay = frame.copy()

    _rounded_rect(
        overlay,
        (x1, y1),
        (x2, y2),
        24,
        (18, 110, 58),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.90,
        frame,
        0.10,
        0,
        frame,
    )

    _rounded_rect(
        frame,
        (x1, y1),
        (x2, y2),
        24,
        (235, 245, 240),
        2,
    )

    lines = str(message).splitlines()
    font_size = (
        34
        if len(lines) == 1
        else 29
    )

    base_y = (
        y1
        + 24
        if len(lines) > 1
        else y1 + 42
    )

    items = []

    for i, line in enumerate(lines):
        items.append(
            centered_text_item(
                w,
                line,
                base_y
                + i * (font_size + 14),
                font_size,
                (255, 255, 255),
                True,
            )
        )

    draw_korean_texts(
        frame,
        items,
    )


# ============================================================
# Minimap: no buttons, no dashboard info
# ============================================================

def _route_pixels(
    renderer,
    points,
):
    return [
        renderer.world_to_pixel(
            float(p[0]),
            float(p[1]),
        )
        for p in points
    ]


def draw_navigation_map(
    map_image: np.ndarray,
    renderer,
    navigator: IndoorNavigator,
    guidance: NavigationGuidance,
    position_world_mm: Optional[np.ndarray],
):
    full_pixels = np.asarray(
        _route_pixels(
            renderer,
            ROUTE_POINTS,
        ),
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    cv2.polylines(
        map_image,
        [full_pixels],
        False,
        (210, 145, 55),
        5,
        cv2.LINE_AA,
    )

    if navigator.route_started:
        progress_s = (
            ROUTE_TOTAL_MM
            if navigator.completed
            else guidance.route_progress_s
        )

        progress_points = route_points_to_s(
            progress_s
        )

        progress_pixels = np.asarray(
            _route_pixels(
                renderer,
                progress_points,
            ),
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        if len(progress_pixels) >= 2:
            cv2.polylines(
                map_image,
                [progress_pixels],
                False,
                (40, 180, 85),
                8,
                cv2.LINE_AA,
            )

    labels = (
        ("앞문", 0),
        ("4번", 5),
        ("2번", 8),
        ("뒷문", 11),
    )

    passed_count = (
        navigator
        .passed_checkpoint_count()
    )

    text_items = [
        centered_text_item(
            map_image.shape[1],
            "앞문  →  4번 강의실  →  2번 강의실  →  뒷문",
            84,
            18,
            (35, 35, 35),
            True,
        )
    ]

    for i, (label, index) in enumerate(labels):
        point = ROUTE_POINTS[index]

        px = renderer.world_to_pixel(
            float(point[0]),
            float(point[1]),
        )

        if (
            navigator.completed
            or i < passed_count
        ):
            node_color = (
                40,
                180,
                85,
            )
        elif (
            i == navigator.stage_index
            and navigator.active
        ):
            node_color = (
                0,
                180,
                235,
            )
        else:
            node_color = (
                125,
                125,
                125,
            )

        cv2.circle(
            map_image,
            px,
            11,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            map_image,
            px,
            8,
            node_color,
            -1,
            cv2.LINE_AA,
        )

        text_items.append(
            (
                label,
                (
                    px[0] + 9,
                    px[1] - 22,
                ),
                16,
                (35, 35, 35),
                True,
            )
        )

    if position_world_mm is not None:
        p = np.asarray(
            position_world_mm,
            dtype=np.float64,
        ).reshape(3)

        current_px = renderer.world_to_pixel(
            float(p[0]),
            float(p[1]),
        )

        cv2.circle(
            map_image,
            current_px,
            13,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            map_image,
            current_px,
            9,
            (210, 80, 210),
            -1,
            cv2.LINE_AA,
        )

    if guidance.desired_point_xy is not None:
        desired_px = renderer.world_to_pixel(
            float(
                guidance
                .desired_point_xy[0]
            ),
            float(
                guidance
                .desired_point_xy[1]
            ),
        )

        cv2.circle(
            map_image,
            desired_px,
            6,
            (0, 210, 235),
            -1,
            cv2.LINE_AA,
        )

    draw_korean_texts(
        map_image,
        text_items,
    )


# ============================================================
# Separate dashboard window
# ============================================================

def _draw_card(
    image,
    rect,
    fill,
    border=(52, 60, 72),
):
    x1, y1, x2, y2 = rect

    _rounded_rect(
        image,
        (x1, y1),
        (x2, y2),
        18,
        fill,
        -1,
    )

    _rounded_rect(
        image,
        (x1, y1),
        (x2, y2),
        18,
        border,
        1,
    )


def _draw_dashboard_button(
    image,
    rect,
    label,
    base_color,
    pressed=False,
    enabled=True,
):
    x1, y1, x2, y2 = rect

    inset = 4 if pressed else 0

    draw_rect = (
        x1 + inset,
        y1 + inset,
        x2 - inset,
        y2 - inset,
    )

    if not enabled:
        color = (58, 62, 70)
        text_color = (135, 140, 150)
    else:
        factor = 0.72 if pressed else 1.0

        color = tuple(
            int(
                max(
                    0,
                    min(
                        255,
                        c * factor,
                    ),
                )
            )
            for c in base_color
        )

        text_color = (
            245,
            245,
            245,
        )

    _rounded_rect(
        image,
        (
            draw_rect[0],
            draw_rect[1],
        ),
        (
            draw_rect[2],
            draw_rect[3],
        ),
        16,
        color,
        -1,
    )

    if enabled:
        border = (
            230,
            235,
            240,
        )
    else:
        border = (
            90,
            95,
            105,
        )

    _rounded_rect(
        image,
        (
            draw_rect[0],
            draw_rect[1],
        ),
        (
            draw_rect[2],
            draw_rect[3],
        ),
        16,
        border,
        1,
    )

    text_w = _text_width(
        label,
        19,
        True,
    )

    tx = (
        draw_rect[0]
        + max(
            8,
            (
                draw_rect[2]
                - draw_rect[0]
                - text_w
            )
            // 2,
        )
    )

    ty = (
        draw_rect[1]
        + 15
        + (2 if pressed else 0)
    )

    draw_korean_texts(
        image,
        [
            (
                label,
                (tx, ty),
                19,
                text_color,
                True,
            )
        ],
    )


def draw_navigation_dashboard(
    width: int,
    height: int,
    navigator: IndoorNavigator,
    guidance: NavigationGuidance,
    position_world_mm: Optional[np.ndarray],
    nearest_door_name: str,
    nearest_door_distance_mm: Optional[float],
    camera_yaw_deg: Optional[float],
    frame_number: int,
    display_fps: float,
    processing_ms: float,
    processing_fps: float,
    pressed_button: Optional[str] = None,
):
    """
    별도의 Navigation Control 창을 만든다.
    미니맵을 가리지 않도록 모든 버튼/상태/성능정보를 이 화면으로 이동.
    """
    width = max(640, int(width))
    height = max(430, int(height))

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    # dark dashboard background
    canvas[:] = (
        18,
        22,
        29,
    )

    # header
    draw_korean_texts(
        canvas,
        [
            (
                "INDOOR NAVIGATION",
                (28, 22),
                16,
                (125, 140, 160),
                True,
            ),
            (
                "실내 길안내",
                (28, 48),
                30,
                (245, 247, 250),
                True,
            ),
        ],
    )

    if navigator.completed:
        status_text = "도착 완료"
        status_color = (
            60,
            205,
            115,
        )
    elif navigator.active:
        status_text = "안내 중"
        status_color = (
            0,
            190,
            235,
        )
    else:
        status_text = "대기"
        status_color = (
            115,
            125,
            140,
        )

    status_rect = (
        width - 138,
        28,
        width - 28,
        68,
    )

    _rounded_rect(
        canvas,
        (
            status_rect[0],
            status_rect[1],
        ),
        (
            status_rect[2],
            status_rect[3],
        ),
        18,
        status_color,
        -1,
    )

    draw_korean_texts(
        canvas,
        [
            centered_text_item(
                status_rect[2] - status_rect[0],
                status_text,
                status_rect[1] + 8,
                17,
                (255, 255, 255),
                True,
            )
        ],
    )

    # correct status centered inside pill
    text_w = _text_width(
        status_text,
        17,
        True,
    )

    status_tx = (
        status_rect[0]
        +
        (
            status_rect[2]
            -
            status_rect[0]
            -
            text_w
        )
        // 2
    )

    draw_korean_texts(
        canvas,
        [
            (
                status_text,
                (
                    status_tx,
                    status_rect[1] + 8,
                ),
                17,
                (255, 255, 255),
                True,
            )
        ],
    )

    # Route card
    route_card = (
        24,
        94,
        width - 24,
        178,
    )

    _draw_card(
        canvas,
        route_card,
        (24, 29, 38),
    )

    draw_korean_texts(
        canvas,
        [
            (
                "ROUTE",
                (44, 110),
                14,
                (115, 130, 150),
                True,
            ),
            centered_text_item(
                width,
                "앞문  →  4번 강의실  →  2번 강의실  →  뒷문",
                136,
                21,
                (242, 244, 248),
                True,
            ),
        ],
    )

    # Main guidance card
    guide_card = (
        24,
        194,
        width - 24,
        294,
    )

    _draw_card(
        canvas,
        guide_card,
        (24, 29, 38),
    )

    headline = (
        guidance.headline
        if guidance.headline
        else
        "네비게이션 시작을 눌러주세요."
    )

    instruction = (
        guidance.instruction
        if guidance.instruction
        else
        "앞문에서 경로가 시작됩니다."
    )

    draw_korean_texts(
        canvas,
        [
            (
                "CURRENT GUIDE",
                (44, 210),
                14,
                (115, 130, 150),
                True,
            ),
            centered_text_item(
                width,
                headline,
                233,
                22,
                (0, 205, 240),
                True,
            ),
            centered_text_item(
                width,
                instruction,
                266,
                18,
                (235, 238, 242),
                False,
            ),
        ],
    )

    # Left info card
    info_left = (
        24,
        310,
        width // 2 - 8,
        height - 92,
    )

    _draw_card(
        canvas,
        info_left,
        (24, 29, 38),
    )

    if position_world_mm is not None:
        p = np.asarray(
            position_world_mm,
            dtype=np.float64,
        ).reshape(3)

        pos_line = (
            f"X {p[0]:.0f}   "
            f"Y {p[1]:.0f}   "
            f"Z {p[2]:.0f} mm"
        )
    else:
        pos_line = "위치 인식 대기"

    nearest_line = (
        (
            f"{nearest_door_name} · "
            f"{nearest_door_distance_mm / 1000.0:.2f} m"
        )
        if nearest_door_distance_mm is not None
        else "-"
    )

    draw_korean_texts(
        canvas,
        [
            (
                "POSITION",
                (44, 326),
                14,
                (115, 130, 150),
                True,
            ),
            (
                pos_line,
                (44, 350),
                18,
                (235, 238, 242),
                True,
            ),
            (
                "최근접 문",
                (44, 380),
                14,
                (115, 130, 150),
                False,
            ),
            (
                nearest_line,
                (44, 402),
                17,
                (210, 216, 224),
                True,
            ),
        ],
    )

    # Right metrics card
    info_right = (
        width // 2 + 8,
        310,
        width - 24,
        height - 92,
    )

    _draw_card(
        canvas,
        info_right,
        (24, 29, 38),
    )

    draw_korean_texts(
        canvas,
        [
            (
                "PERFORMANCE",
                (
                    width // 2 + 28,
                    326,
                ),
                14,
                (115, 130, 150),
                True,
            ),
            (
                f"Frame  {int(frame_number)}",
                (
                    width // 2 + 28,
                    350,
                ),
                17,
                (235, 238, 242),
                True,
            ),
            (
                f"화면 FPS  {display_fps:.1f}",
                (
                    width // 2 + 28,
                    375,
                ),
                16,
                (205, 211, 220),
                False,
            ),
            (
                (
                    f"처리  {processing_ms:.1f} ms"
                    f"   ·   {processing_fps:.1f} FPS"
                ),
                (
                    width // 2 + 28,
                    400,
                ),
                16,
                (205, 211, 220),
                False,
            ),
        ],
    )

    # bottom legend
    legend_y = height - 76

    draw_korean_texts(
        canvas,
        [
            centered_text_item(
                width,
                "흰색 = 카메라 정면   ·   노랑 = 안내 방향   ·   초록 = 실제 이동 방향",
                legend_y,
                15,
                (155, 165, 180),
                False,
            )
        ],
    )

    # buttons in their own dashboard, no map obstruction
    gap = 16
    button_y1 = height - 52
    button_y2 = height - 10

    start_rect = (
        24,
        button_y1,
        width // 2 - gap // 2,
        button_y2,
    )

    stop_rect = (
        width // 2 + gap // 2,
        button_y1,
        width - 24,
        button_y2,
    )

    start_label = (
        "다시 시작"
        if (
            navigator.active
            or navigator.completed
        )
        else
        "네비게이션 시작"
    )

    _draw_dashboard_button(
        canvas,
        start_rect,
        start_label,
        (45, 155, 95),
        pressed=(
            pressed_button == "start"
        ),
        enabled=True,
    )

    _draw_dashboard_button(
        canvas,
        stop_rect,
        "네비게이션 종료",
        (60, 80, 185),
        pressed=(
            pressed_button == "stop"
        ),
        enabled=(
            navigator.active
            or navigator.completed
        ),
    )

    return (
        canvas,
        {
            "start": start_rect,
            "stop": stop_rect,
        },
    )


# Compatibility helper kept for older code paths.
def draw_navigation_buttons(
    map_preview: np.ndarray,
    navigator: IndoorNavigator,
    pressed_button: Optional[str] = None,
):
    return {}
