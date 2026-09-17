"""거리 임계값 기반 격자(grid) 위치 추적 + 실시간 표시용 이미지 렌더링.

각 랜드마크마다 "이 거리(snap_distance_m) 이내로 보이면 갱신 후보"라는
규칙이 있고, 조건을 만족하는 랜드마크가 탐지되면 그 랜드마크의
grid_cell(격자 교차점 좌표)을 기준으로 "지금 계산된 실제 거리와 가장 가까운
격자점"을 찾아 현재 위치로 갱신합니다. 단순히 랜드마크 좌표로 순간이동하는 게
아니라, 거리가 가까워질수록 그 랜드마크 쪽 격자점으로 점점 옮겨가는 식입니다.
조건을 만족하는 게 하나도 없으면 이전 위치를 그대로 유지합니다(스티키).

좌표계: grid_cell은 "칸 번호"가 아니라 격자 교차점(코너) 좌표 (x, y)이고,
왼쪽 아래가 원점(0,0), x는 0~cols, y는 0~rows 범위입니다. 좌회전/우회전만
있고 대각선 이동이 없는, 정해진 시연 경로에 맞춘 단순화입니다.

이동 가능 경로는 x축(y=0), y축(x=0), x=cols축(오른쪽 끝) 이 세 직선
위로만 제한됩니다 (앞문 -> y축을 따라 내려옴 -> x축을 따라 이동 ->
x=cols축을 따라 올라감 -> 뒷문, 이런 L자/U자형 경로). 그래서 "가장 가까운
격자점"을 전체 격자가 아니라 이 세 구간 위의 점들 중에서만 찾습니다 —
안 그러면 임계값이 넓을 때 경로에서 벗어난 엉뚱한 격자점으로 튈 수 있음.

여러 랜드마크가 동시에 조건을 만족하면, distance/snap_distance_m 비율이 가장
작은(=임계값 대비 가장 가까운) 것을 우선합니다.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from ..localization.location_map import GridConfig, Location
from ..landmark_locator import SIGNS, Locator, build_verifier, read_digit
from ..utils.image_utils import put_korean_text

# landmark_locator(Locator)가 내는 원본 YOLO 클래스명 -> locations.yaml의 name.
# water_dispenser는 REAL_SIZE에 물리 크기가 없어서(=Locator가 애초에 못 다룸) 대상에서 제외.
_YOLO_CLASS_TO_LOCATION_NAME = {
    "2_class": "room2",
    "4_class": "room4",
    "front_door": "front_door",
    "rear_door": "rear_door",
    "logo": "logo",
}

# 표지판(2_class/4_class) 글자를 직접 읽어서 실제 숫자로 정정할 때 쓰는 매핑.
# 3강의실은 YOLO 클래스가 따로 없어서 모델이 2_class 또는 4_class로 오분류하기
# 때문에, OCR로 읽은 숫자가 "3"이면 room3로 정정한다.
_DIGIT_TO_LOCATION_NAME = {"2": "room2", "3": "room3", "4": "room4"}

# 지도/목록에 찍는 물체 이름표는 "~앞" 없이 이름만 짧게 표시 (음성/자막 안내
# 문구의 display_name, 예: "정문 앞"/"목적지"와는 별개). render_grid_image()의
# 점 옆 라벨과 webapp/backend/vision_bridge.py의 실시간 탐지 목록 둘 다 씀 -
# public이라 다른 모듈에서 import 가능.
MAP_LABELS = {
    "front_door": "앞문",
    "rear_door": "뒷문",
    "room2": "2강의실",
    "room3": "3강의실",
    "room4": "4강의실",
    "logo": "로고",
}

# 로고가 화면 중앙(0.35~0.65)이 아닌 상태로 이만큼(초) 이상 계속되면 "정렬
# 안 됨" 상태로 보고 "왼쪽/오른쪽으로 회전해주세요" 안내로 바뀐다. 로고가
# 잘 보이고 있어도(탐지는 계속 되고 있어도) 중앙만 아니면 대상이며, 반대로
# 중앙에 잡히거나 아예 안 보이게 되면 즉시 해제된다. 각도 안내는 오직 이
# 경우에만 나온다 - 그 전까지는 화면 중앙이 아니어도 그냥 locations.yaml에
# 정해둔 고정 guidance만 나온다.
_LOGO_OFF_CENTER_TIMEOUT_S = 3.0

# 현재 위치가 한 번 정해지면 최소 이 시간(초) 동안은 다른 위치로 바뀌지
# 않는다 (GridPositionTracker._apply() 참고). 같은 위치 안에서 거리가
# 갱신되는 건 이 잠금과 무관하게 항상 허용됨 - "다른 이름으로의 전환"만
# 막는다. pipeline/locking.py의 LocationLocker(OCR 텍스트 매칭 경로,
# min_lock_seconds=3.0)와 같은 목적이지만, 격자 위치 경로는 이미 votes/
# sign 히스테리시스로 어느 정도 안정화돼 있어 그보다 짧게 잡음.
MIN_LOCATION_LOCK_SECONDS = 1.0


def _direction_from_box(box: tuple[float, float, float, float], frame_width: int, left: float = 0.35, right: float = 0.65) -> str:
    """bbox 중심 x / 화면 너비로 좌/우/직진 판정 (webapp/backend/vision_bridge.py의
    _direction_from_box와 동일 임계값)."""
    cx = (box[0] + box[2]) / 2 / max(1, frame_width)
    return "왼쪽" if cx < left else "오른쪽" if cx > right else "앞쪽"


def _describe_position(location: Location | None, direction: str | None, off_center: bool = False) -> tuple[str | None, str | None]:
    """현재 위치 안내를 (헤드라인, 보조문구) 튜플로 만듭니다.

    location이 없으면 (None, None). 그 외엔 항상 "현재 위치는 'X'입니다."를
    헤드라인으로 쓰고, 보조문구는 보통 locations.yaml의 guidance 그대로입니다.

    각도 안내는 오직 **로고가 화면 중앙이 아닌 상태가 _LOGO_OFF_CENTER_TIMEOUT_S초
    이상 계속됐을 때(off_center=True)만** 나옵니다 - 로고가 중앙이 아니어도
    막 그렇게 됐으면(3초 미만) 그냥 고정 guidance만 나오고, 3초 넘게 계속
    중앙이 아니어야 "왼쪽/오른쪽으로 회전해주세요"로 바뀝니다.
    """
    if location is None:
        return None, None
    name = location.display_name or location.name
    headline = f"현재 위치는 '{name}'입니다."
    if location.name == "logo" and off_center and direction in ("왼쪽", "오른쪽"):
        return headline, f"{direction}으로 회전해주세요."
    return headline, location.guidance


class GridPositionTracker:
    def __init__(
        self,
        grid: GridConfig,
        locations: list[Location],
    ) -> None:
        self.grid = grid
        # 같은 name(예: "logo")이 여러 번 등록될 수 있음 - 멀리서/가까이 등
        # 여러 단계를 다른 grid_cell·display_name·guidance로 표현하기 위함.
        # snap_distance_m 오름차순(가까운 단계부터)으로 정렬해두고, 실제 판정
        # 때는 distance_m이 처음으로 들어맞는(=가장 가까운 조건의) 단계를 씀.
        self._rules: dict[str, list[Location]] = {}
        for loc in locations:
            if loc.grid_cell is not None and loc.snap_distance_m is not None:
                self._rules.setdefault(loc.name, []).append(loc)
        for stages in self._rules.values():
            stages.sort(key=lambda loc: loc.snap_distance_m)
        # 이동 가능 경로: x축(y=0) + y축(x=0) + x=cols축(오른쪽 끝) 위의 점들만.
        # (좌회전/우회전만 있는 L자/U자형 고정 경로라 그 외 지점은 갈 수 없음)
        path_points = set()
        path_points.update((x, 0) for x in range(grid.cols + 1))
        path_points.update((0, y) for y in range(grid.rows + 1))
        path_points.update((grid.cols, y) for y in range(grid.rows + 1))
        self._grid_points = sorted(path_points)
        self.current_cell: tuple[int, int] | None = None
        self.current_label: str | None = None
        self.current_location: Location | None = None
        # 방향(왼쪽/오른쪽/앞쪽): 마지막으로 위치를 갱신시킨 탐지의 bbox 중심
        # x위치 기준. LandmarkGridPositionTracker.update_from_frame()에서 채움.
        self.current_direction: str | None = None
        # 마지막으로 current_location이 실제로 갱신된 시각(t).
        self.current_matched_at: float | None = None
        # 로고가 화면 중앙이 아닌 상태로 있기 시작한 시각(t). 중앙이거나
        # 로고가 아니면 None. LandmarkGridPositionTracker._apply()에서 채움.
        self._logo_off_center_since: float | None = None
        # True면 "로고가 화면 중앙이 아닌 상태가 _LOGO_OFF_CENTER_TIMEOUT_S초
        # 이상 계속됨" (describe()가 회전 안내를 낼지 판단하는 데 씀).
        self.logo_off_center: bool = False
        # 진단용: 이번 프레임에 투표는 충분히 쌓였지만(voting 통과) 아직
        # snap_distance_m 문턱을 못 넘은 랜드마크가 있으면 (위치 이름, 계산된
        # 거리, 문턱값)을 기록. "탐지는 되는데 지도에 안 뜬다"를 확인할 때 씀
        # (render_grid_image가 화면에 표시).
        self.pending_candidate: tuple[str, float, float] | None = None
        # 마지막으로 "다른 위치로" 전환된 시각 + 그로부터 잠금이 풀리는 시각.
        # _apply()가 씀 (LandmarkGridPositionTracker.__init__이 photo_tracker
        # 등에서 0으로 낮춰 잠금을 끌 수 있게 min_lock_seconds는 인스턴스
        # 속성으로 둠).
        self.min_lock_seconds = MIN_LOCATION_LOCK_SECONDS
        self._locked_until = 0.0

    @property
    def landmarks(self) -> list[Location]:
        """격자 위치가 지정된(grid_cell이 있는) 물체 목록 (지도에 고정 표시용)."""
        return [loc for stages in self._rules.values() for loc in stages]

    def describe(self) -> str | None:
        """현재 위치를 사용자에게 보여줄 완성된 한 문장 (헤드라인 + 보조문구).
        위치가 아직 확정 안 됐으면 None (호출자가 알아서 대체 문구를 씀)."""
        headline, detail = _describe_position(self.current_location, self.current_direction, self.logo_off_center)
        if headline is None:
            return None
        return f"{headline} {detail}" if detail else headline

    def _best_stage(self, name: str, distance_m: float | None) -> Location | None:
        """name에 등록된 단계들 중, distance_m이 snap_distance_m 이내로 처음
        들어맞는(=가장 가까운 조건의) 단계를 반환합니다. 하나도 안 맞으면 None."""
        if distance_m is None:
            return None
        for loc in self._rules.get(name, []):
            if distance_m <= loc.snap_distance_m:
                return loc
        return None

    def _apply(self, loc: Location, distance_m: float, direction: str | None = None, t: float | None = None) -> tuple[int, int]:
        now = t if t is not None else time.time()

        # logo/rear_door처럼 같은 name으로 여러 단계(예: "후문 앞"/"목적지")가
        # 등록된 경우, name만 비교하면 두 단계 사이 전환에는 락이 전혀 걸리지
        # 않는다. 거리가 두 단계의 snap_distance_m 경계(예: 1.0m) 근처에서
        # 프레임마다 흔들리면 같은 name이라 매 프레임 다른 단계(다른 안내
        # 문구)로 계속 넘나들어 음성 안내가 번갈아 반복 재생된다
        # (webapp/backend/services/position/guide.py에서 겪은 것과 동일한
        # 버그 - 그쪽의 is_switch도 같은 방식으로 고쳤다). 단계(Location) 객체
        # 자체를 비교해 같은 name 안의 단계 전환도 스티키 락 대상에 포함시킨다.
        is_switch = self.current_location is not None and loc is not self.current_location
        if is_switch and now < self._locked_until:
            # 최소 유지 시간(min_lock_seconds)이 아직 안 지났으면 다른
            # 위치로의 전환을 무시하고 현재 위치를 그대로 유지한다. 같은
            # 위치가 다시 들어와서 셀/거리만 갱신되는 경우는 위 is_switch가
            # False라 이 잠금과 무관하게 항상 통과한다(아래에서 잠금을
            # 계속 연장함).
            return self.current_cell

        target_units = distance_m / self.grid.cell_size_m
        self.current_cell = self._nearest_point_by_distance(loc.grid_cell, target_units)
        self.current_label = loc.display_name or loc.name
        self.current_location = loc
        self.current_matched_at = now
        # 잠금은 "전환된 순간부터 고정 1초"가 아니라, 같은 위치가 계속
        # 재확인될 때마다 매번 now+min_lock_seconds로 연장된다. 그래야
        # 예를 들어 room3가 2초 넘게 계속 재확인되고 있는 도중에 다른
        # 랜드마크가 단발로 끼어들어도(예: OCR 오독) 여전히 막을 수 있다 -
        # 전환 시점 기준 고정 1초만 두면, room3에 머문 지 1초가 지난
        # 뒤에는 그 뒤로도 room3가 계속 잘 보이고 있어도 잠금이 이미
        # 풀려버려 단발 오탐을 못 막는 문제가 있었다.
        self._locked_until = now + self.min_lock_seconds

        # 로고가 중앙(앞쪽)이 아닌 상태가 얼마나 계속됐는지 추적. 방금 막
        # 벗어나기 시작했으면 지금을 시작 시각으로 기록하고, 계속 벗어나 있던
        # 중이면 그 시작 시각을 그대로 두고 경과 시간만 다시 잼. 중앙으로
        # 돌아오거나 로고가 아니면 즉시 해제.
        if loc.name == "logo" and direction in ("왼쪽", "오른쪽"):
            if self._logo_off_center_since is None:
                self._logo_off_center_since = now
            self.logo_off_center = (now - self._logo_off_center_since) >= _LOGO_OFF_CENTER_TIMEOUT_S
        else:
            self._logo_off_center_since = None
            self.logo_off_center = False

        self.current_direction = direction
        return self.current_cell

    def _nearest_point_by_distance(self, landmark_point: tuple[int, int], target_distance_units: float) -> tuple[int, int]:
        """landmark_point에서 target_distance_units(격자 단위)만큼 떨어진 것과
        가장 가까운 격자 교차점을 찾습니다."""
        lx, ly = landmark_point
        return min(
            self._grid_points,
            key=lambda p: abs(math.hypot(p[0] - lx, p[1] - ly) - target_distance_units),
        )


class LandmarkGridPositionTracker(GridPositionTracker):
    """classroom_locator.landmark_locator.Locator의 판정 로직으로 격자 위치를 갱신한다.

    Locator가 프레임을 직접 받아 (자체 YOLO 추론 + REAL_SIZE 실측 물리 크기
    기반 핀홀 거리추정 + 최근 프레임 다수결 투표 + 접근 추세 확인)까지 끝낸
    결과를 쓴다. "계산된 거리와 가장 가까운 격자점을 찾는" 부분
    (_nearest_point_by_distance, 경로 제약, _best_stage/_apply)은 부모
    클래스(GridPositionTracker) 로직을 그대로 재사용한다.
    """

    def __init__(
        self,
        grid: GridConfig,
        locations: list[Location],
        weights: str,
        logo_weights: str | None = None,
        use_ocr: bool = True,
        **locator_kwargs: object,
    ) -> None:
        super().__init__(grid, locations)
        locator_kwargs.setdefault("targets", tuple(_YOLO_CLASS_TO_LOCATION_NAME))
        locator_kwargs.setdefault("route", None)
        if use_ocr and "verifier" not in locator_kwargs:
            locator_kwargs["verifier"] = build_verifier()
        self._locator = Locator(weights, logo_weights=logo_weights, **locator_kwargs)
        # update_from_frame()이 마지막으로 받은 Locator.process() 원본 결과.
        # 같은 프레임에 대해 재추론 없이 detections/landmark를 다시 쓰고 싶은
        # 호출자(예: webapp/backend)를 위해 캐시해둠.
        self.last_result: dict | None = None
        # room2/room3/room4 표지판의 "확정된" 이름 - 프레임마다 새로 읽는
        # raw_name(YOLO 라벨 또는 OCR "3")을 바로 신뢰하지 않고, 같은
        # raw_name이 연속 _sign_name_required_streak번 나와야만 갱신하는
        # 히스테리시스에 씀 (아래 update_from_frame 참고). room3로 들어갈
        # 때뿐 아니라 room3에서 다시 room2/room4로 빠져나올 때도 똑같이
        # 적용해야, OCR 오독 한 프레임 때문에 양방향으로 튀는 걸 막을 수
        # 있다 (49-2에서 room3 진입만 디바운스했다가 이탈 쪽에서 여전히
        # 깜빡였던 것을 49-3에서 양방향으로 확장함).
        # window<=1(예: webapp의 photo_tracker - 사진 한 장짜리 단발 판정)은
        # 애초에 "다음 프레임"이 없어서 디바운스를 요구하면 아무 이름도 확정을
        # 못 하게 되므로 1(=즉시 신뢰)로 두고, 그 외(연속 프레임이 있는
        # 실시간 트래커)는 2로 둬서 단발 오독에 의한 깜빡임을 걸러낸다.
        self._sign_name_required_streak = 1 if self._locator.window <= 1 else 2
        self._sign_name_streak_value: str | None = None
        self._sign_name_streak_count = 0
        self._resolved_sign_name: str | None = None
        # window<=1(사진 한 장짜리 photo_tracker)은 매번 새로운 무관한
        # 사진이 들어오므로, 부모의 min_lock_seconds(위치 전환 최소 유지
        # 시간)를 그대로 두면 직전 사진의 잠금 때문에 다음 사진이 무시될 수
        # 있음 - 그래서 단발 판정 모드는 잠금을 꺼둔다.
        if self._locator.window <= 1:
            self.min_lock_seconds = 0.0

    def _read_sign_digit(self, frame: np.ndarray, box: tuple[float, float, float, float] | None) -> str | None:
        """표지판 bbox 글자를 직접 읽어서 2/3/4 중 하나만 뚜렷하면 그 숫자를 반환.

        Locator 내장 OCR confirm/reject(ocr_verify.OcrVerifier.verify())는
        "YOLO가 예상한 숫자와 일치하는지"만 확인하는 훅이라, YOLO 자체에
        클래스가 없는 3강의실은 항상 reject로 떨어져 정정이 불가능함. 그래서
        여기서는 OcrVerifier.read()로 글자를 직접 읽어 판단한다 (core.py의
        LocatorPipeline과 공용인 read_digit() 재사용).
        """
        return read_digit(self._locator.verifier, frame, box)

    def resolve_location_name(self, det: dict, frame: np.ndarray) -> str:
        """탐지 하나(Locator.detect()가 내는 dict: name/box/conf/distance_m)가
        locations.yaml의 어느 name에 해당하는지 판정합니다.

        표지판(2_class/4_class)은 Locator 내장 OCR 훅과 별개로 글자를 직접
        읽어서(_read_sign_digit) 실제 숫자가 3이면 room3로 정정합니다(YOLO
        모델에 3강의실 전용 클래스가 없어서 2_class/4_class로만 오분류되기
        때문). 글자를 못 읽으면 YOLO가 판단한 라벨(2_class->room2,
        4_class->room4)을 그대로 신뢰합니다. 문/로고 등 나머지 클래스는
        `_YOLO_CLASS_TO_LOCATION_NAME`으로 바로 매핑합니다.
        """
        name = det["name"]
        if name in SIGNS:
            digit = self._read_sign_digit(frame, det.get("box"))
            # OCR은 YOLO가 구조적으로 낼 수 없는 "3"을 잡아낼 때만 신뢰한다.
            # 2/4는 YOLO 라벨을 그대로 따른다 - OCR이 표지판 글자를 "2"로
            # 잘못 읽어도(예: "강의실2") 실제로는 4_class(room4)인 경우가 있어,
            # 여기서 digit을 그대로 신뢰하면 room4 탐지가 room2로 잘못
            # 정정되어 지도 위치가 엉뚱한 곳(room2)으로 튀는 문제가 있었다.
            if digit == "3":
                return _DIGIT_TO_LOCATION_NAME[digit]
        return _YOLO_CLASS_TO_LOCATION_NAME.get(name, name)

    def update_from_frame(self, frame: np.ndarray, t: float | None = None) -> tuple[int, int] | None:
        """Locator.process()로 프레임을 직접 판정해 격자 위치를 갱신합니다.

        표 갱신 보류 조건(부모 클래스와 달리 Locator가 이미 판단해서 넘겨줌):
          - reason이 "votes ..."로 시작 = 아직 프레임 투표 수가 min_votes 미만
          - reason == "not approaching" = 거리가 늘고 있음(스쳐 지나가는 중, 접근 추세 아님)
        이번 프레임에 갱신이 안 되면(위 두 경우거나 애초에 아무것도 안 잡히면)
        이전 위치/상태를 그대로 유지합니다(스티키) - 로고 중앙정렬 타이머
        (_logo_off_center_since)도 실제로 로고가 다시 잡힐 때만 갱신됩니다.
        """
        out = self._locator.process(frame, t)
        self.last_result = out
        landmark = out["landmark"]
        if landmark is None:
            self.pending_candidate = None
            self._sign_name_streak_value = None
            self._sign_name_streak_count = 0
            self._resolved_sign_name = None
            return self.current_cell

        det = next((d for d in out["detections"] if d["name"] == landmark), None)

        # 이 프레임의 표지판 이름을 "확정"하기 전에, 먼저 raw_name(이번
        # 프레임 하나만 보고 판단한 이름)을 구하고 히스테리시스를 거친다.
        # OCR은 YOLO가 구조적으로 낼 수 없는 "3"을 잡아낼 때만 신뢰한다 -
        # 2/4는 YOLO 라벨을 그대로 따른다(OCR이 "2"로 잘못 읽어도 실제로는
        # 4_class인 경우가 있어, digit을 그대로 신뢰하면 room4가 room2로
        # 잘못 정정되는 문제가 있었다 - resolve_location_name()과 동일한
        # 규칙).
        resolved_sign_name = None
        if landmark in SIGNS and det is not None:
            digit = self._read_sign_digit(frame, det.get("box"))
            raw_name = "room3" if digit == "3" else _YOLO_CLASS_TO_LOCATION_NAME.get(landmark, landmark)

            if raw_name == self._sign_name_streak_value:
                self._sign_name_streak_count += 1
            else:
                self._sign_name_streak_value = raw_name
                self._sign_name_streak_count = 1
            # 한 프레임의 오독만으로 즉시 튀지 않도록, 같은 raw_name이 연속
            # _sign_name_required_streak번 나와야만 "확정된" 이름을 바꾼다.
            # room3로 들어갈 때뿐 아니라 room3에서 room2/room4로 나갈 때도
            # 똑같이 적용됨(양방향) - sign_conf를 낮춘 뒤 표지판이 자주
            # 잡히면서 한 프레임짜리 오독으로 room2/3/4가 오락가락하는 오탐이
            # 늘어난 것을 막기 위한 안전장치 (49-2/49-3 항목 참고). 아직
            # 아무 이름도 확정 못 한 첫 프레임(_resolved_sign_name이 None)은
            # 예외로 바로 확정시켜서 불필요한 초기 지연은 안 둔다.
            if self._sign_name_streak_count >= self._sign_name_required_streak or self._resolved_sign_name is None:
                self._resolved_sign_name = raw_name
            resolved_sign_name = self._resolved_sign_name
        else:
            self._sign_name_streak_value = None
            self._sign_name_streak_count = 0
            self._resolved_sign_name = None

        # room3는 YOLO 전용 클래스가 없어 2_class/4_class로만 오분류되는 데다,
        # 표지판 자체가 화면에 짧게만 스쳐 잡히는 경우가 많아 아래 votes 문턱
        # (min_votes/window)을 못 넘기고 매번 걸러지는 일이 잦다(실측: 이
        # 조건을 안 두면 3강의실이 사실상 항상 인식 불가). OCR이 실제로 "3"을
        # (히스테리시스를 거쳐) 확정했다면 그 자체가 이미 min_votes 다수결보다
        # 신뢰도 높은 신호이므로, votes/not approaching로 보류되는 상황이어도
        # room3 판정만은 표결 문턱을 우회해서 즉시 반영한다 (room2/room4/
        # 문/로고의 표결 안정성 로직은 그대로 유지 - 이 예외는 room3 전용).
        if resolved_sign_name == "room3":
            distance_m = det["distance_m"]
            loc = self._best_stage("room3", distance_m)
            if loc is not None:
                self.pending_candidate = None
                direction = _direction_from_box(det["box"], frame.shape[1])
                return self._apply(loc, distance_m, direction, out["t"])

        reason = out.get("reason") or ""
        if reason.startswith("votes") or reason == "not approaching":
            self.pending_candidate = None
            return self.current_cell

        if det is not None and landmark in SIGNS:
            name = resolved_sign_name
        else:
            name = self.resolve_location_name(det, frame) if det is not None else landmark

        distance_m = out["distance_m"]
        loc = self._best_stage(name, distance_m)
        if loc is None:
            # 투표는 통과했는데(=랜드마크로 확실히 인식) 문턱 거리 안에는 아직
            # 안 들어옴 - 진단용으로 기록 (render_grid_image가 보여줌).
            nearest_threshold = min(
                (stage.snap_distance_m for stage in self._rules.get(name, [])), default=None
            )
            if nearest_threshold is not None and distance_m is not None:
                self.pending_candidate = (name, distance_m, nearest_threshold)
            return self.current_cell

        self.pending_candidate = None
        direction = _direction_from_box(det["box"], frame.shape[1]) if det is not None else None
        return self._apply(loc, distance_m, direction, out["t"])


def render_grid_image(
    grid: GridConfig,
    current_cell: tuple[int, int] | None,
    current_label: str | None = None,
    cell_px: int = 60,
    landmarks: list[Location] | None = None,
    current_location: Location | None = None,
    current_direction: str | None = None,
    logo_off_center: bool = False,
    pending_candidate: tuple[str, float, float] | None = None,
) -> np.ndarray:
    """격자 지도를 OpenCV BGR 이미지로 그립니다 (실시간 창 표시용).

    왼쪽 아래가 원점(0,0)이 되도록 그립니다 (이미지 좌표는 위가 원점이라 y를
    뒤집어서 그림). landmarks를 주면 각 물체의 grid_cell 위치에 점 + 이름을
    고정으로 같이 그려서, 지금 위치가 어느 물체 기준인지 지도에서 바로 보이게
    합니다. current_location/current_direction/logo_off_center를 주면
    (=LandmarkGridPositionTracker의 같은 이름 속성) 아래쪽에 "현재 위치는
    'X'입니다." + 안내 문구를 보여줍니다 (_describe_position() 참고 - 로고가
    화면 중앙이 아닌 상태가 3초 넘게 계속되면 "왼쪽/오른쪽으로 회전해주세요"로
    바뀜). pending_candidate(=LandmarkGridPositionTracker.pending_candidate,
    (이름, 계산된 거리, 문턱값))를 주면 오른쪽 위에 진단용으로 "인식은 됐는데
    아직 문턱 거리 밖"인 랜드마크의 실측 거리를 보여줍니다 - "탐지는 되는데
    지도에 안 뜬다"를 확인할 때 씀.
    """
    headline, detail = _describe_position(current_location, current_direction, logo_off_center)
    if headline is None:
        if current_label:
            headline = f"현재 위치: {current_cell} ({current_label} 근처)"
        elif current_cell is not None:
            headline = f"현재 위치: {current_cell}"
        else:
            headline = "현재 위치: 인식 중..."

    margin = 30
    caption_lines = 2 if detail else 1
    width = grid.cols * cell_px + margin * 2
    height = grid.rows * cell_px + margin * 2 + 24 * caption_lines + 16
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return margin + round(x * cell_px), margin + round((grid.rows - y) * cell_px)

    for col in range(grid.cols + 1):
        p1 = to_px(col, 0)
        p2 = to_px(col, grid.rows)
        cv2.line(img, p1, p2, (200, 200, 200), 1)
    for row in range(grid.rows + 1):
        p1 = to_px(0, row)
        p2 = to_px(grid.cols, row)
        cv2.line(img, p1, p2, (200, 200, 200), 1)

    for loc in landmarks or []:
        if loc.grid_cell is None:
            continue
        lx, ly = to_px(loc.grid_cell[0], loc.grid_cell[1])
        cv2.circle(img, (lx, ly), 5, (140, 140, 140), -1, lineType=cv2.LINE_AA)
        near_top = loc.grid_cell[1] >= grid.rows
        near_right = loc.grid_cell[0] >= grid.cols - 1
        text_pos = (lx - 95 if near_right else lx + 8, ly + 6 if near_top else ly - 24)
        map_label = MAP_LABELS.get(loc.name, loc.display_name or loc.name)
        img = put_korean_text(img, map_label, text_pos, font_size=16, color_bgr=(90, 90, 90))

    if current_cell is not None:
        cx, cy = to_px(current_cell[0], current_cell[1])
        cv2.circle(img, (cx, cy), max(8, cell_px // 3), (60, 90, 230), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), max(8, cell_px // 3), (30, 50, 150), 2, lineType=cv2.LINE_AA)

    if pending_candidate is not None:
        name, distance_m, threshold_m = pending_candidate
        candidate_label = MAP_LABELS.get(name, name)
        diag = f"{candidate_label} 감지됨 · {distance_m:.1f}m (기준 {threshold_m:.1f}m 이내)"
        img = put_korean_text(img, diag, (margin, 4), font_size=15, color_bgr=(120, 120, 120))

    if caption_lines == 2:
        img = put_korean_text(img, headline, (margin, height - 24 * 2 - 8), font_size=20, color_bgr=(40, 40, 40))
        img = put_korean_text(img, detail, (margin, height - 24 - 4), font_size=18, color_bgr=(60, 90, 230))
    else:
        img = put_korean_text(img, headline, (margin, height - 32), font_size=20, color_bgr=(40, 40, 40))

    return img
