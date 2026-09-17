"""거리 임계값 기반 격자(grid) 위치 추적 + 실시간 표시용 이미지 렌더링.

VisualLocationEstimator(연속 좌표 + 삼각측량, localization/visual_localization.py)
보다 단순한 방식입니다: 각 랜드마크마다 "이 거리(snap_distance_m) 이내로 보이면
갱신 후보"라는 규칙이 있고, 조건을 만족하는 랜드마크가 탐지되면 그 랜드마크의
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

import logging
import math

import cv2
import numpy as np

from ..detection.base import Detection
from ..localization.location_map import GridConfig, Location
from ..localization.visual_localization import estimate_distance_m
from ..seojiwoo import SIGNS, Locator
from ..utils.image_utils import put_korean_text

# seojiwoo_core(Locator)가 내는 원본 YOLO 클래스명 -> locations.yaml의 name.
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


class GridPositionTracker:
    def __init__(
        self,
        grid: GridConfig,
        locations: list[Location],
        calibration: dict[str, dict[str, float]],
    ) -> None:
        self.grid = grid
        self.calibration = calibration
        self._rules: dict[str, Location] = {
            loc.name: loc
            for loc in locations
            if loc.grid_cell is not None and loc.snap_distance_m is not None
        }
        # 이동 가능 경로: x축(y=0) + y축(x=0) + x=cols축(오른쪽 끝) 위의 점들만.
        # (좌회전/우회전만 있는 L자/U자형 고정 경로라 그 외 지점은 갈 수 없음)
        path_points = set()
        path_points.update((x, 0) for x in range(grid.cols + 1))
        path_points.update((0, y) for y in range(grid.rows + 1))
        path_points.update((grid.cols, y) for y in range(grid.rows + 1))
        self._grid_points = sorted(path_points)
        self.current_cell: tuple[int, int] | None = None
        self.current_label: str | None = None

    @property
    def landmarks(self) -> list[Location]:
        """격자 위치가 지정된(grid_cell이 있는) 물체 목록 (지도에 고정 표시용)."""
        return list(self._rules.values())

    def _nearest_point_by_distance(self, landmark_point: tuple[int, int], target_distance_units: float) -> tuple[int, int]:
        """landmark_point에서 target_distance_units(격자 단위)만큼 떨어진 것과
        가장 가까운 격자 교차점을 찾습니다."""
        lx, ly = landmark_point
        return min(
            self._grid_points,
            key=lambda p: abs(math.hypot(p[0] - lx, p[1] - ly) - target_distance_units),
        )

    def update(
        self, detections: list[Detection], frame_width: int, frame_height: int
    ) -> tuple[int, int] | None:
        """탐지 결과를 보고 현재 격자 위치를 갱신합니다. 갱신 안 되면 이전 위치 유지."""
        best_ratio: float | None = None
        best_loc: Location | None = None
        best_distance_m: float | None = None

        for det in detections:
            loc = self._rules.get(det.class_name)
            if loc is None:
                continue
            distance_m = estimate_distance_m(det, self.calibration, frame_width, frame_height)
            if distance_m is None or distance_m > loc.snap_distance_m:
                continue
            ratio = distance_m / loc.snap_distance_m
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_loc = loc
                best_distance_m = distance_m

        if best_loc is not None:
            target_units = best_distance_m / self.grid.cell_size_m
            self.current_cell = self._nearest_point_by_distance(best_loc.grid_cell, target_units)
            self.current_label = best_loc.display_name or best_loc.name
        return self.current_cell


class SeojiwooGridPositionTracker(GridPositionTracker):
    """세오지우(external/seojiwoo_core) Locator의 판정 로직으로 격자 위치를 갱신하는 버전.

    부모 클래스(GridPositionTracker)는 우리 자체 파이프라인의 Detection 목록 +
    distance_calibration.json 기반 거리추정(estimate_distance_m)을 쓰지만, 이
    클래스는 그 대신 ``classroom_locator.seojiwoo.Locator``가 프레임을 직접 받아
    (자체 YOLO 추론 + REAL_SIZE 실측 물리 크기 기반 핀홀 거리추정 + 최근 프레임
    다수결 투표 + 접근 추세 확인)까지 끝낸 결과를 사용한다. "계산된 거리와 가장
    가까운 격자점을 찾는" 부분(_nearest_point_by_distance, 경로 제약)은 부모
    클래스 로직을 그대로 재사용한다.
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
        super().__init__(grid, locations, calibration={})
        locator_kwargs.setdefault("targets", tuple(_YOLO_CLASS_TO_LOCATION_NAME))
        locator_kwargs.setdefault("route", None)
        if use_ocr and "verifier" not in locator_kwargs:
            locator_kwargs["verifier"] = self._build_ocr_verifier()
        self._locator = Locator(weights, logo_weights=logo_weights, **locator_kwargs)
        # update_from_frame()이 마지막으로 받은 Locator.process() 원본 결과.
        # 같은 프레임에 대해 재추론 없이 detections/landmark를 다시 쓰고 싶은
        # 호출자(예: webapp/backend)를 위해 캐시해둠.
        self.last_result: dict | None = None

    @staticmethod
    def _build_ocr_verifier() -> object | None:
        """ocr_verify.py의 OcrVerifier를 그대로 씀 (근접 판정 시에만 표지판
        글자로 재확인하는 하이브리드). easyocr 미설치/로딩 실패 시엔 조용히
        OCR 없이(YOLO 판정만으로) 계속 진행."""
        try:
            from ..seojiwoo.ocr_verify import OcrVerifier

            try:
                import torch

                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
            return OcrVerifier(gpu=use_gpu)
        except Exception as e:  # noqa: BLE001 - OCR은 선택 기능, 실패해도 YOLO 판정은 계속되어야 함
            logging.getLogger("classroom_locator").warning(
                f"OcrVerifier 초기화 실패, OCR 없이 YOLO 판정만 사용: {e}"
            )
            return None

    def _read_sign_digit(self, frame: np.ndarray, box: tuple[float, float, float, float] | None) -> str | None:
        """표지판 bbox 글자를 직접 읽어서 2/3/4 중 하나만 뚜렷하면 그 숫자를 반환.

        Locator 내장 OCR confirm/reject(ocr_verify.OcrVerifier.verify())는
        "YOLO가 예상한 숫자와 일치하는지"만 확인하는 훅이라, YOLO 자체에
        클래스가 없는 3강의실은 항상 reject로 떨어져 정정이 불가능함. 그래서
        여기서는 OcrVerifier.read()로 글자를 직접 읽어 판단한다.
        """
        verifier = self._locator.verifier
        if verifier is None or box is None:
            return None
        texts = verifier.read(frame, box)
        digits = {ch for text in texts for ch in text if ch in "234"}
        return next(iter(digits)) if len(digits) == 1 else None

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
            if digit is not None:
                return _DIGIT_TO_LOCATION_NAME[digit]
        return _YOLO_CLASS_TO_LOCATION_NAME.get(name, name)

    def update_from_frame(self, frame: np.ndarray, t: float | None = None) -> tuple[int, int] | None:
        """Locator.process()로 프레임을 직접 판정해 격자 위치를 갱신합니다.

        표 갱신 보류 조건(부모 클래스와 달리 Locator가 이미 판단해서 넘겨줌):
          - reason이 "votes ..."로 시작 = 아직 프레임 투표 수가 min_votes 미만
          - reason == "not approaching" = 거리가 늘고 있음(스쳐 지나가는 중, 접근 추세 아님)
        """
        out = self._locator.process(frame, t)
        self.last_result = out
        landmark = out["landmark"]
        if landmark is None:
            return self.current_cell

        reason = out.get("reason") or ""
        if reason.startswith("votes") or reason == "not approaching":
            return self.current_cell

        det = next((d for d in out["detections"] if d["name"] == landmark), None)
        name = self.resolve_location_name(det, frame) if det is not None else landmark

        loc = self._rules.get(name)
        distance_m = out["distance_m"]
        if loc is None or distance_m is None or distance_m > loc.snap_distance_m:
            return self.current_cell

        target_units = distance_m / self.grid.cell_size_m
        self.current_cell = self._nearest_point_by_distance(loc.grid_cell, target_units)
        self.current_label = loc.display_name or loc.name
        return self.current_cell


def render_grid_image(
    grid: GridConfig,
    current_cell: tuple[int, int] | None,
    current_label: str | None = None,
    cell_px: int = 60,
    landmarks: list[Location] | None = None,
) -> np.ndarray:
    """격자 지도를 OpenCV BGR 이미지로 그립니다 (실시간 창 표시용).

    왼쪽 아래가 원점(0,0)이 되도록 그립니다 (이미지 좌표는 위가 원점이라 y를
    뒤집어서 그림). landmarks를 주면 각 물체의 grid_cell 위치에 점 + 이름을
    고정으로 같이 그려서, 지금 위치가 어느 물체 기준인지 지도에서 바로 보이게
    합니다.
    """
    margin = 30
    width = grid.cols * cell_px + margin * 2
    height = grid.rows * cell_px + margin * 2 + 40
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
        img = put_korean_text(img, loc.display_name or loc.name, text_pos, font_size=16, color_bgr=(90, 90, 90))

    if current_cell is not None:
        cx, cy = to_px(current_cell[0], current_cell[1])
        cv2.circle(img, (cx, cy), max(8, cell_px // 3), (60, 90, 230), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), max(8, cell_px // 3), (30, 50, 150), 2, lineType=cv2.LINE_AA)

    label = f"현재 위치: {current_cell}" + (f" ({current_label} 근처)" if current_label else "")
    if current_cell is None:
        label = "현재 위치: 인식 중..."
    img = put_korean_text(img, label, (margin, height - 32), font_size=20, color_bgr=(40, 40, 40))

    return img
