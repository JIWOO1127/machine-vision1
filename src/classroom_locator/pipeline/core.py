"""탐지 -> 클래스 직접 판별 -> 위치 매칭을 한 번에 수행하는 핵심 파이프라인.

realtime_pipeline.py(웹캠/영상)와 batch_pipeline.py(사진 폴더)가 공통으로
이 클래스를 사용합니다. 즉, "한 프레임/한 장의 이미지를 처리하는 로직"은
여기 한 곳에만 있고, 입력을 어디서 받아오는지만 다릅니다.

YOLO가 탐지한 클래스 이름이 곧 locations.yaml의 위치 이름과 같아서 바로
조회합니다 (front_door/rear_door/logo 등). room2/room4(2_class/4_class)만
예외로, 근접 판정 시 표지판 글자를 OCR로 직접 읽어서 확인합니다 (YOLO
모델에 3강의실 전용 클래스가 없어서 2_class/4_class로만 오분류되는 문제
때문 - pipeline/grid_tracker.py의 동일 로직과
`classroom_locator.landmark_locator.ocr_verify`의 `build_verifier`/
`read_digit`을 공유해서 씁니다. 2026-09-17 결정이었던 "메인 파이프라인은
OCR 완전 배제"를 되돌리는 변경이며, 이 좁은 용도(표지판 숫자 하나
재확인)로 한정됩니다 - "OCR 텍스트로 전체 위치를 찾는" 예전 방식(성능이
나빠서 폐기됨, docs/setup_log.md 9번/14번 참고)과는 다릅니다).

OCR이 "3"을 읽으면 즉시 room3로 정정합니다 (YOLO가 애초에 낼 수 없는
클래스라 오판 리스크가 없음). 반대로 OCR이 "2"/"4"를 읽었는데 YOLO
라벨과 다르면(예: 실제 2강의실인데 YOLO가 4_class로 오분류) - 2_class/
4_class는 표지판 생김새가 거의 동일해서(REAL_SIZE 실측값도 둘 다
동일) YOLO 분류기 자체가 가끔 혼동하는 게 확인된 문제라 정정이
필요하지만, OCR도 한 프레임만으로는 오독할 수 있어서(예: "4강의실"을
"강의실2"로 잘못 읽어 room4를 room2로 잘못 정정한 사례가 실제 있었음 -
grid_tracker.py 43번 항목 참고) 같은 불일치가 연속 2프레임 나와야만
정정합니다(`_ocr_disagreement` 스트릭, Locator의 `ocr_confirm_frames=2`와
동일한 관례).

한 프레임에 여러 물체가 동시에 보이면, bbox가 가장 큰(=가장 가까운) 것을
현재 위치로 선택합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..detection import get_detector
from ..detection.base import BaseDetector, Detection
from ..landmark_locator import OcrVerifier, build_verifier, read_sign_info
from ..localization import Location, MatchResult, load_locations

logger = logging.getLogger("classroom_locator")

# YOLO가 2_class/4_class로만 낼 수 있는(=room3 전용 클래스가 없는) 표지판 위치 이름.
# 이 두 위치로 분류된 탐지만 OCR로 숫자를 재확인한다.
_SIGN_LOCATION_NAMES = {"room2", "room4"}
_DIGIT_TO_SIGN_LOCATION = {"2": "room2", "4": "room4"}


@dataclass
class FrameResult:
    detections: list[Detection]
    match: MatchResult | None
    # detections 중 위치가 확정된 것들 전부 (detection, 그 물체 하나만 놓고 봤을 때의
    # 인식 결과). match는 이 중 bbox가 가장 큰 하나일 뿐이고, 물체별로 개별
    # 인식이 맞았는지 보려면(우선순위 로직 제외) 이 리스트를 씁니다.
    object_matches: list[tuple[Detection, MatchResult]]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


class LocatorPipeline:
    """이미지 한 장을 입력받아 현재 위치 추정 결과를 반환하는 파이프라인."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.detector: BaseDetector = get_detector(config["detector"])

        loc_config = config["localization"]
        self.locations: list[Location] = load_locations(loc_config["locations_file"])
        self._location_by_class_name: dict[str, Location] = {loc.name: loc for loc in self.locations}
        # room3 정정에 필요할 때만 OCR을 초기화 (locations.yaml에 room3가 없으면
        # 애초에 정정할 곳이 없으니 스킵 - easyocr 로딩 시간 절약).
        self._ocr_verifier: OcrVerifier | None = (
            build_verifier() if "room3" in self._location_by_class_name else None
        )
        # room2/room4 오분류 정정용 2프레임 연속 확인 상태: (YOLO가 분류한 이름,
        # OCR이 읽은 숫자). 같은 불일치가 다시 들어오면 그때 정정을 적용.
        self._ocr_disagreement: tuple[str, str] | None = None

    def _resolve_location(self, det: Detection, image: np.ndarray) -> Location | None:
        location = self._location_by_class_name.get(det.class_name)
        if location is None or det.class_name not in _SIGN_LOCATION_NAMES or self._ocr_verifier is None:
            return location

        digit, is_meeting_room = read_sign_info(self._ocr_verifier, image, det.bbox)
        if is_meeting_room:
            # "O 회의실" 표지판은 "O 강의실"과 생김새가 거의 같아 YOLO가
            # room2/room4로 오분류함 - 강의실이 아니므로 위치 매칭 자체에서
            # 제외한다(현재 위치로도, "인식된 객체"로도 뜨지 않게).
            self._ocr_disagreement = None
            return None
        if digit == "3":
            # YOLO는 애초에 room3를 낼 수 없는 클래스라, OCR이 "3"을 읽으면
            # 오판 리스크 없이 바로 신뢰한다.
            return self._location_by_class_name.get("room3", location)

        mapped_name = _DIGIT_TO_SIGN_LOCATION.get(digit)
        if mapped_name is None:
            # OCR이 아예 못 읽은 프레임 - 정보가 없을 뿐이니 스트릭은 유지한다
            # (EasyOCR 실패율이 꽤 높아서, 여기서 스트릭을 지우면 실제로는
            # 계속되는 오분류인데도 "연속 확인"에 거의 도달하지 못했었다).
            return location
        if mapped_name == det.class_name:
            self._ocr_disagreement = None  # OCR이 YOLO와 일치 - 불일치 스트릭 해제
            return location

        # OCR이 읽은 숫자가 YOLO 라벨과 다름(2_class/4_class는 생김새가 거의
        # 같아 YOLO가 가끔 혼동함) - 다만 OCR도 한 프레임만으론 오독할 수
        # 있으므로, 같은 불일치를 (중간에 못 읽은 프레임이 껴도) 두 번째로
        # 다시 확인해야 정정한다.
        if self._ocr_disagreement == (det.class_name, digit):
            self._ocr_disagreement = None
            return self._location_by_class_name.get(mapped_name, location)
        self._ocr_disagreement = (det.class_name, digit)
        # 아직 확정 전(처음 불일치)이라고 해서 YOLO의 원래(틀렸을 수 있는)
        # 라벨을 그대로 내보내면 안 된다 - realtime_pipeline.py의
        # LocationLocker가 이 결과를 "새 위치"로 보고 최소 3초간 잠가버려서,
        # 그사이 들어오는 진짜 정답(예: room2)까지 통째로 씹혀버리는 문제가
        # 실제로 있었다. 확정 전에는 "모르겠다"(None)로 답해서 애초에 이
        # 탐지가 위치 후보/화면 잠금 경쟁에 끼지 못하게 한다.
        return None

    def process_image(self, image: np.ndarray) -> FrameResult:
        detections = self.detector.detect(image)

        candidates: list[tuple[Detection, MatchResult]] = []
        for det in detections:
            location = self._resolve_location(det, image)
            if location is not None:
                # OCR로 정정된 경우 박스 라벨(draw_detections)에도 반영되도록
                # class_name 자체를 갱신 - 안 그러면 화면엔 여전히 YOLO가 잘못
                # 분류한 이름(예: room4)이 찍혀서 "여전히 잘못 인식한다"로 보임.
                det.class_name = location.name
                candidates.append(
                    (det, MatchResult(location=location, matched_text=det.class_name, score=det.confidence * 100))
                )

        # 여러 물체가 동시에 보이면, bbox가 가장 큰(=가장 가까운) 것을 현재 위치로 선택.
        best: MatchResult | None = None
        best_area = -1
        for det, match in candidates:
            area = _bbox_area(det.bbox)
            if area > best_area:
                best_area = area
                best = match

        return FrameResult(detections=detections, match=best, object_matches=candidates)
