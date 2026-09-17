"""탐지 -> (OCR 또는 클래스 직접 판별) -> 위치 매칭을 한 번에 수행하는 핵심 파이프라인.

realtime_pipeline.py(웹캠/영상)와 batch_pipeline.py(사진 폴더)가 공통으로
이 클래스를 사용합니다. 즉, "한 프레임/한 장의 이미지를 처리하는 로직"은
여기 한 곳에만 있고, 입력을 어디서 받아오는지만 다릅니다.

YOLO는 두 가지 종류의 클래스를 함께 탐지합니다:

- "sign" (강의실 표지판): 글자가 강의실마다 달라서 재학습 없이 대응하려고,
  YOLO는 위치(bbox)만 찾고 그 안의 글자는 OCR로 읽어서 locations.yaml과
  텍스트 유사도로 비교합니다.
- 그 외 클래스(예: "exit" = 비상구): 글자가 아니라 그림(피토그램) 자체가
  의미를 가지는 표지판이라 OCR이 의미가 없어서, YOLO가 이미지만 보고
  직접 분류하도록 학습합니다. 클래스 이름이 곧 locations.yaml의 위치
  이름과 같아서 바로 조회합니다.

한 프레임에 여러 표지판(강의실 표지판 + 비상구 등)이 동시에 보이면, 종류에
상관없이 bbox가 가장 큰(=가장 가까운) 것을 현재 위치로 선택합니다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

_KOREAN_OR_DIGIT = re.compile(r"[0-9가-힣]")

# OCR로 글자를 읽어서 위치를 판별해야 하는 클래스 이름. 이 클래스가 아닌 다른
# 탐지 클래스(예: "exit")는 클래스 이름 자체가 곧 위치이므로 OCR 없이 바로
# locations.yaml을 조회합니다.
OCR_CLASS_NAME = "sign"

from ..detection import get_detector
from ..detection.base import BaseDetector, Detection
from ..localization import Location, MatchResult, SignMatcher, load_locations
from ..ocr import get_ocr_reader
from ..ocr.base import BaseOcrReader
from ..utils.image_utils import crop_bbox, upscale_for_ocr

logger = logging.getLogger("classroom_locator")


@dataclass
class FrameResult:
    detections: list[Detection]
    ocr_texts_per_detection: list[list[str]]
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

        # target_classes에 "sign"이 없으면(=YOLO가 강의실까지 직접 분류하는 방식이면)
        # OCR 모델을 아예 안 띄움 — EasyOCR 로딩은 느리고 안 쓸 거면 낭비이기 때문.
        target_classes = config["detector"].get("target_classes")
        self._uses_ocr = target_classes is None or OCR_CLASS_NAME in target_classes
        self.ocr_reader: BaseOcrReader | None = get_ocr_reader(config["ocr"]) if self._uses_ocr else None
        self.matcher: SignMatcher | None = (
            SignMatcher(self.locations, match_threshold=loc_config.get("match_threshold", 70))
            if self._uses_ocr
            else None
        )

    def process_image(self, image: np.ndarray) -> FrameResult:
        detections = self.detector.detect(image)

        ocr_texts_per_detection: list[list[str]] = []
        candidates: list[tuple[Detection, MatchResult]] = []

        for det in detections:
            if det.class_name != OCR_CLASS_NAME or not self._uses_ocr:
                # 그림(피토그램)만으로 YOLO가 직접 분류한 클래스 (예: "exit").
                # 글자가 없거나 있어도 의미가 없어서 OCR을 거치지 않음.
                ocr_texts_per_detection.append([])
                location = self._location_by_class_name.get(det.class_name)
                if location is not None:
                    candidates.append(
                        (det, MatchResult(location=location, matched_text=det.class_name, score=det.confidence * 100))
                    )
                continue

            crop = crop_bbox(image, det.bbox)
            if crop.size == 0:
                ocr_texts_per_detection.append([])
                continue
            crop = upscale_for_ocr(crop)
            ocr_results = self.ocr_reader.read_text(crop)
            texts = [r.text for r in ocr_results]
            ocr_texts_per_detection.append(texts)

            # EasyOCR이 "4강의실"을 "4"와 "강의실"처럼 조각내서 따로 인식하는 경우가
            # 많음. 조각을 따로따로 비교하면 숫자(구분 정보)가 매칭에 반영이 안 되고,
            # "강의실"만으로는 room2/3/4가 전부 비슷하게 매칭되어 버리므로, 한 표지판
            # 안에서 인식된 조각들을 하나로 합쳐서 비교해야 함. 표지판에 영어 부제
            # ("Meeting Room" 등)도 같이 있어서 그 조각은 매칭에 도움이 안 되므로
            # 한글/숫자가 하나라도 포함된 조각만 사용 (read_text가 이미 왼쪽->오른쪽
            # 순서로 정렬해서 반환하므로 join 순서도 올바름).
            relevant = [t for t in texts if _KOREAN_OR_DIGIT.search(t)]
            joined_text = " ".join(relevant).strip()

            match = self.matcher.match([joined_text]) if joined_text else None
            logger.debug(
                "매칭 시도: joined_text=%r -> %s",
                joined_text,
                f"{match.location.name} (score={match.score:.1f})" if match else "매칭 실패",
            )
            if match:
                candidates.append((det, match))

        # 여러 표지판(강의실 표지판 + 비상구 등)이 동시에 보이면, 종류에 상관없이
        # bbox가 가장 큰(=가장 가까운) 것을 현재 위치로 선택.
        best: MatchResult | None = None
        best_area = -1
        for det, match in candidates:
            area = _bbox_area(det.bbox)
            if area > best_area:
                best_area = area
                best = match

        return FrameResult(
            detections=detections,
            ocr_texts_per_detection=ocr_texts_per_detection,
            match=best,
            object_matches=candidates,
        )
