"""EasyOCR 기반 구현체.

설치: requirements.txt의 easyocr 주석 해제 후 `pip install -r requirements.txt`
"""

from __future__ import annotations

import logging

import numpy as np

from .base import BaseOcrReader, OcrResult

logger = logging.getLogger("classroom_locator")


class EasyOcrReader(BaseOcrReader):
    def __init__(self, lang: list[str] | None = None, min_confidence: float = 0.4) -> None:
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "easyocr 패키지가 설치되어 있지 않습니다. "
                "requirements.txt의 easyocr 주석을 해제하고 설치하세요."
            ) from e

        self.min_confidence = min_confidence

        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

        # EasyOCR의 한글 코드는 'ko'
        self._reader = easyocr.Reader(lang or ["ko", "en"], gpu=use_gpu)

    def read_text(self, image: np.ndarray) -> list[OcrResult]:
        raw_results = self._reader.readtext(image)
        if raw_results:
            logger.debug(
                "OCR 원본 인식 결과(필터 전): %s",
                [(text.strip(), round(float(conf), 2)) for _bbox, text, conf in raw_results],
            )
        else:
            logger.debug("OCR 원본 인식 결과(필터 전): 없음 (글자를 전혀 못 찾음)")

        filtered = [item for item in raw_results if item[2] >= self.min_confidence]
        # 표지판 안에서 글자가 여러 조각으로 나뉘어 인식될 때(예: "4"와 "강의실"이
        # 따로 인식) 올바른 순서로 합칠 수 있도록, 왼쪽에서 오른쪽 순서(bbox의 가장
        # 왼쪽 x좌표 기준)로 정렬해서 반환합니다.
        filtered.sort(key=lambda item: min(point[0] for point in item[0]))

        return [OcrResult(text=text.strip(), confidence=float(confidence)) for _bbox, text, confidence in filtered]
