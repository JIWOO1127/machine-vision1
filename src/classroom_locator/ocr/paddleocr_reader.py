"""PaddleOCR 기반 구현체.

설치: requirements.txt의 paddlepaddle / paddleocr 주석 해제 후 설치.
한글 포함 다국어 인식 정확도가 좋은 편이라 정확도를 우선한다면 추천.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOcrReader, OcrResult


class PaddleOcrReader(BaseOcrReader):
    def __init__(self, lang: list[str] | None = None, min_confidence: float = 0.4) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "paddleocr 패키지가 설치되어 있지 않습니다. "
                "requirements.txt의 paddlepaddle/paddleocr 주석을 해제하고 설치하세요."
            ) from e

        self.min_confidence = min_confidence
        # PaddleOCR는 lang 하나만 지정 (한글은 'korean')
        primary_lang = "korean" if not lang or "ko" in lang else lang[0]
        self._reader = PaddleOCR(use_angle_cls=True, lang=primary_lang, show_log=False)

    def read_text(self, image: np.ndarray) -> list[OcrResult]:
        raw_results = self._reader.ocr(image, cls=True)
        results: list[OcrResult] = []
        for line in raw_results or []:
            for _bbox, (text, confidence) in line:
                if confidence >= self.min_confidence:
                    results.append(OcrResult(text=text.strip(), confidence=float(confidence)))
        return results
