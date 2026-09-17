"""OCR 리더 팩토리.

configs/default.yaml의 ocr 설정을 받아 알맞은 구현체를 생성합니다.
"""

from __future__ import annotations

from typing import Any

from .base import BaseOcrReader, OcrResult

__all__ = ["BaseOcrReader", "OcrResult", "get_ocr_reader"]


def get_ocr_reader(config: dict[str, Any]) -> BaseOcrReader:
    backend = config.get("backend", "easyocr")
    lang = config.get("lang", ["ko", "en"])
    min_confidence = config.get("min_confidence", 0.4)

    if backend == "easyocr":
        from .easyocr_reader import EasyOcrReader

        return EasyOcrReader(lang=lang, min_confidence=min_confidence)

    if backend == "paddleocr":
        from .paddleocr_reader import PaddleOcrReader

        return PaddleOcrReader(lang=lang, min_confidence=min_confidence)

    if backend == "tesseract":
        from .tesseract_reader import TesseractOcrReader

        return TesseractOcrReader(lang=lang, min_confidence=min_confidence)

    raise ValueError(f"알 수 없는 ocr backend: {backend}")
