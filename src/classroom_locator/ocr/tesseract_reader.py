"""Tesseract 기반 구현체.

설치:
1) requirements.txt의 pytesseract 주석 해제 후 설치
2) 시스템에 tesseract-ocr 엔진 설치 필요 (예: Ubuntu -> `sudo apt install tesseract-ocr tesseract-ocr-kor`)
한글 인식 정확도는 EasyOCR/PaddleOCR보다 낮은 편이니, 정확도가 부족하면 다른 backend로 교체하세요.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOcrReader, OcrResult


class TesseractOcrReader(BaseOcrReader):
    def __init__(self, lang: list[str] | None = None, min_confidence: float = 0.4) -> None:
        try:
            import pytesseract
        except ImportError as e:
            raise ImportError(
                "pytesseract 패키지가 설치되어 있지 않습니다. "
                "requirements.txt의 pytesseract 주석을 해제하고 설치하세요."
            ) from e

        self._pytesseract = pytesseract
        self.min_confidence = min_confidence
        # Tesseract 언어 코드: 한글은 'kor'
        lang = lang or ["ko", "en"]
        mapped = ["kor" if code == "ko" else "eng" for code in lang]
        self.lang_str = "+".join(dict.fromkeys(mapped))

    def read_text(self, image: np.ndarray) -> list[OcrResult]:
        data = self._pytesseract.image_to_data(
            image, lang=self.lang_str, output_type=self._pytesseract.Output.DICT
        )
        results: list[OcrResult] = []
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            conf = float(conf) / 100.0 if conf != "-1" else 0.0
            if text and conf >= self.min_confidence:
                results.append(OcrResult(text=text, confidence=conf))
        return results
