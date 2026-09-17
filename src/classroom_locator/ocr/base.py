"""OCR 리더(Reader) 공용 인터페이스.

EasyOCR / PaddleOCR / Tesseract 중 무엇을 쓰든 이 인터페이스만
맞추면 pipeline 코드는 수정할 필요가 없습니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class OcrResult:
    """OCR 인식 결과 1건."""

    text: str
    confidence: float


class BaseOcrReader(ABC):
    """모든 OCR 구현체가 상속해야 하는 추상 클래스."""

    @abstractmethod
    def read_text(self, image: np.ndarray) -> list[OcrResult]:
        """이미지(보통 탐지된 표지판을 crop한 영역)에서 텍스트를 읽어냅니다."""
        raise NotImplementedError
