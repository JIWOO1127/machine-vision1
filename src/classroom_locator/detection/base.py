"""탐지기(Detector) 공용 인터페이스.

어떤 YOLO 버전/프레임워크를 쓰든 이 인터페이스만 맞추면
pipeline 코드는 전혀 수정할 필요가 없습니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """탐지 결과 1건."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) 픽셀 좌표
    class_name: str
    confidence: float


class BaseDetector(ABC):
    """모든 탐지기 구현체가 상속해야 하는 추상 클래스."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """이미지(BGR, np.ndarray)를 입력받아 탐지 결과 목록을 반환합니다."""
        raise NotImplementedError
