"""detection 모듈 테스트.

ultralytics/yolov5가 설치되어 있지 않아도, 인터페이스와 팩토리 동작만 확인합니다.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.detection.base import BaseDetector, Detection  # noqa: E402


class FakeDetector(BaseDetector):
    """테스트용 가짜 탐지기. 실제 YOLO 없이 pipeline 로직을 검증할 때 사용합니다."""

    def detect(self, image):
        return [Detection(bbox=(0, 0, 100, 50), class_name="sign", confidence=0.9)]


def test_fake_detector_returns_detection():
    detector = FakeDetector()
    detections = detector.detect(image=None)
    assert len(detections) == 1
    assert detections[0].class_name == "sign"


def test_get_detector_raises_on_unknown_backend():
    from classroom_locator.detection import get_detector

    with pytest.raises(ValueError):
        get_detector({"backend": "not-a-real-backend", "weights": "x.pt"})
