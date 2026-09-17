"""ocr 모듈 테스트.

EasyOCR/PaddleOCR/Tesseract가 설치되어 있지 않아도, 인터페이스와 팩토리
동작(알 수 없는 backend 처리 등)만 확인합니다.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.ocr.base import BaseOcrReader, OcrResult  # noqa: E402


class FakeOcrReader(BaseOcrReader):
    """테스트용 가짜 OCR. 실제 OCR 라이브러리 없이 pipeline 로직을 검증할 때 사용합니다."""

    def read_text(self, image):
        return [OcrResult(text="공학관 301호", confidence=0.95)]


def test_fake_ocr_reader_returns_result():
    reader = FakeOcrReader()
    results = reader.read_text(image=None)
    assert len(results) == 1
    assert results[0].text == "공학관 301호"


def test_get_ocr_reader_raises_on_unknown_backend():
    from classroom_locator.ocr import get_ocr_reader

    with pytest.raises(ValueError):
        get_ocr_reader({"backend": "not-a-real-backend"})
