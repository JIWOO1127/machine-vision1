"""localization 모듈 테스트.

실제 YOLO/OCR 라이브러리 설치 없이도 실행 가능합니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.localization import Location, SignMatcher  # noqa: E402


def make_sample_locations() -> list[Location]:
    return [
        Location(name="공학관 301호", aliases=["301호"], building="공학관", floor=3),
        Location(name="공학관 302호", aliases=["302호"], building="공학관", floor=3),
    ]


def test_exact_match():
    matcher = SignMatcher(make_sample_locations(), match_threshold=70)
    result = matcher.match(["공학관 301호"])
    assert result is not None
    assert result.location.name == "공학관 301호"
    assert result.score == 100


def test_noisy_ocr_text_still_matches():
    # OCR이 살짝 잘못 인식한 상황을 가정 (예: '공학관 3O1호' 처럼 숫자 0을 O로 오인식)
    matcher = SignMatcher(make_sample_locations(), match_threshold=70)
    result = matcher.match(["공학관 3O1호"])
    assert result is not None
    assert result.location.name == "공학관 301호"


def test_no_match_returns_none():
    matcher = SignMatcher(make_sample_locations(), match_threshold=90)
    result = matcher.match(["전혀 관련 없는 텍스트"])
    assert result is None
