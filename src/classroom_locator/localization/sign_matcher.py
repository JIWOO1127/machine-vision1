"""OCR로 인식된 텍스트를 위치 목록과 매칭하는 로직.

정확히 일치하지 않아도(OCR 오인식 대비) rapidfuzz로 가장 유사한 위치를 찾습니다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .location_map import Location


@dataclass
class MatchResult:
    location: Location
    matched_text: str
    score: float  # 0~100


class SignMatcher:
    def __init__(self, locations: list[Location], match_threshold: float = 70.0) -> None:
        self.locations = locations
        self.match_threshold = match_threshold

    def match(self, ocr_texts: list[str]) -> MatchResult | None:
        """OCR로 인식된 여러 텍스트 후보 중 가장 잘 맞는 위치 하나를 반환합니다.

        일치하는 위치가 없거나, 최고 점수가 서로 다른 위치 사이에 동점으로
        걸려서 애매하면(예: "강의실"만 읽혀서 room2/3/4가 전부 똑같이 높게
        나오는 경우) None을 반환합니다 — 애매한 추측으로 엉뚱한 위치를
        확정하는 것보다, 판단을 보류하는 게 낫기 때문입니다.
        """
        try:
            from rapidfuzz import fuzz
        except ImportError as e:
            raise ImportError(
                "rapidfuzz 패키지가 설치되어 있지 않습니다. requirements.txt를 확인하세요."
            ) from e

        best: MatchResult | None = None
        ambiguous = False

        for text in ocr_texts:
            text = text.strip()
            if not text:
                continue
            for location in self.locations:
                for candidate_name in location.all_names():
                    score = fuzz.ratio(text, candidate_name)
                    if score < self.match_threshold:
                        continue
                    if best is None or score > best.score:
                        best = MatchResult(location=location, matched_text=text, score=score)
                        ambiguous = False
                    elif score == best.score and location is not best.location:
                        ambiguous = True

        return None if ambiguous else best
