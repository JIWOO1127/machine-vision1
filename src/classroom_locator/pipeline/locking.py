"""위치가 한 번 정해지면 잠깐의 오탐/공백에 흔들리지 않도록 유지하는 로직.

realtime_pipeline.py(화면 표시)와 evaluation.py(성능 평가)가 똑같은 규칙으로
동작해야, 평가 결과가 실제 화면에 보이는 동작과 일치합니다. 그래서 이 로직을
공용 클래스로 분리했습니다.
"""

from __future__ import annotations

from ..localization import MatchResult

MIN_LOCK_SECONDS = 3.0


class LocationLocker:
    """탐지 결과(MatchResult)를 시간 순서대로 넣으면, 화면에 표시해야 할 현재
    위치를 결정합니다. 한 번 정해진 위치는 최소 min_lock_seconds 동안 유지됩니다.
    """

    def __init__(self, min_lock_seconds: float = MIN_LOCK_SECONDS) -> None:
        self.min_lock_seconds = min_lock_seconds
        self.current: MatchResult | None = None
        self._locked_until = 0.0

    def update(self, match: MatchResult | None, now: float) -> tuple[MatchResult | None, bool]:
        """새 탐지 결과를 반영합니다.

        Returns:
            (지금 화면에 표시해야 할 결과, 방금 다른 위치로 전환됐는지 여부)
        """
        if match is None:
            return self.current, False

        is_locked = self.current is not None and now < self._locked_until

        if self.current is not None and match.location.name == self.current.location.name:
            self.current = match  # 같은 위치, 점수만 최신으로 갱신
            return self.current, False

        if is_locked:
            return self.current, False  # 다른 위치가 보여도 잠금 중이면 무시

        self.current = match
        self._locked_until = now + self.min_lock_seconds
        return self.current, True
