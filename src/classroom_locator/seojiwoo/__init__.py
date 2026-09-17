"""서지우님(machine-vision1, feature/seojiwoo-detection) 코드를 이식한 서브패키지.

원본: https://github.com/JIWOO1127/machine-vision1/tree/feature/seojiwoo-detection/seojiwoo/core
(원본 그대로 보존된 복사본은 프로젝트 루트의 `external/seojiwoo_core/`에 있음)

이 패키지는 원본 로직을 그대로 유지하되, `classroom_locator` 패키지 규약에 맞춰
상대 임포트로만 바꾼 버전이다. 우리 파이프라인(`classroom_locator.pipeline`)과는
독립적으로 동작하며, `scripts/run_seojiwoo_demo.py`에서 사용한다.
"""

from .locator import Locator, DISPLAY, REAL_SIZE, SIGNS
from .navigator import Navigator

__all__ = ["Locator", "DISPLAY", "REAL_SIZE", "SIGNS", "Navigator"]
