"""설정 파일(yaml) 로더.

configs/default.yaml 같은 설정 파일을 읽어 딕셔너리로 반환합니다.
프로젝트가 커지면 dataclass로 바꿔도 되지만, 초기 단계에서는
간단한 dict 기반으로 시작합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """yaml 설정 파일을 읽어 dict로 반환합니다."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}
