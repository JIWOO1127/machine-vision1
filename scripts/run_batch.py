#!/usr/bin/env python
"""사진 폴더 배치 처리 실행 스크립트.

사용 예:
    python scripts/run_batch.py --input data/raw --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.config import load_config  # noqa: E402
from classroom_locator.pipeline import run_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="사진 폴더 배치 위치 추적")
    parser.add_argument("--input", type=str, required=True, help="입력 이미지 폴더 (예: data/raw)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_batch(config, args.input)


if __name__ == "__main__":
    main()
