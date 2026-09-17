#!/usr/bin/env python
"""configs/locations.yaml 기반으로 2D 평면도를 이미지로 생성하는 스크립트.

사용 예:
    python scripts/visualize_map.py
    python scripts/visualize_map.py --highlight room3
    python scripts/visualize_map.py --highlight room3 --output outputs/results/map.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.localization import load_locations  # noqa: E402
from classroom_locator.utils import draw_floor_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="평면도 이미지 생성")
    parser.add_argument("--locations", type=str, default="configs/locations.yaml")
    parser.add_argument("--highlight", type=str, default=None, help="현재 위치로 강조할 location name")
    parser.add_argument("--output", type=str, default="outputs/results/map.png")
    args = parser.parse_args()

    locations = load_locations(args.locations)
    output_path = draw_floor_map(locations, args.output, highlight=args.highlight)
    print(f"지도 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
