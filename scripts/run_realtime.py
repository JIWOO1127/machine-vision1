#!/usr/bin/env python
"""웹캠 실시간 위치 추적 실행 스크립트.

사용 예:
    python scripts/run_realtime.py --camera 0 --config configs/default.yaml
    python scripts/run_realtime.py --source data/raw/test_video.mp4
    python scripts/run_realtime.py --source data/raw/test_video.mp4 --snapshot
    python scripts/run_realtime.py --source data/raw/test_video.mp4 --grid-map

스냅샷 저장/격자 지도 창 모두 기본값은 꺼짐(false)이며, 켜려면 위처럼
--snapshot / --grid-map을 명시해야 함 (configs/default.yaml의
realtime.save_snapshots / realtime.show_grid_map 참고).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# src/ 를 import 경로에 추가 (VSCode .vscode/settings.json에서도 동일하게 설정되어 있음)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.config import load_config  # noqa: E402
from classroom_locator.pipeline import run_realtime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="웹캠 실시간 위치 추적 / 영상 파일 재생 테스트")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--camera", type=int, default=None, help="카메라 인덱스 (설정 파일 값 덮어쓰기)")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="웹캠 대신 재생할 영상 파일 경로. 지정하면 --camera보다 우선함",
    )
    parser.add_argument(
        "--snapshot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="위치가 바뀔 때마다 스냅샷 저장 여부 (--snapshot/--no-snapshot). "
        "생략하면 설정 파일의 realtime.save_snapshots 값을 따름 (기본 꺼짐)",
    )
    parser.add_argument(
        "--grid-map",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="격자 지도 위치 창 표시 여부 (--grid-map/--no-grid-map). "
        "생략하면 설정 파일의 realtime.show_grid_map 값을 따름 (기본 꺼짐)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.camera is not None:
        config.setdefault("realtime", {})["camera_index"] = args.camera
    if args.source is not None:
        config.setdefault("realtime", {})["camera_index"] = args.source
    if args.snapshot is not None:
        config.setdefault("realtime", {})["save_snapshots"] = args.snapshot
    if args.grid_map is not None:
        config.setdefault("realtime", {})["show_grid_map"] = args.grid_map

    run_realtime(config)


if __name__ == "__main__":
    main()
