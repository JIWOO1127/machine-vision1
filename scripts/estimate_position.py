#!/usr/bin/env python
"""사진 한 장(또는 여러 장 연속)으로 VisualLocationEstimator를 테스트하는 스크립트.

사용 예:
    python scripts/estimate_position.py --image outputs/eval/sample_results/xxx.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.config import load_config  # noqa: E402
from classroom_locator.localization import VisualLocationEstimator, load_locations  # noqa: E402
from classroom_locator.pipeline import LocatorPipeline  # noqa: E402
from classroom_locator.utils import draw_floor_map  # noqa: E402


def _imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description="사진으로 2D 위치 추정 테스트")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--calibration", type=str, default="data/distance_calibration.json")
    parser.add_argument("--output", type=str, default="outputs/eval/position_estimate.png")
    parser.add_argument("--conf", type=float, default=None, help="detector conf_threshold 임시 덮어쓰기(테스트용)")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = LocatorPipeline(config)
    if args.conf is not None:
        pipeline.detector.conf_threshold = args.conf
    locations = load_locations(config["localization"]["locations_file"])

    with open(args.calibration, encoding="utf-8") as f:
        calibration = json.load(f)
    calibration.pop("_comment", None)

    estimator = VisualLocationEstimator(locations, calibration)

    image = _imread_unicode(Path(args.image))
    h, w = image.shape[:2]

    detections = pipeline.detector.detect(image)
    print("탐지된 물체:")
    for d in detections:
        print(f"  {d.class_name} conf={d.confidence:.2f} bbox={d.bbox}")

    result = estimator.estimate(detections, frame_width=w, frame_height=h)
    print("\n=== 위치 추정 결과 ===")
    print(f"상태: {result.status}")
    print(f"좌표: x={result.x_m}, y={result.y_m}")
    print(f"신뢰도: {result.confidence}")
    print(f"라벨: {result.label}")
    print(f"사용된 랜드마크: {result.anchors}")
    for m in result.measurements:
        print(f"  - {m['class_name']}: 거리 {m['distance_m']}m, 방향각 {m['bearing_deg']}도")

    if result.x_m is not None:
        output_path = draw_floor_map(
            locations, args.output, estimated_position=(result.x_m, result.y_m)
        )
        print(f"\n지도 저장: {output_path}")


if __name__ == "__main__":
    main()
