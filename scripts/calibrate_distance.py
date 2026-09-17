#!/usr/bin/env python
"""distance_data 폴더(파일명이 "클래스_거리(m).jpg"인 사진들)로 YOLO를 돌려서,
bbox 크기와 실제 거리의 대응표를 만드는 스크립트.

파일명 예: sign_1.8.jpg -> 클래스 sign(표지판), 실제 거리 1.8m
          front_4.05.jpg -> 클래스 front_door, 실제 거리 4.05m
          rear_2.7.jpg -> 클래스 rear_door, 실제 거리 2.7m

크기 지표로는 bbox의 sqrt(width * height)를 이미지 대각선 길이로 나눈 비율을
씁니다 (해상도가 달라도 비교 가능하고, 원근 카메라 모델에서 거리에 반비례하는
값이라 다음 단계(크기->거리 변환 공식)에 쓰기 좋음).

사용 예:
    python scripts/calibrate_distance.py --distance-dir "C:\\...\\distance_data\\distance_data"
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.config import load_config  # noqa: E402
from classroom_locator.pipeline import LocatorPipeline  # noqa: E402

# 파일명 접두사 -> YOLO 클래스 이름. sign은 room2/room4 어느 쪽으로 잡히든
# 표지판 실제 물리 크기는 같으므로 구분 없이 하나로 취급.
FILENAME_PREFIX_TO_CLASSES = {
    "front": {"front_door"},
    "rear": {"rear_door"},
    "sign": {"room2", "room3", "room4"},
}

FILENAME_RE = re.compile(r"^([a-zA-Z]+)_([0-9]+(?:\.[0-9]+)?)\.")


def _imread_unicode(path: Path) -> np.ndarray | None:
    """한글/공백이 섞인 경로에서도 안전하게 이미지를 읽음 (cv2.imread는 이런
    경로에서 실패하는 경우가 있어서 우회)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description="크기<->거리 캘리브레이션 데이터 생성")
    parser.add_argument("--distance-dir", type=str, required=True, help="distance_data 사진 폴더")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="data/eval/distance_calibration.csv")
    args = parser.parse_args()

    distance_dir = Path(args.distance_dir)
    image_paths = sorted(p for p in distance_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"거리 캘리브레이션 사진 {len(image_paths)}장 ({distance_dir})")

    config = load_config(args.config)
    pipeline = LocatorPipeline(config)
    # 캘리브레이션은 "실제로 어느 정도 크기로 찍히는지"만 측정하면 되므로,
    # 운영용 conf_threshold(0.5)에 안 걸려도 상관없음. 실제로 sign류는 이
    # 고해상도 원본 사진에서 신뢰도가 낮게 나와서(학습 이미지보다 훨씬 큰
    # 해상도라 리사이즈 시 작아지는 영향으로 추정) 낮춰서 탐지함.
    pipeline.detector.conf_threshold = 0.1

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for image_path in image_paths:
        m = FILENAME_RE.match(image_path.name)
        if not m:
            skipped.append(f"{image_path.name} (파일명 형식 안 맞음)")
            continue
        prefix, distance_str = m.group(1), m.group(2)
        expected_classes = FILENAME_PREFIX_TO_CLASSES.get(prefix)
        if expected_classes is None:
            skipped.append(f"{image_path.name} (알 수 없는 접두사 '{prefix}')")
            continue
        true_distance_m = float(distance_str)

        image = _imread_unicode(image_path)
        if image is None:
            skipped.append(f"{image_path.name} (이미지 읽기 실패)")
            continue
        img_h, img_w = image.shape[:2]
        diagonal = math.hypot(img_w, img_h)

        detections = pipeline.detector.detect(image)
        matches = [d for d in detections if d.class_name in expected_classes]
        if not matches:
            skipped.append(f"{image_path.name} (예상 클래스 {expected_classes} 탐지 안 됨)")
            continue
        # 여러 개 잡히면 가장 크게 잡힌 것(=제일 확실한 것) 사용
        det = max(matches, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        bbox_w = det.bbox[2] - det.bbox[0]
        bbox_h = det.bbox[3] - det.bbox[1]
        size_ratio = math.sqrt(bbox_w * bbox_h) / diagonal

        rows.append(
            {
                "file": image_path.name,
                "class": det.class_name,
                "true_distance_m": true_distance_m,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "img_w": img_w,
                "img_h": img_h,
                "size_ratio": round(size_ratio, 6),
                "confidence": round(det.confidence, 3),
            }
        )
        print(f"{image_path.name}: class={det.class_name} dist={true_distance_m}m size_ratio={size_ratio:.5f}")

    if skipped:
        print("\n건너뜀:")
        for s in skipped:
            print(f"  - {s}")

    if not rows:
        print("추출된 데이터가 없습니다.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n캘리브레이션 데이터 저장 완료: {output_path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
