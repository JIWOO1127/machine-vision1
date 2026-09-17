#!/usr/bin/env python
"""Roboflow에서 export한 라벨링된 테스트셋(예: 다운로드 폴더의 "test sample.v1i.yolov8")을
정답(ground truth)으로 삼아, 실제 파이프라인(YOLO+OCR)의 성능을 평가합니다.

두 가지 모드:
- 기본(위치 모드): 한 이미지에 물체가 여러 개 찍혔으면, bbox 면적이 가장
  큰(=가장 가까운) 물체 하나만 그 이미지의 "정답 위치"로 삼습니다. 실제
  파이프라인도 여러 물체가 동시에 보이면 가장 큰 것 하나를 "현재 위치"로
  뽑기 때문에, 실제 사용 시나리오와 동일한 기준입니다.
- --per-object (물체별 모드): "현재 위치가 어디냐"라는 우선순위 로직은 빼고,
  라벨링된 물체 하나하나가 개별적으로 올바르게 인식되는지만 봅니다 (예: 한
  사진에 표지판과 문이 같이 찍혔으면 둘 다 채점 대상). bbox가 겹치는(IoU가
  가장 높은) 탐지 결과를 그 물체의 예측값으로 매칭합니다.

클래스 이름은 data.yaml의 이름(예: "2_class", "4_class")을 locations.yaml
기준 이름(room2, room4 등)으로 매핑합니다.

사용 예:
    python scripts/evaluate_test_sample.py --dataset-dir "C:\\Users\\...\\Downloads\\test sample.v1i.yolov8"
    python scripts/evaluate_test_sample.py --dataset-dir "..." --per-object
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.config import load_config  # noqa: E402
from classroom_locator.eval_metrics import ConfusionMatrix, plot_confusion_matrix  # noqa: E402
from classroom_locator.pipeline import LocatorPipeline  # noqa: E402

# data.yaml 클래스 이름 -> locations.yaml/우리 시스템 클래스 이름. 여기 없는
# 이름은 그대로 쓰거나(front_door 등 이미 같으면), --class-map으로 추가하세요.
DEFAULT_CLASS_MAP = {
    "2_class": "room2",
    "3_class": "room3",
    "4_class": "room4",
}

MIN_IOU_FOR_MATCH = 0.3


@dataclass
class LabeledObject:
    class_name: str  # 매핑 적용된 이름
    bbox_norm: tuple[float, float, float, float]  # cx, cy, w, h (0~1)

    @property
    def area(self) -> float:
        return self.bbox_norm[2] * self.bbox_norm[3]

    def to_pixel_xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        cx, cy, w, h = self.bbox_norm
        return (
            (cx - w / 2) * img_w,
            (cy - h / 2) * img_h,
            (cx + w / 2) * img_w,
            (cy + h / 2) * img_h,
        )


def _find_split_dir(dataset_dir: Path) -> Path:
    for split in ("train", "valid", "test"):
        if (dataset_dir / split / "images").exists():
            return dataset_dir / split
    raise FileNotFoundError(f"{dataset_dir} 안에서 train/valid/test 이미지 폴더를 못 찾았습니다.")


def _load_labels(label_path: Path, class_names: list[str], class_map: dict[str, str]) -> list[LabeledObject]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    objects = []
    with label_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            raw_name = class_names[cls_id]
            objects.append(LabeledObject(class_map.get(raw_name, raw_name), (cx, cy, w, h)))
    return objects


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _predict_for_object(
    obj: LabeledObject,
    candidates: list[tuple[tuple[int, int, int, int], str]],
    img_w: int,
    img_h: int,
) -> str | None:
    """그 물체(GT bbox)와 IoU가 가장 높은 탐지 결과를 찾아 예측 위치를 반환."""
    gt_box = obj.to_pixel_xyxy(img_w, img_h)
    best_iou = 0.0
    best_label: str | None = None
    for bbox, label in candidates:
        iou = _iou(gt_box, bbox)
        if iou > best_iou:
            best_iou = iou
            best_label = label
    return best_label if best_iou >= MIN_IOU_FOR_MATCH else None


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨링된 테스트셋으로 파이프라인 성능 평가")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Roboflow export 폴더 (data.yaml 포함)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument(
        "--per-object",
        action="store_true",
        help="위치(우선순위) 로직 없이, 라벨링된 물체 하나하나가 개별적으로 맞는지 채점",
    )
    parser.add_argument(
        "--raw-detector",
        action="store_true",
        help="OCR/locations.yaml 매칭을 거치지 않고, YOLO 탐지 결과의 클래스 이름을 "
        "바로 예측값으로 씀 (DEFAULT_CLASS_MAP으로 이름만 맞춤). 클래스 이름이 "
        "locations.yaml과 다른 모델(예: 2_class/4_class 그대로인 모델)을 테스트할 때 사용",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    with (dataset_dir / "data.yaml").open(encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    class_names: list[str] = data_yaml["names"]

    split_dir = _find_split_dir(dataset_dir)
    image_paths = sorted(
        p for p in (split_dir / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    mode_desc = "물체별(위치 제외)" if args.per_object else "위치(최대 bbox 기준)"
    print(f"테스트 이미지 {len(image_paths)}장 ({split_dir}) - 평가 모드: {mode_desc}")

    config = load_config(args.config)
    pipeline = LocatorPipeline(config)

    confusion = ConfusionMatrix()
    skipped = 0

    for image_path in image_paths:
        label_path = split_dir / "labels" / f"{image_path.stem}.txt"
        labeled_objects = _load_labels(label_path, class_names, DEFAULT_CLASS_MAP)
        if not labeled_objects:
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        if args.raw_detector:
            # OCR/위치 매칭 없이, YOLO가 뱉은 클래스 이름을 이름만 맞춰서 바로 사용
            raw_dets = pipeline.detector.detect(image)
            candidates = [(det.bbox, DEFAULT_CLASS_MAP.get(det.class_name, det.class_name)) for det in raw_dets]
        else:
            result = pipeline.process_image(image)
            candidates = [(det.bbox, match.location.name) for det, match in result.object_matches]

        if not args.per_object:
            # 위치 모드: bbox가 가장 큰 라벨 하나만 정답으로 채점 (실제 파이프라인의
            # "현재 위치" 판단과 동일한 기준)
            true_obj = max(labeled_objects, key=lambda o: o.area)
            if candidates:
                predicted = max(candidates, key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]))[1]
            else:
                predicted = None
            confusion.add(true_obj.class_name, predicted)
        else:
            # 물체별 모드: 라벨링된 물체 전부를 각각 채점
            img_h, img_w = image.shape[:2]
            for obj in labeled_objects:
                predicted = _predict_for_object(obj, candidates, img_w, img_h)
                confusion.add(obj.class_name, predicted)

    if skipped:
        print(f"(라벨/이미지 문제로 {skipped}장 건너뜀)")

    if confusion.total == 0:
        print("평가할 이미지가 없습니다.")
        return

    print()
    print(confusion.format_summary())

    print()
    print("=== 클래스별 precision / recall / F1 ===")
    header = f"{'클래스':<18}{'precision':>10}{'recall':>10}{'f1':>10}"
    print(header)
    print("-" * len(header))
    for label in confusion.labels:
        p, r, f1 = confusion.precision_recall_f1(label)
        print(f"{label:<18}{p:>10.3f}{r:>10.3f}{f1:>10.3f}")

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_per_object" if args.per_object else ""
    report_path = output_dir / f"eval_report{suffix}_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(confusion.to_dict(), f, ensure_ascii=False, indent=2)

    plot_path = plot_confusion_matrix(confusion, output_dir / f"confusion_matrix{suffix}_{timestamp}.png")

    print(f"\n상세 결과 저장: {report_path}")
    print(f"혼동행렬 그래프 저장: {plot_path}")


if __name__ == "__main__":
    main()
