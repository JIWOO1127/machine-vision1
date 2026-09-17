from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vision_matplotlib")
)

import matplotlib
import numpy as np
import torch
from matplotlib import font_manager
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faster_rcnn_pipeline import (
    BASE_DIR,
    CLASS_NAMES,
    DetectionDataset,
    build_model,
    collate_fn,
    convert_yolo_to_coco,
)

DEFAULT_TEST_ROOT = BASE_DIR / "test sample.v1i.yolov8"
DEFAULT_WEIGHTS = BASE_DIR / "models" / "faster_rcnn_916machine_4class_best.pt"
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "faster_rcnn_test"
DISPLAY_CLASS_SPECS = [
    ("front_door", "front_door"),
    ("rear_door", "rear_door"),
    ("2_class", "room2"),
    ("4_class", "room4"),
]
DISPLAY_CLASS_NAMES = [display_name for _, display_name in DISPLAY_CLASS_SPECS]
MODEL_NAME_TO_DISPLAY_NAME = dict(DISPLAY_CLASS_SPECS)
CLASS_ID_TO_DISPLAY_INDEX = {
    CLASS_NAMES.index(model_name) + 1: display_index
    for display_index, (model_name, _) in enumerate(DISPLAY_CLASS_SPECS)
}


def configure_plot_font():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_model(weights_path: Path, device: torch.device):
    checkpoint = torch.load(str(weights_path), map_location=device)
    if not checkpoint.get("exif_corrected", False):
        raise RuntimeError(
            "The checkpoint predates the EXIF orientation fix. Retrain with the "
            "current faster_rcnn_pipeline.py before evaluating it."
        )
    class_names = checkpoint.get("class_names", [])
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected checkpoint classes {CLASS_NAMES}, got {class_names}")
    model = build_model(
        len(class_names) + 1,
        pretrained=False,
        min_size=int(checkpoint.get("min_size", 1200)),
        max_size=int(checkpoint.get("max_size", 2000)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def yolo_test_summary(dataset_root: Path, split: str):
    label_dir = dataset_root / split / "labels"
    image_dir = dataset_root / split / "images"
    counts = Counter()
    for label_path in label_dir.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if values:
                counts[int(float(values[0]))] += 1
    return {
        "images": sum(1 for path in image_dir.glob("*") if path.is_file()),
        "objects": {
            CLASS_NAMES[class_id]: counts[class_id]
            for class_id in range(len(CLASS_NAMES))
        },
    }


def match_at_threshold(predictions, targets, confidence: float, iou_threshold=0.5):
    totals = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in range(1, len(CLASS_NAMES) + 1)
    }
    for prediction, target in zip(predictions, targets):
        for class_id in totals:
            pred_mask = (prediction["labels"] == class_id) & (
                prediction["scores"] >= confidence
            )
            pred_boxes = prediction["boxes"][pred_mask]
            pred_scores = prediction["scores"][pred_mask]
            gt_boxes = target["boxes"][target["labels"] == class_id]
            matched_gt = set()

            for pred_index in torch.argsort(pred_scores, descending=True).tolist():
                if len(gt_boxes) == 0:
                    totals[class_id]["fp"] += 1
                    continue
                ious = box_iou(pred_boxes[pred_index].unsqueeze(0), gt_boxes)[0]
                available = [
                    index for index in range(len(gt_boxes)) if index not in matched_gt
                ]
                if not available:
                    totals[class_id]["fp"] += 1
                    continue
                best_gt = max(available, key=lambda index: float(ious[index]))
                if float(ious[best_gt]) >= iou_threshold:
                    matched_gt.add(best_gt)
                    totals[class_id]["tp"] += 1
                else:
                    totals[class_id]["fp"] += 1
            totals[class_id]["fn"] += len(gt_boxes) - len(matched_gt)

    metrics = {}
    for class_id, values in totals.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[CLASS_NAMES[class_id - 1]] = {
            **values,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def confusion_matrix_at_threshold(
    predictions,
    targets,
    confidence: float,
    iou_threshold: float = 0.5,
):
    """Build an actual(row) x predicted(column) matrix with a no-response column.

    A prediction is greedily matched to the unmatched ground-truth box with the
    highest IoU, regardless of class. This makes wrong-class detections appear
    in off-diagonal cells. Unmatched ground truths use the final no-response
    column. Unmatched predictions are returned separately so they affect
    precision without adding a background row to the displayed matrix.
    """
    no_response_index = len(DISPLAY_CLASS_NAMES)
    matrix = np.zeros(
        (len(DISPLAY_CLASS_NAMES), len(DISPLAY_CLASS_NAMES) + 1), dtype=np.int64
    )
    unmatched_predictions = np.zeros(len(DISPLAY_CLASS_NAMES), dtype=np.int64)

    for prediction, target in zip(predictions, targets):
        keep = prediction["scores"] >= confidence
        pred_boxes = prediction["boxes"][keep]
        pred_scores = prediction["scores"][keep]
        pred_labels = prediction["labels"][keep]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        matched_gt = set()

        for pred_index in torch.argsort(pred_scores, descending=True).tolist():
            pred_class = CLASS_ID_TO_DISPLAY_INDEX[int(pred_labels[pred_index])]
            if len(gt_boxes) == 0:
                unmatched_predictions[pred_class] += 1
                continue

            ious = box_iou(pred_boxes[pred_index].unsqueeze(0), gt_boxes)[0]
            available = [
                gt_index
                for gt_index in range(len(gt_boxes))
                if gt_index not in matched_gt
            ]
            if not available:
                unmatched_predictions[pred_class] += 1
                continue

            best_gt = max(available, key=lambda gt_index: float(ious[gt_index]))
            if float(ious[best_gt]) >= iou_threshold:
                actual_class = CLASS_ID_TO_DISPLAY_INDEX[int(gt_labels[best_gt])]
                matrix[actual_class, pred_class] += 1
                matched_gt.add(best_gt)
            else:
                unmatched_predictions[pred_class] += 1

        for gt_index, gt_label in enumerate(gt_labels.tolist()):
            if gt_index not in matched_gt:
                actual_class = CLASS_ID_TO_DISPLAY_INDEX[int(gt_label)]
                matrix[actual_class, no_response_index] += 1

    return matrix, unmatched_predictions


def metrics_from_confusion_matrix(
    matrix: np.ndarray, unmatched_predictions: np.ndarray
):
    metrics = {}
    for class_index, class_name in enumerate(DISPLAY_CLASS_NAMES):
        tp = int(matrix[class_index, class_index])
        fp = int(
            matrix[:, class_index].sum()
            - tp
            + unmatched_predictions[class_index]
        )
        fn = int(matrix[class_index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def save_confusion_matrix_csv(matrix: np.ndarray, output_path: Path):
    predicted_labels = [*DISPLAY_CLASS_NAMES, "(무응답)"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["정답\\예측", *predicted_labels])
        for label, row in zip(DISPLAY_CLASS_NAMES, matrix.tolist()):
            writer.writerow([label, *row])


def save_class_metrics_csv(metrics, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["class", "precision", "recall", "f1", "tp", "fp", "fn"])
        for class_name in DISPLAY_CLASS_NAMES:
            values = metrics[class_name]
            writer.writerow(
                [
                    class_name,
                    f"{values['precision']:.6f}",
                    f"{values['recall']:.6f}",
                    f"{values['f1']:.6f}",
                    values["tp"],
                    values["fp"],
                    values["fn"],
                ]
            )


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    output_path: Path,
    dataset_image_count: int,
):
    configure_plot_font()
    predicted_labels = [*DISPLAY_CLASS_NAMES, "(무응답)"]
    total_ground_truths = int(matrix.sum())
    answered = int(matrix[:, :-1].sum())
    correct = int(np.trace(matrix[:, : len(DISPLAY_CLASS_NAMES)]))
    sign_total = int(matrix[2:4, :].sum())
    sign_answered = int(matrix[2:4, :-1].sum())
    coverage = answered / total_ground_truths if total_ground_truths else 0.0
    response_accuracy = correct / answered if answered else 0.0
    sign_coverage = sign_answered / sign_total if sign_total else 0.0

    max_value = int(matrix.max()) if matrix.size else 0
    color_max = max(5, int(np.ceil(max_value / 5.0) * 5))
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=color_max,
        aspect="auto",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("개수", fontsize=11)
    axis.set(
        xticks=np.arange(len(predicted_labels)),
        yticks=np.arange(len(DISPLAY_CLASS_NAMES)),
        xticklabels=predicted_labels,
        yticklabels=DISPLAY_CLASS_NAMES,
        xlabel="예측",
        ylabel="정답",
    )
    axis.set_title(
        "혼동행렬 (Confusion Matrix) — Faster R-CNN 단독, "
        f"외부 테스트셋 {dataset_image_count}장",
        pad=38,
        fontsize=12,
    )
    axis.text(
        0.5,
        1.02,
        (
            f"커버리지 {coverage:.1%}  ·  응답 정확도 {response_accuracy:.1%}  ·  "
            f"표지판 커버리지 {sign_coverage:.1%}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#263746",
    )
    plt.setp(
        axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
    )

    threshold = color_max / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=12,
            )
    axis.tick_params(axis="both", labelsize=10.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_class_metrics_plot(metrics, output_path: Path):
    configure_plot_font()
    rows = []
    for class_name in DISPLAY_CLASS_NAMES:
        values = metrics[class_name]
        rows.append(
            [
                class_name,
                f"{values['precision']:.3f}",
                f"{values['recall']:.3f}",
                f"{values['f1']:.3f}",
            ]
        )

    figure, axis = plt.subplots(figsize=(7.5, 2.8))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["클래스", "precision", "recall", "f1"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)
    for column in range(4):
        table[(0, column)].set_facecolor("#1f2937")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row in range(1, len(rows) + 1):
        background = "#eef4fb" if row % 2 else "#ffffff"
        for column in range(4):
            table[(row, column)].set_facecolor(background)
    axis.set_title("Per-class Precision / Recall / F1", fontsize=14, pad=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def replot_existing_results(results_dir: Path):
    """Convert an already evaluated legacy 5x5 report to the 4x5 PPT layout."""
    report_path = results_dir / "metrics.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Existing metrics file not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    confusion = report.get("confusion_matrix", {})
    old_labels = confusion.get("labels")
    old_values = confusion.get("values")
    if not old_labels or old_values is None or "background" not in old_labels:
        raise ValueError(
            "--replot-existing expects a legacy confusion matrix containing "
            "the label 'background'."
        )

    old_matrix = np.asarray(old_values, dtype=np.int64)
    display_model_names = [model_name for model_name, _ in DISPLAY_CLASS_SPECS]
    class_indices = [old_labels.index(model_name) for model_name in display_model_names]
    background_index = old_labels.index("background")
    matrix = old_matrix[np.ix_(class_indices, [*class_indices, background_index])]
    unmatched_predictions = old_matrix[background_index, class_indices]
    per_class_metrics = metrics_from_confusion_matrix(
        matrix, unmatched_predictions
    )

    save_confusion_matrix_csv(matrix, results_dir / "confusion_matrix.csv")
    save_confusion_matrix_plot(
        matrix,
        results_dir / "confusion_matrix.png",
        int(report["dataset"]["images"]),
    )
    save_class_metrics_csv(per_class_metrics, results_dir / "class_metrics.csv")
    save_class_metrics_plot(per_class_metrics, results_dir / "class_metrics.png")

    old_dataset_objects = report.get("dataset", {}).get("objects", {})
    report["dataset"]["objects"] = {
        display_name: old_dataset_objects.get(model_name, 0)
        for model_name, display_name in DISPLAY_CLASS_SPECS
    }
    old_coco_metrics = report.get("per_class_coco", {})
    report["per_class_coco"] = {
        display_name: old_coco_metrics.get(model_name, {})
        for model_name, display_name in DISPLAY_CLASS_SPECS
    }
    report["per_class_at_confidence"] = per_class_metrics
    report["confusion_matrix"] = {
        "actual_labels": DISPLAY_CLASS_NAMES,
        "predicted_labels": [*DISPLAY_CLASS_NAMES, "(무응답)"],
        "rows_are_actual": True,
        "columns_are_predicted": True,
        "values": matrix.tolist(),
        "unmatched_predictions_not_shown_in_matrix": {
            class_name: int(unmatched_predictions[class_index])
            for class_index, class_name in enumerate(DISPLAY_CLASS_NAMES)
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Reformatted existing evaluation results in {results_dir}")


def per_class_coco_metrics(coco_eval: COCOeval):
    precision = coco_eval.eval["precision"]  # IoU, recall, class, area, maxDet
    recall = coco_eval.eval["recall"]  # IoU, class, area, maxDet
    metrics = {}
    for model_name, display_name in DISPLAY_CLASS_SPECS:
        class_index = CLASS_NAMES.index(model_name)
        ap_values = precision[:, :, class_index, 0, -1]
        ap50_values = precision[0, :, class_index, 0, -1]
        ar_values = recall[:, class_index, 0, -1]
        metrics[display_name] = {
            "AP50_95": float(np.mean(ap_values[ap_values > -1]))
            if np.any(ap_values > -1)
            else 0.0,
            "AP50": float(np.mean(ap50_values[ap50_values > -1]))
            if np.any(ap50_values > -1)
            else 0.0,
            "AR100": float(np.mean(ar_values[ar_values > -1]))
            if np.any(ar_values > -1)
            else 0.0,
        }
    return metrics


def evaluate(args):
    dataset_root = Path(args.dataset_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.replot_existing:
        replot_existing_results(results_dir)
        return
    annotation_path = convert_yolo_to_coco(dataset_root, args.split, results_dir)
    dataset_summary = yolo_test_summary(dataset_root, args.split)
    print(f"Test dataset: {dataset_summary}")
    print(f"COCO annotations: {annotation_path}")
    if args.prepare_only:
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.weights), device)
    dataset = DetectionDataset(dataset_root, annotation_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    coco_predictions = []
    prediction_tensors = []
    target_tensors = []
    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader, 1):
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"].item())
                cpu_output = {key: value.detach().cpu() for key, value in output.items()}
                cpu_target = {key: value.detach().cpu() for key, value in target.items()}
                prediction_tensors.append(cpu_output)
                target_tensors.append(cpu_target)
                for box, score, label in zip(
                    cpu_output["boxes"],
                    cpu_output["scores"],
                    cpu_output["labels"],
                ):
                    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                    coco_predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )
            if index == 1 or index % 10 == 0:
                print(f"Evaluated {index}/{len(loader)} images", flush=True)

    predictions_path = results_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(coco_predictions, indent=2), encoding="utf-8"
    )
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(str(predictions_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = list(range(1, len(CLASS_NAMES) + 1))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    confusion_matrix, unmatched_predictions = confusion_matrix_at_threshold(
        prediction_tensors,
        target_tensors,
        args.confidence,
        args.iou_threshold,
    )
    per_class_metrics = metrics_from_confusion_matrix(
        confusion_matrix, unmatched_predictions
    )
    confusion_csv_path = results_dir / "confusion_matrix.csv"
    confusion_plot_path = results_dir / "confusion_matrix.png"
    class_metrics_csv_path = results_dir / "class_metrics.csv"
    class_metrics_plot_path = results_dir / "class_metrics.png"
    save_confusion_matrix_csv(confusion_matrix, confusion_csv_path)
    save_confusion_matrix_plot(
        confusion_matrix,
        confusion_plot_path,
        dataset_summary["images"],
    )
    save_class_metrics_csv(per_class_metrics, class_metrics_csv_path)
    save_class_metrics_plot(per_class_metrics, class_metrics_plot_path)

    report = {
        "checkpoint": str(Path(args.weights).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": dataset_summary,
        "confidence_for_prf1": args.confidence,
        "iou_threshold_for_prf1": args.iou_threshold,
        "overall": {
            "mAP50_95": float(coco_eval.stats[0]),
            "mAP50": float(coco_eval.stats[1]),
            "mAP75": float(coco_eval.stats[2]),
            "AR100": float(coco_eval.stats[8]),
        },
        "per_class_coco": per_class_coco_metrics(coco_eval),
        "per_class_at_confidence": per_class_metrics,
        "confusion_matrix": {
            "actual_labels": DISPLAY_CLASS_NAMES,
            "predicted_labels": [*DISPLAY_CLASS_NAMES, "(무응답)"],
            "rows_are_actual": True,
            "columns_are_predicted": True,
            "values": confusion_matrix.tolist(),
            "unmatched_predictions_not_shown_in_matrix": {
                class_name: int(unmatched_predictions[class_index])
                for class_index, class_name in enumerate(DISPLAY_CLASS_NAMES)
            },
        },
    }
    report_path = results_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved metrics: {report_path}")
    print(f"Saved class metrics: {class_metrics_csv_path}")
    print(f"Saved class metrics table: {class_metrics_plot_path}")
    print(f"Saved confusion matrix: {confusion_csv_path}")
    print(f"Saved confusion matrix plot: {confusion_plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the corrected four-class Faster R-CNN on the held-out test set."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_TEST_ROOT))
    parser.add_argument("--split", default="train")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--replot-existing",
        action="store_true",
        help="Reformat existing metrics without running model inference again",
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
