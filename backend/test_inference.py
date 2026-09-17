from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from faster_rcnn_pipeline import BASE_DIR, CLASS_NAMES, build_model
from services.ocr import EasyOCREngine, PaddleOCREngine, draw_ocr_results
from services.visual_localization import VisualLocationEstimator, draw_location_overlay

NEW_MODEL_PATH = BASE_DIR / "models" / "faster_rcnn_916machine_4class_best.pt"
LEGACY_MODEL_PATH = BASE_DIR / "models" / "faster_rcnn_916machine_best.pt"
NUMBER_CLASSES = {"2_class", "4_class"}
DOOR_CLASSES = {"front_door", "rear_door"}
# final_best.pt에는 logo도 포함되어 있다. 위치/길안내의 시작 기준점이므로
# 다른 탐지 클래스와 마찬가지로 웹 API까지 전달해야 한다.
LANDMARK_CLASSES = {"logo"}
ALLOWED_CLASSES = NUMBER_CLASSES | DOOR_CLASSES | LANDMARK_CLASSES


def load_model(weights_path: Path, device: torch.device):
    checkpoint = torch.load(str(weights_path), map_location=device)
    if weights_path.resolve() == NEW_MODEL_PATH.resolve() and not checkpoint.get(
        "exif_corrected", False
    ):
        raise RuntimeError(
            "This four-class checkpoint was trained before the EXIF orientation "
            "fix. Stop training, restart it with the corrected pipeline, and use "
            "the newly saved checkpoint."
        )
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    min_size = int(checkpoint.get("min_size", 800))
    max_size = int(checkpoint.get("max_size", 1333))
    model = build_model(
        len(class_names) + 1,
        pretrained=False,
        min_size=min_size,
        max_size=max_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def detection_threshold(
    class_name: str,
    confidence: float,
    number_confidence: float,
    door_confidence: float,
):
    if class_name in NUMBER_CLASSES:
        return number_confidence
    if class_name in DOOR_CLASSES:
        return door_confidence
    return confidence


def draw_readable_label(image, text: str, x: int, y: int):
    """Draw a high-contrast label that stays readable on bright video frames."""
    image_height, image_width = image.shape[:2]
    font_scale = max(0.9, min(1.4, image_width / 1600.0))
    thickness = max(2, round(image_width / 700))
    padding = max(5, round(image_width / 350))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
    )

    label_width = text_width + padding * 2
    label_height = text_height + baseline + padding * 2
    left = max(0, min(x, image_width - label_width))
    if y - label_height >= 0:
        top = y - label_height
    else:
        top = min(max(0, y), max(0, image_height - label_height))
    right = min(image_width - 1, left + label_width)
    bottom = min(image_height - 1, top + label_height)

    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), -1)
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 2)
    cv2.putText(
        image,
        text,
        (left + padding, top + padding + text_height),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def box_iou(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def draw_detections(
    image,
    source_image,
    outputs,
    class_names,
    confidence: float,
    number_confidence: float,
    door_confidence: float,
    ocr_engine=None,
    ocr_confidence: float = 0.3,
):
    detection_count = 0
    ocr_checks = 0
    rejected_count = 0
    ocr_events = []
    ocr_readings = []
    observations = []
    candidates = []
    for box, score, label in zip(
        outputs["boxes"], outputs["scores"], outputs["labels"]
    ):
        label_id = int(label)
        class_name = (
            class_names[label_id - 1]
            if 0 < label_id <= len(class_names)
            else str(label_id)
        )
        if class_name not in ALLOWED_CLASSES:
            continue

        score_value = float(score)
        if score_value < detection_threshold(
            class_name, confidence, number_confidence, door_confidence
        ):
            continue

        candidates.append(
            {
                "box": [int(value) for value in box.tolist()],
                "score": score_value,
                "class_name": class_name,
                "source_label": class_name,
            }
        )

    # torchvision NMS is class-aware, so the same sign can survive once as
    # 2_class and once as 4_class. Suppress those cross-class duplicates.
    selected = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if candidate["class_name"] in NUMBER_CLASSES and any(
            kept["class_name"] in NUMBER_CLASSES
            and box_iou(candidate["box"], kept["box"]) >= 0.45
            for kept in selected
        ):
            continue
        selected.append(candidate)

    for candidate in selected:
        x1, y1, x2, y2 = candidate["box"]
        score_value = candidate["score"]
        class_name = candidate["class_name"]
        localization_label = class_name
        localization_only = False
        read_text = ""
        if class_name in NUMBER_CLASSES and ocr_engine is not None:
            image_height, image_width = source_image.shape[:2]
            pad_x = max(8, round((x2 - x1) * 0.20))
            pad_y = max(10, round((y2 - y1) * 0.22))
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(image_width, x2 + pad_x)
            crop_y2 = min(image_height, y2 + pad_y)
            crop = source_image[crop_y1:crop_y2, crop_x1:crop_x2]
            decision = ocr_engine.read_room_sign(
                crop,
                expected_number=int(class_name.split("_", 1)[0]),
                confidence=ocr_confidence,
            )
            ocr_checks += 1
            read_text = " + ".join(decision["texts"])
            if decision["digit_texts"]:
                read_text = " + ".join(
                    part for part in (read_text, *decision["digit_texts"]) if part
                )
            read_text = read_text or "읽기 실패"
            ocr_readings.append(
                {
                    "text": read_text,
                    "confidence": float(decision["confidence"]),
                    "room_type": decision.get("room_type"),
                    "observed_numbers": decision.get("observed_numbers", []),
                    "accepted": bool(decision["accepted"]),
                    "reason": decision.get("reason"),
                    "bbox": candidate["box"],
                }
            )

            # OCR may reveal a room-3 sign inside a detector box classified as
            # room 2/4. Do not draw it as a model detection, but preserve it as
            # a map landmark for localization.
            room_numbers = decision.get("observed_numbers", [])
            if decision.get("room_type") == "강의실" and len(room_numbers) == 1:
                observed_room = int(room_numbers[0])
                if observed_room in {2, 3, 4}:
                    localization_label = (
                        f"{observed_room}_class"
                        if observed_room in {2, 4}
                        else "3_ocr"
                    )
                    localization_only = observed_room == 3

            if not decision["accepted"]:
                rejected_count += 1
                reason_text = decision["room_type"] or "강의실 확인 안 됨"
                ocr_events.append(
                    {
                        "text": f"제외 {class_name}: {reason_text} / {read_text}",
                        "confidence": decision["confidence"],
                    }
                )
                if localization_only:
                    observations.append(
                        {
                            "label": localization_label,
                            "display_label": "3_class (OCR)",
                            "score": max(score_value, float(decision["confidence"])),
                            "box": candidate["box"],
                            "area": max(1, x2 - x1) * max(1, y2 - y1),
                            "ocr_verified": True,
                            "localization_only": True,
                            "ocr_text": read_text,
                        }
                    )
                continue

            class_name = decision["resolved_label"]
            localization_label = class_name
            ocr_events.append(
                {
                    "text": f"확정 {class_name}:  {read_text}",
                    "confidence": decision["confidence"],
                }
            )

        observations.append(
            {
                "label": localization_label,
                "display_label": class_name,
                "score": score_value,
                "box": candidate["box"],
                "area": max(1, x2 - x1) * max(1, y2 - y1),
                "ocr_verified": bool(
                    ocr_engine is not None and class_name in NUMBER_CLASSES
                ),
                "localization_only": False,
                "ocr_text": read_text if class_name in NUMBER_CLASSES else "",
            }
        )

        box_thickness = max(3, round(image.shape[1] / 500))
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            box_thickness,
        )
        verified = " OCR" if ocr_engine is not None and class_name in NUMBER_CLASSES else ""
        draw_readable_label(image, f"{class_name}{verified} {score_value:.0%}", x1, y1)
        detection_count += 1
    return {
        "kept": detection_count,
        "ocr_checks": ocr_checks,
        "rejected": rejected_count,
        "ocr_events": ocr_events,
        "ocr_readings": ocr_readings,
        "observations": observations,
    }


def run_image_inference(
    model,
    class_names,
    image_path: Path,
    output_path: Path,
    confidence: float,
    number_confidence: float,
    door_confidence: float,
    device: torch.device,
    ocr_engine=None,
    ocr_confidence: float = 0.3,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    clean_image = image.copy()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
    with torch.inference_mode():
        outputs = model([tensor])[0]

    stats = draw_detections(
        image,
        clean_image,
        outputs,
        class_names,
        confidence,
        number_confidence,
        door_confidence,
        ocr_engine,
        ocr_confidence,
    )
    if stats["ocr_events"]:
        image = draw_ocr_results(image, stats["ocr_events"], draw_boxes=False)
    location_estimator = VisualLocationEstimator(
        BASE_DIR / "data" / "minimap_coordinates.json", image.shape[1]
    )
    location = location_estimator.update(clean_image, stats["observations"])
    image = draw_location_overlay(image, location)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(
        f"Saved {stats['kept']} detections, {stats['ocr_checks']} OCR checks, "
        f"and {stats['rejected']} rejected candidates "
        f"to {output_path}"
    )
    if stats["ocr_events"]:
        print("OCR:", " | ".join(item["text"] for item in stats["ocr_events"]))
    print(
        f"Location: {location['label']} | {location.get('detail', '')} | "
        f"candidates={location.get('candidates', [])}"
    )


def run_video_inference(
    model,
    class_names,
    video_path: Path,
    output_path: Path,
    confidence: float,
    number_confidence: float,
    door_confidence: float,
    frame_step: int,
    device: torch.device,
    ocr_engine=None,
    ocr_confidence: float = 0.3,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_index = 0
    total_detections = 0
    ocr_updates = 0
    rejected_detections = 0
    last_ocr_results = []
    location_estimator = VisualLocationEstimator(
        BASE_DIR / "data" / "minimap_coordinates.json", width
    )
    last_location = {
        "label": "위치 불확실",
        "detail": "분석 대기",
        "motion": "",
    }
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1

        if (frame_index - 1) % max(1, frame_step) == 0:
            clean_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
            with torch.inference_mode():
                outputs = model([tensor])[0]
            stats = draw_detections(
                frame,
                clean_frame,
                outputs,
                class_names,
                confidence,
                number_confidence,
                door_confidence,
                ocr_engine,
                ocr_confidence,
            )
            total_detections += stats["kept"]
            ocr_updates += stats["ocr_checks"]
            rejected_detections += stats["rejected"]
            last_ocr_results = stats["ocr_events"]
            last_location = location_estimator.update(clean_frame, stats["observations"])
            if last_ocr_results:
                texts = " | ".join(item["text"] for item in last_ocr_results)
                print(f"OCR frame {frame_index}: {texts}")

        if last_ocr_results:
            frame = draw_ocr_results(
                frame, last_ocr_results, draw_boxes=False
            )
        frame = draw_location_overlay(frame, last_location)
        writer.write(frame)

        if frame_index % 30 == 1:
            print(f"Processed frame {frame_index} from {video_path.name}")

    capture.release()
    writer.release()
    print(
        f"Saved {total_detections} detections, {ocr_updates} targeted OCR checks, "
        f"and {rejected_detections} rejected candidates "
        f"to {output_path}"
    )


def choose_model_path(requested_path: str) -> Path:
    if requested_path:
        return Path(requested_path)
    if NEW_MODEL_PATH.exists():
        return NEW_MODEL_PATH
    return LEGACY_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Run four-class Faster R-CNN inference with optional OCR."
    )
    parser.add_argument("--image", default="", help="Single image path to test")
    parser.add_argument("--video", default="", help="Single video path to test")
    parser.add_argument("--weights", default="", help="Checkpoint path")
    parser.add_argument(
        "--input-dir",
        default=str(BASE_DIR / "tests" / "input"),
        help="Directory containing input images and videos",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "tests" / "output"),
        help="Directory for annotated output files",
    )
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--number-confidence", type=float, default=0.1)
    parser.add_argument(
        "--door-confidence",
        type=float,
        default=0.65,
        help="Confidence threshold used only for front_door and rear_door",
    )
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Validate only 2_class/4_class candidates with cropped sign OCR",
    )
    parser.add_argument(
        "--ocr-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="OCR device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=("paddle", "easyocr"),
        default="paddle",
        help="OCR backend used for 2_class/4_class verification",
    )
    parser.add_argument("--ocr-confidence", type=float, default=0.3)
    parser.add_argument(
        "--ocr-frame-step",
        type=int,
        default=15,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = choose_model_path(args.weights)
    model, class_names = load_model(weights_path, device)
    print(f"Using {weights_path}")
    print(f"Checkpoint classes: {class_names}; displaying: {CLASS_NAMES}")
    ocr_engine = None
    if args.ocr:
        print(f"Loading {args.ocr_engine} OCR...")
        engine_class = (
            PaddleOCREngine if args.ocr_engine == "paddle" else EasyOCREngine
        )
        ocr_engine = engine_class(device=args.ocr_device)
        print(
            f"Targeted room-sign OCR enabled: engine={args.ocr_engine}, "
            f"device={ocr_engine.device}. "
            "Every 2_class/4_class candidate must be verified."
        )

    output_dir = Path(args.output_dir)
    if args.image:
        image_path = Path(args.image)
        run_image_inference(
            model,
            class_names,
            image_path,
            output_dir / f"{image_path.stem}_annotated.jpg",
            args.confidence,
            args.number_confidence,
            args.door_confidence,
            device,
            ocr_engine,
            args.ocr_confidence,
        )
    elif args.video:
        video_path = Path(args.video)
        run_video_inference(
            model,
            class_names,
            video_path,
            output_dir / f"{video_path.stem}_annotated.mp4",
            args.confidence,
            args.number_confidence,
            args.door_confidence,
            args.frame_step,
            device,
            ocr_engine,
            args.ocr_confidence,
        )
    else:
        for input_path in sorted(Path(args.input_dir).glob("**/*")):
            if not input_path.is_file():
                continue
            if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                run_image_inference(
                    model,
                    class_names,
                    input_path,
                    output_dir / "images" / f"{input_path.stem}_annotated.jpg",
                    args.confidence,
                    args.number_confidence,
                    args.door_confidence,
                    device,
                    ocr_engine,
                    args.ocr_confidence,
                )
            elif input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                run_video_inference(
                    model,
                    class_names,
                    input_path,
                    output_dir / "videos" / f"{input_path.stem}_annotated.mp4",
                    args.confidence,
                    args.number_confidence,
                    args.door_confidence,
                    args.frame_step,
                    device,
                    ocr_engine,
                    args.ocr_confidence,
                )


if __name__ == "__main__":
    main()
