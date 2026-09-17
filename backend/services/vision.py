from __future__ import annotations

import os
from pathlib import Path
import subprocess
from threading import Lock

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from services.ocr import PaddleOCREngine, draw_ocr_results
from services.visual_localization import VisualLocationEstimator, draw_location_overlay
from test_inference import draw_detections, load_model


class VisionEngine:
    """Server-side Faster R-CNN + targeted PaddleOCR inference."""

    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = Path(__file__).resolve().parents[1]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_kind = None
        self.class_names = []
        self.ocr = None
        self.model_path = self._choose_model_path()
        self.location_estimator = None
        self.location_frame_width = None
        self.lock = Lock()
        self.load_error = None
        self._load_models()

    def _choose_model_path(self):
        candidates = (
            self.base_dir / "models" / "final.pt",
            self.base_dir / "models" / "final_best.pt",
            self.base_dir
            / "models"
            / "faster_rcnn_916machine_4class_100epochs_best_epoch16.pt",
            self.base_dir / "models" / "faster_rcnn_916machine_4class_best.pt",
        )
        return next((path for path in candidates if path.exists()), candidates[0])

    def _load_models(self):
        try:
            if self.model_path.name in {"final.pt", "final_best.pt"}:
                yolo_config = Path("C:/yolo_config")
                yolo_config.mkdir(parents=True, exist_ok=True)
                os.environ["YOLO_CONFIG_DIR"] = str(yolo_config)
                from ultralytics import YOLO

                self.model = YOLO(str(self.model_path))
                names = self.model.names
                self.class_names = (
                    [names[index] for index in sorted(names)]
                    if isinstance(names, dict)
                    else list(names)
                )
                self.model_kind = "YOLOv8"
            else:
                self.model, self.class_names = load_model(self.model_path, self.device)
                self.model_kind = "Faster R-CNN"
            # PaddlePaddle on this Windows environment is CPU-only. Faster
            # R-CNN still runs on CUDA when available.
            self.ocr = PaddleOCREngine(device="cpu")
        except Exception as exc:
            self.load_error = str(exc)
            self.model = None
            self.ocr = None

    @property
    def model_ready(self):
        return self.model is not None

    @property
    def yolo_ready(self):
        # Kept for the older health-response field used by the web client.
        return self.model_ready

    @property
    def ocr_ready(self):
        return self.ocr is not None

    @staticmethod
    def _read_phone_image(image_path: Path):
        # Phone cameras frequently store rotation in EXIF instead of rotating
        # pixels. Normalize it before detection and OCR.
        with Image.open(image_path) as pil_image:
            oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
            return cv2.cvtColor(np.asarray(oriented), cv2.COLOR_RGB2BGR)

    def _get_location_estimator(self, frame_width: int):
        if (
            self.location_estimator is None
            or self.location_frame_width != frame_width
        ):
            self.location_estimator = VisualLocationEstimator(
                self.base_dir / "data" / "minimap_coordinates.json",
                frame_width,
            )
            self.location_frame_width = frame_width
        return self.location_estimator

    def _infer_frame(self, clean_image, estimator):
        image = clean_image.copy()
        if self.model_kind == "YOLOv8":
            yolo_result = self.model.predict(
                clean_image,
                imgsz=640,
                conf=0.05,
                device=0 if self.device.type == "cuda" else "cpu",
                verbose=False,
            )[0]
            output = {
                "boxes": yolo_result.boxes.xyxy,
                "scores": yolo_result.boxes.conf,
                # draw_detections uses torchvision's 1-based foreground IDs.
                "labels": yolo_result.boxes.cls.to(torch.int64) + 1,
            }
        else:
            rgb = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(rgb.copy())
                .permute(2, 0, 1)
                .float()
                .div(255)
                .to(self.device)
            )
            with torch.inference_mode():
                output = self.model([tensor])[0]

        stats = draw_detections(
            image,
            clean_image,
            output,
            self.class_names,
            confidence=0.30,
            number_confidence=0.10,
            door_confidence=0.65,
            ocr_engine=self.ocr,
            ocr_confidence=0.25,
        )
        if stats["ocr_events"]:
            image = draw_ocr_results(image, stats["ocr_events"], draw_boxes=False)
        location = estimator.update(clean_image, stats["observations"])
        return draw_location_overlay(image, location), stats, location

    def analyze(self, image_path: Path, image_id: str):
        if not self.model_ready:
            raise RuntimeError(self.load_error or "Faster R-CNN model is not ready")
        if not self.ocr_ready:
            raise RuntimeError(self.load_error or "PaddleOCR is not ready")

        with self.lock:
            clean_image = self._read_phone_image(image_path)
            estimator = self._get_location_estimator(clean_image.shape[1])
            image, stats, location = self._infer_frame(clean_image, estimator)

            result_name = f"{image_id}.jpg"
            result_path = self.result_dir / result_name
            encoded_ok, encoded_image = cv2.imencode(".jpg", image)
            if not encoded_ok:
                raise RuntimeError(f"Could not write result image: {result_path}")
            result_path.write_bytes(encoded_image.tobytes())

            detections = []
            for item in stats["observations"]:
                label = item["display_label"]
                if item.get("localization_only"):
                    label = "강의실 3 (위치 참고)"
                detections.append(
                    {
                        "label": label,
                        "confidence": round(float(item["score"]), 4),
                        "bbox": item["box"],
                        "ocr_verified": bool(item.get("ocr_verified")),
                        "localization_only": bool(item.get("localization_only")),
                    }
                )

            texts = [self._display_reading(item) for item in stats["ocr_readings"]]
            return {
                "detections": detections,
                "texts": texts,
                "location": location,
                "result_name": result_name,
                "model": self.model_kind,
                "weights": self.model_path.name,
                "device": str(self.device),
            }

    @staticmethod
    def _display_detection(item):
        label = item["display_label"]
        if item.get("localization_only"):
            label = "강의실 3 (위치 참고)"
        return {
            "label": label,
            "confidence": round(float(item["score"]), 4),
            "bbox": item["box"],
            "ocr_verified": bool(item.get("ocr_verified")),
            "localization_only": bool(item.get("localization_only")),
        }

    @staticmethod
    def _display_reading(item):
        raw_text = str(item["text"]).strip()
        room_type = item.get("room_type")
        numbers = list(dict.fromkeys(item.get("observed_numbers", [])))
        display_text = raw_text
        if room_type and len(numbers) == 1:
            display_text = f"{numbers[0]}번 {room_type}"
        elif room_type:
            display_text = room_type
        return {
            "text": display_text,
            "raw_text": raw_text,
            "confidence": round(float(item["confidence"]), 4),
            "room_type": room_type,
            "observed_numbers": numbers,
            "accepted": bool(item.get("accepted")),
            "reason": item.get("reason"),
        }

    @staticmethod
    def _landmark_display_name(anchor):
        return {
            "classroom_2": "2번 강의실 표지판",
            "classroom_3": "3번 강의실 표지판",
            "classroom_4": "4번 강의실 표지판",
            "front_door": "앞문",
            "rear_door": "쪽문",
        }.get(anchor, anchor)

    @classmethod
    def _video_timeline_cue(cls, time_seconds, stats, location):
        """Build the text panel shown under the video at this playback time."""
        cue_texts = []
        for reading in stats["ocr_readings"]:
            displayed = cls._display_reading(reading)
            if displayed["text"] and displayed["text"] not in cue_texts:
                cue_texts.append(displayed["text"])

        cue_detections = []
        for observation in stats["observations"]:
            label = cls._display_detection(observation)["label"]
            if label not in cue_detections:
                cue_detections.append(label)

        landmarks = []
        for measurement in location.get("measurements", []):
            bearing = float(measurement.get("bearing_deg", 0.0))
            direction = "왼쪽" if bearing < -8 else "오른쪽" if bearing > 8 else "앞쪽"
            distance_m = float(measurement["distance_mm"]) / 1000.0
            landmarks.append(
                {
                    "name": cls._landmark_display_name(measurement["anchor"]),
                    "distance_m": round(distance_m, 2),
                    "direction": direction,
                    "state": "near" if distance_m <= 2.5 else "far",
                    "state_text": "가까움" if distance_m <= 2.5 else "거리가 있음",
                    "bearing_deg": measurement.get("bearing_deg"),
                }
            )
        landmarks.sort(key=lambda item: item["distance_m"])

        location_summary = {
            "status": location.get("status", "unknown"),
            "label": location.get("label", "위치 불확실"),
            "detail": location.get("detail", ""),
            "motion": location.get("motion", ""),
            "confidence": location.get("confidence", 0.0),
            "candidates": location.get("candidates", []),
        }
        if landmarks:
            nearest = landmarks[0]
            guidance = (
                f"{nearest['direction']} 약 {nearest['distance_m']:.1f}m에 "
                f"{nearest['name']}이(가) 보입니다. "
                f"현재 위치는 {location_summary['label']}로 추정됩니다."
            )
        else:
            guidance = "이 구간에서는 위치를 판단할 기준 객체가 보이지 않습니다."

        return {
            "time_seconds": round(time_seconds, 3),
            "texts": cue_texts,
            "detections": cue_detections,
            "landmarks": landmarks,
            "location": location_summary,
            "guidance": guidance,
        }

    def analyze_video(self, video_path: Path, video_id: str, frame_step: int = 5):
        if not self.model_ready:
            raise RuntimeError(self.load_error or "Faster R-CNN model is not ready")
        if not self.ocr_ready:
            raise RuntimeError(self.load_error or "PaddleOCR is not ready")

        frame_step = max(1, min(int(frame_step), 30))
        raw_path = self.result_dir / f"{video_id}_raw.mp4"
        result_path = self.result_dir / f"{video_id}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("업로드한 영상을 열 수 없습니다.")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if total_frames > 0 else 0.0
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("영상 해상도를 확인할 수 없습니다.")
        if duration_seconds > 300:
            capture.release()
            raise ValueError("현재는 5분 이하 영상만 업로드할 수 있습니다.")

        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("분석 결과 영상을 만들 수 없습니다.")

        estimator = VisualLocationEstimator(
            self.base_dir / "data" / "minimap_coordinates.json", width
        )
        detections_by_label = {}
        texts_by_value = {}
        last_location = {
            "status": "unknown",
            "label": "위치 불확실",
            "detail": "분석 대기",
            "motion": "",
            "candidates": [],
        }
        last_valid_location = None
        timeline = []
        processed_frames = 0
        frame_index = 0

        try:
            with self.lock:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index % frame_step == 0:
                        annotated, stats, last_location = self._infer_frame(frame, estimator)
                        processed_frames += 1
                        if last_location.get("status") != "unknown":
                            last_valid_location = last_location
                        for observation in stats["observations"]:
                            displayed = self._display_detection(observation)
                            key = displayed["label"]
                            previous = detections_by_label.get(key)
                            if previous is None or displayed["confidence"] > previous["confidence"]:
                                displayed["occurrences"] = (previous or {}).get("occurrences", 0) + 1
                                detections_by_label[key] = displayed
                            else:
                                previous["occurrences"] += 1
                        for reading in stats["ocr_readings"]:
                            displayed_reading = self._display_reading(reading)
                            text = displayed_reading["text"]
                            if not text:
                                continue
                            previous = texts_by_value.get(text)
                            current = dict(displayed_reading)
                            current["occurrences"] = (previous or {}).get("occurrences", 0) + 1
                            if previous is None or current["confidence"] >= previous["confidence"]:
                                texts_by_value[text] = current
                            else:
                                previous["occurrences"] += 1
                        timeline.append(
                            self._video_timeline_cue(
                                frame_index / fps,
                                stats,
                                last_location,
                            )
                        )
                    else:
                        annotated = draw_location_overlay(frame, last_location)
                    writer.write(annotated)
                    frame_index += 1
        finally:
            capture.release()
            writer.release()

        if frame_index == 0:
            raw_path.unlink(missing_ok=True)
            raise ValueError("영상에 읽을 수 있는 프레임이 없습니다.")

        try:
            self._transcode_for_mobile(raw_path, video_path, result_path)
        finally:
            raw_path.unlink(missing_ok=True)
        return {
            "detections": sorted(
                detections_by_label.values(),
                key=lambda item: item["confidence"],
                reverse=True,
            ),
            "texts": sorted(
                texts_by_value.values(),
                key=lambda item: (item["occurrences"], item["confidence"]),
                reverse=True,
            ),
            "location": last_valid_location or last_location,
            "result_name": result_path.name,
            "model": self.model_kind,
            "weights": self.model_path.name,
            "device": str(self.device),
            "video": {
                "width": width,
                "height": height,
                "fps": round(fps, 3),
                "frames": frame_index,
                "duration_seconds": round(frame_index / fps, 2),
                "processed_frames": processed_frames,
                "frame_step": frame_step,
            },
            "timeline": timeline,
        }

    @staticmethod
    def _transcode_for_mobile(raw_path: Path, source_path: Path, result_path: Path):
        import imageio_ffmpeg

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(raw_path),
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0 or not result_path.exists():
            raise RuntimeError(
                "모바일용 결과 영상 변환에 실패했습니다: "
                + completed.stderr[-500:]
            )
