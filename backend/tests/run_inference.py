from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CLASS_NAMES = ["2_class", "4_class", "front_door", "rear_door", "water_dispenser"]


def build_model(num_classes):
	model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, weights_backbone=None)
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	return model


def load_model(weights_path: Path, device: torch.device):
	class_names = CLASS_NAMES
	model = build_model(len(class_names) + 1).to(device)
	checkpoint = torch.load(str(weights_path), map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	return model, class_names


def run_video(model, class_names, video_path: Path, output_path: Path, device, confidence, preview, frame_step, seek_seconds, ocr_step, use_ocr):
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
	output_path.parent.mkdir(parents=True, exist_ok=True)
	writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
	window_name = f"Faster R-CNN - {video_path.name}"
	preview_width = 800
	preview_height = max(1, int(preview_width * height / width))
	paused = False
	seek_frames = max(1, int(fps * seek_seconds))
	total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	left_keys = {ord("a"), 81, 2424832}
	right_keys = {ord("d"), 83, 2555904}
	ocr = None
	if use_ocr:
		try:
			import easyocr
			ocr = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
		except Exception as exc:
			print(f"OCR unavailable: {exc}")
	if preview:
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(window_name, preview_width, preview_height)

	frames_with_detections = 0
	total_detections = 0
	frame_number = 0
	ocr_rooms = []
	while True:
		ok, frame = capture.read()
		if not ok:
			break
		frame_number += 1
		if (frame_number - 1) % max(1, frame_step) != 0:
			writer.write(frame)
			continue
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
		with torch.no_grad():
			result = model([image])[0]
		if ocr is not None and (not ocr_rooms or frame_number % max(1, ocr_step) == 0):
			candidate_scores = [float(score) for score, label in zip(result["scores"], result["labels"])
				if float(score) >= confidence and int(label) in {1, 2}]
			if candidate_scores:
				ocr_rooms = []
				enlarged = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
				for text_bbox, text, text_confidence in ocr.readtext(enlarged, detail=1):
					compact_text = re.sub(r"\s+", "", str(text))
					if "강의실" not in compact_text and "회의실" not in compact_text:
						continue
					room_type = "강의실" if "강의실" in compact_text else "회의실"
					x_values = [point[0] / 2 for point in text_bbox]
					y_values = [point[1] / 2 for point in text_bbox]
					ocr_rooms.append((room_type, (min(x_values), min(y_values), max(x_values), max(y_values))))

		frame_detections = 0
		for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
			score_value = float(score)
			if score_value < confidence:
				continue
			x1, y1, x2, y2 = [int(value) for value in box.tolist()]
			label_id = int(label)
			name = class_names[label_id - 1] if 0 < label_id <= len(class_names) else str(label_id)
			if re.fullmatch(r"\d+_class", name):
				if use_ocr and not ocr_rooms:
					continue
				if use_ocr:
					detection_center = ((x1 + x2) / 2, (y1 + y2) / 2)
					room_type, _ = min(ocr_rooms, key=lambda item: (detection_center[0] - (item[1][0] + item[1][2]) / 2) ** 2 + (detection_center[1] - (item[1][1] + item[1][3]) / 2) ** 2)
					name = f"{room_type}{label_id}"
			frame_detections += 1
			total_detections += 1
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
			cv2.putText(frame, f"{name} {score_value:.0%}", (x1, max(35, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

		if frame_detections:
			frames_with_detections += 1
		cv2.putText(frame, f"detections: {frame_detections}", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
		writer.write(frame)

		if preview:
			display = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
			cv2.imshow(window_name, display)
			key = cv2.waitKeyEx(0 if paused else 1)
			seek_requested = False
			if key == ord(" "):
				paused = not paused
				print("Preview paused." if paused else "Preview resumed.")
			elif key in left_keys or key in right_keys:
				direction = -1 if key in left_keys else 1
				target_frame = max(0, min(total_frames - 1, frame_number + direction * seek_frames))
				capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
				frame_number = target_frame
				seek_requested = True
				print(f"Seeked to {target_frame / fps:.1f}s")
			elif key in {ord("q"), 27}:
				break
			while paused:
				key = cv2.waitKeyEx(50)
				if key == ord(" "):
					paused = False
					print("Preview resumed.")
				elif key in left_keys or key in right_keys:
					direction = -1 if key in left_keys else 1
					target_frame = max(0, min(total_frames - 1, frame_number + direction * seek_frames))
					capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
					frame_number = target_frame
					seek_requested = True
					print(f"Seeked to {target_frame / fps:.1f}s")
					break
				elif key in {ord("q"), 27}:
					paused = False
					break
			if seek_requested:
				continue
			if not paused and key in {ord("q"), 27}:
				break

	capture.release()
	writer.release()
	if preview:
		cv2.destroyAllWindows()
	print(f"Detection summary: {total_detections} boxes across {frames_with_detections}/{frame_number} frames")
	print(f"Saved: {output_path}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--video", required=True)
	parser.add_argument("--weights", default="models/faster_rcnn_916machine_best.pt")
	parser.add_argument("--output", default="tests/output/nayeon_916machine_annotated.mp4")
	parser.add_argument("--confidence", type=float, default=0.3)
	parser.add_argument("--preview", action="store_true")
	parser.add_argument("--frame-step", type=int, default=3)
	parser.add_argument("--seek-seconds", type=float, default=2.0)
	parser.add_argument("--ocr-step", type=int, default=10, help="Run OCR every N frames and reuse the last room result")
	parser.add_argument("--no-ocr", action="store_true", help="Disable OCR and run object detection only")
	args = parser.parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model, class_names = load_model(Path(args.weights), device)
	run_video(model, class_names, Path(args.video), Path(args.output), device, args.confidence, args.preview, args.frame_step, args.seek_seconds, args.ocr_step, not args.no_ocr)


if __name__ == "__main__":
	main()
