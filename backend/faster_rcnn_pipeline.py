from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "machine_vision 2.v1i.yolov8"
MODEL_PATH = BASE_DIR / "models" / "faster_rcnn_best.pt"
CLASS_NAMES = ["2_class", "4_class", "front_door", "rear_door", "water_dispenser"]


def load_yaml_names(dataset_root: Path) -> List[str]:
	yaml_path = dataset_root / "data.yaml"
	if not yaml_path.exists():
		return CLASS_NAMES
	for line in yaml_path.read_text(encoding="utf-8").splitlines():
		if line.strip().startswith("names:"):
			values = line.split(":", 1)[1].strip().strip("[]")
			names = [value.strip().strip("'\"") for value in values.split(",") if value.strip()]
			if names:
				return names
	return CLASS_NAMES


def annotation_to_bbox(values: List[str], width: int, height: int):
	if len(values) == 5:
		_, cx, cy, box_width, box_height = map(float, values)
		x = (cx - box_width / 2) * width
		y = (cy - box_height / 2) * height
		return [max(0.0, x), max(0.0, y), max(1.0, box_width * width), max(1.0, box_height * height)]
	if len(values) >= 7 and (len(values) - 1) % 2 == 0:
		points = list(map(float, values[1:]))
		xs, ys = points[0::2], points[1::2]
		x1, y1 = max(0.0, min(xs) * width), max(0.0, min(ys) * height)
		x2, y2 = min(float(width), max(xs) * width), min(float(height), max(ys) * height)
		return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]
	return None


def convert_yolo_to_coco(dataset_root: Path, split: str, output_dir: Path) -> Path:
	image_dir = dataset_root / split / "images"
	label_dir = dataset_root / split / "labels"
	names = load_yaml_names(dataset_root)
	images, annotations = [], []
	annotation_id = 1
	for image_path in sorted(image_dir.glob("*")):
		if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
			continue
		with Image.open(image_path) as image:
			width, height = image.size
		image_id = len(images) + 1
		images.append({"id": image_id, "file_name": image_path.relative_to(dataset_root).as_posix(), "width": width, "height": height})
		label_path = label_dir / f"{image_path.stem}.txt"
		if not label_path.exists():
			continue
		for line in label_path.read_text(encoding="utf-8").splitlines():
			values = line.split()
			if len(values) < 5:
				continue
			try:
				class_id = int(float(values[0]))
				bbox = annotation_to_bbox(values, width, height)
			except ValueError:
				continue
			if not 0 <= class_id < len(names) or bbox is None:
				continue
			x, y, box_width, box_height = bbox
			annotations.append({"id": annotation_id, "image_id": image_id, "category_id": class_id + 1, "bbox": bbox, "area": box_width * box_height, "iscrowd": 0})
			annotation_id += 1
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{split}_coco.json"
	output_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": [{"id": i + 1, "name": name} for i, name in enumerate(names)]}, indent=2), encoding="utf-8")
	return output_path


class DetectionDataset(Dataset):
	def __init__(self, root: Path, annotation_path: Path):
		self.root = root
		self.coco = COCO(str(annotation_path))
		self.image_ids = sorted(self.coco.getImgIds())

	def __len__(self):
		return len(self.image_ids)

	def __getitem__(self, index):
		image_info = self.coco.loadImgs(self.image_ids[index])[0]
		image = Image.open(self.root / image_info["file_name"]).convert("RGB")
		boxes, labels = [], []
		for annotation in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[self.image_ids[index]])):
			x, y, width, height = annotation["bbox"]
			boxes.append([x, y, x + width, y + height])
			labels.append(annotation["category_id"])
		box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
		label_tensor = torch.tensor(labels, dtype=torch.int64)
		target = {"boxes": box_tensor, "labels": label_tensor, "image_id": torch.tensor([self.image_ids[index]]), "area": (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1]), "iscrowd": torch.zeros(len(labels), dtype=torch.int64)}
		image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).div(255)
		return image_tensor, target


def collate_fn(batch):
	return tuple(zip(*batch))


def build_model(num_classes, pretrained=True, min_size=800, max_size=1333):
	"""Build the detector for both training and checkpoint-only inference."""
	weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
	model = fasterrcnn_resnet50_fpn(
		weights=weights,
		weights_backbone=None,
		min_size=int(min_size),
		max_size=int(max_size),
	)
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	return model


def train(args):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dataset_root = Path(args.dataset_root)
	annotation_dir = Path(args.annotation_dir)
	train_json = convert_yolo_to_coco(dataset_root, "train", annotation_dir)
	valid_json = convert_yolo_to_coco(dataset_root, "valid", annotation_dir)
	train_set = DetectionDataset(dataset_root, train_json)
	valid_set = DetectionDataset(dataset_root, valid_json)
	train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
	valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
	model = build_model(len(load_yaml_names(dataset_root)) + 1).to(device)
	optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
	model_path = Path(args.model_path)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	best_loss = float("inf")
	print(f"Using device: {device}; train={len(train_set)} valid={len(valid_set)}")
	for epoch in range(args.epochs):
		model.train()
		train_loss = 0.0
		for batch_index, (images, targets) in enumerate(train_loader, 1):
			images = [image.to(device) for image in images]
			targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
			losses = model(images, targets)
			total_loss = sum(loss for loss in losses.values())
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
			train_loss += float(total_loss.detach().cpu())
			if batch_index == 1 or batch_index % 10 == 0:
				print(f"Epoch {epoch + 1}/{args.epochs} train batch {batch_index}/{len(train_loader)}", flush=True)
			if args.max_batches and batch_index >= args.max_batches:
				break
		model.train()
		valid_loss = 0.0
		with torch.no_grad():
			for batch_index, (images, targets) in enumerate(valid_loader, 1):
				images = [image.to(device) for image in images]
				targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
				losses = model(images, targets)
				valid_loss += float(sum(losses.values()).detach().cpu())
				if batch_index == 1 or batch_index % 10 == 0:
					print(f"Epoch {epoch + 1}/{args.epochs} valid batch {batch_index}/{len(valid_loader)}", flush=True)
				if args.max_batches and batch_index >= args.max_batches:
					break
		train_loss /= max(1, len(train_loader))
		valid_loss /= max(1, len(valid_loader))
		print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
		if valid_loss < best_loss:
			best_loss = valid_loss
			torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "class_names": load_yaml_names(dataset_root), "dataset_root": str(dataset_root)}, model_path)
			print(f"Saved best model to {model_path}")


def main():
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="command", required=True)
	train_parser = subparsers.add_parser("train")
	train_parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
	train_parser.add_argument("--annotation-dir", default=str(BASE_DIR / "annotations"))
	train_parser.add_argument("--model-path", default=str(MODEL_PATH))
	train_parser.add_argument("--epochs", type=int, default=15)
	train_parser.add_argument("--batch-size", type=int, default=1)
	train_parser.add_argument("--learning-rate", type=float, default=0.005)
	train_parser.add_argument("--max-batches", type=int, default=0)
	args = parser.parse_args()
	if args.command == "train":
		train(args)


if __name__ == "__main__":
	main()
