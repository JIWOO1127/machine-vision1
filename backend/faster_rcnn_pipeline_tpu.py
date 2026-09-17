from __future__ import annotations

"""Google Colab TPU trainer for the four-class Faster R-CNN model.

Keep this file next to ``faster_rcnn_pipeline.py`` in Google Drive.  The base
file owns all dataset conversion, EXIF correction, validation, and model
construction logic; this file only replaces the CUDA/CPU training loop with a
PyTorch/XLA loop.

Colab setup (run once after selecting a TPU runtime):

    !pip install -q pycocotools

PyTorch/XLA is normally preinstalled in a Colab TPU runtime.  If the import
below fails, restart the TPU runtime instead of installing an arbitrary XLA
version: torch, torchvision, and torch_xla must be version-compatible.
"""

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

try:
	import torch_xla
	import torch_xla.core.xla_model as xm
	from torch_xla import runtime as xr
except ImportError as error:
	raise SystemExit(
		"torch_xla is not installed. In Colab choose Runtime > Change runtime type "
		"> TPU, reconnect, and run this file again."
	) from error

import faster_rcnn_pipeline as core


def get_tpu_device() -> torch.device:
	device_kind = xr.device_type()
	if device_kind != "TPU":
		raise RuntimeError(
			f"PyTorch/XLA device is {device_kind!r}, not 'TPU'. "
			"In Colab select Runtime > Change runtime type > TPU v5e-1 and reconnect."
		)
	if hasattr(torch_xla, "device"):
		return torch_xla.device()
	return xm.xla_device()


def xla_sync():
	if hasattr(torch_xla, "sync"):
		torch_xla.sync()
	else:
		xm.mark_step()


def explain_detection_op_error(error: RuntimeError):
	message = str(error)
	problem_ops = ("torchvision::nms", "torchvision::roi_align", "aten::nonzero")
	if any(name in message for name in problem_ops):
		raise RuntimeError(
			"This Colab torch/torchvision/torch_xla combination cannot execute a "
			"torchvision Faster R-CNN detection operator on TPU. This is an XLA "
			"operator-compatibility problem, not a dataset problem. Switch the Colab "
			"runtime to an NVIDIA GPU and run faster_rcnn_pipeline.py. Original error:\n"
			f"{message}"
		) from error
	raise error


def train(args):
	device = get_tpu_device()
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	dataset_root = Path(args.dataset_root)
	annotation_dir = Path(args.annotation_dir)
	train_json = core.convert_yolo_to_coco(dataset_root, "train", annotation_dir)
	valid_json = core.convert_yolo_to_coco(dataset_root, "valid", annotation_dir)
	core.validate_coco_annotations(dataset_root, train_json)
	core.validate_coco_annotations(dataset_root, valid_json)

	class_names = core.load_yaml_names(dataset_root)
	if class_names != core.CLASS_NAMES:
		raise ValueError(f"Expected classes {core.CLASS_NAMES}, got {class_names}")

	train_set = core.DetectionDataset(dataset_root, train_json)
	valid_set = core.DetectionDataset(dataset_root, valid_json)
	loader_options = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"pin_memory": False,
		"collate_fn": core.collate_fn,
	}
	# A fixed batch shape avoids an extra XLA compilation for the final odd batch.
	train_loader = torch.utils.data.DataLoader(
		train_set,
		shuffle=True,
		drop_last=args.drop_last,
		**loader_options,
	)
	valid_loader = torch.utils.data.DataLoader(
		valid_set,
		shuffle=False,
		**loader_options,
	)

	model = core.build_model(
		len(class_names) + 1,
		pretrained=True,
		min_size=args.min_size,
		max_size=args.max_size,
	).to(device)
	optimizer = torch.optim.SGD(
		[p for p in model.parameters() if p.requires_grad],
		lr=args.learning_rate,
		momentum=0.9,
		weight_decay=0.0005,
	)
	scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer,
		step_size=args.lr_step_size,
		gamma=args.lr_gamma,
	)

	model_path = Path(args.model_path)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	history_path = Path(args.history_path)
	history_path.parent.mkdir(parents=True, exist_ok=True)
	best_map = -1.0
	history = []

	print(
		f"Using device: {device}; XLA device type={xr.device_type()}; "
		f"BF16={args.bf16}; train={len(train_set)} valid={len(valid_set)}",
		flush=True,
	)
	print(
		"The first batch is slow because XLA compiles the graph. "
		"Later batches should be faster.",
		flush=True,
	)

	for epoch in range(args.epochs):
		model.train()
		train_loss = 0.0
		train_batches = 0
		for batch_index, (images, targets) in enumerate(train_loader, 1):
			images = [image.to(device) for image in images]
			targets = [
				{key: value.to(device) for key, value in target.items()}
				for target in targets
			]
			optimizer.zero_grad(set_to_none=True)
			amp_context = (
				torch.autocast("xla", dtype=torch.bfloat16)
				if args.bf16
				else nullcontext()
			)
			try:
				with amp_context:
					losses = model(images, targets)
					total_loss = sum(loss for loss in losses.values())
				total_loss.backward()
				xm.optimizer_step(optimizer, barrier=True)
			except RuntimeError as error:
				explain_detection_op_error(error)

			# Reading one scalar also materializes the lazy XLA graph.
			train_loss += float(total_loss.detach().cpu())
			train_batches += 1
			if batch_index == 1 or batch_index % 10 == 0:
				print(
					f"Epoch {epoch + 1}/{args.epochs} "
					f"train batch {batch_index}/{len(train_loader)}",
					flush=True,
				)
			if args.max_batches and batch_index >= args.max_batches:
				break

		train_loss /= max(1, train_batches)
		try:
			map50_95, map50 = core.evaluate_map(
				model,
				valid_loader,
				valid_set.coco,
				device,
				list(range(1, len(class_names) + 1)),
			)
		except RuntimeError as error:
			explain_detection_op_error(error)
		xla_sync()

		learning_rate = optimizer.param_groups[0]["lr"]
		record = {
			"epoch": epoch + 1,
			"train_loss": train_loss,
			"mAP50_95": map50_95,
			"mAP50": map50,
			"learning_rate": learning_rate,
		}
		history.append(record)
		history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
		print(
			f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} "
			f"mAP50-95={map50_95:.4f} mAP50={map50:.4f} "
			f"lr={learning_rate:.6f}",
			flush=True,
		)

		if map50_95 > best_map:
			best_map = map50_95
			checkpoint = {
				"epoch": epoch + 1,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"class_names": class_names,
				"dataset_root": str(dataset_root),
				"min_size": args.min_size,
				"max_size": args.max_size,
				"exif_corrected": True,
				"best_mAP50_95": map50_95,
				"best_mAP50": map50,
				"training_device": "TPU",
			}
			xm.save(checkpoint, str(model_path), master_only=True)
			print(f"Saved best model to {model_path}", flush=True)
		scheduler.step()


def main():
	parser = argparse.ArgumentParser(
		description="Train the four-class Faster R-CNN model on a Colab TPU."
	)
	parser.add_argument("--dataset-root", default=str(core.DATASET_ROOT))
	parser.add_argument(
		"--annotation-dir",
		default=str(core.BASE_DIR / "annotations" / "916machine_4class"),
	)
	parser.add_argument("--model-path", default=str(core.MODEL_PATH))
	parser.add_argument(
		"--history-path",
		default=str(core.BASE_DIR / "results" / "faster_rcnn_4class_tpu_history.json"),
	)
	parser.add_argument("--epochs", type=int, default=25)
	parser.add_argument("--batch-size", type=int, default=2)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--learning-rate", type=float, default=0.005)
	parser.add_argument("--lr-step-size", type=int, default=8)
	parser.add_argument("--lr-gamma", type=float, default=0.1)
	parser.add_argument("--min-size", type=int, default=1200)
	parser.add_argument("--max-size", type=int, default=2000)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--max-batches", type=int, default=0)
	parser.add_argument(
		"--bf16",
		action="store_true",
		help="Enable TPU bfloat16 autocast (experimental for torchvision detection ops).",
	)
	parser.add_argument(
		"--drop-last",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Drop the last incomplete training batch to reduce XLA recompilation.",
	)
	train(parser.parse_args())


if __name__ == "__main__":
	main()
