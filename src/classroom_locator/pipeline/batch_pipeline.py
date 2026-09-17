"""미리 촬영해둔 사진 폴더를 순차적으로 처리하는 모드."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from ..utils.image_utils import draw_detections
from ..utils.logger import setup_logger
from .core import LocatorPipeline


def run_batch(config: dict[str, Any], input_dir: str) -> list[dict[str, Any]]:
    log_config = config.get("logging", {})
    logger = setup_logger(log_dir=log_config.get("log_dir"), level=log_config.get("level", "INFO"))
    pipeline = LocatorPipeline(config)

    pipeline_config = config.get("pipeline", {})
    extensions = set(pipeline_config.get("image_extensions", [".jpg", ".jpeg", ".png"]))
    output_dir = Path(pipeline_config.get("output_dir", "outputs/results"))
    save_annotated = pipeline_config.get("save_annotated_image", True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in Path(input_dir).rglob("*") if p.suffix.lower() in extensions
    )
    logger.info(f"{len(image_paths)}장의 이미지를 처리합니다: {input_dir}")

    all_results: list[dict[str, Any]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"이미지를 읽을 수 없습니다: {image_path}")
            continue

        result = pipeline.process_image(image)

        record: dict[str, Any] = {
            "image": str(image_path),
            "num_detections": len(result.detections),
            "matched_location": result.match.location.name if result.match else None,
            "match_score": result.match.score if result.match else None,
        }
        all_results.append(record)

        if result.match:
            logger.info(f"{image_path.name} -> {result.match.location.name} ({result.match.score:.0f})")
        else:
            logger.info(f"{image_path.name} -> 위치를 특정하지 못함")

        if save_annotated:
            annotated = draw_detections(image, result.detections)
            out_path = output_dir / f"annotated_{image_path.name}"
            cv2.imwrite(str(out_path), annotated)

    results_json_path = output_dir / "batch_results.json"
    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"결과 저장 완료: {results_json_path}")

    return all_results
