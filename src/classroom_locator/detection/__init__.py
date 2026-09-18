"""탐지기 팩토리.

configs/default.yaml의 detector 설정을 받아 알맞은 구현체를 생성합니다.
새로운 탐지 백엔드를 추가하려면 이 함수에 분기만 추가하면 됩니다.
"""

from __future__ import annotations

from typing import Any

from .base import BaseDetector, Detection
from .yolo_detector import YoloDetector

__all__ = ["BaseDetector", "Detection", "YoloDetector", "get_detector"]


def get_detector(config: dict[str, Any]) -> BaseDetector:
    backend = config.get("backend", "ultralytics")

    if backend in ("ultralytics", "yolov5"):
        return YoloDetector(
            weights=config["weights"],
            logo_weights=config.get("logo_weights"),
            backend=backend,
            conf_threshold=config.get("conf_threshold", 0.5),
            target_classes=config.get("target_classes"),
            class_name_map=config.get("class_name_map"),
        )

    raise ValueError(f"알 수 없는 detector backend: {backend}")
