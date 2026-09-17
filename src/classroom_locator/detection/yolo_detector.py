"""YOLO 기반 탐지기 구현체.

`backend` 설정값에 따라 Ultralytics(YOLOv8/v11) 또는 YOLOv5(torch.hub) 중
하나를 사용합니다. 필요한 라이브러리는 requirements.txt에서 주석을 해제한 뒤
설치하세요. 두 라이브러리 모두 사용 시점(lazy import)에만 불러오므로,
아직 아무것도 설치하지 않은 상태에서도 프로젝트 구조 탐색에는 문제가 없습니다.
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class YoloDetector(BaseDetector):
    def __init__(
        self,
        weights: str,
        backend: str = "ultralytics",
        conf_threshold: float = 0.5,
        target_classes: list[str] | None = None,
        class_name_map: dict[str, str] | None = None,
    ) -> None:
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.target_classes = set(target_classes) if target_classes else None
        # 모델이 뱉는 원래 클래스 이름을 locations.yaml 기준 이름으로 바꿔치기.
        # (예: 2_class -> room2) 학습 데이터셋마다 클래스명이 달라도 config만
        # 바꾸면 코드 수정 없이 붙일 수 있게 하기 위함.
        self.class_name_map = class_name_map or {}
        self._model = self._load_model(weights)

    def _load_model(self, weights: str):
        if self.backend == "ultralytics":
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError(
                    "ultralytics 패키지가 설치되어 있지 않습니다. "
                    "requirements.txt의 ultralytics 주석을 해제하고 설치하세요."
                ) from e
            return YOLO(weights)

        if self.backend == "yolov5":
            try:
                import torch
            except ImportError as e:
                raise ImportError(
                    "torch 패키지가 설치되어 있지 않습니다. "
                    "requirements.txt의 torch/torchvision/yolov5 주석을 해제하고 설치하세요."
                ) from e
            return torch.hub.load("ultralytics/yolov5", "custom", path=weights)

        raise ValueError(f"알 수 없는 detector backend: {self.backend}")

    def detect(self, image: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []

        if self.backend == "ultralytics":
            results = self._model.predict(image, conf=self.conf_threshold, verbose=False)
            result = results[0]
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_name_map.get(names[cls_id], names[cls_id])
                confidence = float(box.conf[0])
                if self.target_classes and class_name not in self.target_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection((x1, y1, x2, y2), class_name, confidence))

        elif self.backend == "yolov5":
            results = self._model(image)
            for *xyxy, conf, cls_id in results.xyxy[0].tolist():
                raw_name = self._model.names[int(cls_id)]
                class_name = self.class_name_map.get(raw_name, raw_name)
                if conf < self.conf_threshold:
                    continue
                if self.target_classes and class_name not in self.target_classes:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append(Detection((x1, y1, x2, y2), class_name, float(conf)))

        return detections
