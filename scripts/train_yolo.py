#!/usr/bin/env python
"""표지판(sign) 탐지용 YOLO 커스텀 학습 스크립트 (Ultralytics 기준 예시).

라벨링된 데이터셋(data/annotations, YOLO txt 포맷)이 준비되면 사용하세요.
YOLOv5로 학습하는 경우, 각 프레임워크의 공식 학습 커맨드를 참고해 이 스크립트를
그에 맞게 수정하면 됩니다 (구조만 잡아둔 예시 스크립트입니다).

사용 예:
    python scripts/train_yolo.py --data data/dataset.yaml --epochs 100
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 표지판 탐지 모델 학습")
    parser.add_argument("--data", type=str, required=True, help="YOLO 데이터셋 yaml 경로")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="베이스 모델/가중치")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="outputs/results")
    parser.add_argument("--name", type=str, default="sign_detector")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics 패키지가 설치되어 있지 않습니다. "
            "requirements.txt의 ultralytics 주석을 해제하고 설치하세요."
        ) from e

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
