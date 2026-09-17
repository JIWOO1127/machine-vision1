"""이미지 처리 공용 유틸."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..detection.base import Detection

_KOREAN_FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")


def put_korean_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_size: int = 24,
    color_bgr: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """cv2.putText는 한글을 지원하지 않아서(깨짐) PIL로 그린 뒤 다시 BGR로 변환합니다.
    origin은 cv2.putText와 다르게 텍스트 박스의 좌상단 기준입니다.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = (
        ImageFont.truetype(str(_KOREAN_FONT_PATH), font_size)
        if _KOREAN_FONT_PATH.exists()
        else ImageFont.load_default()
    )
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(origin, text, font=font, fill=color_rgb, stroke_width=1, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    label_suffix: dict[int, str] | None = None,
) -> np.ndarray:
    """탐지 결과를 이미지 위에 시각화합니다 (디버깅/데모용)."""
    annotated = image.copy()
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        if label_suffix and idx in label_suffix:
            label += f" | {label_suffix[idx]}"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return annotated
