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


def crop_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 5) -> np.ndarray:
    """bbox 영역을 약간의 여백(margin)을 두고 crop합니다."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return image[y1:y2, x1:x2]


def upscale_for_ocr(image: np.ndarray, min_height: int = 120) -> np.ndarray:
    """crop된 표지판 이미지가 너무 작으면(멀리서 찍혀서 글자가 몇 픽셀 안 됨)
    OCR이 인식을 못 하는 경우가 많아서, 세로 길이가 min_height보다 작으면
    비율을 유지한 채 확대합니다.
    """
    h, w = image.shape[:2]
    if h == 0 or h >= min_height:
        return image
    scale = min_height / h
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


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
