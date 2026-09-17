from __future__ import annotations

import os
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/NanumGothic.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
)


class EasyOCREngine:
    """Lazy EasyOCR wrapper shared by video tests and localization."""

    def __init__(self, device: str = "auto", languages=None):
        import easyocr

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--ocr-device cuda was requested but CUDA is unavailable")
        use_gpu = torch.cuda.is_available() if device == "auto" else device == "cuda"
        self.device = "cuda" if use_gpu else "cpu"
        self.reader = easyocr.Reader(
            languages or ["ko", "en"],
            gpu=use_gpu,
            verbose=False,
        )

    def read(self, image, confidence: float = 0.3, allowlist=None):
        results = []
        for bbox, text, score in self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
        ):
            score_value = float(score)
            normalized_text = str(text).strip()
            if not normalized_text or score_value < confidence:
                continue
            points = [[float(x), float(y)] for x, y in bbox]
            results.append(
                {
                    "text": normalized_text,
                    "confidence": score_value,
                    "bbox": points,
                }
            )
        return results

    def read_room_sign(self, image, expected_number: int, confidence: float = 0.3):
        """Read a cropped room sign and decide whether it is classroom 2 or 4."""
        if image is None or image.size == 0:
            return self._room_decision(expected_number, [], [], confidence)

        height, width = image.shape[:2]
        scale = min(4.0, max(2.0, 280.0 / max(1, min(height, width))))
        enlarged = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        enhanced = cv2.detailEnhance(enlarged, sigma_s=10, sigma_r=0.15)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        contrast = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        general = self.read(enhanced, confidence=max(0.18, confidence * 0.7))
        alternate = self.read(enlarged, confidence=max(0.18, confidence * 0.7))
        seen_texts = {item["text"] for item in general}
        general.extend(item for item in alternate if item["text"] not in seen_texts)
        digits = self.read(
            contrast,
            confidence=max(0.12, confidence * 0.45),
            allowlist="0123456789",
        )
        return self._room_decision(expected_number, general, digits, confidence)

    @staticmethod
    def _room_decision(expected_number, general, digit_results, confidence):
        def reading_order(item):
            points = item.get("bbox") or [[0, 0]]
            return (
                min(float(point[1]) for point in points),
                min(float(point[0]) for point in points),
            )

        general = sorted(general, key=reading_order)
        general_texts = [str(item["text"]).strip() for item in general]
        digit_texts = [str(item["text"]).strip() for item in digit_results]
        combined = " ".join(general_texts)
        compact = re.sub(r"[^0-9A-Za-z가-힣]", "", combined).lower()

        # EasyOCR frequently confuses the final Korean consonant at a distance.
        meeting_room = (
            "meetingroom" in compact
            or re.search(r"[회희최][의이](?:[실심신살])?", compact) is not None
        )
        classroom = (
            "classroom" in compact
            or re.search(r"[강경][의외](?:[실심신설스근])?", compact) is not None
        )

        # Prefer digits from the normal Korean/English OCR. The digit-only pass
        # is a fallback and can otherwise hallucinate values such as "490".
        standalone_numbers = [
            int(text)
            for text in general_texts
            if re.fullmatch(r"[234]", re.sub(r"\s+", "", text))
        ]
        general_numbers = [
            int(value) for value in re.findall(r"[234]", "".join(general_texts))
        ]
        digit_numbers = [
            int(value) for value in re.findall(r"[234]", "".join(digit_texts))
        ]
        # The large number printed at the top is normally returned as its own
        # OCR token. Prefer its first reading over digits hallucinated inside
        # English text such as "C4" or "ClassRoom".
        observed_numbers = (
            standalone_numbers[:1] or general_numbers or digit_numbers
        )
        observed_number = None
        if expected_number in observed_numbers:
            observed_number = expected_number
        else:
            supported = [value for value in observed_numbers if value in {2, 4}]
            if len(set(supported)) == 1:
                observed_number = supported[0]

        scores = [float(item["confidence"]) for item in general + digit_results]
        decision_confidence = max(scores, default=0.0)
        result = {
            "expected_number": int(expected_number),
            "observed_number": observed_number,
            # Keep every plausible room number for localization. In
            # particular, room 3 is not a detector class, but seeing its sign
            # is still a useful map landmark.
            "observed_numbers": list(dict.fromkeys(observed_numbers)),
            "texts": general_texts,
            "digit_texts": digit_texts,
            "combined_text": combined,
            "confidence": decision_confidence,
            "room_type": None,
            "accepted": False,
            "resolved_label": None,
            "reason": "ocr_unconfirmed",
        }

        if meeting_room:
            result.update(room_type="회의실", reason="meeting_room")
            return result
        if not classroom:
            return result

        result["room_type"] = "강의실"
        if 3 in observed_numbers and not any(
            value in observed_numbers for value in {2, 4}
        ):
            result["reason"] = "unsupported_room_number"
            return result
        if observed_number in {2, 4}:
            result.update(
                accepted=True,
                resolved_label=f"{observed_number}_class",
                reason="classroom_number_confirmed",
            )
        return result


class PaddleOCREngine:
    """PaddleOCR 3.x wrapper with the same room-sign interface as EasyOCR."""

    def __init__(self, device: str = "auto", languages=None):
        # Paddle's Windows C++ inference loader cannot reliably open model
        # files under a path containing Korean characters. Keep its cache in
        # an ASCII-only path before importing PaddleX/PaddleOCR.
        paddle_cache = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", "C:/paddle_cache"))
        paddle_cache.mkdir(parents=True, exist_ok=True)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_cache)
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        import paddle
        from paddleocr import PaddleOCR

        if device == "cuda" and not paddle.is_compiled_with_cuda():
            raise RuntimeError(
                "--ocr-device cuda was requested, but this PaddlePaddle build is CPU-only"
            )
        use_gpu = device == "cuda" or (
            device == "auto" and paddle.is_compiled_with_cuda()
        )
        self.device = "gpu:0" if use_gpu else "cpu"
        self.reader = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            device=self.device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def read(self, image, confidence: float = 0.3, allowlist=None):
        del allowlist  # PaddleOCR does not expose EasyOCR-style allowlists.
        pages = self.reader.predict(
            image,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=confidence,
        )
        results = []
        for page in pages or []:
            data = getattr(page, "json", page)
            if callable(data):
                data = data()
            if isinstance(data, str):
                import json

                data = json.loads(data)
            if not isinstance(data, dict):
                continue
            data = data.get("res", data)
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            boxes = data.get("rec_polys") or data.get("dt_polys") or []
            for bbox, text, score in zip(boxes, texts, scores):
                score_value = float(score)
                normalized_text = str(text).strip()
                if not normalized_text or score_value < confidence:
                    continue
                points = [[float(x), float(y)] for x, y in bbox]
                results.append(
                    {
                        "text": normalized_text,
                        "confidence": score_value,
                        "bbox": points,
                    }
                )
        return results

    def read_room_sign(self, image, expected_number: int, confidence: float = 0.3):
        if image is None or image.size == 0:
            return EasyOCREngine._room_decision(expected_number, [], [], confidence)

        height, width = image.shape[:2]
        scale = min(4.0, max(2.0, 280.0 / max(1, min(height, width))))
        enlarged = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        enhanced = cv2.detailEnhance(enlarged, sigma_s=10, sigma_r=0.15)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        contrast = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        general = self.read(enhanced, confidence=max(0.18, confidence * 0.7))
        first_decision = EasyOCREngine._room_decision(
            expected_number, general, [], confidence
        )
        if first_decision["accepted"] or first_decision["reason"] in {
            "meeting_room",
            "unsupported_room_number",
        }:
            return first_decision

        alternate = self.read(enlarged, confidence=max(0.18, confidence * 0.7))
        seen_texts = {item["text"] for item in general}
        general.extend(item for item in alternate if item["text"] not in seen_texts)

        # The room number is on the upper portion of the sign. A separate pass
        # helps when the full multiline sign is merged into one OCR result.
        top = contrast[: max(1, round(contrast.shape[0] * 0.62))]
        digits = self.read(top, confidence=max(0.12, confidence * 0.45))
        return EasyOCREngine._room_decision(
            expected_number, general, digits, confidence
        )


def _load_font(size: int):
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_ocr_results(image, results, draw_boxes: bool = True):
    """Draw Korean-capable OCR labels and a persistent top summary."""
    if not results:
        return image

    image_height, image_width = image.shape[:2]
    font_size = max(22, min(38, round(image_width / 55)))
    font = _load_font(font_size)

    if draw_boxes:
        for result in results:
            points = result.get("bbox") or []
            if len(points) < 4:
                continue
            polygon = [(round(x), round(y)) for x, y in points]
            cv2.polylines(
                image,
                [np.asarray(polygon, dtype=np.int32)],
                True,
                (255, 160, 0),
                max(2, round(image_width / 800)),
                cv2.LINE_AA,
            )

    summary = "OCR: " + " | ".join(
        f"{item['text']} {item['confidence']:.0%}" for item in results[:6]
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    painter = ImageDraw.Draw(canvas)
    left, top = 14, 14
    text_box = painter.textbbox((left, top), summary, font=font)
    right = min(image_width - 1, text_box[2] + 14)
    bottom = min(image_height - 1, text_box[3] + 12)
    painter.rectangle((6, 6, right, bottom), fill=(0, 0, 0))
    painter.text((left, top), summary, font=font, fill=(255, 210, 60))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
