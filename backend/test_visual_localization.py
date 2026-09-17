import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from services.ocr import EasyOCREngine
from services.visual_localization import VisualLocationEstimator
from test_inference import draw_detections


class RoomThreeOCRStub:
    device = "cpu"

    def read_room_sign(self, *_args, **_kwargs):
        return {
            "observed_number": None,
            "observed_numbers": [3],
            "room_type": "강의실",
            "accepted": False,
            "resolved_label": None,
            "reason": "unsupported_room_number",
            "texts": ["3", "강의실"],
            "digit_texts": [],
            "confidence": 0.91,
        }


class VisualLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.map_path = BASE_DIR / "data" / "minimap_coordinates.json"

    def test_room_three_ocr_is_kept_as_localization_only(self):
        output = {
            "boxes": torch.tensor([[100, 100, 250, 250]], dtype=torch.float32),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([1]),
        }
        stats = draw_detections(
            self.frame.copy(),
            self.frame,
            output,
            ["2_class", "4_class", "front_door", "rear_door"],
            0.3,
            0.1,
            0.65,
            RoomThreeOCRStub(),
            0.3,
        )
        self.assertEqual(stats["kept"], 0)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(stats["observations"][0]["label"], "3_ocr")
        self.assertTrue(stats["observations"][0]["localization_only"])

    def test_far_room_three_makes_room_four_a_location_candidate(self):
        estimator = VisualLocationEstimator(self.map_path, self.frame.shape[1])
        distance_mm = 9680.0
        box_height = estimator.focal_length_px * 200.0 / distance_mm
        box_width = estimator.focal_length_px * 130.0 / distance_mm
        observation = {
            "label": "3_ocr",
            "score": 0.9,
            "box": [
                960 - box_width / 2,
                400,
                960 + box_width / 2,
                400 + box_height,
            ],
            "area": box_width * box_height,
        }
        result = estimator.update(self.frame, [observation])
        candidate_names = [item["name"] for item in result["candidates"]]
        self.assertEqual(result["status"], "candidates")
        self.assertIn("강의실 4 앞 구역", candidate_names)
        self.assertEqual(result["anchors"], ["classroom_3"])
        self.assertTrue(math.isclose(result["measurements"][0]["distance_mm"], 9680, abs_tol=1))

    def test_room_decision_exposes_unsupported_room_number(self):
        decision = EasyOCREngine._room_decision(
            2,
            [
                {"text": "3", "confidence": 0.9, "bbox": [[0, 0]]},
                {"text": "강의실", "confidence": 0.9, "bbox": [[0, 2]]},
            ],
            [],
            0.3,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["observed_numbers"], [3])
        self.assertEqual(decision["reason"], "unsupported_room_number")


if __name__ == "__main__":
    unittest.main()
