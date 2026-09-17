import json
import re
from pathlib import Path
from threading import Lock


class LocationStore:
    def __init__(self, path: Path):
        self.path, self.lock = path, Lock()
        if not path.exists():
            path.write_text("[]", encoding="utf-8")

    def all(self):
        with self.lock:
            return json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def count(self):
        return len(self.all())

    def upsert(self, payload):
        location = {"node_id": str(payload["node_id"]).strip(),
            "place_name": str(payload["place_name"]).strip(), "floor": payload.get("floor", 1),
            "map_x": payload.get("map_x"), "map_y": payload.get("map_y"),
            "direction": payload.get("direction"), "keywords": payload.get("keywords", []),
            "next_nodes": payload.get("next_nodes", [])}
        with self.lock:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            rows = [row for row in rows if row.get("node_id") != location["node_id"]] + [location]
            self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return location

    def match_texts(self, texts):
        return self.match_ocr_results([{"text": text} for text in texts])

    def match_ocr_results(self, results):
        candidates = [self._normalize(result.get("text", "")) for result in results if str(result.get("text", "")).strip()]
        if not candidates:
            return None

        # Prefer a complete OCR token so "강의실 4" cannot match "회의실 4".
        for location in self.all():
            words = [location.get("place_name", ""), *location.get("keywords", [])]
            normalized_words = [self._normalize(word) for word in words if word]
            if any(word == candidate or word in candidate for word in normalized_words for candidate in candidates):
                return location

        # Return a structured OCR candidate even before it is added to locations.json.
        for candidate in candidates:
            room_match = re.search(r"(강의실|회의실)([0-9]+)", candidate)
            if room_match:
                room_type, room_number = room_match.groups()
                place_name = f"{room_type}{room_number}"
                return {
                    "node_id": None,
                    "place_name": place_name,
                    "room_type": room_type,
                    "room_number": int(room_number),
                    "floor": None,
                    "map_x": None,
                    "map_y": None,
                    "direction": None,
                    "keywords": [f"{room_type} {room_number}", place_name],
                    "next_nodes": [],
                    "registered": False,
                }
        return None

    def room_candidates_from_ocr(self, results):
        parsed = []
        for result in results:
            text = self._normalize(result.get("text", ""))
            if not text:
                continue
            bbox = result.get("bbox") or []
            if len(bbox) < 4:
                continue
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
            parsed.append({
                "text": text,
                "center_x": (min(xs) + max(xs)) / 2,
                "center_y": (min(ys) + max(ys)) / 2,
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys),
            })

        room_words = [item for item in parsed if re.search(r"(강의실|회의실)", item["text"])]
        number_words = [item for item in parsed if re.fullmatch(r"[0-9]", item["text"])]
        for room in room_words:
            direct = re.search(r"(강의실|회의실)([0-9]+)", room["text"])
            if direct:
                yield direct.group(1), int(direct.group(2))
                continue
            room_type = "강의실" if "강의실" in room["text"] else "회의실"
            nearby = [number for number in number_words
                if number["center_y"] < room["center_y"]
                and abs(number["center_x"] - room["center_x"]) <= max(room["width"], number["width"], 20) * 1.5]
            if nearby:
                number = min(nearby, key=lambda item: abs(item["center_y"] - room["center_y"]))
                yield room_type, int(number["text"])

    @staticmethod
    def _normalize(value):
        return re.sub(r"[^0-9가-힣a-z]", "", str(value).lower())
