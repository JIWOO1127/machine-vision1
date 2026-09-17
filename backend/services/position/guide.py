"""객체 탐지 결과를 위치 정보와 사용자 안내 명령으로 바꾼다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PositionGuide:
    """``position/*.yaml``을 웹 API에서 사용하는 단일 안내 데이터원으로 제공한다."""

    _ANCHOR_TO_LOCATION = {
        "classroom_2": "room2",
        "classroom_3": "room3",
        "classroom_4": "room4",
        "front_door": "front_door",
        "rear_door": "rear_door",
        "side_door": "rear_door",
    }

    def __init__(self, directory: Path):
        locations_data = self._read_yaml(directory / "locations.yaml")
        default_data = self._read_yaml(directory / "default.yaml")
        self.locations = {
            str(item["name"]): item
            for item in locations_data.get("locations", [])
            if item.get("name")
        }
        self.class_name_map = {
            str(source): str(target)
            for source, target in default_data.get("detector", {})
            .get("class_name_map", {})
            .items()
        }
        self.grid = locations_data.get("grid") or {}
        self._last_cell: list[int] | tuple[int, int] | None = None
        self._last_label: str | None = None

    def reset(self) -> None:
        """새 영상/카메라 세션에서는 이전 세션의 점을 지운다."""
        self._last_cell = None
        self._last_label = None

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}

    def resolve(
        self,
        *,
        anchors: list[str],
        labels: list[str],
        measurements: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """거리 계산에 쓰인 지도 anchor를 우선하고, 탐지 클래스명으로 보완한다."""
        candidates = [self._ANCHOR_TO_LOCATION.get(anchor, anchor) for anchor in anchors]
        candidates.extend(self.class_name_map.get(label, label) for label in labels)
        for name in candidates:
            entry = self.locations.get(name)
            if entry is not None:
                current_cell = self._estimate_cell(entry, measurements or [])
                self._last_cell = current_cell
                self._last_label = str(entry.get("display_name") or entry["name"])
                return self._public_entry(entry, current_cell)
        # 기준 객체가 잠깐 화면에서 사라져도 마지막 위치 점은 그대로 둔다.
        return self._empty_entry()

    def _estimate_cell(self, entry: dict[str, Any], measurements: list[dict[str, Any]]):
        """기준 물체까지의 거리로 U자형 복도 위 현재 격자점을 추정한다."""
        landmark = entry.get("grid_cell")
        cols = int(self.grid.get("cols", 0) or 0)
        rows = int(self.grid.get("rows", 0) or 0)
        cell_size = float(self.grid.get("cell_size_m", 0) or 0)
        if not landmark or not cols or not rows or not cell_size:
            return landmark

        anchor_names = {key for key, value in self._ANCHOR_TO_LOCATION.items() if value == entry["name"]}
        measurement = next((item for item in measurements if item.get("anchor") in anchor_names), None)
        if measurement is None:
            return landmark
        try:
            distance_units = float(measurement["distance_mm"]) / 1000.0 / cell_size
        except (KeyError, TypeError, ValueError):
            return landmark

        path = {(x, 0) for x in range(cols + 1)}
        path.update((0, y) for y in range(rows + 1))
        path.update((cols, y) for y in range(rows + 1))
        lx, ly = landmark
        return list(min(path, key=lambda point: abs(((point[0] - lx) ** 2 + (point[1] - ly) ** 2) ** 0.5 - distance_units)))

    def _public_entry(self, entry: dict[str, Any], current_cell: list[int] | tuple[int, int] | None) -> dict[str, Any]:
        name = str(entry["name"])
        display_name = str(entry.get("display_name") or name)
        command = entry.get("guidance")
        if not command:
            command = f"{display_name}이 보입니다. 현재 위치를 확인하세요."
        return {
            "key": name,
            "label": display_name,
            "description": str(entry.get("description") or ""),
            "command": str(command),
            "grid_cell": entry.get("grid_cell"),
            "current_cell": current_cell,
            "snap_distance_m": entry.get("snap_distance_m"),
            **self._map_payload(),
        }

    def _empty_entry(self) -> dict[str, Any]:
        """탐지 전에도 웹이 빈 지도와 고정 기준점을 그릴 수 있게 한다."""
        return {
            "key": None,
            "label": self._last_label or "위치 인식 중",
            "description": "",
            "command": None,
            "grid_cell": None,
            "current_cell": self._last_cell,
            "snap_distance_m": None,
            **self._map_payload(),
        }

    def _map_payload(self) -> dict[str, Any]:
        return {
            "grid": self.grid,
            "landmarks": [
                {
                    "key": item["name"],
                    "label": item.get("display_name") or item["name"],
                    "grid_cell": item.get("grid_cell"),
                }
                for item in self.locations.values()
                if item.get("grid_cell") is not None
            ],
        }
