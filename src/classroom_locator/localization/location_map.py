"""configs/locations.yaml 로더."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Location:
    name: str
    aliases: list[str] = field(default_factory=list)
    building: str | None = None
    floor: int | None = None
    description: str | None = None
    display_name: str | None = None
    guidance: str | None = None
    coords: dict[str, Any] | None = None
    # 격자 지도 방식(GridPositionTracker)에서 쓰는 필드: 이 물체가
    # snap_distance_m 이내로 탐지되면 현재 위치를 grid_cell로 갱신함.
    grid_cell: tuple[int, int] | None = None
    snap_distance_m: float | None = None

    def all_names(self) -> list[str]:
        """매칭에 사용할 모든 이름(원래 이름 + 별칭) 목록."""
        return [self.name, *self.aliases]


@dataclass
class GridConfig:
    cols: int
    rows: int
    cell_size_m: float


def load_locations(locations_file: str | Path) -> list[Location]:
    path = Path(locations_file)
    if not path.exists():
        raise FileNotFoundError(f"위치 매핑 파일을 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    locations = []
    for item in data.get("locations", []):
        grid_cell = item.get("grid_cell")
        locations.append(
            Location(
                name=item["name"],
                aliases=item.get("aliases", []),
                building=item.get("building"),
                floor=item.get("floor"),
                description=item.get("description"),
                display_name=item.get("display_name"),
                guidance=item.get("guidance"),
                coords=item.get("coords"),
                grid_cell=tuple(grid_cell) if grid_cell else None,
                snap_distance_m=item.get("snap_distance_m"),
            )
        )
    return locations


def load_grid_config(locations_file: str | Path) -> GridConfig | None:
    path = Path(locations_file)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    grid = data.get("grid")
    if grid is None:
        return None
    return GridConfig(cols=int(grid["cols"]), rows=int(grid["rows"]), cell_size_m=float(grid["cell_size_m"]))
