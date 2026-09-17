"""configs/locations.yaml의 좌표(x, y)를 이용해 간단한 2D 평면도를 그리는 유틸."""

from __future__ import annotations

from pathlib import Path

from ..localization.location_map import Location

# 복도(직선 경로)를 이루는 위치들. 이 순서대로 있으면 선으로 이어서 표시합니다.
# 그 외 위치(문, 정수기 등)는 독립된 점으로만 표시됩니다.
CORRIDOR_SEQUENCE = ["room4", "room3", "room2"]


def draw_floor_map(
    locations: list[Location],
    output_path: str | Path,
    highlight: str | None = None,
    estimated_position: tuple[float, float] | None = None,
) -> Path:
    """위치들의 (x, y) 좌표를 2D 평면도로 그려 PNG로 저장합니다.

    highlight로 넘긴 이름(또는 별칭)과 일치하는 위치는 "현재 위치"로 강조합니다.
    estimated_position=(x, y)를 넘기면, 정해진 위치 점이 아니라 임의의 연속
    좌표(예: VisualLocationEstimator가 계산한 추정 현재 위치)를 별 모양
    마커로 추가 표시합니다 (highlight와 동시에 써도 됨).
    coords가 없는 위치는 건너뜁니다 (아직 실측 전인 위치).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotted = [loc for loc in locations if loc.coords and "x" in loc.coords and "y" in loc.coords]
    if not plotted:
        raise ValueError("locations에 coords(x, y)가 설정된 위치가 없습니다.")

    by_name = {loc.name: loc for loc in plotted}

    xs = [loc.coords["x"] for loc in plotted]
    ys = [loc.coords["y"] for loc in plotted]
    if estimated_position is not None:
        xs = xs + [estimated_position[0]]
        ys = ys + [estimated_position[1]]
    width = max(6.0, (max(xs) - min(xs)) / 2 + 3)
    height = max(4.0, (max(ys) - min(ys)) / 2 + 3)

    fig, ax = plt.subplots(figsize=(width, height))

    # 알려진 복도 구간(room4-room3-room2)은 선으로 연결해서 경로처럼 보이게 표시
    corridor_points = [by_name[name].coords for name in CORRIDOR_SEQUENCE if name in by_name]
    if len(corridor_points) >= 2:
        ax.plot(
            [p["x"] for p in corridor_points],
            [p["y"] for p in corridor_points],
            color="#888888",
            linewidth=6,
            zorder=1,
            solid_capstyle="round",
        )

    for loc in plotted:
        x, y = loc.coords["x"], loc.coords["y"]
        is_current = highlight is not None and (loc.name == highlight or highlight in loc.aliases)
        color = "#e63946" if is_current else "#1d3557"
        size = 260 if is_current else 160
        label = loc.display_name or loc.name

        ax.scatter([x], [y], s=size, color=color, zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(
            label,
            (x, y),
            xytext=(0, 22),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold" if is_current else "normal",
            color=color,
        )
        if is_current:
            ax.annotate(
                "현재 위치",
                (x, y),
                xytext=(0, -30),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color=color,
            )

    if estimated_position is not None:
        ex, ey = estimated_position
        ax.scatter(
            [ex], [ey], s=380, marker="*", color="#f4a300", zorder=4,
            edgecolors="#1d3557", linewidths=1.2,
        )
        ax.annotate(
            "추정 현재 위치",
            (ex, ey),
            xytext=(0, 24),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#c17f00",
        )

    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
