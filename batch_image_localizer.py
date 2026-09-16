r"""
batch_image_localizer.py

폴더 안의 정지 이미지들을 image_localizer.py로 일괄 판정하고,
CSV / JSON / HTML 요약을 만든다.

예시
----
# 기존 camera_calibration.npz 사용
python batch_image_localizer.py "C:\images"

# 촬영 장비가 불명확한 이미지
python batch_image_localizer.py "C:\images" --no-calibration

# 하위 폴더까지 검색
python batch_image_localizer.py "C:\images" --no-calibration --recursive

결과
----
image_results/
├─ 0001_xxx/
│  ├─ annotated.png
│  ├─ map.png
│  └─ result.json
├─ ...
├─ batch_summary.csv
├─ batch_summary.json
├─ batch_summary.html
└─ failed_images.txt

batch_summary.html을 웹 브라우저로 열면
100장의 annotated 결과와 미니맵을 한 번에 훑어볼 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def find_images(
    input_dir: Path,
    recursive: bool,
) -> list[Path]:
    iterator = (
        input_dir.rglob("*")
        if recursive
        else input_dir.glob("*")
    )

    images = [
        p
        for p in iterator
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    return sorted(
        images,
        key=lambda p: str(p).lower(),
    )


def safe_folder_name(
    index: int,
    image_path: Path,
) -> str:
    stem = "".join(
        ch
        if (
            ch.isalnum()
            or ch in "-_"
        )
        else "_"
        for ch in image_path.stem
    )

    stem = (
        stem[:80]
        if stem
        else "image"
    )

    return (
        f"{index:04d}_{stem}"
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def get_xyz(
    result: dict[str, Any],
) -> tuple[
    float | None,
    float | None,
    float | None,
]:
    fused = result.get(
        "fused"
    )

    if not fused:
        return (
            None,
            None,
            None,
        )

    position = fused.get(
        "position_world_mm"
    )

    if (
        not position
        or len(position) < 3
    ):
        return (
            None,
            None,
            None,
        )

    return (
        float(position[0]),
        float(position[1]),
        float(position[2]),
    )


def join_ids(
    value,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        (list, tuple),
    ):
        return ",".join(
            str(x)
            for x in value
        )

    return str(value)


def rel_url(
    target: Path,
    html_dir: Path,
) -> str:
    try:
        relative = target.resolve().relative_to(
            html_dir.resolve()
        )
    except ValueError:
        import os

        relative = Path(
            os.path.relpath(
                target.resolve(),
                html_dir.resolve(),
            )
        )

    return quote(
        relative.as_posix(),
        safe="/._-",
    )


def make_html(
    rows: list[dict[str, Any]],
    output_dir: Path,
    source_dir: Path,
) -> str:
    ok_count = sum(
        1
        for r in rows
        if r["status"] == "OK"
    )

    no_pose_count = sum(
        1
        for r in rows
        if r["status"] == "NO_POSE"
    )

    error_count = sum(
        1
        for r in rows
        if r["status"] == "ERROR"
    )

    cards = []

    for row in rows:
        status = row["status"]

        if status == "OK":
            css_class = "ok"
        elif status == "NO_POSE":
            css_class = "warn"
        else:
            css_class = "error"

        annotated = (
            Path(row["annotated_path"])
            if row["annotated_path"]
            else None
        )

        map_path = (
            Path(row["map_path"])
            if row["map_path"]
            else None
        )

        annotated_html = (
            f'<img loading="lazy" '
            f'src="{rel_url(annotated, output_dir)}" '
            f'alt="annotated">'
            if (
                annotated is not None
                and annotated.exists()
            )
            else (
                '<div class="missing">'
                'annotated image 없음'
                '</div>'
            )
        )

        map_html = (
            f'<img loading="lazy" '
            f'src="{rel_url(map_path, output_dir)}" '
            f'alt="map">'
            if (
                map_path is not None
                and map_path.exists()
            )
            else ""
        )

        x = row["x_mm"]
        y = row["y_mm"]
        z = row["z_mm"]

        xyz_text = (
            f"X={x:.0f} / Y={y:.0f} / "
            f"Z={z:.0f} mm"
            if (
                x is not None
                and y is not None
                and z is not None
            )
            else "XYZ 없음"
        )

        distance = row[
            "nearest_distance_mm"
        ]

        distance_text = (
            f"{distance:.0f} mm "
            f"({distance / 1000.0:.2f} m)"
            if distance is not None
            else "-"
        )

        error_text = html.escape(
            str(
                row.get(
                    "error",
                    "",
                )
            )
        )

        cards.append(
            f"""
            <article class="card {css_class}">
              <div class="card-head">
                <div>
                  <strong>#{row['index']:03d}</strong>
                  {html.escape(row['image_name'])}
                </div>
                <span class="badge">{status}</span>
              </div>

              <div class="important">
                {html.escape(xyz_text)}
              </div>

              <div class="door">
                최근접 문:
                <strong>
                  {html.escape(row['nearest_landmark'] or '-')}
                </strong>
                &nbsp; 거리:
                <strong>{html.escape(distance_text)}</strong>
              </div>

              <div class="meta">
                판정: {html.escape(row['location_text'] or '-')}<br>
                검출 ID: {html.escape(row['detected_ids'])}<br>
                등록 ID: {html.escape(row['registered_ids'])}<br>
                사용 ID: {html.escape(row['used_ids'])}
              </div>

              {(
                  f'<div class="error-text">{error_text}</div>'
                  if error_text
                  else ''
              )}

              <div class="images">
                <div>{annotated_html}</div>
                <div>{map_html}</div>
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Batch Image Localization Summary</title>
<style>
  body {{
    margin: 0;
    font-family: Arial, "Malgun Gothic", sans-serif;
    background: #f3f4f6;
    color: #111827;
  }}

  header {{
    position: sticky;
    top: 0;
    z-index: 10;
    background: rgba(255,255,255,.96);
    border-bottom: 1px solid #d1d5db;
    padding: 14px 18px;
  }}

  header h1 {{
    margin: 0 0 7px 0;
    font-size: 22px;
  }}

  .summary {{
    font-size: 14px;
    font-weight: 700;
  }}

  main {{
    padding: 16px;
    display: grid;
    grid-template-columns:
      repeat(auto-fit, minmax(460px, 1fr));
    gap: 16px;
  }}

  .card {{
    background: white;
    border: 2px solid #d1d5db;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
  }}

  .card.ok {{
    border-color: #22c55e;
  }}

  .card.warn {{
    border-color: #f59e0b;
  }}

  .card.error {{
    border-color: #ef4444;
  }}

  .card-head {{
    padding: 10px 12px;
    background: #111827;
    color: white;
    display: flex;
    justify-content: space-between;
    gap: 12px;
  }}

  .badge {{
    font-size: 12px;
    border-radius: 999px;
    background: white;
    color: #111827;
    padding: 3px 8px;
    font-weight: 800;
  }}

  .important {{
    padding: 12px 12px 4px 12px;
    font-size: 21px;
    font-weight: 900;
    color: #b91c1c;
  }}

  .door {{
    padding: 6px 12px;
    font-size: 18px;
    background: #fef3c7;
  }}

  .meta {{
    padding: 8px 12px;
    line-height: 1.55;
    font-size: 14px;
  }}

  .error-text {{
    margin: 8px 12px;
    padding: 8px;
    background: #fee2e2;
    color: #991b1b;
    white-space: pre-wrap;
  }}

  .images {{
    padding: 10px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    background: #e5e7eb;
  }}

  .images img {{
    display: block;
    width: 100%;
    height: auto;
    background: #111827;
  }}

  .missing {{
    min-height: 160px;
    display: grid;
    place-items: center;
    background: #ddd;
    color: #555;
  }}

  @media (max-width: 700px) {{
    main {{
      grid-template-columns: 1fr;
      padding: 8px;
    }}

    .images {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1>ArUco 정지 이미지 일괄 판정</h1>
  <div class="summary">
    원본 폴더: {html.escape(str(source_dir))}
    &nbsp; | &nbsp;
    총 {len(rows)}장
    &nbsp; | &nbsp;
    OK {ok_count}
    &nbsp; | &nbsp;
    NO_POSE {no_pose_count}
    &nbsp; | &nbsp;
    ERROR {error_count}
  </div>
</header>
<main>
{''.join(cards)}
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "폴더의 이미지를 image_localizer.py로 "
            "일괄 판정하고 HTML/CSV 요약을 생성합니다."
        )
    )

    parser.add_argument(
        "input_dir",
        help="원본 이미지 폴더",
    )

    parser.add_argument(
        "--output-dir",
        default="image_results",
        help="결과 폴더. 기본 image_results",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="하위 폴더까지 이미지 검색",
    )

    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="HFOV 기반 근사 카메라 모델 사용",
    )

    parser.add_argument(
        "--calibration",
        default=None,
        help="camera_calibration.npz 경로",
    )

    parser.add_argument(
        "--hfov",
        type=float,
        default=60.0,
        help="--no-calibration 사용 시 수평 화각",
    )

    parser.add_argument(
        "--dict",
        dest="dictionary_name",
        default="DICT_4X4_50",
        help="ArUco dictionary",
    )

    parser.add_argument(
        "--outlier-mm",
        type=float,
        default=1500.0,
        help="다중 마커 이상치 제거 기준",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "result.json이 이미 있는 이미지는 "
            "재처리하지 않고 기존 결과 사용"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "앞에서 N장만 처리. "
            "0이면 전체 처리"
        ),
    )

    args = parser.parse_args()

    project_dir = Path(
        __file__
    ).resolve().parent

    localizer_script = (
        project_dir
        /
        "image_localizer.py"
    )

    if not localizer_script.exists():
        print(
            "ERROR: image_localizer.py가 "
            f"없습니다: {localizer_script}"
        )
        return 2

    input_dir = Path(
        args.input_dir
    ).expanduser().resolve()

    if not input_dir.exists():
        print(
            f"ERROR: 이미지 폴더가 없습니다: "
            f"{input_dir}"
        )
        return 2

    if not input_dir.is_dir():
        print(
            f"ERROR: 폴더가 아닙니다: "
            f"{input_dir}"
        )
        return 2

    output_dir = Path(
        args.output_dir
    ).expanduser()

    if not output_dir.is_absolute():
        output_dir = (
            project_dir
            /
            output_dir
        )

    output_dir = (
        output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = find_images(
        input_dir,
        args.recursive,
    )

    if args.limit > 0:
        images = images[
            :args.limit
        ]

    if not images:
        print(
            "처리할 이미지가 없습니다."
        )
        return 1

    print("=" * 72)
    print("BATCH IMAGE LOCALIZER")
    print("=" * 72)
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Images : {len(images)}")
    print(
        "Camera : "
        + (
            f"APPROX HFOV={args.hfov:.1f}"
            if args.no_calibration
            else "CALIBRATED"
        )
    )
    print("=" * 72)

    rows: list[
        dict[str, Any]
    ] = []

    failures: list[str] = []

    for index, image_path in enumerate(
        images,
        start=1,
    ):
        result_dir = (
            output_dir
            /
            safe_folder_name(
                index,
                image_path,
            )
        )

        result_json = (
            result_dir
            /
            "result.json"
        )

        annotated_path = (
            result_dir
            /
            "annotated.png"
        )

        map_path = (
            result_dir
            /
            "map.png"
        )

        print(
            f"[{index:03d}/{len(images):03d}] "
            f"{image_path.name}"
        )

        error_message = ""

        if not (
            args.skip_existing
            and result_json.exists()
        ):
            result_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            cmd = [
                sys.executable,
                str(localizer_script),
                str(image_path),
                "--output-dir",
                str(result_dir),
                "--dict",
                args.dictionary_name,
                "--outlier-mm",
                str(args.outlier_mm),
            ]

            if args.no_calibration:
                cmd.extend(
                    [
                        "--no-calibration",
                        "--hfov",
                        str(args.hfov),
                    ]
                )
            elif args.calibration:
                cmd.extend(
                    [
                        "--calibration",
                        str(
                            Path(
                                args.calibration
                            )
                            .expanduser()
                            .resolve()
                        ),
                    ]
                )

            completed = subprocess.run(
                cmd,
                cwd=str(project_dir),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )

            if completed.returncode != 0:
                error_message = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or (
                        "image_localizer.py "
                        f"exit code "
                        f"{completed.returncode}"
                    )
                )

        result: dict[
            str,
            Any
        ] = {}

        if result_json.exists():
            try:
                result = load_json(
                    result_json
                )
            except Exception as exc:
                if not error_message:
                    error_message = (
                        f"result.json 읽기 실패: "
                        f"{exc}"
                    )

        x, y, z = get_xyz(
            result
        )

        fused = result.get(
            "fused"
        )

        location = (
            result.get(
                "location"
            )
            or {}
        )

        if error_message:
            status = "ERROR"
        elif fused is None:
            status = "NO_POSE"
        else:
            status = "OK"

        detected_ids = join_ids(
            result.get(
                "detected_ids"
            )
        )

        registered_ids = join_ids(
            result.get(
                "registered_ids"
            )
        )

        used_ids = ""

        if fused:
            used_ids = join_ids(
                fused.get(
                    "used_marker_ids"
                )
            )

        nearest_distance = (
            location.get(
                "nearest_landmark_distance_mm"
            )
        )

        if nearest_distance is not None:
            try:
                nearest_distance = float(
                    nearest_distance
                )
            except Exception:
                nearest_distance = None

        row = {
            "index": index,
            "image_name": (
                image_path.name
            ),
            "image_path": str(
                image_path
            ),
            "status": status,
            "detected_ids": (
                detected_ids
            ),
            "registered_ids": (
                registered_ids
            ),
            "used_ids": used_ids,
            "x_mm": x,
            "y_mm": y,
            "z_mm": z,
            "yaw_deg": (
                fused.get(
                    "yaw_deg"
                )
                if fused
                else None
            ),
            "spread_mm": (
                fused.get(
                    "spread_mm"
                )
                if fused
                else None
            ),
            "nearest_landmark": (
                location.get(
                    "nearest_landmark"
                )
                or ""
            ),
            "nearest_distance_mm": (
                nearest_distance
            ),
            "nearest_distance_m": (
                (
                    nearest_distance
                    /
                    1000.0
                )
                if nearest_distance
                is not None
                else None
            ),
            "location_text": (
                location.get(
                    "text_ko"
                )
                or ""
            ),
            "result_dir": str(
                result_dir
            ),
            "annotated_path": (
                str(
                    annotated_path
                )
                if annotated_path.exists()
                else ""
            ),
            "map_path": (
                str(
                    map_path
                )
                if map_path.exists()
                else ""
            ),
            "result_json": (
                str(
                    result_json
                )
                if result_json.exists()
                else ""
            ),
            "error": error_message,
        }

        rows.append(
            row
        )

        if status != "OK":
            failures.append(
                (
                    f"[{status}] "
                    f"{image_path} "
                    f"{error_message}"
                ).strip()
            )

        if status == "OK":
            print(
                "   -> "
                f"XYZ=({x:.0f}, "
                f"{y:.0f}, "
                f"{z:.0f}) mm | "
                f"nearest="
                f"{row['nearest_landmark']} "
                f"{nearest_distance:.0f} mm"
                if nearest_distance
                is not None
                else (
                    "   -> "
                    f"XYZ=({x:.0f}, "
                    f"{y:.0f}, "
                    f"{z:.0f}) mm"
                )
            )
        else:
            print(
                f"   -> {status}"
            )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------
    csv_path = (
        output_dir
        /
        "batch_summary.csv"
    )

    fields = [
        "index",
        "image_name",
        "status",
        "detected_ids",
        "registered_ids",
        "used_ids",
        "x_mm",
        "y_mm",
        "z_mm",
        "yaw_deg",
        "spread_mm",
        "nearest_landmark",
        "nearest_distance_mm",
        "nearest_distance_m",
        "location_text",
        "image_path",
        "result_dir",
        "annotated_path",
        "map_path",
        "result_json",
        "error",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(
                        key,
                        "",
                    )
                    for key in fields
                }
            )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------
    json_path = (
        output_dir
        /
        "batch_summary.json"
    )

    json_path.write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # failures
    # --------------------------------------------------------
    failures_path = (
        output_dir
        /
        "failed_images.txt"
    )

    failures_path.write_text(
        "\n".join(
            failures
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # HTML gallery
    # --------------------------------------------------------
    html_path = (
        output_dir
        /
        "batch_summary.html"
    )

    html_path.write_text(
        make_html(
            rows,
            output_dir,
            input_dir,
        ),
        encoding="utf-8",
    )

    ok_count = sum(
        1
        for r in rows
        if r["status"] == "OK"
    )

    no_pose_count = sum(
        1
        for r in rows
        if r["status"] == "NO_POSE"
    )

    error_count = sum(
        1
        for r in rows
        if r["status"] == "ERROR"
    )

    print()
    print("=" * 72)
    print("BATCH COMPLETE")
    print("=" * 72)
    print(f"TOTAL   : {len(rows)}")
    print(f"OK      : {ok_count}")
    print(f"NO_POSE : {no_pose_count}")
    print(f"ERROR   : {error_count}")
    print(f"CSV     : {csv_path}")
    print(f"JSON    : {json_path}")
    print(f"HTML    : {html_path}")
    print(f"FAILED  : {failures_path}")
    print()
    print(
        "100장 결과를 눈으로 확인하려면 "
        "batch_summary.html을 브라우저로 여세요."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
