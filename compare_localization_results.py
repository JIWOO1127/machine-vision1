"""
compare_localization_results.py

여러 video_localizer 결과 CSV를 비교하는 간단한 비교 도구.

Ground Truth 없이 비교할 경우:
- pose coverage
- 프레임 간 이동량(jitter proxy)
- 결과 존재 프레임 수
를 비교한다.

Ground Truth CSV를 주면:
필수 컬럼:
    frame,x_mm,y_mm,z_mm

각 결과 CSV의 다음 방법을 비교:
- best_single
- fused
- ema

사용 예
-------
python compare_localization_results.py result1/positions.csv result2/positions.csv

GT가 있는 경우:
python compare_localization_results.py result1/positions.csv --ground-truth gt.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


METHODS = {
    "best_single": (
        "best_single_x_mm",
        "best_single_y_mm",
        "best_single_z_mm",
    ),
    "fused": (
        "fused_x_mm",
        "fused_y_mm",
        "fused_z_mm",
    ),
    "ema": (
        "ema_x_mm",
        "ema_y_mm",
        "ema_z_mm",
    ),
}


def parse_float(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def read_rows(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def read_gt(path: Path):
    gt = {}

    for row in read_rows(path):
        try:
            frame = int(row["frame"])
        except Exception:
            continue

        xyz = [
            parse_float(row.get("x_mm")),
            parse_float(row.get("y_mm")),
            parse_float(row.get("z_mm")),
        ]

        if all(
            value is not None
            for value in xyz
        ):
            gt[frame] = np.array(
                xyz,
                dtype=np.float64,
            )

    return gt


def method_positions(
    rows,
    method,
):
    keys = METHODS[
        method
    ]

    result = []

    for row in rows:
        try:
            frame = int(row["frame"])
        except Exception:
            continue

        xyz = [
            parse_float(
                row.get(keys[0])
            ),
            parse_float(
                row.get(keys[1])
            ),
            parse_float(
                row.get(keys[2])
            ),
        ]

        if all(
            value is not None
            for value in xyz
        ):
            result.append(
                (
                    frame,
                    np.array(
                        xyz,
                        dtype=np.float64,
                    ),
                )
            )

    return result


def jitter_proxy(
    positions,
):
    if len(positions) < 2:
        return None

    diffs = []

    previous_frame = None
    previous_position = None

    for frame, position in positions:
        if (
            previous_frame is not None
            and
            frame == previous_frame + 1
        ):
            diffs.append(
                float(
                    np.linalg.norm(
                        position
                        -
                        previous_position
                    )
                )
            )

        previous_frame = frame
        previous_position = position

    if not diffs:
        return None

    return {
        "mean_step_mm": float(
            np.mean(diffs)
        ),
        "median_step_mm": float(
            np.median(diffs)
        ),
        "p95_step_mm": float(
            np.percentile(
                diffs,
                95,
            )
        ),
    }


def gt_metrics(
    positions,
    gt,
):
    errors_3d = []
    errors_xy = []

    for frame, position in positions:
        if frame not in gt:
            continue

        error = (
            position
            -
            gt[frame]
        )

        errors_3d.append(
            float(
                np.linalg.norm(
                    error
                )
            )
        )

        errors_xy.append(
            float(
                np.linalg.norm(
                    error[:2]
                )
            )
        )

    if not errors_3d:
        return None

    e3 = np.asarray(
        errors_3d,
        dtype=np.float64,
    )

    exy = np.asarray(
        errors_xy,
        dtype=np.float64,
    )

    return {
        "matched_frames": int(
            len(e3)
        ),
        "mae_3d_mm": float(
            np.mean(e3)
        ),
        "rmse_3d_mm": float(
            np.sqrt(
                np.mean(
                    e3 ** 2
                )
            )
        ),
        "median_3d_mm": float(
            np.median(e3)
        ),
        "p95_3d_mm": float(
            np.percentile(
                e3,
                95,
            )
        ),
        "mae_xy_mm": float(
            np.mean(exy)
        ),
        "rmse_xy_mm": float(
            np.sqrt(
                np.mean(
                    exy ** 2
                )
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "results",
        nargs="+",
        help="positions.csv 파일들",
    )

    parser.add_argument(
        "--ground-truth",
        default=None,
        help="GT CSV: frame,x_mm,y_mm,z_mm",
    )

    args = parser.parse_args()

    gt = (
        read_gt(
            Path(
                args.ground_truth
            )
        )
        if args.ground_truth
        else None
    )

    for result_path_str in args.results:
        result_path = Path(
            result_path_str
        )

        rows = read_rows(
            result_path
        )

        print()
        print("=" * 80)
        print(
            f"RESULT: {result_path}"
        )
        print("=" * 80)

        total_frames = len(rows)

        for method in METHODS:
            positions = method_positions(
                rows,
                method,
            )

            coverage = (
                len(positions)
                /
                total_frames
                if total_frames
                else 0.0
            )

            jitter = jitter_proxy(
                positions
            )

            print()
            print(
                f"[{method}]"
            )
            print(
                f"coverage : "
                f"{len(positions)}/{total_frames} "
                f"({coverage * 100:.1f}%)"
            )

            if jitter is not None:
                print(
                    "frame-step jitter proxy: "
                    f"mean={jitter['mean_step_mm']:.1f} mm, "
                    f"median={jitter['median_step_mm']:.1f} mm, "
                    f"p95={jitter['p95_step_mm']:.1f} mm"
                )

            if gt is not None:
                metrics = gt_metrics(
                    positions,
                    gt,
                )

                if metrics is None:
                    print(
                        "GT matched frames: 0"
                    )
                else:
                    print(
                        "GT: "
                        f"matched={metrics['matched_frames']}, "
                        f"MAE_3D={metrics['mae_3d_mm']:.1f} mm, "
                        f"RMSE_3D={metrics['rmse_3d_mm']:.1f} mm, "
                        f"P95_3D={metrics['p95_3d_mm']:.1f} mm, "
                        f"MAE_XY={metrics['mae_xy_mm']:.1f} mm"
                    )


if __name__ == "__main__":
    main()
