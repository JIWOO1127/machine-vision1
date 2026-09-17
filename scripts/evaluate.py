#!/usr/bin/env python
"""배치 처리 결과(outputs/results/batch_results.json)를 바탕으로
혼동행렬(Confusion Matrix)과 그로부터 계산되는 모든 지표(정확도, 클래스별
precision/recall/F1, macro F1, 커버리지)를 계산하는 평가 스크립트.

정답 라벨(각 이미지의 실제 위치)을 별도 JSON으로 준비해서 ground_truth
인자로 넘겨주면 정확도를 계산합니다. 정답 데이터 포맷은
{"image_filename": "실제_위치_이름", ...} 형태의 JSON을 예시로 합니다.

결과는 outputs/eval/에 JSON 리포트 + 혼동행렬 그래프(PNG)로 저장됩니다.

사용 예:
    python scripts/evaluate.py --results outputs/results/batch_results.json --ground-truth data/eval/ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.eval_metrics import ConfusionMatrix, plot_confusion_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="위치 추정 성능 평가 (혼동행렬 기반)")
    parser.add_argument("--results", type=str, default="outputs/results/batch_results.json")
    parser.add_argument("--ground-truth", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)
    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    confusion = ConfusionMatrix()
    for record in results:
        filename = Path(record["image"]).name
        if filename not in ground_truth:
            continue
        confusion.add(ground_truth[filename], record.get("matched_location"))

    if confusion.total == 0:
        print("정답 데이터와 매칭되는 결과가 없습니다.")
        return

    print(confusion.format_summary())

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = output_dir / f"eval_report_{timestamp}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(confusion.to_dict(), f, ensure_ascii=False, indent=2)

    plot_path = plot_confusion_matrix(confusion, output_dir / f"confusion_matrix_{timestamp}.png")

    print(f"\n상세 결과 저장: {report_path}")
    print(f"혼동행렬 그래프 저장: {plot_path}")


if __name__ == "__main__":
    main()
