"""혼동행렬(confusion matrix) 기반 분류 성능 지표 계산 + 시각화 공용 유틸.

사진 기반 평가(scripts/evaluate.py)와 영상 기반 평가(evaluation.py)가 똑같은
계산 로직을 쓰도록 여기 한 곳에 모아뒀습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NO_ANSWER = "(무응답)"


@dataclass
class ConfusionMatrix:
    """정답(true) x 예측(predicted) 개수를 누적하고, 그로부터 파생되는
    accuracy/precision/recall/F1을 계산합니다.

    무응답(모델이 판단을 보류한 경우)도 하나의 예측값(NO_ANSWER)으로 취급해서
    recall 계산에 반영합니다 — "틀리게 답함"과 "아예 답을 안 함"을 구분해서
    보되, 정답을 맞히지 못했다는 점에서는 둘 다 실패로 채점하기 위함입니다.
    """

    counts: dict[tuple[str, str], int] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)  # 정답으로 등장하는 클래스들 (NO_ANSWER 제외)

    def add(self, true_label: str, predicted_label: str | None) -> None:
        pred = predicted_label if predicted_label is not None else NO_ANSWER
        key = (true_label, pred)
        self.counts[key] = self.counts.get(key, 0) + 1
        if true_label not in self.labels:
            self.labels.append(true_label)
            self.labels.sort()

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def correct(self) -> int:
        return sum(c for (t, p), c in self.counts.items() if t == p)

    @property
    def answered(self) -> int:
        return sum(c for (_, p), c in self.counts.items() if p != NO_ANSWER)

    @property
    def accuracy(self) -> float:
        """전체 정확도 (무응답도 오답으로 취급)."""
        return self.correct / self.total if self.total else 0.0

    @property
    def coverage(self) -> float:
        """응답률 = 답을 낸 것 / 전체."""
        return self.answered / self.total if self.total else 0.0

    @property
    def accuracy_among_answered(self) -> float:
        """응답한 것 중 정확도."""
        return self.correct / self.answered if self.answered else 0.0

    def precision_recall_f1(self, label: str) -> tuple[float, float, float]:
        tp = self.counts.get((label, label), 0)
        fp = sum(c for (t, p), c in self.counts.items() if p == label and t != label)
        fn = sum(c for (t, p), c in self.counts.items() if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    @property
    def macro_f1(self) -> float:
        if not self.labels:
            return 0.0
        return sum(self.precision_recall_f1(label)[2] for label in self.labels) / len(self.labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "coverage": round(self.coverage, 4),
            "accuracy_among_answered": round(self.accuracy_among_answered, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class": {
                label: {
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f1, 4),
                }
                for label in self.labels
                for p, r, f1 in [self.precision_recall_f1(label)]
            },
            "confusion_matrix": {f"{t}->{p}": c for (t, p), c in sorted(self.counts.items())},
        }

    def format_summary(self) -> str:
        lines = [
            "=== 성능 평가 결과 ===",
            f"전체 정확도: {self.accuracy:.3f} ({self.correct}/{self.total})",
            f"커버리지(응답률): {self.coverage:.3f} ({self.answered}/{self.total})",
            f"응답한 것 중 정확도: {self.accuracy_among_answered:.3f}",
            f"Macro F1: {self.macro_f1:.3f}",
        ]
        for label in self.labels:
            p, r, f1 = self.precision_recall_f1(label)
            lines.append(f"  - {label}: precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
        return "\n".join(lines)


def plot_confusion_matrix(cm: ConfusionMatrix, output_path: str | Path) -> Path:
    """혼동행렬을 히트맵 이미지로 저장합니다. 행=정답, 열=예측(무응답 포함)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_labels = cm.labels
    col_labels = cm.labels + [NO_ANSWER] if any(p == NO_ANSWER for _, p in cm.counts) else cm.labels

    matrix = np.zeros((len(row_labels), len(col_labels)), dtype=int)
    for (t, p), c in cm.counts.items():
        if t in row_labels and p in col_labels:
            matrix[row_labels.index(t), col_labels.index(p)] = c

    fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 1.1), max(3.5, len(row_labels) * 1.0)))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.set_title("혼동행렬 (Confusion Matrix)")

    vmax = matrix.max() if matrix.size else 0
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = matrix[i, j]
            color = "white" if vmax and value > vmax * 0.6 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.8, label="개수")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
