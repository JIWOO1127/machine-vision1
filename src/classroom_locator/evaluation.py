"""영상 + 정답(ground truth) 타임라인을 이용해 위치 추적 성능을 측정하는 모듈.

측정하는 5개 지표:
1. 클래스별 F1 / 혼동행렬 (Confusion Matrix)
2. 응답한 것 중 정확도 (답을 낸 프레임 중 맞춘 비율)
3. 전환 반응 속도 (Latency) — 새 위치 진입 후 정답으로 갱신되기까지 걸린 시간
4. 처리 속도 (FPS / 추론 시간)
5. 커버리지(응답률) — 답을 낸 프레임 수 / 전체 정답 있는 프레임 수

정답 타임라인은 CSV로 준비합니다 (start_sec, end_sec, location):
    start_sec,end_sec,location
    0,5,room4
    5,8,
    8,15,room3

location을 비워두면 "전환 구간(정답 없음)"으로 취급되어 채점에서 제외됩니다.

1/2/5번(혼동행렬/F1/정확도/커버리지)은 eval_metrics.ConfusionMatrix가 계산하고,
여기서는 영상 전용 지표(Latency, FPS)만 추가로 붙입니다.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from .eval_metrics import ConfusionMatrix, plot_confusion_matrix
from .pipeline.core import LocatorPipeline
from .pipeline.locking import LocationLocker


@dataclass
class GroundTruthSegment:
    start_sec: float
    end_sec: float
    location: str | None  # None이면 정답 없음(전환 구간)


def load_ground_truth(path: str | Path) -> list[GroundTruthSegment]:
    segments: list[GroundTruthSegment] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            location = (row.get("location") or "").strip() or None
            segments.append(
                GroundTruthSegment(
                    start_sec=float(row["start_sec"]),
                    end_sec=float(row["end_sec"]),
                    location=location,
                )
            )
    return segments


def _location_at(segments: list[GroundTruthSegment], t: float) -> str | None:
    for seg in segments:
        if seg.start_sec <= t < seg.end_sec:
            return seg.location
    return None


@dataclass
class EvalReport:
    confusion: ConfusionMatrix
    latencies_sec: list[float] = field(default_factory=list)
    failed_transitions: int = 0
    avg_inference_ms: float = 0.0
    avg_fps: float = 0.0

    @property
    def avg_latency_sec(self) -> float | None:
        return sum(self.latencies_sec) / len(self.latencies_sec) if self.latencies_sec else None

    def to_dict(self) -> dict[str, Any]:
        data = self.confusion.to_dict()
        data.update(
            {
                "avg_latency_sec": round(self.avg_latency_sec, 3) if self.avg_latency_sec is not None else None,
                "latency_samples_sec": [round(v, 3) for v in self.latencies_sec],
                "failed_transitions": self.failed_transitions,
                "avg_inference_ms": round(self.avg_inference_ms, 2),
                "avg_fps": round(self.avg_fps, 2),
            }
        )
        return data

    def format_summary(self) -> str:
        lines = [
            "=== 성능 평가 결과 ===",
            f"1. Macro F1: {self.confusion.macro_f1:.3f}",
        ]
        for label in self.confusion.labels:
            p, r, f1 = self.confusion.precision_recall_f1(label)
            lines.append(f"   - {label}: precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
        lines.append(
            f"2. 응답한 것 중 정확도: {self.confusion.accuracy_among_answered:.3f} "
            f"({self.confusion.correct}/{self.confusion.answered})"
        )
        if self.avg_latency_sec is not None:
            lines.append(
                f"3. 평균 전환 반응 속도: {self.avg_latency_sec:.2f}초 "
                f"(성공 {len(self.latencies_sec)}회, 실패 {self.failed_transitions}회)"
            )
        else:
            lines.append(f"3. 평균 전환 반응 속도: 측정 불가 (성공 사례 없음, 실패 {self.failed_transitions}회)")
        lines.append(f"4. 처리 속도: 평균 {self.avg_inference_ms:.1f}ms/프레임 (약 {self.avg_fps:.1f} FPS)")
        lines.append(
            f"5. 커버리지(응답률): {self.confusion.coverage:.3f} "
            f"({self.confusion.answered}/{self.confusion.total})"
        )
        return "\n".join(lines)


def evaluate_video(
    config: dict[str, Any],
    video_path: str,
    ground_truth_path: str,
    process_every_n_frames: int | None = None,
) -> EvalReport:
    """영상을 처음부터 끝까지 돌리며 정답 타임라인과 비교해 성능 지표를 계산합니다."""
    pipeline = LocatorPipeline(config)
    segments = load_ground_truth(ground_truth_path)

    rt_config = config.get("realtime", {})
    n = process_every_n_frames or rt_config.get("process_every_n_frames", 5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    locker = LocationLocker()
    confusion = ConfusionMatrix()
    inference_times_ms: list[float] = []

    # 구간 전환 시점마다 "정답으로 바뀌기까지 걸린 시간"을 재기 위한 상태
    latencies: list[float] = []
    failed_transitions = 0
    pending_target: str | None = None
    pending_start: float | None = None
    last_gt_location: str | None = None

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            video_time = frame_index / fps
            frame_index += 1

            gt = _location_at(segments, video_time)

            # 새로운 실제 위치 구간에 막 진입했으면 latency 측정 시작
            if gt is not None and gt != last_gt_location:
                if pending_target is not None:
                    failed_transitions += 1  # 이전 목표는 끝내 못 맞춘 채 다음 구간으로 넘어감
                pending_target = gt
                pending_start = video_time
            last_gt_location = gt

            if frame_index % n != 0:
                continue

            t0 = time.perf_counter()
            result = pipeline.process_image(frame)
            inference_times_ms.append((time.perf_counter() - t0) * 1000)

            current, _ = locker.update(result.match, video_time)
            predicted = current.location.name if current else None

            if pending_target is not None and predicted == pending_target:
                latencies.append(video_time - pending_start)
                pending_target = None
                pending_start = None

            if gt is None:
                continue

            confusion.add(gt, predicted)
    finally:
        cap.release()

    if pending_target is not None:
        failed_transitions += 1

    avg_ms = sum(inference_times_ms) / len(inference_times_ms) if inference_times_ms else 0.0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return EvalReport(
        confusion=confusion,
        latencies_sec=latencies,
        failed_transitions=failed_transitions,
        avg_inference_ms=avg_ms,
        avg_fps=avg_fps,
    )


def save_report(report: EvalReport, output_path: str | Path, plot_path: str | Path | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    if plot_path is not None:
        plot_confusion_matrix(report.confusion, plot_path)
    return output_path
