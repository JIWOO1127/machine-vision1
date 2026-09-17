"""Flask 백엔드와 root의 classroom_locator 위치 판정 파이프라인
(`LandmarkGridPositionTracker`)을 연결하는 어댑터.

예전에 `backend/vision_bridge.py`(현재는 삭제됨, git 이력의 df5cdb4 커밋 참고)로
한 번 시도됐던 초기 버전을 이어받아, 그동안 root 쪽에서 더 발전한 로직
(로고 중앙정렬 회전 안내, 3강의실 OCR 히스테리시스, 접근추세 확인, 스티키
위치 잠금 등 - docs/setup_log.md 참고)까지 반영해서 다시 작성했다.

실시간 카메라(/api/analyze-frame), 사진 업로드(/api/analyze), 영상 업로드
(/api/analyze-video) 세 경로 모두 이 모듈이 담당한다. 세 경로는 판정 이력이
서로 섞이면 안 되므로(예: 사진 업로드 중에 실시간 카메라의 현재 위치가
바뀌어버리면 안 됨) 각자 독립된 `LandmarkGridPositionTracker` 인스턴스를 쓴다
- 실제로 쓰일 때만 만들어(지연 생성) 안 쓰는 경로의 모델 로딩 비용은 내지
않는다.

과거 UI 담당자가 만든 옛 파이프라인(services/vision.py와 그게 불러오던
services/position/, services/visual_localization.py, services/ocr.py,
test_inference.py, faster_rcnn_pipeline.py)은 이 모듈로 완전히 교체되어
삭제됐다 - 영상 변환 유틸(_transcode_for_mobile)만 이 파일로 옮겨왔다.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from classroom_locator.localization.location_map import (  # noqa: E402
    GridConfig,
    Location,
    load_grid_config,
    load_locations,
)
from classroom_locator.pipeline.grid_tracker import (  # noqa: E402
    MAP_LABELS,
    LandmarkGridPositionTracker,
)
from classroom_locator.utils.image_utils import put_korean_text  # noqa: E402

DEFAULT_WEIGHTS = _PROJECT_ROOT / "models" / "yolo" / "final_best.pt"
LOCATIONS_FILE = _PROJECT_ROOT / "configs" / "locations.yaml"


def decode_image(payload: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("재생 중인 영상 프레임을 읽을 수 없습니다.")
    return frame


def _read_phone_image(image_path: Path) -> np.ndarray:
    """폰 카메라는 회전 정보를 픽셀이 아니라 EXIF에 저장하는 경우가 많아서,
    탐지 전에 정규화한다 (기존 services/vision.py의 동일 로직 재사용)."""
    with Image.open(image_path) as pil_image:
        oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
        return cv2.cvtColor(np.asarray(oriented), cv2.COLOR_RGB2BGR)


def _direction_from_box(
    box: tuple[float, float, float, float], frame_width: int, left: float = 0.35, right: float = 0.65
) -> str:
    cx = (box[0] + box[2]) / 2 / max(1, frame_width)
    return "왼쪽" if cx < left else "오른쪽" if cx > right else "앞쪽"


def _transcode_for_mobile(raw_path: Path, source_path: Path, result_path: Path) -> None:
    """분석 결과 영상(무압축에 가까운 mp4v)을 폰/브라우저에서 바로 재생되는
    H.264+AAC로 변환한다 (원래 services/vision.py의 동일 로직 - 옛 파이프라인
    삭제 전에 이 함수만 옮겨왔다)."""
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(raw_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-shortest",
        str(result_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(
            "모바일용 결과 영상 변환에 실패했습니다: " + completed.stderr[-500:]
        )


def _reset_tracker_state(tracker: LandmarkGridPositionTracker) -> None:
    tracker._locator.reset()
    tracker.current_cell = None
    tracker.current_label = None
    tracker.current_location = None
    tracker.current_direction = None
    tracker.current_matched_at = None
    tracker._logo_off_center_since = None
    tracker.logo_off_center = False
    tracker.pending_candidate = None
    tracker._locked_until = 0.0
    tracker.last_result = None


class LandmarkVisionEngine:
    """카메라/사진/영상 입력을 `LandmarkGridPositionTracker`로 판정해, 프론트
    (`App.jsx`)가 기대하는 JSON 모양으로 변환한다."""

    def __init__(self, result_dir: Path) -> None:
        self.result_dir = Path(result_dir)
        self.load_error: str | None = None
        self.grid: GridConfig | None = None
        self.locations: list[Location] = []
        self.by_name: dict[str, Location] = {}
        self.live_tracker: LandmarkGridPositionTracker | None = None
        self._photo_tracker: LandmarkGridPositionTracker | None = None
        self._video_tracker: LandmarkGridPositionTracker | None = None
        self.lock = threading.Lock()
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - 로딩 실패해도 서버 전체는 계속 뜨게 함
            self.load_error = str(exc)

    def _load(self) -> None:
        self.grid = load_grid_config(LOCATIONS_FILE)
        self.locations = load_locations(LOCATIONS_FILE)
        self.by_name = {loc.name: loc for loc in self.locations}
        self.live_tracker = LandmarkGridPositionTracker(
            self.grid,
            self.locations,
            weights=str(DEFAULT_WEIGHTS),
        )

    @property
    def photo_tracker(self) -> LandmarkGridPositionTracker | None:
        # 사진 한 장짜리 단발 판정: window=1이라 매 호출마다 버퍼가 통째로
        # 새로 채워져서 이전 사진의 흔적이 안 남는다. min_votes=1/approach_only
        # =False로 둬서 "여러 프레임 접근 추세 확인" 같은, 연속 프레임 전제
        # 조건 없이 즉시 판정한다.
        if self._photo_tracker is None and self.grid is not None:
            self._photo_tracker = LandmarkGridPositionTracker(
                self.grid,
                self.locations,
                weights=str(DEFAULT_WEIGHTS),
                window=1,
                min_votes=1,
                approach_only=False,
            )
        return self._photo_tracker

    @property
    def video_tracker(self) -> LandmarkGridPositionTracker | None:
        # 라이브 카메라(live_tracker)와 별도 인스턴스 - 영상 업로드 분석이
        # 진행 중인 실시간 카메라 세션의 현재 위치를 건드리면 안 되기 때문.
        if self._video_tracker is None and self.grid is not None:
            self._video_tracker = LandmarkGridPositionTracker(
                self.grid,
                self.locations,
                weights=str(DEFAULT_WEIGHTS),
            )
        return self._video_tracker

    # -- app.py의 /api/health --
    @property
    def model_ready(self) -> bool:
        return self.live_tracker is not None and self.live_tracker._locator.model is not None

    @property
    def ocr_ready(self) -> bool:
        return self.live_tracker is not None and self.live_tracker._locator.verifier is not None

    @property
    def model_kind(self) -> str:
        return "YOLOv8 (landmark_locator)"

    @property
    def model_path(self) -> Path:
        return DEFAULT_WEIGHTS

    @property
    def device(self) -> str:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def reset_tracking(self) -> None:
        if self.live_tracker is None:
            return
        with self.lock:
            _reset_tracker_state(self.live_tracker)

    # ---------------------------------------------------------------- cue 빌드
    def _landmarks_and_detections(
        self, tracker: LandmarkGridPositionTracker, out: dict, frame: np.ndarray
    ) -> tuple[list[dict], list[str]]:
        frame_width = frame.shape[1]
        landmarks = []
        detection_labels: list[str] = []
        for det in out.get("detections", []):
            name = tracker.resolve_location_name(det, frame)
            loc = self.by_name.get(name)
            display = MAP_LABELS.get(name, (loc.display_name or loc.name) if loc else name)
            snap = loc.snap_distance_m if (loc and loc.snap_distance_m) else 3.0
            distance_m = det["distance_m"]
            state = "near" if distance_m <= snap else "far"
            landmarks.append(
                {
                    "name": display,
                    "distance_m": round(distance_m, 2),
                    "direction": _direction_from_box(det["box"], frame_width),
                    "state": state,
                    "state_text": "가까움" if state == "near" else "거리가 있음",
                }
            )
            if display not in detection_labels:
                detection_labels.append(display)
        landmarks.sort(key=lambda item: item["distance_m"])
        return landmarks, detection_labels

    def _location_summary(self, tracker: LandmarkGridPositionTracker) -> dict:
        location = tracker.current_location
        if location is None:
            return {"status": "unknown", "label": "위치 인식 중", "detail": "", "candidates": []}
        return {
            "status": "localized",
            "label": location.display_name or location.name,
            "detail": location.description or "",
            "candidates": [],
        }

    def _position_payload(self, tracker: LandmarkGridPositionTracker) -> dict:
        location = tracker.current_location
        grid = tracker.grid
        return {
            "key": location.name if location else None,
            "label": (location.display_name or location.name) if location else (tracker.current_label or "위치 인식 중"),
            "description": (location.description or "") if location else "",
            "grid_cell": list(location.grid_cell) if location and location.grid_cell else None,
            "current_cell": list(tracker.current_cell) if tracker.current_cell else None,
            "snap_distance_m": location.snap_distance_m if location else None,
            "grid": {"cols": grid.cols, "rows": grid.rows, "cell_size_m": grid.cell_size_m} if grid else {},
            "landmarks": [
                {
                    "key": loc.name,
                    "label": MAP_LABELS.get(loc.name, loc.display_name or loc.name),
                    "grid_cell": list(loc.grid_cell) if loc.grid_cell else None,
                }
                for loc in tracker.landmarks
            ],
        }

    def _stability_payload(self, tracker: LandmarkGridPositionTracker, out: dict) -> dict:
        locator = tracker._locator
        votes = Counter(name for name, _, _ in locator.buf if name)
        vote_count = votes.most_common(1)[0][1] if votes else 0
        reason = out.get("reason") or ""
        confirmed = out.get("landmark") is not None and not reason.startswith("votes")
        return {
            "confirmed": confirmed,
            "count": vote_count,
            "required": locator.min_votes,
            "collected": len(locator.buf),
            "window": locator.window,
        }

    def _build_cue(
        self, tracker: LandmarkGridPositionTracker, frame: np.ndarray, time_seconds: float
    ) -> tuple[dict, dict]:
        """프레임 하나를 판정하고 (cue, Locator 원본 결과)를 반환한다.
        실시간 카메라/영상 타임라인이 공유하는 공통 경로."""
        tracker.update_from_frame(frame, time_seconds)
        out = tracker.last_result or {"detections": [], "landmark": None, "reason": ""}
        landmarks, detection_labels = self._landmarks_and_detections(tracker, out, frame)
        location = self._location_summary(tracker)
        position = self._position_payload(tracker)
        stability = self._stability_payload(tracker, out)
        described = tracker.describe()

        if described:
            command = described
            guidance = described
        elif landmarks:
            nearest = landmarks[0]
            command = None
            guidance = (
                f"{nearest['direction']} 약 {nearest['distance_m']:.1f}m에서 "
                f"{nearest['name']}이 보입니다."
            )
        else:
            # 위치 미확정 + 탐지물도 없음: 빈 문자열로 두고 프론트 자체 기본
            # 문구가 대신 뜨게 한다 (docs/setup_log.md 33번 항목과 동일한 정책).
            command = None
            guidance = ""

        cue = {
            "time_seconds": round(time_seconds, 3),
            "texts": [],
            "detections": detection_labels,
            "landmarks": landmarks,
            "location": location,
            "position": position,
            "command": command,
            "guidance": guidance,
            "stability": stability,
        }
        return cue, out

    def _draw_annotated(self, frame: np.ndarray, out: dict, headline: str | None) -> np.ndarray:
        annotated = frame.copy()
        for det in out.get("detections", []):
            x1, y1, x2, y2 = (int(v) for v in det["box"])
            distance_m = det.get("distance_m")
            near = distance_m is not None and distance_m <= 3.0
            color = (0, 200, 90) if near else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            distance_text = f" {distance_m:.1f}m" if distance_m is not None else ""
            label = f"{det['name']} {det['conf']:.2f}{distance_text}"
            cv2.putText(
                annotated, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        if headline:
            annotated = put_korean_text(annotated, headline, (14, 12), font_size=26, color_bgr=(255, 255, 255))
        return annotated

    # -------------------------------------------------------------- 실시간 카메라
    def analyze_frame_bytes(self, payload: bytes, time_seconds: float = 0.0) -> dict[str, Any]:
        if self.live_tracker is None:
            raise RuntimeError(self.load_error or "위치 추적 모델이 준비되지 않았습니다.")
        frame = decode_image(payload)
        with self.lock:
            cue, _out = self._build_cue(self.live_tracker, frame, time_seconds)
        return {
            "detections": [],
            "texts": [],
            "location": cue["location"],
            "cue": cue,
            "model": self.model_kind,
            "weights": self.model_path.name,
            "device": self.device,
        }

    # -------------------------------------------------------------- 사진 업로드
    def analyze_photo(self, image_path: Path, image_id: str) -> dict[str, Any]:
        tracker = self.photo_tracker
        if tracker is None:
            raise RuntimeError(self.load_error or "위치 추적 모델이 준비되지 않았습니다.")
        with self.lock:
            frame = _read_phone_image(image_path)
            cue, out = self._build_cue(tracker, frame, 0.0)
            headline = cue["command"] or cue["guidance"] or None
            annotated = self._draw_annotated(frame, out, headline)

            result_name = f"{image_id}.jpg"
            result_path = self.result_dir / result_name
            encoded_ok, encoded = cv2.imencode(".jpg", annotated)
            if not encoded_ok:
                raise RuntimeError(f"결과 이미지를 저장하지 못했습니다: {result_path}")
            result_path.write_bytes(encoded.tobytes())

            detections = [
                {
                    "label": MAP_LABELS.get(tracker.resolve_location_name(det, frame), det["name"]),
                    "confidence": round(float(det["conf"]), 4),
                    "bbox": [round(float(v), 1) for v in det["box"]],
                    "ocr_verified": False,
                    "localization_only": False,
                }
                for det in out.get("detections", [])
            ]

        return {
            "detections": detections,
            "texts": [],
            "location": cue["location"],
            "result_name": result_name,
            "model": self.model_kind,
            "weights": self.model_path.name,
            "device": self.device,
        }

    # -------------------------------------------------------------- 영상 업로드
    def analyze_video(self, video_path: Path, video_id: str, frame_step: int = 5) -> dict[str, Any]:
        tracker = self.video_tracker
        if tracker is None:
            raise RuntimeError(self.load_error or "위치 추적 모델이 준비되지 않았습니다.")
        frame_step = max(1, min(int(frame_step), 30))

        raw_path = self.result_dir / f"{video_id}_raw.mp4"
        result_path = self.result_dir / f"{video_id}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("업로드한 영상을 열 수 없습니다.")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if total_frames > 0 else 0.0
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("영상 해상도를 확인할 수 없습니다.")
        if duration_seconds > 300:
            capture.release()
            raise ValueError("현재는 5분 이하 영상만 업로드할 수 있습니다.")

        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("분석 결과 영상을 만들 수 없습니다.")

        detections_by_label: dict[str, dict] = {}
        timeline: list[dict] = []
        processed_frames = 0
        frame_index = 0
        last_headline: str | None = None

        try:
            with self.lock:
                _reset_tracker_state(tracker)
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index % frame_step == 0:
                        cue, out = self._build_cue(tracker, frame, frame_index / fps)
                        processed_frames += 1
                        last_headline = cue["command"] or cue["guidance"] or last_headline
                        for det in out.get("detections", []):
                            name = tracker.resolve_location_name(det, frame)
                            display = MAP_LABELS.get(name, det["name"])
                            confidence = round(float(det["conf"]), 4)
                            previous = detections_by_label.get(display)
                            if previous is None or confidence > previous["confidence"]:
                                occurrences = (previous or {}).get("occurrences", 0) + 1
                                detections_by_label[display] = {
                                    "label": display,
                                    "confidence": confidence,
                                    "bbox": [round(float(v), 1) for v in det["box"]],
                                    "ocr_verified": False,
                                    "localization_only": False,
                                    "occurrences": occurrences,
                                }
                            else:
                                previous["occurrences"] += 1
                        timeline.append(cue)
                        annotated = self._draw_annotated(frame, out, last_headline)
                    else:
                        annotated = self._draw_annotated(frame, {"detections": []}, last_headline)
                    writer.write(annotated)
                    frame_index += 1
        finally:
            capture.release()
            writer.release()

        if frame_index == 0:
            raw_path.unlink(missing_ok=True)
            raise ValueError("영상에 읽을 수 있는 프레임이 없습니다.")

        try:
            _transcode_for_mobile(raw_path, video_path, result_path)
        finally:
            raw_path.unlink(missing_ok=True)

        final_location = (
            timeline[-1]["location"]
            if timeline
            else {"status": "unknown", "label": "위치 불확실", "detail": "", "candidates": []}
        )

        return {
            "detections": sorted(
                detections_by_label.values(), key=lambda item: item["confidence"], reverse=True
            ),
            "texts": [],
            "location": final_location,
            "result_name": result_path.name,
            "model": self.model_kind,
            "weights": self.model_path.name,
            "device": self.device,
            "video": {
                "width": width,
                "height": height,
                "fps": round(fps, 3),
                "frames": frame_index,
                "duration_seconds": round(frame_index / fps, 2),
                "processed_frames": processed_frames,
                "frame_step": frame_step,
            },
            "timeline": timeline,
        }
