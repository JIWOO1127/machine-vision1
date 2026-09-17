"""웹캠으로부터 실시간 프레임을 받아 위치를 추적하는 모드."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from ..localization import Location, load_grid_config
from ..utils.image_utils import draw_detections, put_korean_text
from ..utils.logger import setup_logger
from .core import LocatorPipeline
from .grid_tracker import SeojiwooGridPositionTracker, render_grid_image
from .locking import LocationLocker

MAX_DISPLAY_DIM = 900


def _resize_for_display(frame: Any, max_dim: int = MAX_DISPLAY_DIM) -> Any:
    """화면(모니터) 밖으로 창이 넘쳐서 잘리는 걸 막기 위해, 세로/가로 중 큰 쪽이
    max_dim을 넘으면 비율을 유지한 채 축소합니다. 탐지 자체는 원본 해상도로
    이미 끝난 뒤라 정확도에는 영향 없습니다.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def _build_guidance_lines(location: Location) -> list[str]:
    name = location.display_name or location.name
    lines = [f"현재 위치는 {name} 앞입니다."]
    if location.guidance:
        lines.append(location.guidance)
    return lines


def run_realtime(config: dict[str, Any]) -> None:
    log_config = config.get("logging", {})
    logger = setup_logger(log_dir=log_config.get("log_dir"), level=log_config.get("level", "INFO"))
    pipeline = LocatorPipeline(config)

    rt_config = config.get("realtime", {})
    camera_index = rt_config.get("camera_index", 0)
    frame_width = rt_config.get("frame_width", 1280)
    frame_height = rt_config.get("frame_height", 720)
    process_every_n_frames = rt_config.get("process_every_n_frames", 5)

    # 위치가 바뀔 때마다(=화면에 새 안내 문구가 뜰 때마다) 그 순간의 탐지 결과를
    # 저장해서, "왜 이게 이 위치로 인식됐는지" 나중에 화면으로 확인할 수 있게 함.
    # --snapshot/--no-snapshot(CLI) 또는 realtime.save_snapshots(설정 파일)로 끄고 켤 수 있음.
    save_snapshots = rt_config.get("save_snapshots", True)
    snapshot_dir = Path(config.get("pipeline", {}).get("output_dir", "outputs/results")) / "realtime_snapshots"
    if save_snapshots:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 격자 지도(SeojiwooGridPositionTracker) 준비: locations.yaml에 grid 설정과
    # grid_cell/snap_distance_m이 있으면 활성화. 없으면(또는 --no-grid-map/
    # realtime.show_grid_map: false면) 그리드 창은 표시 안 함.
    # 거리 판정은 external/seojiwoo_core의 Locator 로직(REAL_SIZE 실측 물리크기
    # 기반 핀홀 거리추정 + 최근 프레임 다수결 투표 + 접근추세 확인)을 그대로
    # 쓰고, "계산된 거리와 가장 가까운 격자점 찾기"만 우리 쪽 로직을 씀.
    show_grid_map = rt_config.get("show_grid_map", True)
    loc_config = config["localization"]
    grid_config = load_grid_config(loc_config["locations_file"]) if show_grid_map else None
    grid_tracker: SeojiwooGridPositionTracker | None = None
    if grid_config is not None:
        # detector.weights와 동일한 모델을 씀 (final_best.pt는 이 팀원 저장소의
        # v1_best.pt와 해시까지 동일한 5클래스 모델 - docs/setup_log.md 18~19번 참고).
        # logo 클래스는 이 5클래스 모델엔 없어서, 있으면 6클래스 모델
        # (final_best_v2_with_logo.pt)을 logo 전용으로 같이 돌림.
        weights_path = Path(config["detector"]["weights"])
        logo_weights_path = weights_path.with_name("final_best_v2_with_logo.pt")
        grid_tracker = SeojiwooGridPositionTracker(
            grid_config,
            pipeline.locations,
            weights=str(weights_path),
            logo_weights=str(logo_weights_path) if logo_weights_path.exists() else None,
        )

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다 (index={camera_index})")

    logger.info("실시간 위치 추적을 시작합니다. 'q'를 누르면 종료됩니다.")

    frame_count = 0
    current_lines = ["위치 인식 중..."]
    locker = LocationLocker()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("프레임을 읽어오지 못했습니다.")
                break

            frame_count += 1
            annotated = frame

            if frame_count % process_every_n_frames == 0:
                result = pipeline.process_image(frame)
                annotated = draw_detections(frame, result.detections)

                now = time.monotonic()
                current, switched = locker.update(result.match, now)

                if current:
                    current_lines = _build_guidance_lines(current.location)

                if switched:
                    logger.info(" ".join(current_lines))

                    if save_snapshots:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snapshot_path = (
                            snapshot_dir / f"{timestamp}_{current.location.name}_score{current.score:.0f}.jpg"
                        )
                        cv2.imwrite(str(snapshot_path), annotated)
                        logger.info(f"인식 순간 스냅샷 저장: {snapshot_path}")

                if grid_tracker is not None:
                    grid_tracker.update_from_frame(frame, time.monotonic())
                    grid_img = render_grid_image(
                        grid_tracker.grid,
                        grid_tracker.current_cell,
                        grid_tracker.current_label,
                        landmarks=grid_tracker.landmarks,
                    )
                    cv2.imshow("classroom-locator - 격자 위치", grid_img)

            for i, line in enumerate(current_lines):
                annotated = put_korean_text(annotated, line, (20, 20 + i * 40), font_size=28)
            cv2.imshow("classroom-locator (press q to quit)", _resize_for_display(annotated))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
