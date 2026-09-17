"""classroom-locator 웹 데모 서버.

실제 위치 판정은 ``vision_bridge.VisionBridge``를 통해 classroom-locator
파이프라인이 처리한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from vision_bridge import VisionBridge, decode_image

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(FRONTEND_DIST) if FRONTEND_DIST.exists() else None,
        static_url_path="",
    )
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
    vision = VisionBridge()

    @app.get("/api/health")
    def health():
        return {
            "status": "ok" if vision.model_ready else "degraded",
            "model_ready": vision.model_ready,
            "device": vision.device,
            "load_error": vision.load_error,
            "location_count": len(vision.locations),
        }

    @app.post("/api/analyze")
    def analyze():
        started = perf_counter()
        image = request.files.get("image")
        if not image or not image.filename:
            return jsonify(error="이미지 파일이 필요합니다."), 400
        extension = Path(secure_filename(image.filename)).suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify(error="JPG, PNG, WEBP 또는 BMP 이미지만 사용할 수 있습니다."), 400
        if not vision.model_ready:
            return jsonify(error=vision.load_error or "모델이 준비되지 않았습니다."), 503

        try:
            frame = decode_image(image.read())
            output = vision.analyze_photo(frame)
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"이미지 분석 중 오류가 발생했습니다: {exc}"), 500

        result_name = f"{uuid4().hex}.jpg"
        cv2.imwrite(str(RESULT_DIR / result_name), output["annotated_image"])

        return jsonify(
            detections=output["detections"],
            location=output["location"],
            annotated_image_url=f"/api/results/{result_name}",
            model_ready=vision.model_ready,
            processing_ms=round((perf_counter() - started) * 1000),
        )

    @app.post("/api/analyze-frame")
    def analyze_frame():
        started = perf_counter()
        frame_file = request.files.get("frame")
        if not frame_file:
            return jsonify(error="분석할 영상 프레임이 필요합니다."), 400
        if not vision.model_ready:
            return jsonify(error=vision.load_error or "모델이 준비되지 않았습니다."), 503
        try:
            time_seconds = max(0.0, float(request.form.get("time_seconds", 0.0)))
            frame = decode_image(frame_file.read())
            cue = vision.analyze_live_frame(frame, time_seconds)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"재생 프레임 분석 중 오류가 발생했습니다: {exc}"), 500
        return jsonify(cue=cue, processing_ms=round((perf_counter() - started) * 1000))

    @app.post("/api/reset-tracking")
    def reset_tracking():
        vision.reset_live_tracking()
        return jsonify(status="ok")

    @app.post("/api/analyze-video")
    def analyze_video():
        return jsonify(
            error="영상 업로드 분석은 아직 지원하지 않습니다. 사진 촬영 또는 실시간 카메라를 사용해 주세요."
        ), 501

    @app.get("/api/results/<path:filename>")
    def result_file(filename):
        return send_from_directory(RESULT_DIR, filename)

    @app.get("/api/locations")
    def list_locations():
        return jsonify(
            [
                {
                    "name": loc.name,
                    "display_name": loc.display_name,
                    "grid_cell": loc.grid_cell,
                    "snap_distance_m": loc.snap_distance_m,
                }
                for loc in vision.locations
            ]
        )

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify(error="파일이 너무 큽니다. 300MB 이하 파일을 선택해 주세요."), 413

    @app.get("/")
    def frontend_index():
        if not FRONTEND_DIST.exists():
            return jsonify(error="frontend/dist가 없습니다. frontend에서 npm run build를 실행하세요."), 503
        return send_from_directory(FRONTEND_DIST, "index.html")

    @app.get("/<path:filename>")
    def frontend_file(filename):
        if FRONTEND_DIST.exists() and (FRONTEND_DIST / filename).is_file():
            return send_from_directory(FRONTEND_DIST, filename)
        if FRONTEND_DIST.exists():
            return send_from_directory(FRONTEND_DIST, "index.html")
        return jsonify(error="frontend/dist가 없습니다."), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
