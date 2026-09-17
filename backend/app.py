import hmac
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from services.locations import LocationStore
from services.vision import VisionEngine

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
UPLOAD_DIR, RESULT_DIR, DATA_DIR = BASE_DIR / "uploads", BASE_DIR / "results", BASE_DIR / "data"
for directory in (UPLOAD_DIR, RESULT_DIR, DATA_DIR):
    directory.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}
DEMO_ACCESS_KEY = os.environ.get("DEMO_ACCESS_KEY", "").strip()


def create_app():
    app = Flask(
        __name__,
        static_folder=str(FRONTEND_DIST) if FRONTEND_DIST.exists() else None,
        static_url_path="",
    )
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
    vision = VisionEngine(RESULT_DIR)
    locations = LocationStore(DATA_DIR / "locations.json")

    def is_quick_tunnel_request():
        hostname = request.host.split(":", 1)[0].lower()
        return hostname.endswith(".trycloudflare.com")

    @app.before_request
    def protect_quick_tunnel():
        if not DEMO_ACCESS_KEY or not is_quick_tunnel_request():
            return None
        provided = request.args.get("key", "") or request.cookies.get(
            "demo_access", ""
        )
        if hmac.compare_digest(provided, DEMO_ACCESS_KEY):
            return None
        return jsonify(error="유효한 시연 접속 키가 필요합니다."), 403

    @app.after_request
    def remember_quick_tunnel_key(response):
        provided = request.args.get("key", "")
        if (
            DEMO_ACCESS_KEY
            and is_quick_tunnel_request()
            and hmac.compare_digest(provided, DEMO_ACCESS_KEY)
        ):
            response.set_cookie(
                "demo_access",
                DEMO_ACCESS_KEY,
                secure=True,
                httponly=True,
                samesite="Lax",
            )
        return response

    @app.get("/api/health")
    def health():
        return {
            "status": "ok" if vision.model_ready else "degraded",
            "model_ready": vision.model_ready,
            "ocr_ready": vision.ocr_ready,
            "ocr_enabled": False,
            "device": str(vision.device),
            "model": vision.model_kind,
            "weights": vision.model_path.name,
            "load_error": vision.load_error,
            "location_count": locations.count,
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
        image_id = uuid4().hex
        input_path = UPLOAD_DIR / f"{image_id}.{extension}"
        image.save(input_path)
        try:
            output = vision.analyze(input_path, image_id)
        except Exception as exc:
            return jsonify(error=f"이미지 분석 중 오류가 발생했습니다: {exc}"), 500
        location = output.get("location")
        if location is None:
            location = locations.match_ocr_results(output["texts"])
        if location is None:
            for room_type, room_number in locations.room_candidates_from_ocr(output["texts"]):
                location = locations.match_texts([f"{room_type}{room_number}"])
                if location:
                    break
        return jsonify(image_id=image_id, filename=image.filename,
                       detections=output["detections"], texts=output["texts"], location=location,
                       annotated_image_url=(f"/api/results/{output['result_name']}" if output["result_name"] else None),
                       model=output.get("model"), weights=output.get("weights"),
                       device=output.get("device"),
                       model_ready=vision.model_ready,
                       processing_ms=round((perf_counter() - started) * 1000))

    @app.post("/api/analyze-video")
    def analyze_video():
        started = perf_counter()
        video = request.files.get("video")
        if not video or not video.filename:
            return jsonify(error="영상 파일이 필요합니다."), 400
        extension = Path(secure_filename(video.filename)).suffix.lower().lstrip(".")
        if extension not in ALLOWED_VIDEO_EXTENSIONS:
            return jsonify(error="MP4, MOV, AVI, MKV, WEBM 또는 M4V 영상만 사용할 수 있습니다."), 400
        video_id = uuid4().hex
        input_path = UPLOAD_DIR / f"{video_id}.{extension}"
        video.save(input_path)
        try:
            frame_step = int(request.form.get("frame_step", 5))
            output = vision.analyze_video(input_path, video_id, frame_step=frame_step)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception as exc:
            return jsonify(error=f"영상 분석 중 오류가 발생했습니다: {exc}"), 500
        finally:
            input_path.unlink(missing_ok=True)
        return jsonify(
            video_id=video_id,
            filename=video.filename,
            media_type="video",
            detections=output["detections"],
            texts=output["texts"],
            location=output["location"],
            annotated_video_url=f"/api/results/{output['result_name']}",
            video=output["video"],
            timeline=output.get("timeline", []),
            model=output.get("model"),
            weights=output.get("weights"),
            device=output.get("device"),
            model_ready=vision.model_ready,
            processing_ms=round((perf_counter() - started) * 1000),
        )

    @app.post("/api/reset-tracking")
    def reset_tracking():
        vision.reset_tracking()
        return jsonify(status="ok")

    @app.post("/api/analyze-frame")
    def analyze_frame():
        started = perf_counter()
        frame = request.files.get("frame")
        if not frame:
            return jsonify(error="분석할 영상 프레임이 필요합니다."), 400
        try:
            time_seconds = max(0.0, float(request.form.get("time_seconds", 0.0)))
            output = vision.analyze_frame_bytes(frame.read(), time_seconds)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception as exc:
            return jsonify(error=f"재생 프레임 분석 중 오류가 발생했습니다: {exc}"), 500
        return jsonify(
            **output,
            processing_ms=round((perf_counter() - started) * 1000),
        )

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify(error="파일이 너무 큽니다. 300MB 이하 파일을 선택해 주세요."), 413

    @app.get("/api/results/<path:filename>")
    def result_file(filename):
        return send_from_directory(RESULT_DIR, filename)

    @app.get("/api/locations")
    def list_locations():
        return jsonify(locations.all())

    @app.post("/api/locations")
    def save_location():
        payload = request.get_json(silent=True) or {}
        if any(not str(payload.get(field, "")).strip() for field in ("node_id", "place_name")):
            return jsonify(error="node_id와 place_name이 필요합니다."), 400
        return jsonify(locations.upsert(payload)), 201

    @app.get("/")
    def frontend_index():
        if not FRONTEND_DIST.exists():
            return jsonify(
                error="frontend/dist가 없습니다. frontend에서 npm run build를 실행하세요."
            ), 503
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
