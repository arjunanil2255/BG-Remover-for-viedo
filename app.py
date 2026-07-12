"""
Flask web server for Video Background Removal.
Run:  python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import uuid
import time
import json
import threading
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

from remove_video_bg import remove_video_background, MODELS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory progress store: job_id -> {status, progress, total, message, output_path}
jobs = {}
jobs_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify(error="No video file provided"), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400

    job_id = uuid.uuid4().hex[:12]
    safe_name = secure_filename(f.filename) or f"input_{job_id}.mp4"
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")
    f.save(input_path)

    # Get video info
    cap = cv2.VideoCapture(input_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()

    with jobs_lock:
        jobs[job_id] = {
            "status": "uploaded",
            "input_path": input_path,
            "info": info,
            "filename": safe_name,
        }

    return jsonify(job_id=job_id, info=info)


@app.route("/api/upload-bg", methods=["POST"])
def upload_bg():
    if "bg_image" not in request.files:
        return jsonify(error="No image provided"), 400
    f = request.files["bg_image"]
    ext = os.path.splitext(secure_filename(f.filename) or "bg.jpg")[1] or ".jpg"
    name = f"bg_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    return jsonify(path=path)


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    if not job_id:
        return jsonify(error="Missing job_id"), 400

    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify(error="Unknown job"), 404
        if job.get("status") == "processing":
            return jsonify(error="Already processing"), 409

    # Settings
    model = data.get("model", "u2net_human_seg")
    bg = data.get("bg", None)
    if bg in ("", "null"):
        bg = None
    skip_frames = int(data.get("skip_frames", 0))
    erode = int(data.get("erode", 0))
    feather = int(data.get("feather", 2))
    transparent = bool(data.get("transparent", False))
    no_alpha_matting = bool(data.get("no_alpha_matting", False))
    fg_threshold = int(data.get("fg_threshold", 240))
    bg_threshold = int(data.get("bg_threshold", 10))

    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_output.mp4")

    with jobs_lock:
        jobs[job_id].update({
            "status": "processing",
            "progress": 0,
            "total": 0,
            "message": "Initializing...",
            "output_path": output_path,
        })

    def run_job():
        input_path = jobs[job_id]["input_path"]
        try:
            remove_video_background(
                input_path=input_path,
                output_path=output_path,
                model_name=model,
                bg_arg=bg,
                skip_frames=skip_frames,
                erode_px=erode,
                feather_px=feather,
                transparent=transparent,
                alpha_matting=not no_alpha_matting,
                alpha_matting_foreground_threshold=fg_threshold,
                alpha_matting_background_threshold=bg_threshold,
            )
            with jobs_lock:
                jobs[job_id].update({
                    "status": "done",
                    "progress": 100,
                    "message": "Complete!",
                })
        except Exception as e:
            with jobs_lock:
                jobs[job_id].update({
                    "status": "error",
                    "message": str(e),
                })

    t = threading.Thread(target=run_job, daemon=True)
    t.start()

    return jsonify(status="processing")


@app.route("/api/progress/<job_id>")
def progress(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify(error="Unknown job"), 404

    # Compute a rough percentage from frame count if available
    total = job.get("total", 0)
    # For progress, we try to read from tqdm's output via a shared variable
    # Since we can't easily intercept tqdm, we'll return status info
    return jsonify(
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0),
        message=job.get("message", ""),
    )


@app.route("/api/stream/<job_id>")
def stream_progress(job_id):
    """SSE endpoint that pushes progress updates."""
    def event_stream():
        while True:
            with jobs_lock:
                job = jobs.get(job_id)
                if not job:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Unknown job'})}\n\n"
                    break
                payload = {
                    "status": job.get("status", "unknown"),
                    "progress": job.get("progress", 0),
                    "message": job.get("message", ""),
                }
                if job.get("status") in ("done", "error"):
                    yield f"data: {json.dumps(payload)}\n\n"
                    break
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.5)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify(error="Unknown job"), 404
        if job.get("status") != "done":
            return jsonify(error="Not ready"), 400

    output_path = job.get("output_path", "")
    if not os.path.isfile(output_path):
        return jsonify(error="Output file not found"), 404

    original = job.get("filename", "output.mp4")
    name_no_ext = os.path.splitext(original)[0]
    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"{name_no_ext}_bg_removed.mp4",
    )


@app.route("/api/cleanup/<job_id>", methods=["POST"])
def cleanup(job_id):
    with jobs_lock:
        job = jobs.pop(job_id, None)
    if job:
        for key in ("input_path", "output_path"):
            p = job.get(key)
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    return jsonify(ok=True)


if __name__ == "__main__":
    print("\n  BG Remover Web UI")
    print("  http://localhost:5000\n")
    app.run(debug=True, port=5000)
