"""RoadSense backend: serves the UI and exposes the repo's YOLOv8 model.

Setup (inside a clone of Anoushka-kanchi/road-damage-detection):
    mkdir weights && mv best.pt weights/best.pt     # inference.py loads weights/best.pt
    pip install flask pillow numpy ultralytics torch
    mkdir static && cp index.html static/index.html
    python app.py                                   # open http://localhost:5000
"""
import io, time
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

import inference  # the repo's own inference.py

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
MODEL = inference.load(DEVICE)  # loaded once at startup
NAMES = [str(MODEL.yolo.names[i]) for i in sorted(MODEL.yolo.names)]  # real class labels from best.pt


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", device=DEVICE, imgsz=inference.IMGSZ, names=NAMES)


@app.post("/api/predict")
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No image uploaded (field name: image)."), 400
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(f.read()))).convert("RGB")
    except Exception:
        return jsonify(error="Could not read that file as an image."), 400

    t0 = time.time()
    dets = inference.predict(MODEL, np.array(img))  # (cls, score, x1, y1, x2, y2)
    return jsonify(
        width=img.width, height=img.height,
        ms=round((time.time() - t0) * 1000), device=DEVICE,
        detections=[dict(cls=c, score=s, box=[x1, y1, x2, y2]) for c, s, x1, y1, x2, y2 in dets],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
