"""
dino_server.py — Jetson inference server for RockDinosaurMartin
===============================================================
The Arduino (or any HTTP client) POSTs a JPEG to /classify and gets back JSON.

Architecture
------------
  Arduino  ──(HTTP POST /classify)──►  Jetson (this server)
                                            │
                                     ResNet-18 classifier
                                            │
                                      JSON response
                                            │
  Arduino  ◄──────────────────────── {"class":"triceratops","confidence":0.91,...}

Endpoints
---------
  GET  /health          → {"status":"ok","classes":[...]}
  POST /classify        → classify a JPEG you send (body = raw JPEG bytes,
                           or multipart field "image")

Usage on the Jetson
-------------------
  # 1. Install extra deps
  pip install flask

  # 2. Run
  python dino_server.py [--port 5000] [--device cuda]
                        [--checkpoint artifacts/dinosaur_classifier.pt]
                        [--classes artifacts/class_names.json]
                        [--host 0.0.0.0]

  # 3. Test
  curl http://<jetson-ip>:5000/health
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

# ── lazy PyTorch ───────────────────────────────────────────────────────────────
torch: Any = None
nn: Any = None
F: Any = None
models: Any = None
transforms: Any = None

def _import_heavy() -> None:
    global torch, nn, F, models, transforms
    if torch is not None:
        return
    print("Loading PyTorch…", flush=True)
    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torchvision import models as _m, transforms as _tr
    torch, nn, F, models, transforms = _t, _nn, _F, _m, _tr
    print("PyTorch ready.", flush=True)


# ── model helpers ──────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> Any:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_classifier(checkpoint_path: Path, classes_json: Path, device: Any):
    _import_heavy()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not classes_json.is_file():
        raise FileNotFoundError(f"Class list not found: {classes_json}")
    print(f"Loading weights from {checkpoint_path.name}…", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    num_classes = int(ckpt["num_classes"])
    image_size = int(ckpt.get("image_size", 224))
    with open(classes_json, encoding="utf-8") as f:
        meta = json.load(f)
    class_names: list[str] = meta["class_names"]
    model = build_model(num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model ready: {num_classes} classes, image_size={image_size}", flush=True)
    return model, class_names, image_size


def make_transform(image_size: int) -> Any:
    _import_heavy()
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def center_square_roi(h: int, w: int, fraction: float = 0.42,
                      max_side: int = 480) -> tuple[int, int, int, int]:
    side = min(int(min(w, h) * fraction), max_side, min(w, h))
    side = max(32, side)
    cx, cy = w // 2, h // 2
    half = side // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + side, y1 + side
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return x1, y1, x2, y2


def classify_frame(frame_bgr: np.ndarray, model, class_names, tfm, device,
                   roi_fraction: float = 0.42, max_roi_side: int = 480) -> dict:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = center_square_roi(h, w, roi_fraction, max_roi_side)
    crop = frame_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    conf, pred_i = float(probs.max().item()), int(probs.argmax().item())
    all_scores = {name: round(float(probs[i].item()), 4)
                  for i, name in enumerate(class_names)}
    return {
        "class": class_names[pred_i],
        "class_index": pred_i,
        "confidence": round(conf, 4),
        "scores": all_scores,
        "roi": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "timestamp": time.time(),
    }


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Globals filled in main()
_model = None
_class_names: list[str] = []
_tfm = None
_device = None
_roi_fraction: float = 0.42
_max_roi_side: int = 480


def _classify_bgr(frame_bgr: np.ndarray) -> dict:
    return classify_frame(
        frame_bgr, _model, _class_names, _tfm, _device,
        _roi_fraction, _max_roi_side,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "classes": _class_names,
        "num_classes": len(_class_names),
        "device": str(_device),
    })


@app.route("/classify", methods=["POST"])
def classify_post():
    """
    Classify an image sent from the client.
    Accepts:
      - Raw JPEG/PNG bytes in the request body (Content-Type: image/jpeg)
      - Multipart form with a field named 'image'
    """
    img_bytes: bytes | None = None

    # Multipart upload
    if request.files and "image" in request.files:
        img_bytes = request.files["image"].read()
    # Raw body
    elif request.data:
        img_bytes = request.data

    if not img_bytes:
        return jsonify({"error": "No image provided"}), 400

    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = _classify_bgr(frame)
    return jsonify(result)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Jetson HTTP server: classify dinos for Arduino clients."
    )
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind address (default 0.0.0.0 = all interfaces)")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | auto")
    p.add_argument("--checkpoint", type=Path,
                   default=root / "artifacts" / "dinosaur_classifier.pt")
    p.add_argument("--classes", type=Path,
                   default=root / "artifacts" / "class_names.json")
    p.add_argument("--roi-fraction", type=float, default=0.42)
    p.add_argument("--max-roi-side", type=int, default=480)
    return p.parse_args()


def main() -> None:
    global _model, _class_names, _tfm, _device
    global _roi_fraction, _max_roi_side

    args = parse_args()
    _roi_fraction = args.roi_fraction
    _max_roi_side = args.max_roi_side

    _import_heavy()

    if args.device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(args.device)
    print(f"Device: {_device}", flush=True)

    _model, _class_names, img_size = load_classifier(
        args.checkpoint, args.classes, _device
    )
    _tfm = make_transform(img_size)

    print(f"\nServer starting on http://{args.host}:{args.port}", flush=True)
    print(f"  GET  http://<jetson-ip>:{args.port}/health", flush=True)
    print(f"  POST http://<jetson-ip>:{args.port}/classify  (body=JPEG)", flush=True)
    print()

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
