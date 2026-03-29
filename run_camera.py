"""
Live webcam inference using the ResNet-18 classifier.

Classifies only a **center square ROI** (crosshair region), not the full frame—so the
model answers “what’s in the middle of the screen?” Place the toy in that region.

Expects artifacts/dinosaur_classifier.pt and artifacts/class_names.json (or --checkpoint / --classes).

For best results, train on **similar center crops** (same camera FOV and ROI fraction).

Capture requests **1920×1080**, then after **rotation** the frame is **center-cropped to 16:9 landscape** (edges removed as needed). The classifier ROI is a centered square **at most 480×480** px.

Defaults: **camera index 3**, **90°** rotation. Quit: Q or Esc. Switch camera: keys **0–9**. **R**: cycle rotation 90°.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def _try_windows_directshow_names() -> list[str] | None:
    """Optional friendly device names on Windows (same index order as OpenCV usually)."""
    if sys.platform != "win32":
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph

        return FilterGraph().get_input_devices()
    except Exception:
        return None


def probe_cameras(
    max_index: int = 10,
    dshow_names: list[str] | None = None,
) -> list[tuple[int, str]]:
    """Return (index, description) for each working camera index."""
    if dshow_names is None:
        dshow_names = _try_windows_directshow_names()
    found: list[tuple[int, str]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        try:
            if not cap.isOpened():
                continue
            backend = cap.getBackendName()
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if dshow_names is not None and i < len(dshow_names):
                name = dshow_names[i]
                extra = f"{name} ({backend})"
            else:
                extra = backend
            if fw > 0 and fh > 0:
                extra = f"{extra}, {fw}x{fh}"
            found.append((i, extra))
        finally:
            cap.release()
    return found


def print_camera_list(max_index: int = 10) -> None:
    dshow = _try_windows_directshow_names()
    rows = probe_cameras(max_index, dshow_names=dshow)
    print("Cameras (number key → index):")
    if not rows:
        print(f"  (none found among indices 0–{max_index - 1})")
        return
    for idx, label in rows:
        print(f"  [{idx}] {label}")
    if sys.platform == "win32" and dshow is None:
        print(
            "  Tip: pip install pygrabber  →  show USB camera product names on Windows."
        )


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch != "resnet18":
        raise ValueError(f"Unsupported arch in checkpoint: {arch!r} (expected 'resnet18')")
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_classifier(
    checkpoint_path: Path,
    classes_json: Path,
    device: torch.device,
) -> tuple[nn.Module, list[str], int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not classes_json.is_file():
        raise FileNotFoundError(f"Class list not found: {classes_json}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    num_classes = int(ckpt["num_classes"])
    arch = ckpt.get("arch", "resnet18")
    image_size = int(ckpt.get("image_size", 224))

    with open(classes_json, encoding="utf-8") as f:
        meta = json.load(f)
    class_names: list[str] = meta["class_names"]
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names.json has {len(class_names)} entries but checkpoint expects {num_classes}"
        )

    model = build_model(arch, num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, class_names, image_size


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def crop_to_horizontal_16_9(frame: np.ndarray) -> np.ndarray:
    """
    Center-crop to 16:9 **landscape** (width : height = 16 : 9).
    After rotation, portrait frames become landscape by cropping the top/bottom (or sides if too wide).
    """
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame
    target = 16.0 / 9.0
    aspect = w / h
    if abs(aspect - target) < 1e-4:
        return frame
    if aspect > target:
        new_w = max(1, int(round(h * target)))
        new_w = min(new_w, w)
        x0 = (w - new_w) // 2
        return frame[:, x0 : x0 + new_w]
    new_h = max(1, int(round(w / target)))
    new_h = min(new_h, h)
    y0 = (h - new_h) // 2
    return frame[y0 : y0 + new_h, :]


def center_square_roi(
    h: int,
    w: int,
    fraction: float,
    max_roi_side: int = 480,
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) inclusive-exclusive crop box; square side ≤ max_roi_side and ≤ frame."""
    side = int(min(w, h) * fraction)
    side = min(side, max_roi_side, min(w, h))
    side = max(32, side)
    cx, cy = w // 2, h // 2
    half = side // 2
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + side
    y2 = y1 + side
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= x2 - w
        x2 = w
    if y2 > h:
        y1 -= y2 - h
        y2 = h
    x1 = max(0, x1)
    y1 = max(0, y1)
    return x1, y1, x2, y2


def crop_to_tensor(
    frame_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    tfm: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty ROI crop")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return tfm(pil).unsqueeze(0).to(device)


def apply_rotation(frame: np.ndarray, rot_k: int) -> np.ndarray:
    """Rotate by rot_k * 90° clockwise (rot_k in 0..3)."""
    rot_k = rot_k % 4
    if rot_k == 0:
        return frame
    if rot_k == 1:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot_k == 2:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def draw_roi_crosshair(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_rect: tuple[int, int, int] = (0, 255, 255),
    color_cross: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw ROI rectangle and horizontal/vertical cross through its center (BGR)."""
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color_rect, 2, lineType=cv2.LINE_AA)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.line(frame, (x1, cy), (x2 - 1, cy), color_cross, 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (cx, y1), (cx, y2 - 1), color_cross, 1, lineType=cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Classify the center square ROI (crosshair) with the dinosaur classifier."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "artifacts" / "dinosaur_classifier.pt",
        help="Path to dinosaur_classifier.pt",
    )
    p.add_argument(
        "--classes",
        type=Path,
        default=root / "artifacts" / "class_names.json",
        help="Path to class_names.json",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=3,
        help="OpenCV camera index (default 3).",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu, or auto",
    )
    p.add_argument(
        "--roi-fraction",
        type=float,
        default=0.42,
        help="Center square side as fraction of min(w,h); side is also capped by --max-roi-side (default 480).",
    )
    p.add_argument(
        "--ema",
        type=float,
        default=0.88,
        help="EMA factor for softmax smoothing (0 disables). Higher = smoother, laggier.",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum top-1 probability to show the label (else show uncertain).",
    )
    p.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Run the model every N frames (1 = every frame).",
    )
    p.add_argument(
        "--max-camera-index",
        type=int,
        default=10,
        help="Probe indices 0..N-1 when listing cameras and for hotkeys 0-9.",
    )
    p.add_argument(
        "--capture-width",
        type=int,
        default=1920,
        help="Requested capture width (default 1920 for 1080p).",
    )
    p.add_argument(
        "--capture-height",
        type=int,
        default=1080,
        help="Requested capture height (default 1080).",
    )
    p.add_argument(
        "--max-roi-side",
        type=int,
        default=480,
        help="Maximum center ROI square size in pixels (default 480).",
    )
    return p.parse_args()


def configure_capture_resolution(
    cap: cv2.VideoCapture, width: int, height: int
) -> tuple[int, int]:
    """Request resolution; return (width, height) as reported by the device."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return rw, rh


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    print("Device:", device)
    print(
        "ROI: fraction",
        args.roi_fraction,
        "| max square",
        args.max_roi_side,
        "| capture request",
        f"{args.capture_width}x{args.capture_height}",
        "| output after rotation: 16:9 crop",
    )
    print()
    print_camera_list(args.max_camera_index)
    print()

    try:
        model, class_names, image_size = load_classifier(
            args.checkpoint, args.classes, device
        )
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    tfm = make_transform(image_size)
    camera_index = int(args.camera)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}", file=sys.stderr)
        return 1

    rw, rh = configure_capture_resolution(
        cap, args.capture_width, args.capture_height
    )
    print(
        f"Camera {camera_index}: requested {args.capture_width}x{args.capture_height}, "
        f"reports {rw}x{rh}",
    )

    ema_probs: torch.Tensor | None = None
    frame_idx = 0
    last_label = "—"
    last_conf = 0.0
    rot_k = 1  # 0,1,2,3 → 0°,90°,180°,270° clockwise; default 1 = 90° CW

    print("Press Q or Esc to quit. Keys 0–9 switch camera. R rotates 90°.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed; exiting.", file=sys.stderr)
            break

        frame = apply_rotation(frame, rot_k)
        frame = crop_to_horizontal_16_9(frame)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = center_square_roi(
            h, w, args.roi_fraction, max_roi_side=args.max_roi_side
        )
        roi_side = x2 - x1

        frame_idx += 1
        run_infer = frame_idx % max(1, args.every_n) == 0

        if run_infer:
            x = crop_to_tensor(frame, x1, y1, x2, y2, tfm, device)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)[0]

            if args.ema > 0 and ema_probs is not None:
                ema_probs = args.ema * ema_probs + (1.0 - args.ema) * probs
            else:
                ema_probs = probs

            p = ema_probs
            conf, pred_i = float(p.max().item()), int(p.argmax().item())
            label = class_names[pred_i]

            if conf < args.min_confidence:
                last_label = "uncertain"
                last_conf = conf
            else:
                last_label = label
                last_conf = conf

        draw_roi_crosshair(frame, x1, y1, x2, y2)

        text = f"ROI {roi_side}x{roi_side}: {last_label}  ({last_conf:.2f})"
        cv2.putText(
            frame,
            text,
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"cam {camera_index}  |  rot {rot_k * 90}°  |  {w}x{h} 16:9  |  r rotate  0-9  q/esc",
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("dinosaur classifier (center ROI)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("r"), ord("R")):
            rot_k = (rot_k + 1) % 4
            ema_probs = None
            frame_idx = 0
            print(f"Rotation: {rot_k * 90}° clockwise (relative to sensor)")

        if ord("0") <= key <= ord("9"):
            new_idx = key - ord("0")
            if new_idx != camera_index:
                trial = cv2.VideoCapture(new_idx)
                if trial.isOpened():
                    tw, th = configure_capture_resolution(
                        trial, args.capture_width, args.capture_height
                    )
                    cap.release()
                    cap = trial
                    camera_index = new_idx
                    ema_probs = None
                    frame_idx = 0
                    print(
                        f"Switched to camera {camera_index}, "
                        f"requested {args.capture_width}x{args.capture_height}, reports {tw}x{th}"
                    )
                else:
                    trial.release()
                    print(f"Camera index {new_idx} is not available.", file=sys.stderr)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
