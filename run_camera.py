"""
Live webcam inference using the ResNet-18 classifier.

Classifies a **square ROI** (crosshair), not the full frame—by default **centered horizontally
and aligned to the bottom** of the view (``--roi-vertical``). Place the toy in that region.

Expects artifacts/dinosaur_classifier.pt and artifacts/class_names.json (or --checkpoint / --classes).

For best results, train on **similar center crops** (same camera FOV and ROI fraction).

Capture requests **1920×1080**, then after **rotation** the frame is **center-cropped to 16:9 landscape** (edges removed as needed). The classifier ROI is a square **at most 480×480** px, **centered horizontally** and by default **aligned to the bottom** of the view (see `--roi-vertical`).

Defaults: **camera index 3**, **90°** rotation, **`--device cpu`** (avoids slow CUDA probing; use **`--device auto`** or **`cuda`** for GPU). **Serial** is prompted at startup (empty = camera only), except **`--seek COLOR`** / **`--shoot-mission`** headless modes (port required). Quit: Q/Esc. **0–9** cameras. **P** view rotate. **R/B/G/Y** servo seek. **`--seek COLOR`**, **`--shoot-mission`**, **`--seek-from-state`**, or **`--mission-chain`** (preview: lock → laser ``--laser-fire-sec`` → clear bit → seek next) tie the camera to shared mission bits. With **`.env`** from **`setup_and_test_voice.py`**, caveman lines from **`state.PHRASES`** play during hunts unless **`--no-voice`**. Flash **sketch_mar28a.ino** (115200 baud).
"""

from __future__ import annotations

import sys
from typing import Any

print("run_camera: starting (OpenCV first; PyTorch loads later — faster startup)…", flush=True)

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import serial
except ImportError:
    serial = None  # type: ignore[misc, assignment]

# Populated by _import_heavy() — defer PyTorch until after serial prompt / optional camera scan.
torch: Any = None
nn: Any = None
F: Any = None
models: Any = None
transforms: Any = None

print("run_camera: light imports OK.", flush=True)


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _voice_line(args: argparse.Namespace, phrase_key: str) -> None:
    """Play ``state.PHRASES[phrase_key]`` via ElevenLabs when ``--no-voice`` is not set."""
    if getattr(args, "no_voice", False):
        print(f"[voice] run_camera SKIPPED (--no-voice): {phrase_key!r}", flush=True)
        return
    try:
        from caveman_voice import _voice_log, speak_phrase_key

        _voice_log(f"run_camera PLAY: {phrase_key!r}")
        speak_phrase_key(phrase_key, enabled=True)
    except Exception as e:
        print(f"[voice] run_camera error ({phrase_key}): {e}", flush=True)


def _import_heavy() -> None:
    """Load PyTorch/torchvision once (often the slowest part of startup)."""
    global torch, nn, F, models, transforms
    if torch is not None:
        return
    _progress("Loading PyTorch & torchvision (first time can take several seconds)…")
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torchvision import models as _models, transforms as _transforms

    torch, nn, F, models, transforms = _torch, _nn, _F, _models, _transforms
    _progress("PyTorch ready.")


# Servo (sketch_mar28a.ino)
SERIAL_BAUD = 115200


def build_sweep_angles(step: int) -> list[int]:
    """0→180→0 ping-pong by step (degrees)."""
    step = max(1, step)
    fwd = list(range(0, 181, step))
    back = list(range(180 - step, -1, -step))
    return fwd + back


@dataclass
class SeekState:
    target_i: int
    label: str
    sweep_idx: int = 0
    hits: int = 0
    settle_until: float = 0.0
    frames_at_angle: int = 0
    # Earliest time to advance to next angle (after servo settle + scoring budget).
    next_angle_time: float = 0.0


def prompt_serial_port() -> str | None:
    print("Servo serial port (sketch_mar28a.ino @ 115200). Leave empty for camera-only.")
    line = input("Port (e.g. COM3): ").strip()
    return line if line else None


def open_servo_serial(port: str, wait_after_open: float = 0.5):
    if serial is None:
        print("Install pyserial: pip install pyserial", file=sys.stderr)
        return None
    try:
        _progress(f"Opening serial {port!r} @ {SERIAL_BAUD} baud…")
        ser = serial.Serial(port, SERIAL_BAUD, timeout=0.1)
        if wait_after_open > 0:
            _progress(f"Serial open; waiting {wait_after_open}s for USB/Arduino reset…")
            time.sleep(wait_after_open)
        _progress("Serial ready.")
        return ser
    except Exception as e:
        print(e, file=sys.stderr)
        return None


def send_servo_angle(ser, angle: int) -> None:
    ser.write(f"{int(angle)}\n".encode("ascii"))
    ser.flush()


def send_laser(ser, on: bool) -> None:
    """Arduino sketch: L1 = laser on (pin A2), L0 = off."""
    ser.write(b"L1\n" if on else b"L0\n")
    ser.flush()


def laser_on_for_target(
    probs: Any,
    target_i: int,
    *,
    min_prob: float,
) -> bool:
    """True iff ROI softmax says target class is top-1 and P(target) >= min_prob."""
    pi = int(probs.argmax().item())
    return pi == target_i and float(probs[target_i].item()) >= min_prob


def class_index_for_color(class_names: list[str], color: str) -> int | None:
    c = color.lower().strip()
    for i, name in enumerate(class_names):
        if name.lower() == c:
            return i
    return None


def run_headless_seek(
    ser,
    cap: cv2.VideoCapture,
    model: Any,
    class_names: list[str],
    tfm: Any,
    device: Any,
    target_color: str,
    rot_k: int,
    args: argparse.Namespace,
) -> int:
    """
    Blocking sweep (no GUI): same pipeline as live seek — rotate, 16:9 crop, ROI — until
    target class hits confidence or sweeps complete.
    """
    ix = class_index_for_color(class_names, target_color)
    if ix is None:
        print(f"No class '{target_color}' in {class_names}", file=sys.stderr)
        return 1
    if class_names[ix].lower() == "none":
        print("Cannot seek class 'none'.", file=sys.stderr)
        return 1

    step = max(1, args.seek_angle_step)
    angles_forward = list(range(0, 181, step))
    angles_back = list(range(180, -1, -step))
    sweep_lists = [angles_forward]
    if not args.seek_forward_only:
        sweep_lists.append(angles_back)

    target_i = ix
    print("Headless seek:", class_names[target_i], "| sweeps:", len(sweep_lists))

    send_laser(ser, False)
    last_laser_on: bool | None = False

    found_angle: int | None = None

    try:
        for sweep in sweep_lists:
            for angle in sweep:
                send_servo_angle(ser, angle)
                time.sleep(args.seek_settle)

                hits = 0
                fidx = 0
                for _ in range(args.seek_max_frames):
                    ok, frame = cap.read()
                    if not ok:
                        print("Camera read failed.", file=sys.stderr)
                        return 1

                    fidx += 1
                    if fidx % max(1, args.every_n) != 0:
                        continue

                    frame = apply_rotation(frame, rot_k)
                    frame = crop_to_horizontal_16_9(frame)
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = roi_box_from_args(h, w, args)

                    x = crop_to_tensor(frame, x1, y1, x2, y2, tfm, device)
                    with torch.no_grad():
                        logits = model(x)
                        probs = F.softmax(logits, dim=1)[0]
                    conf = float(probs[target_i].item())
                    pred_i = int(probs.argmax().item())

                    desired_laser = laser_on_for_target(
                        probs, target_i, min_prob=args.seek_min_confidence
                    )
                    if desired_laser != last_laser_on:
                        send_laser(ser, desired_laser)
                        last_laser_on = desired_laser

                    if pred_i == target_i and conf >= args.seek_min_confidence:
                        hits += 1
                        print(
                            f"angle={angle}°  hit {hits}/{args.seek_hits}  "
                            f"P({class_names[target_i]})={conf:.3f}"
                        )
                    else:
                        if hits > 0:
                            print(
                                f"angle={angle}°  reset (saw {class_names[pred_i]} @ {float(probs[pred_i]):.3f})"
                            )
                        hits = 0

                    if hits >= args.seek_hits:
                        found_angle = angle
                        break

                if found_angle is not None:
                    break
            if found_angle is not None:
                break
    finally:
        send_laser(ser, False)

    if found_angle is not None:
        send_servo_angle(ser, found_angle)
        time.sleep(0.2)
        print(f"Stopped at servo angle {found_angle}° (class {class_names[target_i]}).")
        return 0

    print(
        "Target not found. Try smaller --seek-angle-step, lower --seek-min-confidence, "
        "higher --seek-max-frames, or lighting.",
        file=sys.stderr,
    )
    return 1


def run_shoot_mission(args: argparse.Namespace) -> int:
    """
    Read ``state.load_target_bits()`` (same order as ``state.TARGET_NAMES``: red, green, blue, yellow).
    For each bit set, run headless seek, then laser ON for ``args.laser_fire_sec``, then OFF.
    """
    import state

    bits = state.load_target_bits()
    names = tuple(getattr(state, "TARGET_NAMES", ("red", "green", "blue", "yellow")))
    n = min(len(bits), len(names))
    queue = [names[i] for i in range(n) if bits[i]]
    if not queue:
        print(
            "No targets set (all bits zero). Use the gesture client SEND, call "
            "state.save_target_bits(...) from Python, or edit state.py / state_target_bits.json.",
            file=sys.stderr,
        )
        return 1

    print("Shoot mission — targets in order:", queue)
    print("Serial port required.")
    port_line = input("Port (e.g. COM3): ").strip()
    if not port_line:
        print("Aborted.", file=sys.stderr)
        return 1
    ser = open_servo_serial(port_line, wait_after_open=args.serial_wait)
    if ser is None:
        return 1
    _import_heavy()
    device = pick_device(args.device)
    print("Device:", device)
    print(
        "ROI: fraction",
        args.roi_fraction,
        "| max square",
        args.max_roi_side,
        "| vertical",
        getattr(args, "roi_vertical", "bottom"),
        "| capture request",
        f"{args.capture_width}x{args.capture_height}",
        "| output after rotation: 16:9 crop",
    )
    print()
    try:
        model, class_names, image_size = load_classifier(
            args.checkpoint, args.classes, device
        )
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        ser.close()
        return 1
    tfm = make_transform(image_size)
    cap = open_camera_capture(int(args.camera))
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        ser.close()
        return 1
    configure_capture_resolution(cap, args.capture_width, args.capture_height)

    fire = float(getattr(args, "laser_fire_sec", 1.0))
    try:
        for i, color in enumerate(queue):
            print(f"\n=== Mission {i + 1}/{len(queue)}: seek {color!r} ===")
            _voice_line(args, f"cmd_{color}")
            rc = run_headless_seek(
                ser,
                cap,
                model,
                class_names,
                tfm,
                device,
                color,
                1,
                args,
            )
            if rc != 0:
                _voice_line(args, "no_target")
                print(f"Mission failed: could not find {color!r}.", file=sys.stderr)
                return 1
            print(
                f"Locked on {color}. Laser ON for {fire:.2f}s (shoot)...",
                flush=True,
            )
            _voice_line(args, "target_locked")
            send_laser(ser, True)
            _voice_line(args, "fired")
            time.sleep(fire)
            send_laser(ser, False)
            print("Laser OFF.", flush=True)
            _voice_line(args, "target_down")
            cur = list(state.load_target_bits())
            try:
                bi = names.index(color)
            except ValueError:
                bi = -1
            if 0 <= bi < len(cur):
                cur[bi] = 0
                state.save_target_bits(cur)
                print(
                    f"Cleared bit for {color!r}; state TARGET_BITS = {cur}",
                    flush=True,
                )
        _voice_line(args, "all_done")
    finally:
        send_laser(ser, False)
        cap.release()
        ser.close()

    print("\nAll targeted dinosaurs shot. Mission complete.")
    return 0


def start_seek_for_next_mission_bit(
    ser,
    class_names: list[str],
    sweep_angles: list[int],
    args: argparse.Namespace,
) -> SeekState | None:
    """Begin interactive seek for the first color with bit 1 in ``state.load_target_bits()``."""
    import state

    bits = state.load_target_bits()
    for j, name in enumerate(state.TARGET_NAMES):
        if not bits[j]:
            continue
        ix = class_index_for_color(class_names, name)
        if ix is None or class_names[ix].lower() == "none":
            continue
        seek = SeekState(target_i=ix, label=class_names[ix])
        seek.sweep_idx = 0
        t0 = time.monotonic()
        send_servo_angle(ser, sweep_angles[seek.sweep_idx])
        seek.settle_until = t0 + args.seek_settle
        seek.next_angle_time = t0 + args.seek_settle + args.seek_per_angle_sec
        seek.frames_at_angle = 0
        seek.hits = 0
        print(
            f"Seeking {name!r} (TARGET_BITS={bits})",
            flush=True,
        )
        return seek
    print("No targets left in state (all bits zero).", flush=True)
    return None


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
        _progress(f"  Probing camera index {i}…")
        cap = open_camera_capture(i)
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
    _progress("Enumerating cameras (probing indices 0–{})…".format(max_index - 1))
    dshow = _try_windows_directshow_names()
    rows = probe_cameras(max_index, dshow_names=dshow)
    _progress(f"Camera scan done ({len(rows)} found).")
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


def build_model(arch: str, num_classes: int) -> Any:
    _import_heavy()
    if arch != "resnet18":
        raise ValueError(f"Unsupported arch in checkpoint: {arch!r} (expected 'resnet18')")
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_classifier(
    checkpoint_path: Path,
    classes_json: Path,
    device: Any,
) -> tuple[Any, list[str], int]:
    _import_heavy()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not classes_json.is_file():
        raise FileNotFoundError(f"Class list not found: {classes_json}")

    _progress(f"Loading weights from {checkpoint_path.name}…")
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

    _progress(f"Building ResNet-18 ({num_classes} classes) on {device}…")
    model = build_model(arch, num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    _progress("Classifier loaded and ready.")
    return model, class_names, image_size


def make_transform(image_size: int) -> Any:
    _import_heavy()
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
    vertical: str = "bottom",
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) inclusive-exclusive crop box; square side ≤ max_roi_side and ≤ frame.

    vertical: ``center`` — square centered in the frame; ``bottom`` — same size, centered
    horizontally with the **bottom edge** flush against the bottom of the frame.
    """
    side = int(min(w, h) * fraction)
    side = min(side, max_roi_side, min(w, h))
    side = max(32, side)

    if vertical == "bottom":
        cx = w // 2
        half = side // 2
        x1 = cx - half
        x2 = x1 + side
        y2 = h
        y1 = h - side
    else:
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
    tfm: Any,
    device: Any,
) -> Any:
    _import_heavy()
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


def roi_box_from_args(h: int, w: int, args: argparse.Namespace) -> tuple[int, int, int, int]:
    """ROI used for inference, on-screen crosshair, and seek (find-dinosaur) — one consistent box."""
    v = getattr(args, "roi_vertical", "bottom")
    if v not in ("center", "bottom"):
        v = "bottom"
    return center_square_roi(
        h, w, args.roi_fraction, max_roi_side=args.max_roi_side, vertical=v
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Dinosaur classifier on the ROI crosshair (default: bottom-aligned; see --roi-vertical). "
        "R/B/G/Y seek sweeps the servo until the target class locks in that ROI."
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
        default="cpu",
        help="cpu (default, avoids CUDA probe), cuda, or auto (cuda if available — auto can take minutes on some Windows setups).",
    )
    p.add_argument(
        "--roi-fraction",
        type=float,
        default=0.42,
        help="Square ROI side as fraction of min(w,h); side is also capped by --max-roi-side (default 480).",
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
        help="Maximum ROI square size in pixels (default 480).",
    )
    p.add_argument(
        "--roi-vertical",
        type=str,
        default="bottom",
        choices=("center", "bottom"),
        help="ROI vertical placement: centered in frame, or centered horizontally along the bottom edge (default).",
    )
    p.add_argument(
        "--seek",
        type=str,
        default=None,
        choices=("red", "blue", "green", "yellow"),
        metavar="COLOR",
        help="Headless: sweep servo until this color is found, then exit (no GUI). Requires serial.",
    )
    p.add_argument(
        "--seek-from-state",
        action="store_true",
        help="Interactive: after startup, start seek for the first color with bit 1 in "
        "state.load_target_bits() (gesture SEND / state_target_bits.json). Requires serial.",
    )
    p.add_argument(
        "--mission-chain",
        action="store_true",
        help="Interactive (requires serial): after each successful seek, hold laser ON for "
        "--laser-fire-sec, clear that color's bit in state, then seek the next color with bit 1 "
        "(or stop when empty). Use with --seek-from-state and/or R/B/G/Y.",
    )
    p.add_argument(
        "--seek-angle-step",
        type=int,
        default=1,
        help="Servo degrees per step (interactive R/B/G/Y and --seek). Default 1° (use 5 for coarser steps).",
    )
    p.add_argument(
        "--seek-settle",
        type=float,
        default=0.1,
        help="Seconds after each servo move before scoring (interactive seek). Default 0.1s.",
    )
    p.add_argument(
        "--seek-min-confidence",
        type=float,
        default=0.45,
        help="Min softmax P(target) to count as a hit during seek.",
    )
    p.add_argument(
        "--seek-hits",
        type=int,
        default=1,
        help="Consecutive confident detections required to end seek (default 1 = stop as soon as "
        "the target is seen; use 2–3 to reduce false stops from noisy frames).",
    )
    p.add_argument(
        "--seek-reacquire-frames",
        type=int,
        default=3,
        help="After seek succeeds, resume servo sweep if the target is not confidently seen in the "
        "ROI for this many consecutive frames (interactive only; 0 = never auto-resume).",
    )
    p.add_argument(
        "--seek-max-frames",
        type=int,
        default=35,
        help="Max inference frames per angle (safety cap). Interactive seek advances when "
        "this is reached OR after --seek-settle + --seek-per-angle-sec (whichever first).",
    )
    p.add_argument(
        "--seek-per-angle-sec",
        type=float,
        default=0.0,
        help="Interactive seek only: extra time after --seek-settle before stepping to the next "
        "angle (0 = advance as soon as settle ends and one frame is scored; larger = slower).",
    )
    p.add_argument(
        "--seek-forward-only",
        action="store_true",
        help="Headless --seek: only sweep 0→180 once (default also sweeps back).",
    )
    p.add_argument(
        "--probe-cameras",
        action="store_true",
        help="Scan camera indices 0..--max-camera-index (slow on some PCs). Default: skip scan.",
    )
    p.add_argument(
        "--serial-wait",
        type=float,
        default=0.5,
        help="Seconds to wait after opening serial (USB reset). Increase if servo misses commands.",
    )
    p.add_argument(
        "--shoot-mission",
        action="store_true",
        help="Read state.load_target_bits() [R,G,B,Y]; for each 1, seek that color then laser ON for "
        "--laser-fire-sec (requires serial, no GUI).",
    )
    p.add_argument(
        "--laser-fire-sec",
        type=float,
        default=1.0,
        dest="laser_fire_sec",
        help="After each lock: seconds to hold laser ON (--shoot-mission; --mission-chain in preview). "
        "Default 1.",
    )
    p.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable ElevenLabs caveman lines from state.PHRASES during --shoot-mission and "
        "--mission-chain (requires .env from setup_and_test_voice.py otherwise).",
    )
    return p.parse_args()


def open_camera_capture(index: int) -> cv2.VideoCapture:
    """Open a camera index. On Windows, DirectShow is usually faster than the default MSMF backend."""
    idx = int(index)
    if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    return cv2.VideoCapture(idx)


def configure_capture_resolution(
    cap: cv2.VideoCapture, width: int, height: int
) -> tuple[int, int]:
    """Request resolution; return (width, height) as reported by the device."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return rw, rh


def pick_device(name: str) -> Any:
    _import_heavy()
    if name == "auto":
        _progress(
            "Device auto: probing CUDA (skip long waits with --device cpu)…"
        )
        use_cuda = torch.cuda.is_available()
        _progress("CUDA probe done." if use_cuda else "CUDA not used.")
        return torch.device("cuda" if use_cuda else "cpu")
    return torch.device(name)


def main() -> int:
    _progress("Parsing command line…")
    args = parse_args()
    sweep_angles = build_sweep_angles(args.seek_angle_step)

    if args.shoot_mission:
        return run_shoot_mission(args)

    if args.seek is not None:
        print("Headless seek mode — serial port required.")
        port_line = input("Port (e.g. COM3): ").strip()
        if not port_line:
            print("Aborted.", file=sys.stderr)
            return 1
        ser = open_servo_serial(port_line, wait_after_open=args.serial_wait)
        if ser is None:
            return 1
        _import_heavy()
        device = pick_device(args.device)
        print("Device:", device)
        print(
            "ROI: fraction",
            args.roi_fraction,
            "| max square",
            args.max_roi_side,
            "| vertical",
            getattr(args, "roi_vertical", "bottom"),
            "| capture request",
            f"{args.capture_width}x{args.capture_height}",
            "| output after rotation: 16:9 crop",
        )
        print()
        try:
            model, class_names, image_size = load_classifier(
                args.checkpoint, args.classes, device
            )
        except (FileNotFoundError, ValueError) as e:
            print(e, file=sys.stderr)
            ser.close()
            return 1
        tfm = make_transform(image_size)
        _progress(f"Opening camera index {args.camera} for headless seek…")
        cap = open_camera_capture(int(args.camera))
        if not cap.isOpened():
            print(f"Could not open camera {args.camera}", file=sys.stderr)
            ser.close()
            return 1
        configure_capture_resolution(cap, args.capture_width, args.capture_height)
        _progress("Camera open; starting seek sweep.")
        print(f"Seeking {args.seek!r} on serial {port_line!r} …")
        rc = run_headless_seek(
            ser,
            cap,
            model,
            class_names,
            tfm,
            device,
            args.seek,
            1,
            args,
        )
        cap.release()
        ser.close()
        return rc

    _progress("Enter serial port when prompted (or empty for camera-only).")
    port_str = prompt_serial_port()
    ser = None
    if port_str:
        ser = open_servo_serial(port_str, wait_after_open=args.serial_wait)
        if ser:
            print(f"Serial open: {port_str} @ {SERIAL_BAUD} baud")
        else:
            print("Could not open serial; continuing without servo.", file=sys.stderr)
    else:
        print("No serial port — R/B/G/Y seek disabled until you restart with a port.")

    if args.probe_cameras:
        print()
        print_camera_list(args.max_camera_index)
        print()
    else:
        _progress(
            "Skipping camera scan (fast). Use --probe-cameras to list indices 0–"
            f"{args.max_camera_index - 1}."
        )
        print()

    _import_heavy()
    device = pick_device(args.device)
    print("Device:", device)
    print(
        "ROI: fraction",
        args.roi_fraction,
        "| max square",
        args.max_roi_side,
        "| vertical",
        getattr(args, "roi_vertical", "bottom"),
        "| capture request",
        f"{args.capture_width}x{args.capture_height}",
        "| output after rotation: 16:9 crop",
    )
    print()

    try:
        model, class_names, image_size = load_classifier(
            args.checkpoint, args.classes, device
        )
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        if ser:
            ser.close()
        return 1

    tfm = make_transform(image_size)
    camera_index = int(args.camera)
    _progress(f"Opening preview camera index {camera_index}…")
    cap = open_camera_capture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}", file=sys.stderr)
        if ser:
            ser.close()
        return 1

    rw, rh = configure_capture_resolution(
        cap, args.capture_width, args.capture_height
    )
    print(
        f"Camera {camera_index}: requested {args.capture_width}x{args.capture_height}, "
        f"reports {rw}x{rh}",
    )
    _progress("Opening preview window — use Q or Esc to quit.")

    ema_probs: torch.Tensor | None = None
    frame_idx = 0
    last_label = "—"
    last_conf = 0.0
    rot_k = 1  # 0,1,2,3 → 0°,90°,180°,270° clockwise; default 1 = 90° CW
    seek: SeekState | None = None
    # After seek succeeds: keep laser tied to live ROI until a new seek / cancel.
    laser_follow_class_i: int | None = None
    last_laser_sent: bool | None = None
    lost_streak: int = 0  # frames without target after lock; triggers re-sweep
    resume_sweep_idx: int | None = None  # sweep index when seek last locked; used to resume after loss
    mission_shoot_until: float | None = None  # --mission-chain: monotonic time when fixed laser ends
    mission_shoot_class_i: int | None = None

    print(
        "Q/Esc quit | P rotate | R/B/G/Y seek | X cancel seek | 0–9 camera",
    )
    if args.mission_chain and ser is not None:
        print(
            f"Mission chain: after each lock → laser {args.laser_fire_sec:.2f}s → clear bit → "
            "seek next in state.",
            flush=True,
        )
    elif args.mission_chain and ser is None:
        print(
            "--mission-chain ignored (no serial port).",
            file=sys.stderr,
        )

    if args.seek_from_state and ser is not None:
        import state

        bits = state.load_target_bits()
        names = state.TARGET_NAMES
        started = False
        for i in range(min(len(bits), len(names))):
            if not bits[i]:
                continue
            col = names[i]
            ix = class_index_for_color(class_names, col)
            if ix is None:
                print(
                    f"seek-from-state: no class for {col!r} in {class_names}",
                    file=sys.stderr,
                )
                continue
            if class_names[ix].lower() == "none":
                print("seek-from-state: cannot seek class 'none'.", file=sys.stderr)
                continue
            send_laser(ser, False)
            last_laser_sent = False
            laser_follow_class_i = None
            lost_streak = 0
            resume_sweep_idx = None
            seek = SeekState(target_i=ix, label=class_names[ix])
            seek.sweep_idx = 0
            t0 = time.monotonic()
            send_servo_angle(ser, sweep_angles[seek.sweep_idx])
            seek.settle_until = t0 + args.seek_settle
            seek.next_angle_time = t0 + args.seek_settle + args.seek_per_angle_sec
            seek.frames_at_angle = 0
            seek.hits = 0
            ema_probs = None
            frame_idx = 0
            print(
                f"Seek from state bits {bits}: {seek.label} "
                f"(step {args.seek_angle_step}° / {args.seek_settle}s settle)"
            )
            started = True
            break
        if not started:
            if any(bits[:4]):
                print(
                    "seek-from-state: bits are set but no matching class — check class_names.json.",
                    file=sys.stderr,
                )
            else:
                print(
                    "seek-from-state: no targets (all bits zero). Use gesture SEND or edit state.",
                    file=sys.stderr,
                )
    elif args.seek_from_state and ser is None:
        print(
            "seek-from-state ignored (no serial port). Restart with a COM port.",
            file=sys.stderr,
        )

    import state as _state_mod

    _bits_path = _state_mod.runtime_bits_path()
    _b0 = _state_mod.load_target_bits()
    last_bits_snapshot: tuple[int, int, int, int] = (
        int(_b0[0]),
        int(_b0[1]),
        int(_b0[2]),
        int(_b0[3]),
    )
    last_state_bits_mtime_ns: int | None = (
        int(_bits_path.stat().st_mtime_ns) if _bits_path.is_file() else None
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed; exiting.", file=sys.stderr)
            if ser is not None:
                send_laser(ser, False)
                last_laser_sent = False
            break

        now_mono = time.monotonic()
        if (
            args.mission_chain
            and ser is not None
            and mission_shoot_until is not None
            and mission_shoot_class_i is not None
            and now_mono >= mission_shoot_until
        ):
            send_laser(ser, False)
            last_laser_sent = False
            import state

            bits = list(state.load_target_bits())
            cn = class_names[mission_shoot_class_i].lower()
            bi = state.CLASS_TO_TARGET_IDX.get(cn)
            if bi is not None:
                bits[bi] = 0
                state.save_target_bits(bits)
                print(
                    f"Mission: cleared bit for {cn}; TARGET_BITS={bits}",
                    flush=True,
                )
            _voice_line(args, "target_down")
            mission_shoot_until = None
            mission_shoot_class_i = None
            seek = start_seek_for_next_mission_bit(
                ser, class_names, sweep_angles, args
            )
            if seek is None:
                _voice_line(args, "all_done")
            laser_follow_class_i = None
            lost_streak = 0
            ema_probs = None
            frame_idx = 0

        # Gesture SEND (or any writer) updates state_target_bits.json — detect mtime or bit
        # changes so preview starts seeking without --seek-from-state or R/B/G/Y.
        if (
            ser is not None
            and seek is None
            and laser_follow_class_i is None
            and mission_shoot_until is None
            and _bits_path.is_file()
        ):
            bits = _state_mod.load_target_bits()
            t = (int(bits[0]), int(bits[1]), int(bits[2]), int(bits[3]))
            m_ns = int(_bits_path.stat().st_mtime_ns)
            state_changed = (last_state_bits_mtime_ns is None or m_ns != last_state_bits_mtime_ns) or (
                t != last_bits_snapshot
            )
            if state_changed:
                last_state_bits_mtime_ns = m_ns
                last_bits_snapshot = t
                if any(t):
                    new_seek = start_seek_for_next_mission_bit(
                        ser, class_names, sweep_angles, args
                    )
                    if new_seek is not None:
                        seek = new_seek
                        laser_follow_class_i = None
                        lost_streak = 0
                        resume_sweep_idx = None
                        ema_probs = None
                        frame_idx = 0
                        print(
                            "Started seek from updated state (gesture SEND / file).",
                            flush=True,
                        )

        frame = apply_rotation(frame, rot_k)
        frame = crop_to_horizontal_16_9(frame)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi_box_from_args(h, w, args)
        roi_side = x2 - x1

        frame_idx += 1
        _need_fast_infer = seek is not None or (
            laser_follow_class_i is not None and args.seek_reacquire_frames > 0
        )
        infer_every = 1 if _need_fast_infer else max(1, args.every_n)
        run_infer = frame_idx % infer_every == 0

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

            if seek is not None and ser is not None:
                if now_mono >= seek.settle_until:
                    seek.frames_at_angle += 1
                    conf_t = float(probs[seek.target_i].item())
                    pi = int(probs.argmax().item())
                    if pi == seek.target_i and conf_t >= args.seek_min_confidence:
                        seek.hits += 1
                    else:
                        seek.hits = 0
                    cur_ang = sweep_angles[seek.sweep_idx]
                    if seek.hits >= args.seek_hits:
                        print(
                            f"Seek done: {seek.label} @ servo {cur_ang}° "
                            f"(P({seek.label})={conf_t:.2f})"
                        )
                        resume_sweep_idx = seek.sweep_idx
                        locked_i = seek.target_i
                        seek = None
                        lost_streak = 0
                        if args.mission_chain and ser is not None:
                            laser_follow_class_i = None
                            mission_shoot_class_i = locked_i
                            mission_shoot_until = now_mono + float(args.laser_fire_sec)
                            _voice_line(args, "target_locked")
                            _voice_line(args, "fired")
                            print(
                                f"Mission: locked on {class_names[locked_i]}; "
                                f"laser {args.laser_fire_sec:.2f}s then next in state…",
                                flush=True,
                            )
                        else:
                            laser_follow_class_i = locked_i
                    elif (
                        now_mono >= seek.next_angle_time
                        or seek.frames_at_angle >= args.seek_max_frames
                    ):
                        seek.sweep_idx = (seek.sweep_idx + 1) % len(sweep_angles)
                        nang = sweep_angles[seek.sweep_idx]
                        send_servo_angle(ser, nang)
                        seek.settle_until = now_mono + args.seek_settle
                        seek.next_angle_time = (
                            now_mono + args.seek_settle + args.seek_per_angle_sec
                        )
                        seek.frames_at_angle = 0
                        seek.hits = 0

            if (
                seek is None
                and laser_follow_class_i is not None
                and ser is not None
                and args.seek_reacquire_frames > 0
            ):
                if laser_on_for_target(
                    probs, laser_follow_class_i, min_prob=args.min_confidence
                ):
                    lost_streak = 0
                else:
                    lost_streak += 1
                    if lost_streak >= args.seek_reacquire_frames:
                        ix = laser_follow_class_i
                        print(
                            f"Lost track of {class_names[ix]}; resuming sweep.",
                            flush=True,
                        )
                        laser_follow_class_i = None
                        lost_streak = 0
                        seek = SeekState(target_i=ix, label=class_names[ix])
                        seek.sweep_idx = (
                            resume_sweep_idx
                            if resume_sweep_idx is not None
                            else 0
                        ) % len(sweep_angles)
                        t0 = time.monotonic()
                        send_servo_angle(ser, sweep_angles[seek.sweep_idx])
                        seek.settle_until = t0 + args.seek_settle
                        seek.next_angle_time = (
                            t0 + args.seek_settle + args.seek_per_angle_sec
                        )
                        seek.frames_at_angle = 0
                        seek.hits = 0
                        ema_probs = None
                        frame_idx = 0

            if ser is not None:
                if (
                    args.mission_chain
                    and mission_shoot_until is not None
                    and now_mono < mission_shoot_until
                ):
                    desired_laser = True
                else:
                    t = seek.target_i if seek is not None else laser_follow_class_i
                    if t is None:
                        desired_laser = False
                    else:
                        thr = (
                            args.seek_min_confidence
                            if seek is not None
                            else args.min_confidence
                        )
                        desired_laser = laser_on_for_target(probs, t, min_prob=thr)
                if desired_laser != last_laser_sent:
                    send_laser(ser, desired_laser)
                    last_laser_sent = desired_laser

        if (
            ser is not None
            and args.mission_chain
            and mission_shoot_until is not None
            and now_mono < mission_shoot_until
            and not run_infer
        ):
            if last_laser_sent is not True:
                send_laser(ser, True)
                last_laser_sent = True

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

        if seek is not None:
            sought_str = seek.label
        elif (
            mission_shoot_class_i is not None
            and mission_shoot_until is not None
            and now_mono < mission_shoot_until
        ):
            sought_str = f"{class_names[mission_shoot_class_i]} (shoot)"
        elif laser_follow_class_i is not None:
            sought_str = class_names[laser_follow_class_i]
        else:
            sought_str = "—"

        if ser is None:
            laser_str = "n/a (no serial)"
            laser_color = (140, 140, 140)
        else:
            laser_on = bool(last_laser_sent)
            laser_str = "ON" if laser_on else "OFF"
            laser_color = (0, 255, 0) if laser_on else (160, 160, 160)

        cv2.putText(
            frame,
            f"Sought: {sought_str}",
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Laser: {laser_str}",
            (16, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            laser_color,
            2,
            cv2.LINE_AA,
        )
        if seek is not None:
            sa = sweep_angles[seek.sweep_idx]
            seek_hud = f"Sweep: servo={sa}°  hits={seek.hits}/{args.seek_hits}"
            cv2.putText(
                frame,
                seek_hud,
                (16, 144),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"cam {camera_index}  |  rot {rot_k * 90}°  |  {w}x{h}  |  p rotate  rbgy seek  x end  0-9  q",
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("dinosaur classifier (center ROI)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            if ser is not None:
                send_laser(ser, False)
                last_laser_sent = False
            break

        if key in (ord("p"), ord("P")):
            rot_k = (rot_k + 1) % 4
            ema_probs = None
            frame_idx = 0
            print(f"View rotation: {rot_k * 90}° CW")

        if key in (ord("x"), ord("X")):
            if mission_shoot_until is not None:
                if ser is not None:
                    send_laser(ser, False)
                    last_laser_sent = False
                mission_shoot_until = None
                mission_shoot_class_i = None
                print("Mission shoot cancelled.", flush=True)
            elif seek is not None:
                if ser is not None:
                    send_laser(ser, False)
                    last_laser_sent = False
                laser_follow_class_i = None
                seek = None
                lost_streak = 0
                resume_sweep_idx = None
                print("Seek cancelled.")

        color_key = {ord("r"): "red", ord("R"): "red", ord("b"): "blue", ord("B"): "blue",
                     ord("g"): "green", ord("G"): "green", ord("y"): "yellow", ord("Y"): "yellow"}
        if key in color_key:
            col = color_key[key]
            if ser is None:
                print("No serial port — restart and enter a COM port to use seek.", file=sys.stderr)
            else:
                ix = class_index_for_color(class_names, col)
                if ix is None:
                    print(f"No class '{col}' in {class_names}", file=sys.stderr)
                elif class_names[ix].lower() == "none":
                    print("Cannot seek class 'none'.", file=sys.stderr)
                else:
                    if mission_shoot_until is not None:
                        if ser is not None:
                            send_laser(ser, False)
                            last_laser_sent = False
                        mission_shoot_until = None
                        mission_shoot_class_i = None
                    laser_follow_class_i = None
                    lost_streak = 0
                    resume_sweep_idx = None
                    if ser is not None:
                        send_laser(ser, False)
                        last_laser_sent = False
                    seek = SeekState(target_i=ix, label=class_names[ix])
                    seek.sweep_idx = 0
                    t0 = time.monotonic()
                    send_servo_angle(ser, sweep_angles[seek.sweep_idx])
                    seek.settle_until = t0 + args.seek_settle
                    seek.next_angle_time = t0 + args.seek_settle + args.seek_per_angle_sec
                    seek.frames_at_angle = 0
                    seek.hits = 0
                    ema_probs = None
                    frame_idx = 0
                    print(
                        f"Seeking: {seek.label} "
                        f"(step {args.seek_angle_step}° / {args.seek_settle}s settle)"
                    )

        if ord("0") <= key <= ord("9"):
            new_idx = key - ord("0")
            if new_idx != camera_index:
                trial = open_camera_capture(new_idx)
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
    if ser is not None:
        ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
