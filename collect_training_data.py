"""
Capture 255×255 JPEG crops of the ROI (same crosshair box as run_camera.py: horizontally centered, bottom-aligned by default).

Keys: **r** red | **b** blue | **g** green | **y** yellow | **n** none — select save folder.
**Space** saves the current ROI to ``<dataset-root>/<class>/``. **p** rotate view, **0–9** camera, **q**/Esc quit.

Default output layout matches training: ``dataset/red``, ``dataset/blue``, …
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from run_camera import (
    apply_rotation,
    configure_capture_resolution,
    crop_to_horizontal_16_9,
    draw_roi_crosshair,
    open_camera_capture,
    roi_box_from_args,
)

SAVE_SIZE = 255

KEY_TO_CLASS: dict[int, str] = {
    ord("r"): "red",
    ord("R"): "red",
    ord("b"): "blue",
    ord("B"): "blue",
    ord("g"): "green",
    ord("G"): "green",
    ord("y"): "yellow",
    ord("Y"): "yellow",
    ord("n"): "none",
    ord("N"): "none",
}

VALID_CLASSES = frozenset({"red", "blue", "green", "yellow", "none"})


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Save 255×255 ROI JPEGs into dataset/<class>/ (see module docstring)."
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "dataset",
        help="Root folder; images go to <root>/red, <root>/blue, … (default: ./dataset).",
    )
    p.add_argument("--camera", type=int, default=3, help="OpenCV camera index (default 3).")
    p.add_argument(
        "--roi-fraction",
        type=float,
        default=0.42,
        help="Center square side as fraction of min(w,h), capped by --max-roi-side.",
    )
    p.add_argument("--max-roi-side", type=int, default=480, help="Max ROI square size in pixels.")
    p.add_argument(
        "--roi-vertical",
        type=str,
        default="bottom",
        choices=("center", "bottom"),
        help="ROI vertical placement: centered or bottom-aligned (default: bottom).",
    )
    p.add_argument("--capture-width", type=int, default=1920, help="Requested capture width.")
    p.add_argument("--capture-height", type=int, default=1080, help="Requested capture height.")
    p.add_argument(
        "--max-camera-index",
        type=int,
        default=10,
        help="Largest index for hotkeys 0–9.",
    )
    return p.parse_args()


def ensure_class_dirs(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    for name in sorted(VALID_CLASSES):
        (dataset_root / name).mkdir(parents=True, exist_ok=True)


def count_jpg(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return len(list(folder.glob("*.jpg")))


def save_roi_jpg(
    frame_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    out_path: Path,
) -> bool:
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    resized = cv2.resize(
        crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))


def main() -> int:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    ensure_class_dirs(dataset_root)

    current_class = "red"
    rot_k = 1
    camera_index = int(args.camera)

    cap = open_camera_capture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}", file=sys.stderr)
        return 1
    rw, rh = configure_capture_resolution(cap, args.capture_width, args.capture_height)
    print(
        f"Camera {camera_index}: requested {args.capture_width}x{args.capture_height}, "
        f"reports {rw}x{rh}",
    )
    print(
        "r/b/g/y/n = class | Space = save ROI | p = rotate | 0–9 = camera | q = quit",
        flush=True,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed; exiting.", file=sys.stderr)
            break

        frame = apply_rotation(frame, rot_k)
        frame = crop_to_horizontal_16_9(frame)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi_box_from_args(h, w, args)
        roi_side = x2 - x1

        disp = frame.copy()
        draw_roi_crosshair(disp, x1, y1, x2, y2)

        out_dir = dataset_root / current_class
        n_in_class = count_jpg(out_dir)

        cv2.putText(
            disp,
            f"class: {current_class}  |  saved in folder: {n_in_class}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            f"ROI {roi_side}x{roi_side} -> {SAVE_SIZE}x{SAVE_SIZE} jpg  |  Space save",
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 255, 200),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            f"cam {camera_index}  |  rot {rot_k * 90}deg  |  rbgy n class  |  q quit",
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("collect training data (center ROI)", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        if key in KEY_TO_CLASS:
            current_class = KEY_TO_CLASS[key]
            print(f"Class: {current_class}", flush=True)
        if key == ord(" "):
            fname = f"{int(time.time() * 1000)}.jpg"
            out_path = out_dir / fname
            if save_roi_jpg(frame, x1, y1, x2, y2, out_path):
                n_after = count_jpg(out_dir)
                print(f"Saved {out_path} (total in {current_class}: {n_after})", flush=True)
            else:
                print("Save failed (empty ROI?).", file=sys.stderr)
        if key in (ord("p"), ord("P")):
            rot_k = (rot_k + 1) % 4
            print(f"View rotation: {rot_k * 90} deg CW", flush=True)
        if ord("0") <= key <= ord("9"):
            new_idx = key - ord("0")
            if new_idx > args.max_camera_index:
                continue
            if new_idx != camera_index:
                trial = open_camera_capture(new_idx)
                if trial.isOpened():
                    tw, th = configure_capture_resolution(
                        trial, args.capture_width, args.capture_height
                    )
                    cap.release()
                    cap = trial
                    camera_index = new_idx
                    print(
                        f"Switched to camera {camera_index}, "
                        f"requested {args.capture_width}x{args.capture_height}, reports {tw}x{th}",
                        flush=True,
                    )
                else:
                    trial.release()
                    print(f"Camera index {new_idx} not available.", file=sys.stderr)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
