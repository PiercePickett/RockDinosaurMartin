"""
Live webcam inference using the ResNet-18 classifier trained in train_dinosaur_classifier.ipynb.

Expects artifacts/dinosaur_classifier.pt and artifacts/class_names.json (or paths via --checkpoint / --classes).

Quit: press Q or Esc on the preview window.
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


def frame_to_tensor(
    frame_bgr: np.ndarray, tfm: transforms.Compose, device: torch.device
) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = tfm(pil).unsqueeze(0).to(device)
    return x


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run dinosaur classifier on the default webcam.")
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
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0)")
    p.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu, or auto",
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
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    print("Device:", device)

    try:
        model, class_names, image_size = load_classifier(
            args.checkpoint, args.classes, device
        )
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    tfm = make_transform(image_size)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}", file=sys.stderr)
        return 1

    ema_probs: torch.Tensor | None = None
    frame_idx = 0
    last_label = "—"
    last_conf = 0.0

    print("Press Q or Esc to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed; exiting.", file=sys.stderr)
            break

        frame_idx += 1
        run_infer = frame_idx % max(1, args.every_n) == 0

        if run_infer:
            x = frame_to_tensor(frame, tfm, device)
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

        h, w = frame.shape[:2]
        text = f"{last_label}  ({last_conf:.2f})"
        cv2.putText(
            frame,
            text,
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{w}x{h}  |  q/esc quit",
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("dinosaur classifier", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
