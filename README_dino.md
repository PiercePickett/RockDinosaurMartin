# RockDinosaurMartin

Run the dinosaur classifier on a **local webcam** with OpenCV. Provide a trained checkpoint and class list (see below); training is out of scope here.

## Setup

```bash
cd RockDinosaurMartin
python -m venv .venv
```

**Windows (PowerShell):** `.venv\Scripts\activate`  
**macOS/Linux:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

Use **Python 3.10+**. A **GPU** is optional; CPU works.

On startup, `run_camera.py` **prints detected cameras** with their index numbers. On Windows, `pip install pygrabber` adds USB **product names** next to each index (optional).

## Model files

Place (or point `--checkpoint` / `--classes` at):

- `artifacts/dinosaur_classifier.pt`
- `artifacts/class_names.json`

## Run

**Pipeline:** opens **camera index 3** by default (`--camera` to change). Starts **rotated 90°** clockwise; press **R** to cycle rotation. Capture requests **1920×1080** (actual size is printed). After rotation, **center-crop to 16:9 landscape**. Classification uses a **center square** (side **≤ 480 px**, `--max-roi-side`). Align the toy with the yellow box.

Override defaults: `--camera`, `--capture-width`, `--capture-height`, `--max-roi-side`, `--roi-fraction`.

For reliable results, the model should be trained on **similar crops** (centered object, comparable scale).

```bash
python run_camera.py
```

- **Quit:** **Q** or **Esc** in the preview window.
- **Switch camera:** number keys **0–9** pick that device index while the window is focused (no need to restart).
- **Rotate view:** **R** cycles the feed **90°** at a time (0° → 90° → 180° → 270° → …). Classification uses the rotated image; train with the same orientation if it matters for your setup.
- **Crosshair size:** `python run_camera.py --roi-fraction 0.35` (smaller) or `0.55` (larger).
- **Different camera index:** `python run_camera.py --camera 0`
- **Device:** `python run_camera.py --device cpu` or `--device cuda` (default: auto)
- **Smoother overlay:** EMA is on by default; `python run_camera.py --ema 0` to disable.
- **Higher FPS:** `python run_camera.py --every-n 2` (infer every other frame).

Custom paths:

```bash
python run_camera.py --checkpoint path/to/model.pt --classes path/to/class_names.json
```

## Troubleshooting

- **Checkpoint not found:** Ensure the two files exist under `artifacts/` or pass explicit paths.
- **Wrong or black camera:** Try `--camera 0`, `1`, …
- **Slow:** Use `--every-n` or `--device cuda` if available.