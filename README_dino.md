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

**Startup speed:** PyTorch is loaded **after** the serial prompt (and after an optional camera scan). By default **camera indices are not scanned** (that scan is often very slow on Windows). Use **`--probe-cameras`** when you need a list of devices. **`--serial-wait`** controls the pause after opening USB serial (default **0.5s**; increase if the Arduino misses commands).

When you start `run_camera.py`, it **asks for the Arduino serial port** first (e.g. `COM3`, or Enter for **camera only**). On Windows, `pip install pygrabber` adds friendly names when you use **`--probe-cameras`**.

## Model files

Place (or point `--checkpoint` / `--classes` at):

- `artifacts/dinosaur_classifier.pt`
- `artifacts/class_names.json`

## Run

**Pipeline:** **Serial port prompt** → camera list → preview. Opens **camera index 3** by default (`--camera` to change). Starts **rotated 90°** clockwise; press **P** to cycle view rotation. With a serial port, **R / B / G / Y** start a **fast servo sweep** until that color class locks in the ROI; **X** cancels seek. Capture requests **1920×1080**. After rotation, **16:9** crop; ROI **≤ 480 px** (`--max-roi-side`).

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

## Servo seek (Arduino + classifier)

Flash **`sketch_mar28a.ino`** (115200 baud): **one integer angle per line** (`0`–`180`), servo on **A1**.

- **Live preview:** run `python run_camera.py`, enter a **COM port** when asked (or Enter for camera-only). **R / B / G / Y** starts seek; **P** rotates the view; **X** cancels seek.
- **Headless (no window):** sweep until a color is found, then exit — you’ll be prompted for the port:

```bash
python run_camera.py --seek red
```

Tune with **`--seek-angle-step`**, **`--seek-settle`**, **`--seek-min-confidence`**, **`--seek-hits`**, **`--seek-max-frames`**, **`--seek-forward-only`**.

Class names in `class_names.json` must match (e.g. `"red"`, `"blue"`).

## Troubleshooting

- **Checkpoint not found:** Ensure the two files exist under `artifacts/` or pass explicit paths.
- **Wrong or black camera:** Try `--camera 0`, `1`, …
- **Slow:** Use `--every-n` or `--device cuda` if available.