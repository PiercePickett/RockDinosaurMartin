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

## Model files

Place (or point `--checkpoint` / `--classes` at):

- `artifacts/dinosaur_classifier.pt`
- `artifacts/class_names.json`

## Run

```bash
python run_camera.py
```

- **Quit:** **Q** or **Esc** in the preview window.
- **Different camera:** `python run_camera.py --camera 1`
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