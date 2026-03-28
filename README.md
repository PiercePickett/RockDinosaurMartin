# RockDinosaurMartin

Train a small image classifier for toy dinosaur variants (plus a “no dinosaur” class), then run it on a **local webcam** with OpenCV.

## Setup

Clone the repository (if needed), then:

```bash
cd RockDinosaurMartin
python -m venv .venv
```

**Windows (PowerShell):** `.venv\Scripts\activate`  
**macOS/Linux:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

You need **Python 3.10+** recommended.

## Dataset layout

Put images in **one folder per class** (folder names become labels; they are sorted alphabetically for class indices):

```
data/dataset/
  no_dinosaur/
  variant_red/
  variant_blue/
  ...
```

Use `.jpg`, `.jpeg`, `.png`, or other formats supported by torchvision.

## Train the model

### Option A: Jupyter on your computer

1. Copy or arrange your dataset under `data/dataset/` (or change `DATA_ROOT` in the notebook).
2. Open `train_dinosaur_classifier.ipynb` and run all cells.

### Option B: Google Colab

1. Upload the notebook to Colab (**Runtime → Change runtime type → GPU** recommended).
2. Upload your dataset zip or mount Google Drive; set `DATASET_ZIP` and/or `DATA_ROOT` / `OUTPUT_DIR` as described in the notebook.
3. Run all cells.

Training writes:

- `artifacts/dinosaur_classifier.pt` — weights and metadata  
- `artifacts/class_names.json` — class names in training order  

Copy those two files into this repo’s `artifacts/` folder on the machine where you run the camera if you trained on Colab.

## Run the live camera (local)

From the project root, with `artifacts/dinosaur_classifier.pt` and `artifacts/class_names.json` present:

```bash
python run_camera.py
```

- **Quit:** press **Q** or **Esc** in the preview window.
- **Another camera:** `python run_camera.py --camera 1` (try `0`, `1`, … if the default is wrong).
- **CPU vs GPU:** `python run_camera.py --device cpu` or `--device cuda` (default is auto).
- **Smoother labels:** default EMA is on; disable with `python run_camera.py --ema 0`.
- **Throttle inference:** `python run_camera.py --every-n 2` runs the model every 2 frames.

Custom artifact paths:

```bash
python run_camera.py --checkpoint path/to/dinosaur_classifier.pt --classes path/to/class_names.json
```

The camera script classifies **the full frame**, matching training if you trained on full-frame images. If you trained on crops, add cropping in `run_camera.py` to match.

## Project files

| File | Purpose |
|------|---------|
| `train_dinosaur_classifier.ipynb` | Train ResNet-18; Colab-aware paths and optional zip unzip |
| `run_camera.py` | Webcam inference with OpenCV |
| `requirements.txt` | Python dependencies |

## Troubleshooting

- **“Checkpoint not found”:** Train first, or copy `artifacts/` from Colab and point `--checkpoint` / `--classes`.
- **Black or wrong camera:** Change `--camera` index.
- **Low FPS:** Use `--every-n 2` or higher; use `--device cuda` if you have a GPU.
