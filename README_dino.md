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

- # Jetson Dino Classification Server

How to wire the RockDinosaurMartin repo into an HTTP server so your Arduino
can call it over WiFi.

---

## File layout

```
RockDinosaurMartin/          ← your existing repo clone
├── artifacts/
│   ├── dinosaur_classifier.pt
│   └── class_names.json
├── dino_server.py           ← DROP THIS HERE  (new)
├── run_camera.py            ← unchanged (still works standalone)
├── requirements.txt
└── ...
```

`dino_client.ino` goes in your Arduino sketch folder.

---

## 1 — Jetson setup

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/PiercePickett/RockDinosaurMartin
cd RockDinosaurMartin

# Create / activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install existing deps + Flask
pip install -r requirements.txt
pip install flask

# Copy dino_server.py here
cp /path/to/dino_server.py .
```

---

## 2 — Start the server

```bash
# Basic (CPU, camera index 0, port 5000)
python dino_server.py

# With CUDA (Jetson GPU)
python dino_server.py --device cuda

# Different camera
python dino_server.py --camera 1

# All options
python dino_server.py --help
```

The server binds to `0.0.0.0:5000` by default, so it's reachable from any
device on your local network.

**Find the Jetson's IP:**
```bash
hostname -I
# example output: 192.168.1.42
```

---

## 3 — Test from a PC / phone

```bash
# Health check
curl http://192.168.1.42:5000/health

# One-shot classification (Jetson grabs from its camera)
curl http://192.168.1.42:5000/classify

# Send your own image
curl -X POST http://192.168.1.42:5000/classify \
     -H "Content-Type: image/jpeg" \
     --data-binary @photo.jpg

# Live SSE stream (open in browser)
http://192.168.1.42:5000/classify/stream
```

---

## 4 — Arduino setup

1. Open `dino_client.ino` in the Arduino IDE.
2. Edit the top of the file:
   ```cpp
   #define WIFI_SSID   "YourNetworkName"
   #define WIFI_PASS   "YourNetworkPassword"
   #define JETSON_IP   "192.168.1.42"   // ← your Jetson's IP
   ```
3. Edit `DINO_MAP[]` to match your `class_names.json` and desired servo angles.
4. Install required libraries via **Tools → Manage Libraries**:
   - **ArduinoJson** (Benoit Blanchon)
   - ESP8266 or ESP32 board packages include WiFi + HTTPClient automatically.
5. Select your board (NodeMCU / Wemos D1 / ESP32) and flash.

The Arduino will poll `/classify` every 500 ms and move its servo to the
angle mapped to the detected dinosaur class.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status + class list |
| GET | `/classify` | Grab camera frame, classify, return JSON |
| POST | `/classify` | Classify image you send (raw JPEG body) |
| GET | `/classify/stream` | SSE stream of continuous results |

### Response format

```json
{
  "class": "triceratops",
  "class_index": 2,
  "confidence": 0.9134,
  "scores": {
    "triceratops": 0.9134,
    "trex": 0.0521,
    "stegosaurus": 0.0345
  },
  "roi": {"x1": 400, "y1": 140, "x2": 880, "y2": 620},
  "timestamp": 1743000000.12
}
```

---

## Run on boot (systemd)

```ini
# /etc/systemd/system/dino-server.service
[Unit]
Description=Dino Classification Server
After=network.target

[Service]
User=your_username
WorkingDirectory=/home/your_username/RockDinosaurMartin
ExecStart=/home/your_username/RockDinosaurMartin/.venv/bin/python dino_server.py --device cuda
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp dino-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dino-server
sudo systemctl start dino-server
sudo systemctl status dino-server
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: Checkpoint not found` | Verify `artifacts/dinosaur_classifier.pt` exists; pass `--checkpoint path/to/file.pt` |
| Arduino gets HTTP -1 | Wrong IP or Jetson firewall; run `curl` from a PC first |
| `Camera read failed` | Try `--camera 0` or `--camera 1`; check `ls /dev/video*` on Jetson |
| Slow inference | Add `--device cuda`; or reduce resolution with `--capture-width 640 --capture-height 480` |
| `ModuleNotFoundError: flask` | `pip install flask` inside your venv |
