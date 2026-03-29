"""
Gesture client: MediaPipe Hands (Tasks API) + UDP + state.save_target_bits.

Requires mediapipe>=0.10 (legacy ``mp.solutions`` was removed from the PyPI wheel).
Run from the repo root: ``python src/gestureClient.py``
"""
from __future__ import annotations

import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import state  # noqa: E402

# Official float16 hand landmarker (downloaded once into .cache/)
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_CACHE_DIR = _ROOT / ".cache"
_HAND_MODEL_PATH = _CACHE_DIR / "hand_landmarker.task"

# --- 1. Network Setup ---
# UDP send destination: never use 0.0.0.0 here (invalid on Windows; bind-only address).
# Server: ``python src/gestureServer.py`` listens on 0.0.0.0:8000 and receives localhost.
SERVER_IP = os.environ.get("GESTURE_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("GESTURE_SERVER_PORT", "8000"))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def _ensure_hand_model() -> Path:
    override = os.environ.get("MEDIAPIPE_HAND_MODEL_PATH")
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"MEDIAPIPE_HAND_MODEL_PATH not found: {p}")
        return p
    if _HAND_MODEL_PATH.is_file():
        return _HAND_MODEL_PATH
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {_HAND_MODEL_PATH} …", flush=True)
    urllib.request.urlretrieve(_HAND_MODEL_URL, _HAND_MODEL_PATH)
    return _HAND_MODEL_PATH


def _open_camera(index: int = 0) -> cv2.VideoCapture:
    if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def _count_fingers(lm_list: list, label: str) -> int:
    tip_ids = [8, 12, 16, 20]
    fingers: list[int] = []
    for i in tip_ids:
        lm = lm_list[i]
        pip = lm_list[i - 2]
        fingers.append(1 if lm.y < pip.y else 0)
    t_tip = lm_list[4].x
    t_base = lm_list[2].x
    if label == "Right":
        fingers.append(1 if t_tip < t_base else 0)
    else:
        fingers.append(1 if t_tip > t_base else 0)
    return sum(fingers)


def _draw_hand_skeleton(
    frame_bgr: np.ndarray, landmarks: list, connections: list
) -> None:
    h, w = frame_bgr.shape[:2]
    for conn in connections:
        a = landmarks[conn.start]
        b = landmarks[conn.end]
        cv2.line(
            frame_bgr,
            (int(a.x * w), int(a.y * h)),
            (int(b.x * w), int(b.y * h)),
            (255, 255, 255),
            2,
        )
    for lm in landmarks:
        cv2.circle(
            frame_bgr, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1
        )


def main() -> None:
    # Create state_target_bits.json from state.TARGET_BITS if missing (same folder as state.py).
    state.load_target_bits()

    # --- 2. MediaPipe HandLandmarker (Tasks) ---
    model_path = _ensure_hand_model()
    options = HandLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.8,
        min_tracking_confidence=0.8,
    )
    landmarker = HandLandmarker.create_from_options(options)
    connections = HandLandmarksConnections.HAND_CONNECTIONS

    cap = _open_camera(0)

    # --- 3. Timing + State ---
    dwell_time = 4.0

    last_gesture = None
    gesture_start_time = 0.0
    gesture_locked = False

    command_mode = False
    command_bits = [0, 0, 0, 0]

    frame_ts_ms = 0

    print(f"Streaming to {SERVER_IP}:{PORT}...")

    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Camera error")
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = np.ascontiguousarray(img_rgb)

            frame_ts_ms += 33
            mp_frame = mp_image.Image(mp_image.ImageFormat.SRGB, img_rgb)
            result = landmarker.detect_for_video(mp_frame, frame_ts_ms)

            hand_counts = {"Right": 0, "Left": 0}

            if result.hand_landmarks:
                for i, hand_lms in enumerate(result.hand_landmarks):
                    label = "Right"
                    if i < len(result.handedness) and result.handedness[i]:
                        cats = result.handedness[i]
                        if cats and cats[0].category_name:
                            label = cats[0].category_name
                    _draw_hand_skeleton(img, hand_lms, connections)
                    if label in hand_counts:
                        hand_counts[label] = _count_fingers(hand_lms, label)

            msg = f"{hand_counts['Right']}/{hand_counts['Left']}"
            current_gesture = msg
            now = time.time()

            if current_gesture != last_gesture:
                last_gesture = current_gesture
                gesture_start_time = now
                gesture_locked = False
            else:
                if (now - gesture_start_time) >= dwell_time and not gesture_locked:
                    gesture_locked = True

                    print(f"Confirmed: {current_gesture}")

                    if current_gesture == "4/4" and not command_mode:
                        command_mode = True
                        command_bits = [0, 0, 0, 0]
                        print(">>> ENTERED ATTENTION MODE <<<")

                    elif command_mode:
                        if current_gesture == "1/0":
                            command_bits[0] = 1
                        elif current_gesture == "2/0":
                            command_bits[1] = 1
                        elif current_gesture == "3/0":
                            command_bits[2] = 1
                        elif current_gesture == "4/0":
                            command_bits[3] = 1
                        elif current_gesture == "0/4":
                            command_bits = [1 ^ b for b in command_bits]
                        elif current_gesture == "0/0":
                            command_bits = [0, 0, 0, 0]
                        elif current_gesture == "3/3":
                            command_bits = [0, 0, 0, 0]
                            print("Command Cleared")
                        elif current_gesture == "2/2":
                            bit_string = "".join(map(str, command_bits))
                            sock.sendto(bit_string.encode(), (SERVER_IP, PORT))
                            state.save_target_bits(command_bits)
                            print(
                                f">>> SENT: {bit_string} "
                                f"(saved to state / state_target_bits.json) <<<"
                            )
                            command_mode = False

            # When IDLE, show bits from state_target_bits.json so run_camera --shoot-mission
            # clearing bits updates the HUD. In ATTENTION mode, show the in-progress command_bits.
            bits_display = command_bits if command_mode else state.load_target_bits()

            cv2.putText(
                img,
                f"Gesture: {msg}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f"Mode: {'ATTENTION' if command_mode else 'IDLE'}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                img,
                f"Bits: {''.join(map(str, bits_display))}"
                + ("  (editing)" if command_mode else "  (file)"),
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Gesture Controller", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()
