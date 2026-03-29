"""
Combined Jetson hardware test: USB webcam + pan servo + laser.

Controls (click the OpenCV window first):
  A / Left arrow   : pan left
  D / Right arrow  : pan right
  Space            : toggle laser on/off
  R                : center servo (90)
  Q / ESC          : quit

Run with: sudo python3 test_all.py
"""

import sys
import threading
import time

import cv2
import numpy as np

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("WARNING: Jetson.GPIO not found — running in webcam-only mode.")
    GPIO_AVAILABLE = False

# --- Pin configuration (board numbering) ---
PAN_PIN = 32
LASER_PIN = 29

# --- Servo PWM spec ---
PWM_FREQ_HZ = 50
DUTY_MIN = 2.5
DUTY_MAX = 12.5

# --- Webcam ---
WEBCAM_INDEX = 0
DISPLAY_SCALE = 1

# --- Control step size per keypress ---
ANGLE_STEP = 3


def angle_to_duty(angle: float) -> float:
    angle = max(0.0, min(180.0, float(angle)))
    return DUTY_MIN + (DUTY_MAX - DUTY_MIN) * angle / 180.0


class HardwareController:
    """Manages servo PWM and laser GPIO in a dedicated thread."""

    def __init__(self):
        self.pan = 90
        self.laser_on = False
        self._lock = threading.Lock()
        self._running = False
        self.pan_pwm = None

    def start(self):
        if not GPIO_AVAILABLE:
            return
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(PAN_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(LASER_PIN, GPIO.OUT, initial=GPIO.LOW)

        self.pan_pwm = GPIO.PWM(PAN_PIN, PWM_FREQ_HZ)
        self.pan_pwm.start(angle_to_duty(90))
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            with self._lock:
                pan = self.pan
                laser = self.laser_on
            if self.pan_pwm:
                self.pan_pwm.ChangeDutyCycle(angle_to_duty(pan))
            GPIO.output(LASER_PIN, GPIO.HIGH if laser else GPIO.LOW)
            time.sleep(0.02)

    def set(self, pan: int, laser: bool):
        with self._lock:
            self.pan = max(0, min(180, pan))
            self.laser_on = laser

    def stop(self):
        self._running = False
        if not GPIO_AVAILABLE:
            return
        time.sleep(0.05)
        if self.pan_pwm:
            self.pan_pwm.ChangeDutyCycle(angle_to_duty(90))
            time.sleep(0.2)
            self.pan_pwm.stop()
        GPIO.output(LASER_PIN, GPIO.LOW)
        GPIO.cleanup()


def draw_hud(frame: np.ndarray, pan: int,
             laser_on: bool, fps: float) -> np.ndarray:
    out = frame.copy()
    lines = [
        f"PAN: {pan:3d}  LASER: {'ON ' if laser_on else 'OFF'}",
        f"FPS: {fps:.1f}",
        "A/D or Left/Right arrows: pan servo",
        "Space: laser   R: center   Q/ESC: quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 26
    return out


def main() -> int:
    hw = HardwareController()
    hw.start()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam at index {WEBCAM_INDEX}.")
        hw.stop()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win_name = "Jetson All-in-One Test"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    pan = 90
    laser_on = False
    fps = 0.0
    frame_count = 0
    t0 = time.time()

    print("Running. Click the window and use controls listed on screen.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            now = time.time()
            elapsed = now - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = now

            display = draw_hud(frame, pan, laser_on, fps)
            if DISPLAY_SCALE != 1:
                h, w = display.shape[:2]
                display = cv2.resize(display, (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
                                     interpolation=cv2.INTER_LINEAR)
            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            changed = False

            if key in (ord("q"), 27):
                break
            elif key in (ord("a"), 81):   # left arrow = 81
                pan = max(0, pan - ANGLE_STEP)
                changed = True
            elif key in (ord("d"), 83):   # right arrow = 83
                pan = min(180, pan + ANGLE_STEP)
                changed = True
            elif key == ord(" "):
                laser_on = not laser_on
                changed = True
            elif key == ord("r"):
                pan = 90
                changed = True

            if changed:
                hw.set(pan, laser_on)

    except KeyboardInterrupt:
        pass
    finally:
        hw.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
