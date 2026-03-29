"""
Shared caveman phrase map, gesture-to-command mapping, and classifier index table.

Imported by:
  - run_camera.py              (preview + ``--shoot-mission``)
  - src/gestureClient.py       (writes TARGET_BITS on SEND)
  - src/caveman_controller.py  (runtime)
  - setup_and_test_voice.py    (voice testing)

Cross-process control: ``save_target_bits`` / ``load_target_bits`` persist
``TARGET_BITS`` to ``state_target_bits.json`` next to this file so the gesture
program and ``run_camera`` (separate processes) share the same mission bits.
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET_NAMES = ("red", "green", "blue", "yellow")

_RUNTIME_FILE = Path(__file__).resolve().with_name("state_target_bits.json")


def runtime_bits_path() -> Path:
    """Path to ``state_target_bits.json`` (e.g. for mtime polling from ``run_camera``)."""
    return _RUNTIME_FILE

# Shoot-mission order: for each index, 1 = find and shoot this color, 0 = skip.
# Same order as TARGET_NAMES — red, green, blue, yellow.
# Used by ``python run_camera.py --shoot-mission`` (requires serial + Arduino laser).
# At runtime, prefer ``load_target_bits()`` so values from ``state_target_bits.json``
# (written by the gesture client on SEND) override these defaults.
TARGET_BITS = [1, 0, 1, 0]


def load_target_bits() -> list[int]:
    """Return [R,G,B,Y] bits from ``state_target_bits.json`` if valid.

    If the file is missing or unreadable, it is **created** from the current
    ``TARGET_BITS`` values in this module (defaults ``[1, 0, 1, 0]`` unless
    you edited them before calling ``load_target_bits``).
    """
    if _RUNTIME_FILE.is_file():
        try:
            data = json.loads(_RUNTIME_FILE.read_text(encoding="utf-8"))
            bits = data.get("TARGET_BITS")
            if isinstance(bits, list) and len(bits) >= 4:
                return [1 if int(x) else 0 for x in bits[:4]]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    seeded = [1 if int(x) else 0 for x in TARGET_BITS[:4]]
    save_target_bits(seeded)
    return seeded


def save_target_bits(bits: list[int]) -> None:
    """Persist [R,G,B,Y] to ``state_target_bits.json`` and update ``TARGET_BITS`` in-memory."""
    b = [1 if int(x) else 0 for x in bits[:4]]
    while len(b) < 4:
        b.append(0)
    b = b[:4]
    TARGET_BITS[:] = b
    _RUNTIME_FILE.write_text(
        json.dumps({"TARGET_BITS": b}, indent=2) + "\n",
        encoding="utf-8",
    )

# Classifier class name -> bit-array index  [R, G, B, Y]
CLASS_TO_TARGET_IDX = {
    "red":    0,
    "green":  1,
    "blue":   2,
    "yellow": 3,
}

# Gesture finger-count string -> 4-bit target array [R, G, B, Y]
# "0/4" is special: it inverts the current bit array rather than setting fixed bits.
GESTURE_TO_BITS = {
    "1/0": [1, 0, 0, 0],   # 1 right finger  -> red
    "2/0": [0, 0, 0, 1],   # 2 right fingers -> yellow
    "3/0": [0, 1, 0, 0],   # 3 right fingers -> green
    "4/0": [0, 0, 1, 0],   # 4 right fingers -> blue
}
INVERT_GESTURE = "0/4"

# Caveman command phrases — spoken when a new command arrives
TARGET_COMMAND_PHRASES = {
    "red":    "UGA! RED FANG GET BONK FIRST! LASER FIND RED NOW!",
    "green":  "UGA! GREEN BUSH-LIZARD HIDE NO MORE! COME FACE CLUB!",
    "blue":   "UGA! BLUE ICE-LIZARD COLD-SKIN! ME SMASH WITH LIGHT!",
    "yellow": "UGA! YELLOW SNEAKY ONE GET SKY-THUNDER! SUN-SKIN GO DOWN!",
    "invert": "RAAAH! ZERO-FOUR! FLIP THE BONES! HUNT CHANGE NOW!",
}

# Caveman phrases spoken on state-machine transitions
STATE_PHRASES = {
    "target_locked":  "EYE OF TRIBE SEE BEAST! LASER LOCK!",
    "aiming":         "BONE ARM TURN... TURN... SPEAR OF LIGHT POINT TRUE!",
    "fired":          "THROW LIGHT-SPEAR! NOW!",
    "target_down":    "BEAST FALL! TRIBE STRONG!",
    "all_done":       "HUNT DONE. ALL LIZARD GONE. FIRE CIRCLE SAFE.",
    "no_target":      "NO BEAST IN SIGHT. TRIBE WAIT.",
    "waiting":        "TRIBE REST. WAITING FOR COMMAND.",
}

# ---------------------------------------------------------------------------
# Combined dict for setup_and_test_voice.py compatibility
# ---------------------------------------------------------------------------
PHRASES: dict[str, dict[str, str]] = {}

for _key, _phrase in TARGET_COMMAND_PHRASES.items():
    PHRASES[f"cmd_{_key}"] = {
        "caveman": _phrase,
        "english": f"Command: target {_key} dinosaur(s)",
    }

_STATE_ENGLISH = {
    "target_locked": "Camera has found the target",
    "aiming":        "Servo is moving to aim",
    "fired":         "Laser fired at target",
    "target_down":   "Target has been eliminated",
    "all_done":      "All targets cleared, mission complete",
    "no_target":     "No target visible during sweep",
    "waiting":       "Idle, waiting for next command",
}

for _key, _phrase in STATE_PHRASES.items():
    PHRASES[_key] = {
        "caveman": _phrase,
        "english": _STATE_ENGLISH.get(_key, _key),
    }
