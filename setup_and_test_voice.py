"""
RockDinosaurMartin — ElevenLabs voice system setup and test.

This script:
  1. Installs all required Python packages.
  2. Prompts for your ElevenLabs API key and voice ID, saves them to .env.
  3. Lists all available voices on your account so you can pick one.
  4. Tests every phrase in the caveman phrase map (or a single phrase for a quick check).

Run with:
  python setup_and_test_voice.py          # full interactive setup
  python setup_and_test_voice.py --test   # just test all phrases (needs .env already set)
  python setup_and_test_voice.py --list   # list your voices then exit
"""

import argparse
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Step 1 — Install dependencies
# ---------------------------------------------------------------------------

# elevenlabs>=1.x has PyPI wheels for Python 3.13 on Windows (old 0.3.x pulled pydantic-core source builds).
REQUIRED_PACKAGES = [
    "elevenlabs>=1.50.0",
    "python-dotenv>=1.0.0",
    "pygame>=2.5.0",
]


def install_packages() -> None:
    print("Installing required packages ...")
    for pkg in REQUIRED_PACKAGES:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", pkg]
        )
    print("All packages installed.\n")


# ---------------------------------------------------------------------------
# Step 2 — .env setup
# ---------------------------------------------------------------------------

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)


def save_env(api_key: str, voice_id: str) -> None:
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("# RockDinosaurMartin — ElevenLabs (used by caveman_voice.py)\n")
        f.write(f"ELEVENLABS_API_KEY={api_key}\n")
        f.write(f"G5uOcBecoT17CX53fnzP={voice_id}\n")
    print(f"Saved to {ENV_PATH}\n")


def prompt_credentials() -> tuple[str, str]:
    print("=" * 55)
    print("  ElevenLabs API Setup")
    print("=" * 55)
    print("Get your API key at: https://elevenlabs.io → Profile → API Keys")
    print("Get Voice ID at:     Voices → pick one → click </> icon\n")

    api_key = input("Paste your ElevenLabs API key: ").strip()
    if not api_key:
        print("ERROR: API key cannot be empty.")
        sys.exit(1)

    voice_id = input("Paste your Voice ID (or press Enter to list voices first): ").strip()
    return api_key, voice_id


# ---------------------------------------------------------------------------
# Step 3 — Voice utilities
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set. Run without --test first.")
        sys.exit(1)
    return api_key


def list_voices() -> None:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=get_api_key())
    resp = client.voices.search()
    all_voices = getattr(resp, "voices", None) or []
    print("\nAvailable voices on your account:")
    print(f"  {'Name':<30} {'Voice ID'}")
    print("  " + "-" * 62)
    for v in all_voices:
        vid = getattr(v, "voice_id", None) or getattr(v, "voiceId", "")
        name = getattr(v, "name", "") or "(unnamed)"
        print(f"  {name:<30} {vid}")
    print()


# ---------------------------------------------------------------------------
# Step 4 — Speak a phrase
# ---------------------------------------------------------------------------

from caveman_voice import speak


# ---------------------------------------------------------------------------
# Step 5 — Test all phrases
# ---------------------------------------------------------------------------

def test_all_phrases(voice_id: str) -> None:
    sys.path.insert(0, os.path.dirname(__file__))
    from state import PHRASES

    total = len(PHRASES)
    print(f"\nTesting {total} phrases ...\n")

    for i, (key, entry) in enumerate(PHRASES.items(), start=1):
        caveman = entry["caveman"]
        english = entry["english"]
        print(f"[{i}/{total}] {key}")
        print(f"         Caveman : {caveman}")
        print(f"         English : {english}")
        speak(caveman, voice_id)
        input("         Press Enter for next phrase ... ")
        print()

    print("All phrases tested.")


def test_single_phrase(voice_id: str) -> None:
    phrase = "GRAKA! BIG TOOTH BEAST! TRIBE DANGER! OOK!"
    print(f"\nQuick test phrase: {phrase}")
    speak(phrase, voice_id)
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="RockDinosaurMartin voice setup and test")
    ap.add_argument("--test",   action="store_true", help="Test all phrases (requires .env)")
    ap.add_argument("--quick",  action="store_true", help="Test one phrase only (requires .env)")
    ap.add_argument("--list",   action="store_true", help="List voices then exit")
    args = ap.parse_args()

    # Always install packages first.
    install_packages()

    if args.test or args.quick or args.list:
        load_env()
        if args.list:
            list_voices()
            return 0
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
        if not voice_id or voice_id == "your_voice_id_here":
            print("ERROR: ELEVENLABS_VOICE_ID not set in .env — run without flags first.")
            return 1
        if args.quick:
            test_single_phrase(voice_id)
        else:
            test_all_phrases(voice_id)
        return 0

    # --- Interactive setup ---
    print("\nRockDinosaurMartin — Voice System Setup\n")

    api_key, voice_id = prompt_credentials()

    save_env(api_key, voice_id or "PLACEHOLDER")
    load_env()

    if not voice_id:
        list_voices()
        voice_id = input("Enter Voice ID from the list above: ").strip()
        if not voice_id:
            print("ERROR: Voice ID required.")
            return 1
        save_env(api_key, voice_id)
        load_env()

    print("\nRunning quick voice test ...")
    test_single_phrase(voice_id)

    run_full = input("\nRun full phrase test for all caveman lines? [y/N]: ").strip().lower()
    if run_full == "y":
        test_all_phrases(voice_id)

    print("\nSetup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
