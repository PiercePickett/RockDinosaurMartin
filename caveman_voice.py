"""
ElevenLabs TTS for caveman lines defined in ``state.PHRASES``.

Uses the current ``elevenlabs`` client SDK (prebuilt wheels on Python 3.13+ Windows).
Requires ``.env`` with ``ELEVENLABS_API_KEY`` and ``ELEVENLABS_VOICE_ID`` (see setup script).

Logs every intended playback to stdout with the prefix ``[voice]``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _voice_log(msg: str) -> None:
    print(f"[voice] {msg}", flush=True)


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass


def _audio_to_bytes(audio: object) -> bytes:
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    chunks: list[bytes] = []
    for chunk in audio:  # type: ignore[union-attr]
        if isinstance(chunk, (bytes, bytearray)):
            chunks.append(bytes(chunk))
    return b"".join(chunks)


def speak(text: str, voice_id: str | None = None) -> None:
    """Convert text to speech and play via pygame (blocking until playback ends)."""
    import io

    import pygame
    from elevenlabs.client import ElevenLabs

    load_env()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    vid = voice_id or os.getenv("ELEVENLABS_VOICE_ID")
    preview = text if len(text) <= 120 else text[:117] + "..."
    if not api_key or not vid:
        _voice_log(f"speak SKIPPED (missing API key or voice id) would have played: {preview!r}")
        print(
            "Voice: skipped (set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env).",
            flush=True,
        )
        return

    _voice_log(f"speak START ({len(text)} chars): {preview!r}")

    client = ElevenLabs(api_key=api_key)
    # Signature: convert(voice_id, *, text=..., model_id=..., ...)
    audio = client.text_to_speech.convert(
        vid,
        text=text,
        model_id="eleven_turbo_v2_5",
        output_format="mp3_44100_128",
    )
    data = _audio_to_bytes(audio)

    pygame.mixer.init()
    sound = pygame.mixer.Sound(io.BytesIO(data))
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.wait(50)
    pygame.mixer.quit()


def speak_phrase_key(phrase_key: str, *, enabled: bool = True) -> None:
    """Play ``state.PHRASES[phrase_key][\"caveman\"]`` if enabled and API is configured."""
    if not enabled:
        _voice_log(f"phrase SKIPPED (enabled=False): {phrase_key!r}")
        return
    try:
        import state

        entry = state.PHRASES.get(phrase_key)
        if not entry:
            _voice_log(f"phrase SKIPPED (unknown key): {phrase_key!r}")
            return
        cave = entry.get("caveman", "")
        if not cave:
            _voice_log(f"phrase SKIPPED (empty caveman): {phrase_key!r}")
            return
        _voice_log(f"phrase PLAY {phrase_key!r} → {cave[:100]}{'…' if len(cave) > 100 else ''}")
        speak(cave)
    except Exception as e:
        _voice_log(f"phrase ERROR {phrase_key!r}: {e}")
        print(f"Voice skipped ({phrase_key}): {e}", flush=True)


def speak_command_bits(bits: list[int], *, enabled: bool = True) -> None:
    """
    For each set bit in ``bits`` [R,G,B,Y], play ``cmd_red`` / ``cmd_green`` / … in order.
    If all bits are zero, play ``waiting`` (tribe idle).
    Call this after gesture SEND so each new command is heard.
    """
    if not enabled:
        _voice_log("command_bits SKIPPED (voice disabled for this process)")
        return
    import state

    b = [1 if int(x) else 0 for x in bits[:4]]
    while len(b) < 4:
        b.append(0)
    _voice_log(f"command_bits PLAY sequence for bits={b}")
    any_hit = False
    for i, name in enumerate(state.TARGET_NAMES):
        if b[i]:
            speak_phrase_key(f"cmd_{name}", enabled=True)
            any_hit = True
    if not any_hit:
        _voice_log("command_bits all zeros → playing 'waiting'")
        speak_phrase_key("waiting", enabled=True)
