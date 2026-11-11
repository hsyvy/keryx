"""The AI-disclosure line, rendered to audio and played by the plugin itself (never left to the model).

Sources, in order: a configured WAV file; OpenAI TTS (cached per text+voice); espeak-ng. Rendering
happens BEFORE dialing, so a call never connects without its disclosure ready (fail closed)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import shutil
import subprocess
import urllib.request
import wave
from pathlib import Path
from typing import Optional

from .audio import resample

TTS_URL = "https://api.openai.com/v1/audio/speech"
TTS_RATE = 24000


class DisclosureError(RuntimeError):
    pass


def wav_to_pcm(data: bytes, rate: int) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != 2:
            raise DisclosureError("disclosure WAV must be 16-bit PCM")
        ch, src_rate = w.getnchannels(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if ch > 1:
        import array
        a = array.array("h", frames)
        frames = array.array("h", (int(sum(a[i:i + ch]) / ch) for i in range(0, len(a), ch))).tobytes()
    return resample(frames, src_rate, rate)


def _openai_tts(text: str, voice: str, api_key: str, timeout: float = 20.0) -> bytes:
    body = json.dumps({"model": "gpt-4o-mini-tts", "voice": voice, "input": text, "response_format": "pcm",
                       "instructions": "Friendly, clear and unhurried, like a polite phone call."}).encode()
    req = urllib.request.Request(TTS_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        return resp.read()


def _espeak(text: str) -> bytes:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise DisclosureError("no TTS available (espeak-ng not installed)")
    out = subprocess.run([exe, "--stdout", text], capture_output=True, timeout=20, check=True)
    return out.stdout


async def render_disclosure(text: str, rate: int, *, voice: str = "marin", api_key: str = "",
                            audio_file: str = "", cache_dir: Optional[Path] = None) -> bytes:
    if not text.strip() and not audio_file:
        raise DisclosureError("disclosure_text is empty; Keryx will not call without a disclosure")
    if audio_file:
        try:
            return wav_to_pcm(Path(audio_file).expanduser().read_bytes(), rate)
        except (OSError, wave.Error) as e:
            raise DisclosureError(f"cannot read disclosure_audio_file: {e}") from e
    errors = []
    if api_key:
        key = hashlib.sha256(f"{voice}|{text}".encode()).hexdigest()[:24]
        cached = cache_dir / f"{key}.pcm" if cache_dir else None
        if cached and cached.exists() and cached.stat().st_size > 0:
            return resample(cached.read_bytes(), TTS_RATE, rate)
        try:
            pcm = await asyncio.to_thread(_openai_tts, text, voice, api_key)
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(pcm)
            return resample(pcm, TTS_RATE, rate)
        except Exception as e:
            errors.append(f"OpenAI TTS: {e}")
    try:
        return wav_to_pcm(await asyncio.to_thread(_espeak, text), rate)
    except Exception as e:
        errors.append(f"espeak: {e}")
    raise DisclosureError("could not render the disclosure line (" + "; ".join(errors) + ")")
