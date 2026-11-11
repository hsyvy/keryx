"""Audio bridge between the phone line and the voice model.

- converts sample rates (transport <-> model),
- paces model audio into the call in real time, so barge-in can drop what has not been played yet,
- gates the mic while we speak on echo-prone transports (speakerphone),
- optionally records each side to WAV.

All PCM is s16le mono.
"""

from __future__ import annotations

import array
import asyncio
import contextlib
import math
import time
import wave
from pathlib import Path
from typing import Optional

try:  # numpy is in Hermes's venv; keep a pure-python fallback for bare environments
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    if _np is not None:
        a = _np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(_np.float32)
        return float(_np.sqrt(_np.mean(a * a))) if a.size else 0.0
    a = array.array("h", pcm[: len(pcm) // 2 * 2])
    return math.sqrt(sum(x * x for x in a) / len(a)) if a else 0.0


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampler (good enough for speech at 8-48 kHz)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    n_in = len(pcm) // 2
    n_out = max(1, round(n_in * dst_rate / src_rate))
    if _np is not None:
        x = _np.frombuffer(pcm[: n_in * 2], dtype="<i2").astype(_np.float32)
        pos = _np.arange(n_out, dtype=_np.float64) * (src_rate / dst_rate)
        y = _np.interp(pos, _np.arange(n_in), x)
        return _np.clip(_np.round(y), -32768, 32767).astype("<i2").tobytes()
    src = array.array("h", pcm[: n_in * 2])
    out = array.array("h")
    step = src_rate / dst_rate
    for i in range(n_out):
        p = i * step
        j = int(p)
        frac = p - j
        a = src[min(j, n_in - 1)]
        b = src[min(j + 1, n_in - 1)]
        out.append(int(round(a + (b - a) * frac)))
    return out.tobytes()


class _Recorder:
    def __init__(self, directory: Path, rate: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._files = {}
        for side in ("callee", "agent"):
            w = wave.open(str(directory / f"{side}.wav"), "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            self._files[side] = w

    def write(self, side: str, pcm: bytes) -> None:
        w = self._files.get(side)
        if w is not None:
            w.writeframes(pcm)

    def close(self) -> None:
        for w in self._files.values():
            with contextlib.suppress(Exception):
                w.close()
        self._files = {}


class AudioBridge:
    def __init__(self, transport, model_in_rate: int, model_out_rate: int, *,
                 record_dir: Optional[Path] = None, barge_in_rms: float = 2500.0,
                 lead_ms: Optional[int] = None) -> None:
        self.t = transport
        self.rate = transport.sample_rate
        self.frame_bytes = transport.frame_bytes
        self.frame_s = transport.frame_ms / 1000.0
        self.model_in_rate, self.model_out_rate = model_in_rate, model_out_rate
        self.barge_in_rms = barge_in_rms
        # Keep this much audio queued ahead of real time in the transport. pw-play reads stdin once per
        # graph cycle (~21 ms); if we hand it each frame just in time it often finds the pipe empty and
        # plays silence. Costs this much extra latency; barge-in can't recall audio already handed over.
        self.lead_frames = max(0, int((getattr(transport, "playback_lead_ms", 0) if lead_ms is None else lead_ms)
                                      / transport.frame_ms))
        self._out = bytearray()
        self._played_bytes = 0          # bytes of the current model response actually played
        self._last_voice_out = 0.0
        self._pacer: Optional[asyncio.Task] = None
        self._drained = asyncio.Event()
        self._drained.set()
        self._recorder = _Recorder(record_dir, self.rate) if record_dir else None
        self.recording_dir = str(record_dir) if record_dir else None

    # --- downlink (model -> phone) ---------------------------------------------------------
    def start(self) -> None:
        if self._pacer is None:
            self._pacer = asyncio.create_task(self._pace())

    def enqueue_model_audio(self, pcm: bytes) -> None:
        self._enqueue(resample(pcm, self.model_out_rate, self.rate))

    def play(self, pcm: bytes) -> None:
        """Queue audio already at the transport rate (e.g. the disclosure line)."""
        self._enqueue(pcm)

    def _enqueue(self, pcm: bytes) -> None:
        if pcm:
            self._out.extend(pcm)
            self._drained.clear()

    def new_response(self) -> None:
        self._played_bytes = 0

    def barge_in(self) -> int:
        """Drop unplayed audio; return milliseconds of the current response that were heard."""
        self._out.clear()
        self._drained.set()
        return int(self._played_bytes / 2 / self.rate * 1000)

    @property
    def speaking(self) -> bool:
        return bool(self._out) or (time.monotonic() - self._last_voice_out) < 0.3

    async def wait_drained(self, timeout: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._drained.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _pace(self) -> None:
        silence = bytes(self.frame_bytes)
        try:
            for _ in range(self.lead_frames):
                await self.t.write_frame(silence)
            next_at = time.monotonic()
            while True:
                if self._out:
                    frame = bytes(self._out[: self.frame_bytes])
                    del self._out[: self.frame_bytes]
                    if len(frame) < self.frame_bytes:
                        frame += bytes(self.frame_bytes - len(frame))
                    self._played_bytes += len(frame)
                    self._last_voice_out = time.monotonic()
                    if not self._out:
                        self._drained.set()
                else:
                    frame = silence
                await self.t.write_frame(frame)
                if self._recorder:
                    self._recorder.write("agent", frame)
                next_at += self.frame_s
                delay = next_at - time.monotonic()
                if delay < -0.2:  # fell far behind (stall): resync rather than burst
                    next_at = time.monotonic()
                    delay = 0
                await asyncio.sleep(max(0.0, delay))
        except asyncio.CancelledError:
            pass

    # --- uplink (phone -> model) -----------------------------------------------------------
    def process_inbound(self, frame: bytes) -> bytes:
        """Record, echo-gate and resample one inbound frame for the model."""
        if self._recorder:
            self._recorder.write("callee", frame)
        if getattr(self.t, "echo_prone", False) and self.speaking and rms(frame) < self.barge_in_rms:
            frame = bytes(len(frame))  # our own voice leaking back through the room
        return resample(frame, self.rate, self.model_in_rate)

    async def close(self) -> None:
        if self._pacer:
            self._pacer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pacer
            self._pacer = None
        if self._recorder:
            self._recorder.close()
