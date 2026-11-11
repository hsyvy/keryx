"""speakerphone: fallback for any computer. You dial on your phone and put it on speaker next to the
laptop; Keryx listens on the default mic and talks through the default speaker.

The call is detected as answered when the mic hears sustained speech (the callee's "hello").
Echo-prone: the audio bridge gates the mic while Keryx is speaking (use PipeWire's echo-cancel
module for real barge-in on speakerphone)."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import time
from typing import Callable, Optional

from ..audio import rms
from .base import ANSWERED, NO_ANSWER, CallTransport, TransportError
from .pipewire import PwStream


class SpeakerphoneTransport(CallTransport):
    name = "speakerphone"
    sample_rate = 24000
    # pw-play pulls ~21.3 ms per graph cycle while we pace 20 ms frames; without a head start it
    # periodically finds the pipe short and drops a 20 ms slice (~1 every 0.6 s, measured).
    playback_lead_ms = 80
    echo_prone = True

    def __init__(self, notify: Optional[Callable[[str], None]] = None, source: Optional[str] = None,
                 sink: Optional[str] = None, speech_rms: float = 700.0, speech_ms: int = 300,
                 max_silence_s: float = 25.0) -> None:
        super().__init__(notify)
        self.source, self.sink = source, sink
        self.speech_rms, self.speech_ms, self.max_silence_s = speech_rms, speech_ms, max_silence_s
        self._capture: Optional[PwStream] = None
        self._playback: Optional[PwStream] = None
        self._ended = False
        self._number = ""
        self._last_voice = time.monotonic()

    def approval_note(self) -> str:
        return ("Speakerphone mode: after you approve, dial the number on your phone yourself and put it on "
                "speaker next to this computer. Keryx starts talking when the other side answers.")

    async def preflight(self) -> list[str]:
        return [f"{exe} not found" for exe in ("pw-record", "pw-play") if not shutil.which(exe)]

    async def dial(self, number: str) -> None:
        self._number = number
        self._notify(f"📞 Dial {number} on your phone now and put it on speaker next to the computer.")
        self._capture = PwStream("capture", self.sample_rate, self.source)
        await self._capture.start()

    async def wait_answered(self, timeout: float) -> str:
        if self._capture is None:
            raise TransportError("dial() first")
        need = max(1, self.speech_ms // self.frame_ms)
        voiced = 0
        deadline = time.monotonic() + timeout + 20  # the user still has to dial by hand
        while time.monotonic() < deadline:
            frame = await self._capture.read(self.frame_bytes)
            if frame is None:
                return NO_ANSWER
            voiced = voiced + 1 if rms(frame) >= self.speech_rms else 0
            if voiced >= need:
                self._last_voice = time.monotonic()
                return ANSWERED
        return NO_ANSWER

    async def start_audio(self) -> None:
        self._playback = PwStream("playback", self.sample_rate, self.sink)
        await self._playback.start()

    async def read_frame(self) -> Optional[bytes]:
        if self._capture is None:
            return None
        frame = await self._capture.read(self.frame_bytes)
        if frame is not None and rms(frame) >= self.speech_rms:
            self._last_voice = time.monotonic()
        return frame

    async def write_frame(self, pcm: bytes) -> None:
        if self._playback is not None:
            await self._playback.write(pcm)

    def remote_ended(self) -> bool:
        # No call signalling on speakerphone: treat long silence from the room as the call having ended.
        return self._ended or (time.monotonic() - self._last_voice > self.max_silence_s)

    async def hangup(self) -> None:
        if not self._ended:
            self._notify(f"📞 Call with {self._number} is finished. Please hang up your phone.")
        self._ended = True

    async def close(self) -> None:
        self._ended = True
        for s in (self._capture, self._playback):
            if s:
                with contextlib.suppress(Exception):
                    await s.stop()
        self._capture = self._playback = None
