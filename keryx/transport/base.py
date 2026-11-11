"""CallTransport: the standard "dial, hang up, audio in/out" interface, one implementation per phone line.

Audio contract: signed 16-bit little-endian mono PCM at ``sample_rate``, in ``frame_bytes`` chunks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

ANSWERED, NO_ANSWER, BUSY, FAILED = "answered", "no_answer", "busy", "failed"


class TransportError(RuntimeError):
    pass


class CallTransport(ABC):
    name = "base"
    sample_rate = 24000
    playback_lead_ms = 0  # audio the bridge keeps queued ahead of real time in this transport
    frame_ms = 20
    echo_prone = False  # True when the mic can hear our own speaker (speakerphone)

    def __init__(self, notify: Optional[Callable[[str], None]] = None) -> None:
        self._notify = notify or (lambda _msg: None)

    @property
    def frame_bytes(self) -> int:
        return self.sample_rate * self.frame_ms // 1000 * 2

    def approval_note(self) -> str:
        """Extra text for the approval prompt (e.g. manual steps the user must take)."""
        return ""

    @abstractmethod
    async def preflight(self) -> list[str]:
        """Problems that would stop a call (empty list = ready). Must not dial."""

    @abstractmethod
    async def dial(self, number: str) -> None:
        """Start dialing *number* (E.164). Raises TransportError."""

    @abstractmethod
    async def wait_answered(self, timeout: float) -> str:
        """Block until the callee answers or the attempt ends: ANSWERED, NO_ANSWER, BUSY or FAILED."""

    @abstractmethod
    async def start_audio(self) -> None:
        """Open the call audio streams (after answer)."""

    @abstractmethod
    async def read_frame(self) -> Optional[bytes]:
        """Next frame of callee audio, or None when the stream ended."""

    @abstractmethod
    async def write_frame(self, pcm: bytes) -> None:
        """Play one frame into the call."""

    async def send_dtmf(self, digits: str) -> None:
        raise TransportError(f"{self.name} cannot send touch tones")

    @abstractmethod
    def remote_ended(self) -> bool:
        """True once the other side hung up or the line dropped."""

    @abstractmethod
    async def hangup(self) -> None:
        """End the call. Idempotent, never raises."""

    async def close(self) -> None:
        """Release audio streams/processes. Idempotent, never raises."""
