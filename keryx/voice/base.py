"""Voice engine adapter interface: one realtime speech-to-speech session per call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

# Event kinds
AUDIO = "audio"                    # data: PCM at output_rate
SPEECH_STARTED = "speech_started"  # the callee started talking (barge-in)
TRANSCRIPT = "transcript"          # role: "callee" | "agent", text
TOOL_CALL = "tool_call"            # call_id, name, arguments (JSON string)
RESPONSE_STARTED = "response_started"
RESPONSE_DONE = "response_done"
ERROR = "error"                    # text
CLOSED = "closed"


@dataclass
class VoiceEvent:
    kind: str
    data: bytes = b""
    text: str = ""
    role: str = ""
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    fatal: bool = False


class VoiceEngine(ABC):
    name = "base"
    input_rate = 24000
    output_rate = 24000

    @abstractmethod
    async def connect(self, instructions: str, tools: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None: ...

    @abstractmethod
    def events(self) -> AsyncIterator[VoiceEvent]: ...

    @abstractmethod
    async def send_tool_result(self, call_id: str, output: str, respond: bool = True) -> None: ...

    @abstractmethod
    async def system_message(self, text: str, respond: bool = True) -> None:
        """Out-of-band instruction from the plugin (e.g. 'time is nearly up')."""

    @abstractmethod
    async def start_response(self) -> None:
        """Make the model speak now (e.g. its opening line after the disclosure)."""

    @abstractmethod
    async def interrupted(self, audio_heard_ms: int) -> None:
        """The callee barged in: stop generating and forget audio they never heard."""

    @abstractmethod
    async def close(self) -> None: ...

    async def summarize_outcome(self, instructions: str, transcript: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Write end_call-style outcome from the transcript when the model never reported one
        (other side hung up, time cap, goodbye without end_call). None = not supported / failed."""
        return None
