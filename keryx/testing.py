"""Fake phone + scripted voice model: exercise the whole call path with no Bluetooth and no API key.
Used by the test suite and `hermes keryx simulate`."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from .transport.base import ANSWERED, CallTransport
from .voice import base as V


def tone(ms: int, rate: int = 24000, freq: float = 440.0, amp: int = 6000) -> bytes:
    n = rate * ms // 1000
    return b"".join(struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / rate))) for i in range(n))


class FakePhone(CallTransport):
    name = "fake"
    sample_rate = 24000

    def __init__(self, answer: str = ANSWERED, ring_s: float = 0.05, hangup_after_s: Optional[float] = None,
                 preflight_problems: Optional[list[str]] = None, echo_prone: bool = False) -> None:
        super().__init__()
        self.answer, self.ring_s, self.hangup_after_s = answer, ring_s, hangup_after_s
        self.preflight_problems = preflight_problems or []
        self.echo_prone = echo_prone
        self.dialed: list[str] = []
        self.dtmf: list[str] = []
        self.played = bytearray()
        self.hung_up = False
        self.closed = False
        self._ended = False
        self._audio_started_at: Optional[float] = None
        self.inbound = asyncio.Queue()  # tests may push callee frames

    async def preflight(self) -> list[str]:
        return list(self.preflight_problems)

    async def dial(self, number: str) -> None:
        self.dialed.append(number)

    async def wait_answered(self, timeout: float) -> str:
        await asyncio.sleep(self.ring_s)
        return self.answer

    async def start_audio(self) -> None:
        self._audio_started_at = asyncio.get_running_loop().time()

    async def read_frame(self) -> Optional[bytes]:
        if self._ended:
            return None
        try:
            return self.inbound.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(self.frame_ms / 1000)
            return bytes(self.frame_bytes)

    async def write_frame(self, pcm: bytes) -> None:
        self.played.extend(pcm)

    async def send_dtmf(self, digits: str) -> None:
        self.dtmf.append(digits)

    def remote_ended(self) -> bool:
        if self.hangup_after_s is not None and self._audio_started_at is not None:
            if asyncio.get_event_loop().time() - self._audio_started_at >= self.hangup_after_s:
                self._ended = True
        return self._ended

    async def hangup(self) -> None:
        self.hung_up = True
        self._ended = True

    async def close(self) -> None:
        self.closed = True


class ScriptedVoice(V.VoiceEngine):
    """Plays a script once start_response() is called. Steps:
        ("say", text[, wait_s])           agent speaks (tone audio + transcript)
        ("hear", text)                    callee speaks (transcript; barge-in if ("hear", text, True))
        ("tool", name, args)              model calls a function; waits for its result
        ("sleep", seconds)
    """
    name = "scripted"

    def __init__(self, script: list[tuple], say_ms: int = 120, fail_connect: bool = False) -> None:
        self.script = script
        self.say_ms = say_ms
        self.fail_connect = fail_connect
        self.instructions = ""
        self.tools: list[dict[str, Any]] = []
        self.audio_in = 0
        self.audio_in_before_start = 0
        self.tool_results: dict[str, Any] = {}
        self.system_messages: list[str] = []
        self.interruptions: list[int] = []
        self.connected = self.closed = False
        self._q: asyncio.Queue = asyncio.Queue()
        self._results: dict[str, asyncio.Future] = {}
        self._started = False
        self._runner: Optional[asyncio.Task] = None

    async def connect(self, instructions: str, tools: list[dict[str, Any]]) -> None:
        if self.fail_connect:
            raise RuntimeError("voice engine unavailable")
        self.instructions, self.tools, self.connected = instructions, tools, True

    async def send_audio(self, pcm: bytes) -> None:
        self.audio_in += len(pcm)
        if not self._started:
            self.audio_in_before_start += len(pcm)

    async def events(self) -> AsyncIterator[V.VoiceEvent]:
        while True:
            ev = await self._q.get()
            yield ev
            if ev.kind == V.CLOSED:
                return

    async def send_tool_result(self, call_id: str, output: str, respond: bool = True) -> None:
        try:
            self.tool_results[call_id] = json.loads(output)
        except ValueError:
            self.tool_results[call_id] = output
        fut = self._results.get(call_id)
        if fut and not fut.done():
            fut.set_result(output)

    async def system_message(self, text: str, respond: bool = True) -> None:
        self.system_messages.append(text)

    async def start_response(self) -> None:
        if not self._started:
            self._started = True
            self._runner = asyncio.create_task(self._run())

    async def interrupted(self, audio_heard_ms: int) -> None:
        self.interruptions.append(audio_heard_ms)

    async def _run(self) -> None:
        q = self._q
        for i, step in enumerate(self.script):
            kind = step[0]
            if kind == "say":
                q.put_nowait(V.VoiceEvent(V.RESPONSE_STARTED))
                q.put_nowait(V.VoiceEvent(V.AUDIO, data=tone(self.say_ms, self.output_rate)))
                q.put_nowait(V.VoiceEvent(V.TRANSCRIPT, role="agent", text=step[1]))
                q.put_nowait(V.VoiceEvent(V.RESPONSE_DONE))
                await asyncio.sleep(step[2] if len(step) > 2 else self.say_ms / 1000)
            elif kind == "hear":
                if len(step) > 2 and step[2]:
                    q.put_nowait(V.VoiceEvent(V.SPEECH_STARTED))
                q.put_nowait(V.VoiceEvent(V.TRANSCRIPT, role="callee", text=step[1]))
                await asyncio.sleep(0.02)
            elif kind == "tool":
                call_id = f"call_{i}"
                fut = asyncio.get_running_loop().create_future()
                self._results[call_id] = fut
                q.put_nowait(V.VoiceEvent(V.TOOL_CALL, call_id=call_id, name=step[1],
                                          arguments=json.dumps(step[2] if len(step) > 2 else {})))
                if step[1] != "end_call":
                    await fut
            elif kind == "sleep":
                await asyncio.sleep(step[1])

    async def close(self) -> None:
        self.closed = True
        if self._runner:
            self._runner.cancel()
        self._q.put_nowait(V.VoiceEvent(V.CLOSED))


async def fake_disclosure(text: str, rate: int, **_kw) -> bytes:
    return tone(200, rate, freq=660.0)


def demo_script() -> list[tuple]:
    return [
        ("say", "Hi! I'm calling to book a haircut for Saturday at 11 am."),
        ("hear", "Saturday 11 is taken, we have 12:30. It's 650 rupees."),
        ("say", "One moment please, let me check."),
        ("tool", "ask_principal", {"question": "Is 12:30 on Saturday okay, at 650 rupees?"}),
        ("say", "12:30 works. Could you do it for 600?"),
        ("hear", "Okay, 600 is fine. Name please?"),
        ("say", "It's for Asha. Thank you so much, goodbye!"),
        ("tool", "end_call", {"status": "achieved", "summary": "Booked Saturday 12:30 for ₹600.",
                              "details": {"time": "Saturday 12:30", "price": "₹600", "name": "Asha"},
                              "follow_ups": ["Add calendar event: haircut Saturday 12:30"]}),
    ]


async def simulate_call(goal: str = "Book a haircut for Saturday 11am, max ₹600") -> dict[str, Any]:
    from .config import Settings
    from .tools import Keryx

    settings = Settings(principal_name="Asha", quiet_hours="", transport="fake")
    phone = FakePhone()
    voice = ScriptedVoice(demo_script())
    rt = Keryx(None, data_dir=Path(tempfile.mkdtemp(prefix="keryx-sim-")), settings_override=settings,
                  transport_factory=lambda s, n: phone, engine_factory=lambda s: voice,
                  disclosure_renderer=fake_disclosure, cancel_check_factory=lambda: (lambda: False))
    args = {"to": "+919812345678", "callee_name": "Looks Salon", "goal": goal, "limits": ["max ₹600"],
            "may_share": ["Name: Asha"]}
    hook = rt.pre_tool_call("place_call", args)
    assert hook and hook["action"] == "approve", hook
    out = await asyncio.to_thread(rt.handle_place_call, args)
    return {"approval_prompt": hook["message"], "result": json.loads(out)}
