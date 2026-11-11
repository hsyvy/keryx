"""Call orchestrator: runs one call end to end.

    created -> dialing -> connected -> disclosing -> talking -> wrapping_up -> done
                   \\-> no_answer | busy | failed            \\-> failed | cancelled

Safety properties enforced here (not in the prompt):
- the voice session is connected and the disclosure audio is rendered BEFORE dialing;
- the plugin plays the disclosure itself, and the callee's audio reaches the model only after it;
- a hard time cap with a wrap-up warning 30 s before it;
- touch tones restricted to 0-9 * #.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .audio import AudioBridge
from .brief import VOICE_TOOLS, Brief, build_instructions, normalize_outcome
from .resolver import PrincipalResolver
from .transport.base import ANSWERED, BUSY, FAILED, NO_ANSWER, CallTransport, TransportError
from .voice import base as V

logger = logging.getLogger(__name__)

CREATED, DIALING, CONNECTED, DISCLOSING, TALKING, WRAPPING_UP = (
    "created", "dialing", "connected", "disclosing", "talking", "wrapping_up")
DONE, S_FAILED, S_NO_ANSWER, S_BUSY, CANCELLED = "done", "failed", "no_answer", "busy", "cancelled"
WRAP_UP_WARNING_S = 30
_DTMF_RE = re.compile(r"^[0-9*#]{1,32}$")


class _Cancelled(Exception):
    pass


@dataclass
class CallResult:
    call_id: str
    number: str
    callee: str
    state: str = CREATED
    outcome: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    talk_s: float = 0.0
    error: Optional[str] = None
    recording: Optional[str] = None
    principal_questions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.state == DONE:
            return self.outcome.get("status", "partial")
        return self.state

    def to_tool_result(self, max_transcript_chars: int = 6000) -> dict[str, Any]:
        lines, total = [], 0
        for turn in reversed(self.transcript):
            line = f"{turn['role']}: {turn['text']}"
            total += len(line)
            if total > max_transcript_chars:
                lines.append("… (earlier turns omitted; full transcript in /calls " + self.call_id[:8] + ")")
                break
            lines.append(line)
        res = {
            "success": self.state == DONE,
            "call_id": self.call_id,
            "status": self.status,
            "call_state": self.state,
            "number": self.number,
            "callee": self.callee,
            "duration_s": round(self.duration_s, 1),
            "talk_s": round(self.talk_s, 1),
            "summary": self.outcome.get("summary", ""),
            "details": self.outcome.get("details", {}),
            "follow_ups": self.outcome.get("follow_ups", []),
            "transcript": list(reversed(lines)),
        }
        if self.principal_questions:
            res["principal_questions"] = self.principal_questions
        if self.error:
            res["error"] = self.error
        if self.recording:
            res["recording"] = self.recording
        res["next"] = ("Tell the user the outcome in a sentence or two. Handle follow_ups with your normal tools "
                       "(e.g. add a calendar event) if the user's request implied them.")
        return res


class CallOrchestrator:
    def __init__(self, *, call_id: str, number: str, brief: Brief, transport: CallTransport,
                 engine: V.VoiceEngine, resolver: PrincipalResolver, disclosure_text: str,
                 disclosure_pcm: bytes, max_call_seconds: int = 300, ring_timeout_seconds: int = 45,
                 record_dir: Optional[Path] = None, cancel_check: Optional[Callable[[], bool]] = None,
                 on_state: Optional[Callable[[str, str], None]] = None, call_log=None) -> None:
        self.brief, self.t, self.engine, self.resolver = brief, transport, engine, resolver
        self.disclosure_text, self.disclosure_pcm = disclosure_text, disclosure_pcm
        self.max_call_seconds, self.ring_timeout = max_call_seconds, ring_timeout_seconds
        self.record_dir = record_dir
        self.cancel_check = cancel_check or (lambda: False)
        self.on_state = on_state or (lambda _s, _d: None)
        self.call_log = call_log
        self.result = CallResult(call_id=call_id, number=number, callee=brief.callee)
        self.bridge: Optional[AudioBridge] = None
        self._forward_uplink = False
        self._connected_at: Optional[float] = None
        self._end_requested_at: Optional[float] = None
        self._model_reported = False  # True once the voice model itself called end_call
        self._tasks: set[asyncio.Task] = set()

    # --- helpers -----------------------------------------------------------------------------
    def _set(self, state: str, detail: str = "") -> None:
        self.result.state = state
        with contextlib.suppress(Exception):
            self.on_state(state, detail)

    def _say(self, role: str, text: str, **extra: Any) -> None:
        t = round(time.monotonic() - self._connected_at, 1) if self._connected_at else 0.0
        self.result.transcript.append({"t": t, "role": role, "text": text, **extra})

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _cancellable(self, coro):
        task = asyncio.ensure_future(coro)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=0.25)
                if done:
                    return task.result()
                if self.cancel_check():
                    raise _Cancelled()
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    # --- main --------------------------------------------------------------------------------
    async def run(self) -> CallResult:
        start = time.monotonic()
        try:
            await self._run()
        except _Cancelled:
            self.result.error = "cancelled by the user"
            self._set(CANCELLED)
        except TransportError as e:
            self.result.error = f"phone line: {e}"
            self._set(S_FAILED, str(e))
        except Exception as e:
            logger.exception("keryx call failed")
            self.result.error = f"{type(e).__name__}: {e}"
            self._set(S_FAILED, str(e))
        finally:
            await self._teardown()
            await self._summarize_if_unreported()
            self.result.duration_s = time.monotonic() - start
            if self._connected_at:
                self.result.talk_s = time.monotonic() - self._connected_at
            self.result.principal_questions = list(self.resolver.log)
            if self.bridge and self.bridge.recording_dir:
                self.result.recording = self.bridge.recording_dir
            if self.call_log is not None:
                with contextlib.suppress(Exception):
                    self.call_log.finish(self.result.call_id, status=self.result.status, state=self.result.state,
                                         duration_s=self.result.duration_s, outcome=self.result.outcome,
                                         transcript=self.result.transcript, recording=self.result.recording,
                                         error=self.result.error)
        return self.result

    async def _run(self) -> None:
        # Brain first: never dial unless the voice session is up.
        await self._cancellable(asyncio.wait_for(
            self.engine.connect(build_instructions(self.brief, self.disclosure_text), VOICE_TOOLS), 15))

        self._set(DIALING, self.result.number)
        if self.call_log is not None:
            self.call_log.mark_dialed(self.result.call_id)
        await self._cancellable(self.t.dial(self.result.number))
        answer = await self._cancellable(self.t.wait_answered(self.ring_timeout))
        if answer != ANSWERED:
            self._set({NO_ANSWER: S_NO_ANSWER, BUSY: S_BUSY, FAILED: S_FAILED}.get(answer, S_FAILED))
            self.result.error = {NO_ANSWER: "nobody answered", BUSY: "line busy or call rejected"}.get(
                answer, "the call could not be placed")
            return

        self._connected_at = time.monotonic()
        self._set(CONNECTED)
        await self.t.start_audio()
        self.bridge = AudioBridge(self.t, self.engine.input_rate, self.engine.output_rate,
                                  record_dir=self.record_dir)
        self.bridge.start()
        self._spawn(self._uplink())

        # Disclosure: played by the plugin; the model hears nothing until it has finished.
        self._set(DISCLOSING)
        self.bridge.play(self.disclosure_pcm)
        self._say("agent", self.disclosure_text, source="plugin_disclosure")
        await self.bridge.wait_drained(timeout=len(self.disclosure_pcm) / 2 / self.t.sample_rate + 5)

        self._set(TALKING)
        self._forward_uplink = True
        await self.engine.start_response()
        await self._converse()

    async def _uplink(self) -> None:
        while True:
            frame = await self.t.read_frame()
            if frame is None:
                return
            pcm = self.bridge.process_inbound(frame)
            if self._forward_uplink:
                with contextlib.suppress(Exception):
                    await self.engine.send_audio(pcm)

    async def _converse(self) -> None:
        assert self._connected_at is not None and self.bridge is not None
        deadline = self._connected_at + self.max_call_seconds
        warned = False
        events = self.engine.events().__aiter__()
        nxt: Optional[asyncio.Future] = asyncio.ensure_future(events.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({nxt}, timeout=0.25)
                if done:
                    try:
                        ev = nxt.result()
                    except StopAsyncIteration:
                        ev = V.VoiceEvent(V.CLOSED)
                    if await self._handle(ev):
                        return
                    nxt = asyncio.ensure_future(events.__anext__())
                now = time.monotonic()
                if self.cancel_check():
                    raise _Cancelled()
                if self.t.remote_ended():
                    self._finish_default("partial", "The other side hung up.")
                    return
                if self._end_requested_at is not None:
                    drained = not self.bridge.speaking
                    if drained or now - self._end_requested_at > 10:
                        await asyncio.sleep(0.4)  # let the last syllable leave the phone
                        self._set(DONE)
                        return
                if not warned and now >= deadline - WRAP_UP_WARNING_S:
                    warned = True
                    await self.engine.system_message(
                        "Time is nearly up. Wrap up now: thank them, say goodbye in one sentence, then call end_call.")
                if now >= deadline:
                    self._finish_default("partial", f"Time limit of {self.max_call_seconds}s reached.")
                    return
        finally:
            if nxt is not None and not nxt.done():
                nxt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception, StopAsyncIteration):
                    await nxt

    def _finish_default(self, status: str, summary: str) -> None:
        if not self.result.outcome:
            self.result.outcome = normalize_outcome({"status": status, "summary": summary})
        self._set(DONE)

    async def _handle(self, ev: V.VoiceEvent) -> bool:
        """Process one engine event; True means the conversation is over."""
        b = self.bridge
        assert b is not None
        if ev.kind == V.AUDIO:
            b.enqueue_model_audio(ev.data)
        elif ev.kind == V.RESPONSE_STARTED:
            b.new_response()
        elif ev.kind == V.SPEECH_STARTED:
            if b.speaking and self._end_requested_at is None:
                heard = b.barge_in()
                await self.engine.interrupted(heard)
        elif ev.kind == V.TRANSCRIPT:
            self._say(ev.role or "agent", ev.text)
        elif ev.kind == V.TOOL_CALL:
            await self._tool_call(ev)
        elif ev.kind == V.ERROR:
            logger.warning("keryx voice engine error: %s", ev.text)
            if ev.fatal:
                self.result.error = ev.text
                self._finish_default("failed", "The voice model disconnected mid-call.")
                return True
        elif ev.kind == V.CLOSED:
            if self._end_requested_at is None:
                self._finish_default("failed", "The voice model session closed mid-call.")
            else:
                self._set(DONE)
            return True
        return False

    async def _tool_call(self, ev: V.VoiceEvent) -> None:
        try:
            args = json.loads(ev.arguments or "{}")
        except ValueError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if ev.name == "ask_principal":
            question = str(args.get("question") or "")
            self._say("system", f"ask_principal: {question}")

            async def answer() -> None:
                res = await self.resolver.resolve(question)
                self._say("system", f"principal answer ({res.get('source')}): {res.get('answer')}")
                await self.engine.send_tool_result(ev.call_id, json.dumps(res, ensure_ascii=False))
            self._spawn(answer())
        elif ev.name == "press_keys":
            digits = str(args.get("digits") or "").replace(" ", "")
            if not _DTMF_RE.match(digits):
                out = {"ok": False, "error": "digits must be 0-9, * or # (max 32)"}
            else:
                try:
                    await self.t.send_dtmf(digits)
                    out = {"ok": True}
                    self._say("system", f"pressed {digits}")
                except Exception as e:
                    out = {"ok": False, "error": str(e)}
            await self.engine.send_tool_result(ev.call_id, json.dumps(out))
        elif ev.name == "end_call":
            self.result.outcome = normalize_outcome(args)
            self._model_reported = not ev.call_id.startswith("local_")
            self._say("system", f"end_call: {self.result.outcome['status']}")
            self._end_requested_at = time.monotonic()
            self._set(WRAPPING_UP)
            await self.engine.send_tool_result(ev.call_id, json.dumps({"ok": True}), respond=False)
        else:
            await self.engine.send_tool_result(ev.call_id, json.dumps({"ok": False, "error": "unknown tool"}))

    async def _summarize_if_unreported(self) -> None:
        """After hang-up (never while the line is open): if the model didn't report an outcome but the
        two sides did talk, ask the engine to write one from the transcript."""
        if self._model_reported or self.result.state != DONE:
            return
        if not any(t.get("role") == "callee" for t in self.result.transcript):
            return
        try:
            out = await asyncio.wait_for(self.engine.summarize_outcome(
                build_instructions(self.brief, self.disclosure_text), list(self.result.transcript)), 20)
        except Exception as e:
            logger.info("keryx: outcome summary failed: %s", e)
            return
        if out:
            why = (self.result.outcome or {}).get("summary") or ""
            outcome = normalize_outcome(out)
            if why:
                outcome["details"] = {**outcome.get("details", {}), "how_it_ended": why}
            self.result.outcome = outcome
            self._say("system", f"outcome summarized from transcript: {outcome['status']}")

    async def _teardown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        with contextlib.suppress(Exception):
            await self.t.hangup()
        if self.bridge:
            with contextlib.suppress(Exception):
                await self.bridge.close()
        with contextlib.suppress(Exception):
            await self.t.close()
        with contextlib.suppress(Exception):
            await self.engine.close()


def result_dict(result: CallResult) -> dict[str, Any]:
    return asdict(result)
