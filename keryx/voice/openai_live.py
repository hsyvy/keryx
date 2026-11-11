"""OpenAI GPT-Live adapter (full-duplex voice model, wss://api.openai.com/v1/live/sessions).

GPT-Live differs from the Realtime API in ways that shape this adapter (all observed against the live
service, not just the docs):

- It listens and speaks at the same time and handles interruptions server-side. Output audio is a
  continuous real-time stream (100 ms chunks, exact zeros when silent), so there is no client barge-in
  logic: we just play what arrives and drop the all-zero chunks.
- The voice model has no tools. It *delegates* to a backend model (here via Responses delegation), and
  the backend calls our functions: ask_principal / press_keys / end_call arrive as function calls inside
  ``response.event`` envelopes, and we answer with ``response.item.create`` + ``response.create``.
- Transcripts arrive as ~200 ms word fragments; we stitch them into utterances.
- ``session.instructions.append`` (field ``content``) both nudges the model mid-call and makes it speak
  first once the plugin's disclosure has played.
- After saying goodbye the voice model does not always delegate end_call, so a watchdog nudges it and,
  failing that, ends the call itself (status "partial").
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator, Optional

from .base import AUDIO, CLOSED, ERROR, TOOL_CALL, TRANSCRIPT, VoiceEngine, VoiceEvent

logger = logging.getLogger(__name__)

LIVE_URL = "wss://api.openai.com/v1/live/sessions"

DELEGATION_NOTE = """
# How to use your backend (important)
You cannot call functions yourself. Wherever these instructions say to call ask_principal, press_keys or
end_call, delegate it to your backend instead, starting the request with the tool name:
- "ask_principal: <the exact question>". Say "One moment please, let me check." first, then use the answer.
- "press_keys: <digits>" for automated phone menus.
- "end_call: status=<achieved|partial|failed|declined|voicemail|wrong_number>; summary=<one honest sentence>;
  details=<key facts: prices, dates, times, names, reference numbers>; follow_ups=<things your principal should do>"
Delegation requests are silent: never say them, the tool names, or words like "end call" out loud.
The call only ends when you delegate end_call. Right after your goodbye, in the same turn, delegate end_call.
Never leave the line open after saying goodbye.
"""

BACKEND_INSTRUCTIONS = """You are the backend for a voice agent on a live phone call. Each request from the voice
agent maps to at most one tool call:
- "ask_principal: ..." -> call ask_principal with that exact question.
- "press_keys: ..." -> call press_keys with those digits.
- "end_call: ..." -> call end_call, copying status, summary, details and follow_ups from the request.
Use end_call ONLY when the request explicitly asks to end or hang up the call. Never end the call on your own
initiative, and never call a tool the request did not ask for. After a tool returns, reply with one short
sentence the voice agent can use (for ask_principal: the principal's answer, or that there is no answer yet and
the agent must not guess). If no tool fits the request, reply briefly in text. Never invent answers."""

GOODBYE_RE = re.compile(r"\b(good ?bye|bye|take care|have a (good|great|nice|wonderful) (day|evening|night))\b", re.I)


class _Stitcher:
    """Turns transcript fragments into utterances: flush on speaker change, a gap, or silence."""

    def __init__(self, gap_ms: int = 1500, idle_s: float = 1.6) -> None:
        self.gap_ms, self.idle_s = gap_ms, idle_s
        self.buf: dict[str, str] = {"agent": "", "callee": ""}
        self.last_end: dict[str, int] = {"agent": -1, "callee": -1}
        self.last_wall: dict[str, float] = {"agent": 0.0, "callee": 0.0}

    def add(self, role: str, text: str, start_ms: Optional[int], end_ms: Optional[int]) -> list[tuple[str, str]]:
        out = []
        other = "callee" if role == "agent" else "agent"
        if self.buf[other].strip():
            out += self.flush(other)
        if (self.buf[role].strip() and start_ms is not None and self.last_end[role] >= 0
                and start_ms - self.last_end[role] > self.gap_ms):
            out += self.flush(role)
        self.buf[role] += text
        if end_ms is not None:
            self.last_end[role] = end_ms
        self.last_wall[role] = time.monotonic()
        return out

    def due(self) -> list[tuple[str, str]]:
        now = time.monotonic()
        out = []
        for role in ("agent", "callee"):
            if self.buf[role].strip() and now - self.last_wall[role] > self.idle_s:
                out += self.flush(role)
        return out

    def flush(self, role: str) -> list[tuple[str, str]]:
        text = " ".join(self.buf[role].split())
        self.buf[role] = ""
        return [(role, text)] if text else []

    def flush_all(self) -> list[tuple[str, str]]:
        return self.flush("agent") + self.flush("callee")


class OpenAILiveEngine(VoiceEngine):
    name = "openai-live"
    input_rate = 24000
    output_rate = 24000

    def __init__(self, api_key: str, model: str = "gpt-live-1", voice: str = "marin",
                 backend_model: str = "gpt-5.4-mini", goodbye_nudge_s: float = 2.5,
                 goodbye_hangup_s: float = 6.0) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (keryx needs it for the live voice model)")
        self.api_key, self.model, self.voice, self.backend_model = api_key, model, voice, backend_model
        self.goodbye_nudge_s, self.goodbye_hangup_s = goodbye_nudge_s, goodbye_hangup_s
        self._ws: Any = None
        self._queue: asyncio.Queue[VoiceEvent] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._send_lock = asyncio.Lock()
        self._speaking_allowed = False   # drop model audio until the plugin's disclosure has played
        self._stitch = _Stitcher()
        self._seen_calls: set[str] = set()
        self._end_call_tool: Optional[dict[str, Any]] = None
        self._goodbye_at: Optional[float] = None          # the agent said goodbye
        self._callee_goodbye_at: Optional[float] = None   # the callee said goodbye
        self._last_speech = time.monotonic()              # last transcript fragment from either side
        self._nudged = False
        self._callee_nudged = False
        self._wrap_up = False   # the plugin's "time is nearly up" warning went out
        self._ended = False
        self._ended_at: Optional[float] = None
        self._closed = False

    # --- session ---------------------------------------------------------------------------
    async def connect(self, instructions: str, tools: list[dict[str, Any]]) -> None:
        try:
            from websockets.asyncio.client import connect as ws_connect  # websockets >= 13
            kw = {"additional_headers": {"Authorization": f"Bearer {self.api_key}"}}
        except ImportError:  # pragma: no cover
            from websockets import connect as ws_connect  # type: ignore
            kw = {"extra_headers": {"Authorization": f"Bearer {self.api_key}"}}
        self._ws = await ws_connect(LIVE_URL, max_size=None, **kw)
        backend_tools = [{k: t[k] for k in ("type", "name", "description", "parameters") if k in t} for t in tools]
        self._end_call_tool = next((t for t in backend_tools if t.get("name") == "end_call"), None)
        await self._send({"type": "session.start", "event_id": f"evt_{uuid.uuid4().hex[:12]}", "session": {
            "model": self.model,
            "instructions": instructions + "\n" + DELEGATION_NOTE,
            "audio": {"format": {"type": "audio/pcm", "rate": self.input_rate},
                      "output": {"voice": self.voice}},
            "delegation": {"type": "responses", "responses": {
                "model": self.backend_model, "instructions": BACKEND_INSTRUCTIONS, "tools": backend_tools}},
        }})
        # Wait for session.started (or an error) before anything else.
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), 15)
            ev = json.loads(raw)
            if ev.get("type") == "session.started":
                break
            if ev.get("type") == "error":
                raise RuntimeError(f"GPT-Live refused the session: {_err_text(ev)}")
        self._tasks.append(asyncio.create_task(self._read_loop()))
        self._tasks.append(asyncio.create_task(self._tick()))

    async def _send(self, msg: dict[str, Any]) -> None:
        if self._ws is None or self._closed:
            return
        async with self._send_lock:
            await self._ws.send(json.dumps(msg))

    # --- inbound ---------------------------------------------------------------------------
    def _emit(self, ev: VoiceEvent) -> None:
        self._queue.put_nowait(ev)

    def _emit_transcripts(self, items: list[tuple[str, str]]) -> None:
        for role, text in items:
            self._emit(VoiceEvent(TRANSCRIPT, role=role, text=text))
            self._note_goodbye(role, text)

    def _note_goodbye(self, role: str, text: str) -> None:
        if not GOODBYE_RE.search(text):
            return
        if role == "agent" and self._goodbye_at is None:
            self._goodbye_at = time.monotonic()
        elif role == "callee" and self._callee_goodbye_at is None:
            self._callee_goodbye_at = time.monotonic()

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    self._dispatch(json.loads(raw))
                except Exception:  # never let one odd event kill the call
                    logger.exception("keryx live: bad event")
        except Exception as e:
            if not self._closed:
                self._emit(VoiceEvent(ERROR, text=f"live connection lost: {e}", fatal=True))
        finally:
            self._emit_transcripts(self._stitch.flush_all())
            self._emit(VoiceEvent(CLOSED))

    def _dispatch(self, ev: dict[str, Any]) -> None:
        t = ev.get("type", "")
        if t != "session.output_audio.delta" and logger.isEnabledFor(logging.DEBUG):
            inner = (ev.get("event") or {}).get("type") if t == "response.event" else ""
            if inner not in ("response.function_call_arguments.delta", "response.output_text.delta"):
                logger.debug("live event %s %s", t + (f"/{inner}" if inner else ""), json.dumps(ev)[:600])
        if t == "session.output_audio.delta":
            if not self._speaking_allowed or self._muted():
                return
            pcm = base64.b64decode(ev.get("delta") or "")
            if pcm and any(pcm):  # all-zero chunks are the model being silent
                self._emit(VoiceEvent(AUDIO, data=pcm))
        elif t in ("session.output_transcript.delta", "session.input_transcript.delta"):
            role = "agent" if t.startswith("session.output") else "callee"
            if role == "agent" and (not self._speaking_allowed or self._muted()):
                return
            if (ev.get("delta") or "").strip():
                self._last_speech = time.monotonic()
            self._emit_transcripts(self._stitch.add(role, ev.get("delta") or "", ev.get("start_ms"), ev.get("end_ms")))
            # Detect goodbyes as the words arrive (fragments split words: check the running utterance),
            # so a legitimate end_call right after "goodbye" is not refused.
            self._note_goodbye(role, self._stitch.buf[role])
        elif t == "response.event":
            inner = ev.get("event") or {}
            item = inner.get("item") or {}
            if inner.get("type") == "response.output_item.done" and item.get("type") == "function_call":
                call_id = item.get("call_id") or item.get("id") or ""
                if call_id and call_id not in self._seen_calls:
                    self._seen_calls.add(call_id)
                    name = item.get("name") or ""
                    args = item.get("arguments") or "{}"
                    if name == "end_call" and not self._end_allowed(args):
                        # The backend sometimes hangs up on its own right after a tool result. Refuse in
                        # code: the call ends only after goodbyes, the time warning, or voicemail.
                        logger.info("keryx live: refused premature end_call %s", args[:200])
                        self._spawn(self.send_tool_result(call_id, json.dumps({
                            "ok": False, "error": "The call is still in progress and nobody has said goodbye. "
                                                  "Do not end it; reply to the voice agent in text instead."})))
                        return
                    if name == "end_call":
                        self._ended = True
                        self._ended_at = time.monotonic()
                    self._emit(VoiceEvent(TOOL_CALL, call_id=call_id, name=name, arguments=args))
            elif inner.get("type") == "error" or (inner.get("type") == "response.failed"):
                self._emit(VoiceEvent(ERROR, text=f"backend: {json.dumps(inner)[:300]}"))
        elif t == "error":
            self._emit(VoiceEvent(ERROR, text=_err_text(ev)))
        elif t == "session.closed":
            self._closed = True

    def _end_allowed(self, args: str) -> bool:
        try:
            status = str((json.loads(args) or {}).get("status") or "")
        except (ValueError, AttributeError):
            status = ""
        if status in ("voicemail", "wrong_number") or self._wrap_up:
            return True
        now = time.monotonic()
        return any(t is not None and now - t < 30 for t in (self._goodbye_at, self._callee_goodbye_at))

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.append(task)

    def _local_end(self, why: str) -> None:
        self._ended = True
        self._ended_at = time.monotonic()
        self._emit(VoiceEvent(TOOL_CALL, call_id=f"local_{uuid.uuid4().hex[:8]}", name="end_call",
                              arguments=json.dumps({"status": "partial", "summary": why})))

    def _muted(self) -> bool:
        """Once end_call has been delegated the goodbye is already out (audio arrives in real time);
        anything after it is the model narrating its own hang-up, which the callee must not hear."""
        return self._ended_at is not None and time.monotonic() - self._ended_at > 0.3

    async def _tick(self) -> None:
        """Flush stitched transcripts on silence, and make sure a goodbye ends the call."""
        try:
            while not self._closed:
                await asyncio.sleep(0.3)
                self._emit_transcripts(self._stitch.due())
                if self._ended:
                    continue
                quiet = time.monotonic() - self._last_speech
                if self._goodbye_at is not None:
                    # The agent said goodbye: it should delegate end_call; nudge, then hang up ourselves.
                    waited = time.monotonic() - self._goodbye_at
                    if waited > self.goodbye_nudge_s and not self._nudged:
                        self._nudged = True
                        await self.system_message("You said goodbye. Delegate end_call now with the honest outcome.")
                    elif waited > self.goodbye_hangup_s and quiet > 1.5:
                        self._local_end("Said goodbye; the voice model did not report an outcome.")
                elif self._callee_goodbye_at is not None:
                    # The callee said goodbye and the model tends to go silent: prompt it, then hang up.
                    if quiet > 1.5 and not self._callee_nudged:
                        self._callee_nudged = True
                        await self.system_message("They said goodbye. Say one short goodbye now, then delegate "
                                                  "end_call with the honest outcome.")
                    elif self._callee_nudged and quiet > 5.0:
                        self._local_end("The other side said goodbye; the voice model did not report an outcome.")
        except asyncio.CancelledError:
            pass

    # --- outbound ----------------------------------------------------------------------------
    async def send_audio(self, pcm: bytes) -> None:
        await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode()})

    async def events(self) -> AsyncIterator[VoiceEvent]:
        while True:
            ev = await self._queue.get()
            yield ev
            if ev.kind == CLOSED:
                return

    async def send_tool_result(self, call_id: str, output: str, respond: bool = True) -> None:
        if call_id.startswith("local_"):
            return
        await self._send({"type": "response.item.create",
                          "item": {"type": "function_call_output", "call_id": call_id, "output": output}})
        if respond:
            await self._send({"type": "response.create"})

    async def system_message(self, text: str, respond: bool = True) -> None:
        if "nearly up" in text.lower():
            self._wrap_up = True
        await self._send({"type": "session.instructions.append", "event_id": f"evt_{uuid.uuid4().hex[:12]}",
                          "delegation_id": None, "content": text})

    async def start_response(self) -> None:
        self._speaking_allowed = True
        await self.system_message("The disclosure has just finished playing. Speak now: greet them and say why "
                                  "you are calling.")

    async def interrupted(self, audio_heard_ms: int) -> None:
        """GPT-Live handles barge-in server-side; nothing to do."""

    async def close(self) -> None:
        if self._ws is not None and not self._closed:
            with contextlib.suppress(Exception):
                await self._send({"type": "session.close"})
        self._closed = True
        for t in self._tasks:
            if t is not asyncio.current_task():
                t.cancel()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        # make sure a waiting consumer wakes up
        self._emit(VoiceEvent(CLOSED))


    async def summarize_outcome(self, instructions: str, transcript: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Have the backend model write the end_call outcome from the transcript (one Responses call,
        forced to the end_call tool). Used after hang-up when the voice model never reported."""
        if not self._end_call_tool:
            return None
        lines = "\n".join(f"{t.get('role')}: {t.get('text')}" for t in transcript if t.get("role") in ("agent", "callee", "system"))
        body = {
            "model": self.backend_model,
            "instructions": ("You write the outcome report for a finished phone call made by an AI assistant. Use only "
                             "what the transcript shows; never invent facts. Be honest about anything unclear or not "
                             "confirmed. The call brief follows.\n\n" + instructions),
            "input": "Transcript of the finished call:\n" + lines[-12000:],
            "tools": [self._end_call_tool],
            "tool_choice": {"type": "function", "name": "end_call"},
        }
        data = await asyncio.to_thread(_post_json, "https://api.openai.com/v1/responses", body, self.api_key)
        for item in data.get("output") or []:
            if item.get("type") == "function_call" and item.get("name") == "end_call":
                try:
                    return json.loads(item.get("arguments") or "{}")
                except ValueError:
                    return None
        return None


def _post_json(url: str, body: dict[str, Any], api_key: str, timeout: float = 20.0) -> dict[str, Any]:
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _err_text(ev: dict[str, Any]) -> str:
    err = ev.get("error") or {}
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err)
    return str(err)
