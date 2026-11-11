"""Hermes-facing surface: the place_call tool, its approval hook, and /calls.

Approval flow (enforced in code):
1. Hermes calls place_call -> our pre_tool_call hook compiles the brief and runs the guardrails.
   A violation BLOCKS the call outright. Otherwise the hook returns {"action": "approve"}, which makes
   Hermes ask the user on whatever surface they are on (CLI, Telegram, Desktop). Denial/timeout blocks.
2. The handler only dials if the hook saw these exact arguments (one approval = one dial), so a code
   path that skips the hook cannot place a call.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .brief import BriefError, compile_brief
from .calllog import CallLog
from .config import Settings, load_settings
from .guardrails import check_call

logger = logging.getLogger(__name__)

PLUGIN_ID = "keryx"
APPROVAL_TTL_S = 15 * 60

PLACE_CALL_SCHEMA: dict[str, Any] = {
    "name": "place_call",
    "description": (
        "Place a real outbound phone call from the user's own phone and let a realtime voice assistant hold the "
        "conversation to achieve a goal (book, enquire, confirm, reschedule...). The user must approve every call. "
        "This tool BLOCKS until the call ends (max 5 minutes) and returns the outcome, extracted details, follow-ups "
        "and transcript. Write a complete brief: the voice assistant can only use what you put here; anything else it "
        "will ask the user mid-call. Never use for emergency services."),
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Phone number, ideally E.164 (+919812345678)."},
            "callee_name": {"type": "string", "description": "Who/what is being called, e.g. 'Dr. Mehta's clinic'."},
            "goal": {"type": "string", "description": "What the call must achieve, in one or two sentences."},
            "context": {"type": "string", "description": "Background the assistant needs (dates, prior bookings...)."},
            "limits": {"type": "array", "items": {"type": "string"},
                       "description": "Hard limits, e.g. 'max ₹600', 'only weekday mornings', 'do not prepay'."},
            "may_share": {"type": "array", "items": {"type": "string"},
                          "description": "Facts the assistant may tell the callee (name, callback number...)."},
            "must_not_share": {"type": "array", "items": {"type": "string"},
                               "description": "Facts it must never reveal."},
            "language": {"type": "string", "description": "Language to speak, e.g. 'English', 'Hindi', 'Hinglish'."},
            "success_criteria": {"type": "string", "description": "How to tell the goal is achieved."},
            "voicemail_message": {"type": "string",
                                  "description": "Short message to leave on voicemail; omit to leave none."},
        },
        "required": ["to", "goal"],
        "additionalProperties": False,
    },
}


def _fingerprint(args: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class Keryx:
    """Plugin runtime: owns settings access, the call log, approvals and the one active call."""

    def __init__(self, ctx: Any = None, *, data_dir: Optional[Path] = None,
                 settings_override: Optional[Settings] = None, transport_factory=None, engine_factory=None,
                 disclosure_renderer=None, cancel_check_factory=None) -> None:
        self.ctx = ctx
        self._data_dir = data_dir
        self._settings_override = settings_override
        self._log: Optional[CallLog] = None
        self._approved: dict[str, float] = {}
        self._approved_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._stop = threading.Event()   # set by Hermes's agent_loop_stopped hook (/stop) during a call
        self._transport_factory = transport_factory
        self._engine_factory = engine_factory
        self._disclosure_renderer = disclosure_renderer
        self._cancel_check_factory = cancel_check_factory

    # --- environment -------------------------------------------------------------------------
    def settings(self) -> Settings:
        if self._settings_override is not None:
            return self._settings_override
        get_secret = None
        try:
            from agent.secret_scope import get_secret as _gs
            get_secret = _gs
        except Exception:
            pass
        return load_settings(getattr(self.ctx, "get_config", None), get_secret)

    def data_dir(self) -> Path:
        if self._data_dir is not None:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            return self._data_dir
        from plugins.plugin_storage import plugin_data_dir
        return plugin_data_dir(PLUGIN_ID)

    def log(self) -> CallLog:
        if self._log is None:
            if self._data_dir is None:
                try:
                    from plugins.plugin_storage import plugin_db
                    self._log = CallLog(plugin_db(PLUGIN_ID))
                    return self._log
                except Exception:
                    pass
            self._log = CallLog.at_path(self.data_dir() / "data.db")
        return self._log

    def _guard(self, args: dict[str, Any], settings: Settings):
        brief = compile_brief(args, principal=settings.principal)
        calls = self.log().calls_since(time.time() - 3600)
        return brief, check_call(brief.to, settings, calls_last_hour=calls)

    # --- approval hook -----------------------------------------------------------------------
    def pre_tool_call(self, tool_name: str = "", args: Optional[dict] = None, **_kw) -> Optional[dict]:
        if tool_name != "place_call":
            return None
        args = args if isinstance(args, dict) else {}
        try:
            settings = self.settings()
            brief, guard = self._guard(args, settings)
        except BriefError as e:
            return {"action": "block", "message": json.dumps({"success": False, "error": f"Bad brief: {e}"})}
        except Exception as e:  # fail closed
            return {"action": "block", "message": json.dumps({"success": False, "error": f"Keryx check failed: {e}"})}
        if not guard.ok:
            return {"action": "block", "message": json.dumps({
                "success": False, "error": "Keryx refused this call: " + "; ".join(guard.violations)})}
        with self._approved_lock:
            now = time.time()
            self._approved = {k: v for k, v in self._approved.items() if now - v < APPROVAL_TTL_S}
            self._approved[_fingerprint(args)] = now
        return {"action": "approve", "message": self.approval_text(brief, guard.number, settings),
                # unique per request: an "always allow" answer never carries over to the next call
                "rule_key": f"keryx.place_call.{uuid.uuid4().hex[:12]}"}

    def approval_text(self, brief, number: str, settings: Settings) -> str:
        lines = [f"📞 Place a phone call from YOUR phone to {brief.callee} ({number})?",
                 f"Goal: {brief.goal}"]
        if brief.limits:
            lines.append("Limits: " + "; ".join(brief.limits))
        if brief.may_share:
            lines.append("May share: " + "; ".join(brief.may_share))
        lines.append(f"Language: {brief.language} · Max {settings.max_call_seconds // 60} min · "
                     f"Recording: {'ON' if settings.record else 'off'} · Via: {settings.transport}")
        lines.append(f"It will open with: \"{settings.disclosure}\"")
        if settings.transport == "speakerphone":
            lines.append("Speakerphone mode: after approving, dial the number yourself with your phone on speaker "
                         "next to this computer.")
        return "\n".join(lines)

    def _consume_approval(self, args: dict[str, Any]) -> bool:
        with self._approved_lock:
            ts = self._approved.pop(_fingerprint(args), None)
        return ts is not None and time.time() - ts < APPROVAL_TTL_S

    # --- tool handler ------------------------------------------------------------------------
    def handle_place_call(self, args: dict[str, Any], **kwargs) -> str:
        try:
            return self._handle_place_call(args if isinstance(args, dict) else {}, kwargs)
        except Exception as e:  # never raise into Hermes
            logger.exception("keryx place_call failed")
            return json.dumps({"success": False, "error": f"{type(e).__name__}: {e}"})

    def _handle_place_call(self, args: dict[str, Any], kw: dict[str, Any]) -> str:
        if not self._consume_approval(args):
            return json.dumps({"success": False, "error": (
                "place_call was not approved through Hermes's approval prompt; no call was placed. "
                "Call the tool normally so the user is asked to approve it.")})
        settings = self.settings()
        try:
            brief, guard = self._guard(args, settings)  # re-check: time may have passed while approving
        except BriefError as e:
            return json.dumps({"success": False, "error": f"Bad brief: {e}"})
        if not guard.ok:
            return json.dumps({"success": False, "error": "Keryx refused this call: " + "; ".join(guard.violations)})
        if not self._call_lock.acquire(blocking=False):
            return json.dumps({"success": False, "error": "Another call is already in progress."})
        try:
            parent_agent = kw.get("parent_agent")
            self._stop.clear()
            cancel_check = self._cancel_check_factory() if self._cancel_check_factory else self._stop.is_set
            notifier = _Notifier(self.ctx)
            try:
                coro = self._place_call_async(brief, guard.number, settings, parent_agent, cancel_check, notifier)
                result = _run_in_thread(coro)
            finally:
                notifier.close()
            return json.dumps(result, ensure_ascii=False)
        finally:
            self._stop.clear()
            self._call_lock.release()

    async def _place_call_async(self, brief, number: str, settings: Settings, parent_agent, cancel_check,
                                notifier: "_Notifier") -> dict[str, Any]:
        from .disclosure import DisclosureError, render_disclosure
        from .orchestrator import CallOrchestrator
        from .resolver import PrincipalResolver
        from .transport import TransportError, make_transport
        from .voice import make_engine

        try:
            transport = (self._transport_factory or (lambda s, n: make_transport(s.transport, settings=s, notify=n)))(
                settings, notifier.send)
        except TransportError as e:
            return {"success": False, "error": f"phone line: {e}"}
        problems = await transport.preflight()
        if problems:
            return {"success": False, "error": "Phone line not ready: " + "; ".join(problems),
                    "hint": "Run `hermes keryx doctor` for setup help."}
        try:
            engine = (self._engine_factory or make_engine)(settings)
        except Exception as e:
            return {"success": False, "error": f"voice engine: {e}"}
        try:
            render = self._disclosure_renderer or render_disclosure
            disclosure_pcm = await render(settings.disclosure, transport.sample_rate, voice=settings.voice,
                                          api_key=settings.openai_api_key, audio_file=settings.disclosure_audio_file,
                                          cache_dir=self.data_dir() / "disclosure")
        except DisclosureError as e:
            return {"success": False, "error": f"No disclosure audio, so no call: {e}"}

        call_id = uuid.uuid4().hex
        record_dir = self.data_dir() / "recordings" / call_id if settings.record else None
        resolver = PrincipalResolver(
            brief, llm=self._llm_fn(), dispatch_tool=self._tool_fn(settings),
            readonly_tools=[t for t in settings.readonly_tools if t != "place_call"],
            ask_user=self._ask_user_fn(parent_agent, notifier), timeout_s=settings.principal_timeout_seconds)
        log = self.log()
        log.start(call_id, number, brief.callee, brief.goal, brief.to_dict())
        orch = CallOrchestrator(
            call_id=call_id, number=number, brief=brief, transport=transport, engine=engine, resolver=resolver,
            disclosure_text=settings.disclosure, disclosure_pcm=disclosure_pcm,
            max_call_seconds=settings.max_call_seconds, ring_timeout_seconds=settings.ring_timeout_seconds,
            record_dir=record_dir, cancel_check=cancel_check, call_log=log,
            on_state=lambda state, _d: notifier.status(f"📞 {brief.callee}: {_STATE_LABELS.get(state, state)}"))
        result = await orch.run()
        return result.to_tool_result()

    # --- Hermes services for the resolver --------------------------------------------------------
    def _llm_fn(self) -> Optional[Callable]:
        llm = getattr(self.ctx, "llm", None) if self.ctx is not None else None
        if llm is None:
            return None

        def complete(messages):
            res = llm.complete(messages, max_tokens=300, timeout=10, purpose="keryx.ask_principal")
            return getattr(res, "text", "") or ""
        return complete

    def _tool_fn(self, settings: Settings) -> Optional[Callable]:
        if self.ctx is None or not settings.readonly_tools:
            return None
        allowed = {t for t in settings.readonly_tools if t != "place_call"}

        def dispatch(name, args):
            if name not in allowed:
                return json.dumps({"error": f"{name} is not an allowed read-only tool"})
            return self.ctx.dispatch_tool(name, args)
        return dispatch

    def _ask_user_fn(self, parent_agent, notifier: "_Notifier") -> Optional[Callable]:
        callback = getattr(parent_agent, "clarify_callback", None) if parent_agent is not None else None
        if self.ctx is None or callback is None:
            return None

        def ask(question: str, _timeout: float) -> Optional[str]:
            out = self.ctx.dispatch_tool("clarify", {"questions": [{"question": question}]}, callback=callback)
            try:
                data = json.loads(out)
                resp = (data.get("responses") or [{}])[0]
                if resp.get("status") == "answered" and resp.get("user_response"):
                    val = resp["user_response"]
                    return ", ".join(val) if isinstance(val, list) else str(val)
            except (ValueError, AttributeError, IndexError):
                pass
            return None
        return ask

    # --- /calls ----------------------------------------------------------------------------------
    def on_agent_loop_stopped(self, **_kw: Any) -> None:
        """Hermes /stop (agent_loop_stopped hook): hang up the call in progress. Keryx places one call
        at a time, so any stop while a call is up ends it; erring toward hanging up is the safe side."""
        if self._call_lock.locked():
            self._stop.set()

    def calls_command(self, raw_args: str = "", **_kw) -> str:
        arg = (raw_args or "").strip()
        log = self.log()
        if arg:
            call = log.get(arg)
            if not call:
                return f"No call matching {arg!r}."
            o = call["outcome"] or {}
            lines = [f"📞 {call['callee']} ({call['number']}) — {call['status']}, "
                     f"{(call['duration_s'] or 0):.0f}s, {datetime.fromtimestamp(call['created_at']):%d %b %H:%M}",
                     f"Goal: {call['goal']}"]
            if o.get("summary"):
                lines.append(f"Summary: {o['summary']}")
            for k, v in (o.get("details") or {}).items():
                lines.append(f"  • {k}: {v}")
            if call.get("error"):
                lines.append(f"Error: {call['error']}")
            lines.append("Transcript:")
            lines += [f"  [{t.get('t', 0):>5}] {t['role']}: {t['text']}" for t in call["transcript"]]
            return "\n".join(lines)
        rows = log.recent(10)
        if not rows:
            return "No calls yet."
        return "\n".join(
            f"{r['id'][:8]}  {datetime.fromtimestamp(r['created_at']):%d %b %H:%M}  {r['status']:<11} "
            f"{r['callee']} — {r['summary'] or r['goal']}" for r in rows) + "\n\n/calls <id> for details."


_STATE_LABELS = {"dialing": "dialing…", "connected": "connected", "disclosing": "connected, introducing",
                 "talking": "in conversation", "wrapping_up": "wrapping up", "done": "call ended",
                 "no_answer": "no answer", "busy": "busy / rejected", "failed": "call failed",
                 "cancelled": "cancelled"}


class _Notifier:
    """Best-effort messages to the user's conversation, off the audio loop's thread."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="keryx-notify")
        self._ctxvars = contextvars.copy_context()
        self._status_handle = None

    def _submit(self, fn) -> None:
        if self.ctx is None or not hasattr(self.ctx, "post_message"):
            return
        ctxvars = self._ctxvars.copy()
        self._pool.submit(lambda: ctxvars.run(fn))

    def send(self, text: str) -> None:
        self._submit(lambda: self.ctx.post_message(text))

    def status(self, text: str) -> None:
        def update():
            h = self._status_handle
            if h is not None and getattr(h, "ok", True) and hasattr(h, "edit"):
                h.edit(text)
            else:
                self._status_handle = self.ctx.post_message(text)
        self._submit(update)

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)


def _run_in_thread(coro) -> Any:
    """Run *coro* on a fresh event loop in a worker thread, keeping this thread's contextvars
    (Hermes binds the session there, which post_message and clarify rely on)."""
    ctx = contextvars.copy_context()
    box: dict[str, Any] = {}

    def target():
        try:
            box["result"] = ctx.run(asyncio.run, coro)
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    th = threading.Thread(target=target, name="keryx-call", daemon=True)
    th.start()
    th.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")
