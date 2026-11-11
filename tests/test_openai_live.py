"""GPT-Live adapter against a fake socket. Event shapes are copied from a real gpt-live-1 session."""

import asyncio
import base64
import json

import pytest

from keryx.brief import VOICE_TOOLS
from keryx.voice import base as V
from keryx.voice import openai_live as L
from keryx.voice.openai_live import OpenAILiveEngine


class FakeWS:
    def __init__(self, first=None, script=None):
        self.first = list(first or [{"type": "session.started", "session": {"id": "live_x"}}])
        self.script = list(script or [])   # (delay_s, event) pairs streamed after connect
        self.sent = []
        self.closed = False

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def recv(self):
        return json.dumps(self.first.pop(0))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.script:
            await asyncio.sleep(0.05)
            raise StopAsyncIteration
        delay, ev = self.script.pop(0)
        await asyncio.sleep(delay)
        return json.dumps(ev)

    async def close(self):
        self.closed = True


def _engine(ws, monkeypatch, **kw):
    async def fake_connect(url, **_):
        assert url == L.LIVE_URL
        return ws
    import websockets.asyncio.client as wac
    monkeypatch.setattr(wac, "connect", fake_connect)
    return OpenAILiveEngine(api_key="sk-test", **kw)


def audio(pcm: bytes) -> dict:
    return {"type": "session.output_audio.delta", "delta": base64.b64encode(pcm).decode()}


def tr(kind: str, text: str, start: int) -> dict:
    return {"type": f"session.{kind}_transcript.delta", "delta": text, "start_ms": start, "end_ms": start + 200}


def fn_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"type": "response.event", "delegation_id": "item_1", "event": {
        "type": "response.output_item.done",
        "item": {"type": "function_call", "status": "completed", "call_id": call_id, "name": name,
                 "arguments": json.dumps(args)}}}


async def collect(eng, until=V.CLOSED, timeout=5.0):
    out = []

    async def go():
        async for ev in eng.events():
            out.append(ev)
            if ev.kind == until:
                return
    await asyncio.wait_for(go(), timeout)
    return out


def test_session_start_declares_backend_tools(monkeypatch):
    ws = FakeWS()
    eng = _engine(ws, monkeypatch, backend_model="gpt-5.4-mini")

    async def go():
        await eng.connect("Base instructions.", VOICE_TOOLS)
        await eng.close()
    asyncio.run(go())
    start = ws.sent[0]
    assert start["type"] == "session.start"
    s = start["session"]
    assert s["model"] == "gpt-live-1" and s["audio"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert "Base instructions." in s["instructions"] and "delegate" in s["instructions"]
    d = s["delegation"]
    assert d["type"] == "responses" and d["responses"]["model"] == "gpt-5.4-mini"
    assert sorted(t["name"] for t in d["responses"]["tools"]) == ["ask_principal", "end_call", "press_keys"]
    assert "ONLY when the request explicitly asks to end" in d["responses"]["instructions"]
    assert ws.sent[-1]["type"] == "session.close"


def test_refused_session_raises(monkeypatch):
    ws = FakeWS(first=[{"type": "error", "error": {"code": "invalid_model", "message": "bad model"}}])
    eng = _engine(ws, monkeypatch)
    with pytest.raises(RuntimeError, match="bad model"):
        asyncio.run(eng.connect("x", VOICE_TOOLS))


def test_audio_gated_until_disclosure_and_silence_dropped(monkeypatch):
    speech = b"\x10\x00" * 2400
    ws = FakeWS(script=[(0.0, audio(b"\x05\x00" * 2400)),       # before start_response: dropped
                        (0.15, audio(bytes(4800))),               # silence chunk: dropped
                        (0.0, audio(speech))])
    eng = _engine(ws, monkeypatch)

    async def go():
        await eng.connect("x", VOICE_TOOLS)
        await asyncio.sleep(0.05)
        await eng.start_response()
        evs = await collect(eng)
        return evs
    evs = asyncio.run(go())
    assert [e.data for e in evs if e.kind == V.AUDIO] == [speech]
    nudge = [m for m in ws.sent if m["type"] == "session.instructions.append"]
    assert nudge and nudge[0]["delegation_id"] is None and "Speak now" in nudge[0]["content"]


def test_transcripts_stitched_into_utterances(monkeypatch):
    ws = FakeWS(script=[(0.0, tr("output", " Hi,", 5000)), (0.0, tr("output", " I'd like", 6400)),  # 1.2 s pause: same line
                        (0.0, tr("output", " a haircut.", 5400)),
                        (0.0, tr("input", " Saturday", 9000)), (0.0, tr("input", " is taken.", 9200)),
                        (0.0, tr("output", " Okay,", 9800))])
    eng = _engine(ws, monkeypatch)

    async def go():
        await eng.connect("x", VOICE_TOOLS)
        await eng.start_response()
        return await collect(eng)
    evs = asyncio.run(go())
    lines = [(e.role, e.text) for e in evs if e.kind == V.TRANSCRIPT]
    assert lines == [("agent", "Hi, I'd like a haircut."), ("callee", "Saturday is taken."), ("agent", "Okay,")]


def test_tool_calls_from_delegation_and_results(monkeypatch):
    ws = FakeWS(script=[(0.0, {"type": "session.delegation.created", "delegation": {"id": "item_1"}}),
                        (0.0, tr("input", " Thanks, good", 100)), (0.0, tr("input", "bye!", 300)),
                        (0.0, fn_call("ask_principal", {"question": "Is 650 ok?"}, "call_a")),
                        (0.0, fn_call("ask_principal", {"question": "Is 650 ok?"}, "call_a")),  # duplicate ignored
                        (0.0, fn_call("end_call", {"status": "achieved", "summary": "Booked."}, "call_b"))])
    eng = _engine(ws, monkeypatch)

    async def go():
        await eng.connect("x", VOICE_TOOLS)
        evs = await collect(eng)
        await eng.send_tool_result("call_a", json.dumps({"answer": "yes"}))
        await eng.send_tool_result("call_b", json.dumps({"ok": True}), respond=False)
        return evs
    evs = asyncio.run(go())
    calls = [(e.name, e.call_id, json.loads(e.arguments)) for e in evs if e.kind == V.TOOL_CALL]
    assert calls == [("ask_principal", "call_a", {"question": "Is 650 ok?"}),
                     ("end_call", "call_b", {"status": "achieved", "summary": "Booked."})]
    sent = [m for m in ws.sent if m["type"] in ("response.item.create", "response.create")]
    assert sent == [
        {"type": "response.item.create", "item": {"type": "function_call_output", "call_id": "call_a",
                                                  "output": "{\"answer\": \"yes\"}"}},
        {"type": "response.create"},
        {"type": "response.item.create", "item": {"type": "function_call_output", "call_id": "call_b",
                                                  "output": "{\"ok\": true}"}},
    ]


def test_goodbye_without_end_call_is_nudged_then_ended(monkeypatch):
    ws = FakeWS(script=[(0.0, tr("output", " Thanks a lot! Goodbye.", 1000)), (2.0, tr("input", " ", 9000))])
    eng = _engine(ws, monkeypatch, goodbye_nudge_s=0.3, goodbye_hangup_s=0.9)

    async def go():
        await eng.connect("x", VOICE_TOOLS)
        eng._stitch.idle_s = 0.1
        await eng.start_response()
        return await collect(eng, until=V.TOOL_CALL, timeout=5)
    evs = asyncio.run(go())
    end = evs[-1]
    assert end.kind == V.TOOL_CALL and end.name == "end_call" and end.call_id.startswith("local_")
    assert json.loads(end.arguments)["status"] == "partial"
    nudges = [m["content"] for m in ws.sent if m["type"] == "session.instructions.append"]
    assert any("Delegate end_call now" in n for n in nudges)


def test_errors_are_reported_and_connection_loss_is_fatal(monkeypatch):
    class Dropping(FakeWS):
        async def __anext__(self):
            if self.script:
                return await super().__anext__()
            raise ConnectionError("socket reset")
    ws = Dropping(script=[(0.0, {"type": "error", "error": {"message": "Missing required parameter: 'content'."}})])
    eng = _engine(ws, monkeypatch)

    async def go():
        await eng.connect("x", VOICE_TOOLS)
        return await collect(eng)
    evs = asyncio.run(go())
    errs = [e for e in evs if e.kind == V.ERROR]
    assert errs[0].text.startswith("Missing required parameter") and not errs[0].fatal
    assert errs[1].fatal and "socket reset" in errs[1].text
    assert evs[-1].kind == V.CLOSED


def test_make_engine_selects_live():
    from keryx.config import Settings
    from keryx.voice import make_engine
    eng = make_engine(Settings(voice_engine="openai-live", openai_api_key="sk-x", live_backend_model="gpt-5.4-nano"))
    assert isinstance(eng, OpenAILiveEngine) and eng.model == "gpt-live-1" and eng.backend_model == "gpt-5.4-nano"


def test_nothing_spoken_after_end_call(monkeypatch):
    speech = b"\x10\x00" * 2400
    ws = FakeWS(script=[(0.0, audio(speech)), (0.0, tr("output", " Goodbye.", 1000)),
                        (0.0, fn_call("end_call", {"status": "achieved", "summary": "ok"})),
                        (0.5, audio(speech)), (0.0, tr("output", " end_call:", 2000))])
    eng = _engine(ws, monkeypatch)

    async def go():
        await eng.connect("x", VOICE_TOOLS)
        await eng.start_response()
        return await collect(eng)
    evs = asyncio.run(go())
    assert len([e for e in evs if e.kind == V.AUDIO]) == 1
    assert [e.text for e in evs if e.kind == V.TRANSCRIPT] == ["Goodbye."]
    start = [m for m in [*eng._ws.sent] if m["type"] == "session.start"][0]
    assert "never say them" in start["session"]["instructions"]


def test_summarize_outcome_forces_end_call(monkeypatch):
    ws = FakeWS()
    eng = _engine(ws, monkeypatch, backend_model="gpt-5.4-mini")
    seen = {}

    def fake_post(url, body, key, timeout=20.0):
        seen.update(url=url, body=body)
        return {"output": [{"type": "message"}, {"type": "function_call", "name": "end_call",
                                                 "arguments": json.dumps({"status": "achieved", "summary": "Booked 12:30."})}]}
    monkeypatch.setattr(L, "_post_json", fake_post)

    async def go():
        await eng.connect("brief", VOICE_TOOLS)
        out = await eng.summarize_outcome("brief", [{"role": "agent", "text": "Can I book 12:30?"},
                                                    {"role": "callee", "text": "Yes, booked."}])
        await eng.close()
        return out
    out = asyncio.run(go())
    assert out == {"status": "achieved", "summary": "Booked 12:30."}
    assert seen["url"].endswith("/v1/responses") and seen["body"]["tool_choice"] == {"type": "function", "name": "end_call"}
    assert "callee: Yes, booked." in seen["body"]["input"] and seen["body"]["model"] == "gpt-5.4-mini"


def test_callee_goodbye_then_silence_ends_the_call(monkeypatch):
    ws = FakeWS(script=[(0.0, tr("input", " Okay, you're booked. Thank you, goodbye!", 1000)), (8.0, tr("input", " ", 9000))])
    eng = _engine(ws, monkeypatch)

    async def go():
        await eng.connect("x", VOICE_TOOLS)
        eng._stitch.idle_s = 0.1
        await eng.start_response()
        return await collect(eng, until=V.TOOL_CALL, timeout=10)
    evs = asyncio.run(go())
    end = evs[-1]
    assert end.name == "end_call" and end.call_id.startswith("local_")
    assert "other side said goodbye" in json.loads(end.arguments)["summary"]
    nudges = [m["content"] for m in ws.sent if m["type"] == "session.instructions.append"]
    assert any("They said goodbye" in n for n in nudges)


def test_premature_end_call_is_refused(monkeypatch):
    ws = FakeWS(script=[(0.0, fn_call("end_call", {"status": "achieved", "summary": "Booked."}, "call_early")),
                        (0.1, tr("output", " Great, thanks so much. Goodbye!", 3000)),
                        (1.8, fn_call("end_call", {"status": "achieved", "summary": "Booked."}, "call_late"))])
    eng = _engine(ws, monkeypatch)

    async def go():
        await eng.connect("x", VOICE_TOOLS)
        eng._stitch.idle_s = 0.1
        await eng.start_response()
        evs = await collect(eng)
        await asyncio.sleep(0.05)
        return evs
    evs = asyncio.run(go())
    ends = [e.call_id for e in evs if e.kind == V.TOOL_CALL]
    assert ends == ["call_late"]
    refusal = [m for m in ws.sent if m["type"] == "response.item.create" and m["item"]["call_id"] == "call_early"]
    assert refusal and "still in progress" in refusal[0]["item"]["output"]


def test_voicemail_and_wrap_up_may_end_without_goodbye(monkeypatch):
    ws = FakeWS(script=[(0.0, fn_call("end_call", {"status": "voicemail", "summary": "Voicemail."}, "vm"))])
    eng = _engine(ws, monkeypatch)

    async def go():
        await eng.connect("x", VOICE_TOOLS)
        return await collect(eng)
    assert [e.call_id for e in asyncio.run(go()) if e.kind == V.TOOL_CALL] == ["vm"]
    eng2 = _engine(FakeWS(), monkeypatch)

    async def go2():
        await eng2.connect("x", VOICE_TOOLS)
        await eng2.system_message("Time is nearly up. Wrap up now.")
        ok = eng2._end_allowed('{"status": "partial"}')
        await eng2.close()
        return ok
    assert asyncio.run(go2())
