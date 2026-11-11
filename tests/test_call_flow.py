"""End-to-end call flow with the fake phone and scripted voice model."""

import asyncio
import json
from pathlib import Path

import pytest

from keryx.audio import AudioBridge, resample, rms
from keryx.brief import compile_brief
from keryx.config import Settings
from keryx.orchestrator import CallOrchestrator
from keryx.resolver import PrincipalResolver
from keryx.testing import FakePhone, ScriptedVoice, demo_script, fake_disclosure, simulate_call, tone
from keryx.tools import Keryx
from keryx.transport.base import NO_ANSWER

ARGS = {"to": "+919812345678", "callee_name": "Looks Salon", "goal": "Book a haircut Saturday 11am",
        "limits": ["max ₹600"], "may_share": ["Name: Asha"]}


def run(coro):
    return asyncio.run(coro)


def make_orch(phone, voice, *, resolver=None, max_s=300, cancel=None, disclosure_ms=200):
    brief = compile_brief(ARGS, principal="Asha")
    return CallOrchestrator(
        call_id="c1", number="+919812345678", brief=brief, transport=phone, engine=voice,
        resolver=resolver or PrincipalResolver(brief), disclosure_text="Hi, this is an AI assistant.",
        disclosure_pcm=tone(disclosure_ms, 24000, 660), max_call_seconds=max_s, ring_timeout_seconds=5,
        cancel_check=cancel)


def test_happy_path_disclosure_first_and_outcome():
    phone, voice = FakePhone(), ScriptedVoice(demo_script())
    brief = compile_brief(ARGS, principal="Asha")
    llm = lambda msgs: '{"action": "answer", "answer": "Yes, 12:30 is fine if it is 600.", "confident": true}'
    orch = make_orch(phone, voice, resolver=PrincipalResolver(brief, llm=llm))
    res = run(orch.run())
    assert res.state == "done" and res.status == "achieved", (res.state, res.error)
    assert res.outcome["details"]["price"] == "₹600"
    # Disclosure was played by the plugin before anything the model said, and is first in the transcript.
    assert res.transcript[0]["source"] == "plugin_disclosure"
    disclosure = tone(200, 24000, 660)
    assert bytes(phone.played[:len(disclosure)]) == disclosure
    # ask_principal got the resolver's answer.
    answers = [r for r in voice.tool_results.values() if isinstance(r, dict) and "answer" in r]
    assert answers and answers[0]["answer"].startswith("Yes")
    assert phone.hung_up and phone.closed and voice.closed
    assert any(t["role"] == "callee" for t in res.transcript)
    tr = res.to_tool_result()
    assert tr["success"] and tr["follow_ups"] and tr["transcript"][0].startswith("agent: Hi, this is")


def test_model_hears_nothing_during_disclosure():
    class Spy(ScriptedVoice):
        async def send_audio(self, pcm):
            assert orch.result.state not in ("connected", "disclosing"), "callee audio leaked before disclosure ended"
            await super().send_audio(pcm)
    phone, voice = FakePhone(), Spy([("sleep", 0.2), ("tool", "end_call", {"status": "achieved", "summary": "ok"})])
    orch = make_orch(phone, voice, disclosure_ms=400)
    res = run(orch.run())
    assert res.state == "done" and voice.audio_in > 0


def test_voice_connects_before_dial_and_failure_never_dials():
    phone, voice = FakePhone(), ScriptedVoice([], fail_connect=True)
    res = run(make_orch(phone, voice).run())
    assert res.state == "failed" and phone.dialed == []


def test_no_answer():
    phone, voice = FakePhone(answer=NO_ANSWER), ScriptedVoice([])
    res = run(make_orch(phone, voice).run())
    assert res.state == "no_answer" and res.status == "no_answer"
    assert phone.dialed == ["+919812345678"] and voice.closed and phone.played == bytearray()


def test_remote_hangup():
    phone = FakePhone(hangup_after_s=0.6)
    voice = ScriptedVoice([("say", "Hello?"), ("sleep", 5)])
    res = run(make_orch(phone, voice).run())
    assert res.state == "done" and res.outcome["status"] == "partial"
    assert "hung up" in res.outcome["summary"]


def test_time_cap_warns_and_ends():
    phone, voice = FakePhone(), ScriptedVoice([("sleep", 10)])
    res = run(make_orch(phone, voice, max_s=1).run())
    assert res.state == "done" and "Time limit" in res.outcome["summary"]
    assert voice.system_messages and "nearly up" in voice.system_messages[0]


def test_cancel_hangs_up():
    flag = {"stop": False}
    phone, voice = FakePhone(), ScriptedVoice([("sleep", 0.3), ("sleep", 10)])

    async def go():
        orch = make_orch(phone, voice, cancel=lambda: flag["stop"])
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.6)
        flag["stop"] = True
        return await task
    res = run(go())
    assert res.state == "cancelled" and phone.hung_up


def test_barge_in_drops_unplayed_audio():
    phone = FakePhone()
    voice = ScriptedVoice([("say", "A long sentence", 0.5), ("hear", "Sorry, who is this?", True), ("sleep", 0.2),
                           ("tool", "end_call", {"status": "partial", "summary": "x"})], say_ms=2000)
    res = run(make_orch(phone, voice).run())
    assert res.state == "done"
    assert voice.interruptions and voice.interruptions[0] < 2000
    assert len(phone.played) < (200 + 2000) * 48  # the 2 s reply was not fully played


def test_press_keys_validated():
    phone = FakePhone()
    voice = ScriptedVoice([("tool", "press_keys", {"digits": "1#"}), ("tool", "press_keys", {"digits": "12; rm"}),
                           ("tool", "end_call", {"status": "achieved", "summary": "menu"})])
    res = run(make_orch(phone, voice).run())
    assert phone.dtmf == ["1#"]
    results = list(voice.tool_results.values())
    assert results[0] == {"ok": True} and results[1]["ok"] is False
    assert res.state == "done"


def test_resolver_falls_back_to_user_then_unknown():
    brief = compile_brief(ARGS, principal="Asha")
    unsure = lambda msgs: '{"action": "answer", "answer": "maybe", "confident": false}'
    r = PrincipalResolver(brief, llm=unsure, ask_user=lambda q, t: "Yes, book it")
    assert run(r.resolve("Can we do 650?")) ["source"] == "principal"
    r2 = PrincipalResolver(brief, llm=unsure, ask_user=lambda q, t: None)
    out = run(r2.resolve("Can we do 650?"))
    assert out["answer"] is None and "do not guess" in out["note"]


def test_resolver_uses_only_allowlisted_tools():
    brief = compile_brief(ARGS, principal="Asha")
    calls = []
    steps = iter(['{"action": "tool", "tool": "calendar_list", "args": {"day": "sat"}}',
                  '{"action": "answer", "answer": "Free after 12.", "confident": true}'])
    r = PrincipalResolver(brief, llm=lambda m: next(steps), dispatch_tool=lambda n, a: calls.append(n) or "[]",
                          readonly_tools=["calendar_list"])
    assert run(r.resolve("Is Saturday afternoon ok?"))["answer"] == "Free after 12."
    assert calls == ["calendar_list"]
    r2 = PrincipalResolver(brief, llm=lambda m: '{"action": "tool", "tool": "terminal", "args": {}}',
                           dispatch_tool=lambda n, a: calls.append(n) or "", readonly_tools=["calendar_list"])
    assert run(r2.resolve("x"))["answer"] is None and calls == ["calendar_list"]


# --- plugin surface: approval hook + handler ------------------------------------------------------

def make_runtime(tmp_path: Path, phone=None, voice=None, **settings):
    settings.setdefault("quiet_hours", "")
    settings.setdefault("principal_name", "Asha")
    phone = phone or FakePhone()
    voice = voice or ScriptedVoice([("say", "Hi"), ("tool", "end_call", {"status": "achieved", "summary": "done"})])
    rt = Keryx(None, data_dir=tmp_path, settings_override=Settings(**settings),
                  transport_factory=lambda s, n: phone, engine_factory=lambda s: voice,
                  disclosure_renderer=fake_disclosure, cancel_check_factory=lambda: (lambda: False))
    return rt, phone, voice


def test_hook_ignores_other_tools(tmp_path):
    rt, *_ = make_runtime(tmp_path)
    assert rt.pre_tool_call("terminal", {"command": "ls"}) is None


def test_hook_blocks_emergency_and_bad_briefs(tmp_path):
    rt, phone, _ = make_runtime(tmp_path)
    blocked = rt.pre_tool_call("place_call", {**ARGS, "to": "112"})
    assert blocked["action"] == "block" and "emergency" in blocked["message"]
    assert rt.pre_tool_call("place_call", {"to": "+919812345678"})["action"] == "block"
    # and the handler refuses even if invoked directly
    out = json.loads(rt.handle_place_call({**ARGS, "to": "112"}))
    assert not out["success"] and phone.dialed == []


def test_handler_requires_hook_approval_once(tmp_path):
    rt, phone, _ = make_runtime(tmp_path)
    out = json.loads(rt.handle_place_call(dict(ARGS)))
    assert not out["success"] and "approv" in out["error"] and phone.dialed == []

    hook = rt.pre_tool_call("place_call", dict(ARGS))
    assert hook["action"] == "approve" and "Looks Salon" in hook["message"] and "max ₹600" in hook["message"]
    assert hook["rule_key"].startswith("keryx.place_call.")
    out = json.loads(rt.handle_place_call(dict(ARGS)))
    assert out["success"] and out["status"] == "achieved" and phone.dialed == ["+919812345678"]

    again = json.loads(rt.handle_place_call(dict(ARGS)))  # approval is single-use
    assert not again["success"] and phone.dialed == ["+919812345678"]

    listing = rt.calls_command("")
    assert "Looks Salon" in listing and "achieved" in listing
    detail = rt.calls_command(out["call_id"][:8])
    assert "Transcript:" in detail and "AI assistant" in detail


def test_rate_limit_counts_dialed_calls(tmp_path):
    rt, *_ = make_runtime(tmp_path, max_calls_per_hour=1)
    rt.pre_tool_call("place_call", dict(ARGS))
    assert json.loads(rt.handle_place_call(dict(ARGS)))["success"]
    blocked = rt.pre_tool_call("place_call", dict(ARGS))
    assert blocked["action"] == "block" and "rate limit" in blocked["message"]


def test_preflight_problem_stops_before_dial(tmp_path):
    rt, phone, voice = make_runtime(tmp_path, phone=FakePhone(preflight_problems=["no phone connected"]))
    rt.pre_tool_call("place_call", dict(ARGS))
    out = json.loads(rt.handle_place_call(dict(ARGS)))
    assert not out["success"] and "no phone connected" in out["error"] and phone.dialed == []


def test_simulate_command():
    out = run(simulate_call())
    assert out["result"]["status"] == "achieved" and "Looks Salon" in out["approval_prompt"]


# --- audio helpers ----------------------------------------------------------------------------------

def test_resample_and_rms():
    pcm = tone(100, 24000)
    assert len(resample(pcm, 24000, 16000)) == 1600 * 2
    assert len(resample(pcm, 24000, 8000)) == 800 * 2
    assert rms(bytes(960)) == 0 and rms(pcm) > 3000


def test_echo_gate_on_speakerphone():
    async def go():
        phone = FakePhone(echo_prone=True)
        b = AudioBridge(phone, 24000, 24000)
        b.play(tone(500, 24000))
        quiet = tone(20, 24000, amp=500)
        loud = tone(20, 24000, amp=12000)
        return rms(b.process_inbound(quiet)), rms(b.process_inbound(loud))
    q, l = run(go())
    assert q == 0 and l > 5000


def test_bridge_keeps_a_head_start_in_the_transport():
    async def go():
        phone = FakePhone()
        phone.playback_lead_ms = 80
        b = AudioBridge(phone, 24000, 24000)
        b.start()
        await asyncio.sleep(0.005)  # the head start is written at once, before real-time pacing
        early = len(phone.played)
        await b.close()
        return early
    assert run(go()) >= 4 * 960


def test_unreported_outcome_is_summarized_after_hangup():
    class Summarizing(ScriptedVoice):
        async def summarize_outcome(self, instructions, transcript):
            self.summarized_with = [t["role"] for t in transcript]
            assert "Book a haircut" in instructions
            return {"status": "achieved", "summary": "Salon confirmed 12:30.", "details": {"time": "12:30"}}
    phone = FakePhone(hangup_after_s=0.6)
    voice = Summarizing([("say", "Hello?"), ("hear", "Booked for 12:30, bye"), ("sleep", 5)])
    res = run(make_orch(phone, voice).run())
    assert res.outcome["status"] == "achieved" and res.outcome["details"]["time"] == "12:30"
    assert res.outcome["details"]["how_it_ended"] == "The other side hung up."
    assert "callee" in voice.summarized_with and phone.hung_up


def test_model_reported_outcome_is_not_overwritten():
    class Summarizing(ScriptedVoice):
        async def summarize_outcome(self, instructions, transcript):
            raise AssertionError("must not summarize when the model reported")
    voice = Summarizing([("say", "Hi"), ("hear", "hello"), ("tool", "end_call", {"status": "declined", "summary": "No slots."})])
    res = run(make_orch(FakePhone(), voice).run())
    assert res.outcome["status"] == "declined"


def test_stop_hook_hangs_up_the_call(tmp_path):
    from keryx.config import Settings
    phone = FakePhone()
    voice = ScriptedVoice([("say", "Hi"), ("sleep", 10)])
    rt = Keryx(None, data_dir=tmp_path, settings_override=Settings(quiet_hours="", principal_name="Asha"),
               transport_factory=lambda s, n: phone, engine_factory=lambda s: voice,
               disclosure_renderer=fake_disclosure)
    rt.pre_tool_call("place_call", dict(ARGS))
    import threading, time
    threading.Timer(0.8, rt.on_agent_loop_stopped, kwargs={"session_key": "s", "reason": "stop"}).start()
    t0 = time.monotonic()
    out = json.loads(rt.handle_place_call(dict(ARGS)))
    assert time.monotonic() - t0 < 6 and out["call_state"] == "cancelled" and phone.hung_up
    rt.on_agent_loop_stopped()  # no call in progress: harmless
    assert not rt._stop.is_set()
