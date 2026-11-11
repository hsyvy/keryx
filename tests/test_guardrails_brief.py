from datetime import datetime

import pytest

from keryx.brief import BriefError, build_instructions, compile_brief, normalize_outcome
from keryx.config import Settings, load_settings
from keryx.guardrails import GuardrailViolation, check_call, in_quiet_hours, normalize_number

NOON = datetime(2026, 11, 11, 12, 0)


def s(**kw):
    kw.setdefault("quiet_hours", "")
    return Settings(**kw)


@pytest.mark.parametrize("raw,expected", [
    ("+91 98123 45678", "+919812345678"),
    ("098123-45678", "+919812345678"),
    ("9812345678", "+919812345678"),
    ("0044 20 7946 0958", "+442079460958"),
])
def test_normalize(raw, expected):
    assert normalize_number(raw, "+91") == expected


@pytest.mark.parametrize("raw", ["112", "911", "100", "108", "+91 112", "+1 911", "999", "1098", "+44 999"])
def test_emergency_numbers_blocked(raw):
    res = check_call(raw, s(), now=NOON)
    assert not res.ok
    assert any("emergency" in v or "short" in v for v in res.violations)


@pytest.mark.parametrize("raw", ["*99#", "#21#", "*123*1#"])
def test_ussd_blocked(raw):
    with pytest.raises(GuardrailViolation):
        normalize_number(raw)


def test_short_codes_and_letters_blocked():
    assert not check_call("12345", s(), now=NOON).ok
    assert not check_call("1-800-FLOWERS", s(), now=NOON).ok


def test_premium_blocked():
    assert not check_call("+1 900 555 0100", s(), now=NOON).ok


def test_ordinary_number_ok():
    res = check_call("+91 98123 45678", s(), now=NOON)
    assert res.ok, res.violations
    assert res.number == "+919812345678"


def test_allow_and_block_lists():
    assert not check_call("+919812345678", s(allowed_numbers=["+919800000000"]), now=NOON).ok
    assert check_call("+919812345678", s(allowed_numbers=["98123 45678"]), now=NOON).ok
    assert not check_call("+919812345678", s(blocked_numbers=["+9198123"]), now=NOON).ok


def test_quiet_hours_wrap_midnight():
    assert in_quiet_hours(datetime(2026, 1, 1, 22, 30), "21:00-09:00")
    assert in_quiet_hours(datetime(2026, 1, 1, 8, 59), "21:00-09:00")
    assert not in_quiet_hours(datetime(2026, 1, 1, 12, 0), "21:00-09:00")
    res = check_call("+919812345678", Settings(quiet_hours="21:00-09:00"), now=datetime(2026, 1, 1, 23, 0))
    assert not res.ok and "quiet hours" in res.violations[0]


def test_bad_quiet_hours_fails_closed():
    assert not check_call("+919812345678", Settings(quiet_hours="late"), now=NOON).ok


def test_rate_limit():
    assert not check_call("+919812345678", s(max_calls_per_hour=2), now=NOON, calls_last_hour=2).ok


def test_call_cap_cannot_be_raised():
    assert Settings(max_call_seconds=3600).max_call_seconds == 300
    assert Settings(max_call_seconds=120).max_call_seconds == 120


def test_load_settings_from_getter_and_secret():
    cfg = {"principal_name": "Asha", "max_call_seconds": 900, "readonly_tools": "cal_list, cal_get"}
    st = load_settings(lambda k, d=None: cfg.get(k, d), lambda name: "sk-test" if name == "OPENAI_API_KEY" else None)
    assert st.principal == "Asha" and st.max_call_seconds == 300
    assert st.readonly_tools == ["cal_list", "cal_get"] and st.openai_api_key == "sk-test"
    assert "Asha" in st.disclosure


def test_brief_validation():
    with pytest.raises(BriefError):
        compile_brief({"goal": "x"})
    with pytest.raises(BriefError):
        compile_brief({"to": "+919812345678"})
    with pytest.raises(BriefError):
        compile_brief({"to": "1", "goal": "x", "limits": 5})
    b = compile_brief({"to": "+919812345678", "goal": "Book", "limits": "max ₹600"}, principal="Asha")
    assert b.limits == ["max ₹600"] and b.principal == "Asha" and b.callee == "+919812345678"


def test_instructions_carry_brief_and_disclosure():
    b = compile_brief({"to": "+919812345678", "goal": "Book a haircut", "limits": ["max ₹600"],
                       "may_share": ["Name: Asha"], "language": "Hinglish"}, principal="Asha")
    text = build_instructions(b, "Hi, this is an AI assistant calling on behalf of Asha.")
    for needle in ("Book a haircut", "max ₹600", "Name: Asha", "Hinglish", "ALREADY", "ask_principal", "end_call"):
        assert needle in text


def test_normalize_outcome():
    o = normalize_outcome('{"status": "weird", "summary": "s", "details": [1], "follow_ups": ["a"]}')
    assert o == {"status": "partial", "summary": "s", "details": {}, "follow_ups": ["a"]}
    assert normalize_outcome(None)["status"] == "partial"
