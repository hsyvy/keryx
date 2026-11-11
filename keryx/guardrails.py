"""Guardrails enforced in code before any dial: number normalization, blocked number classes,
allow/deny lists, quiet hours and rate limiting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Optional

from .config import Settings

# Emergency and public-service numbers (national form). Anything this short is blocked anyway by the
# short-code rule; the explicit list also catches them when written with a country code (+91 112).
EMERGENCY_NUMBERS = frozenset({
    "112", "911", "999", "000", "08", "110", "118", "119", "120", "122", "15", "17", "18", "100", "101",
    "102", "103", "104", "108", "1091", "1098", "1099", "1930", "14567", "106", "116", "117", "191",
    "192", "193", "194", "195", "196", "197", "198", "199",
})
# Premium-rate / chargeable prefixes (E.164). Extend with Settings.blocked_numbers.
PREMIUM_PREFIXES = ("+1900", "+1976", "+449", "+4487", "+4470", "+61190", "+911900", "+91900")

MIN_NATIONAL_DIGITS = 7  # shorter national numbers are service/short codes, never dialed


class GuardrailViolation(Exception):
    pass


def normalize_number(raw: str, default_country_code: str = "+91") -> str:
    """Return E.164 (``+<digits>``) or raise GuardrailViolation."""
    s = str(raw or "").strip()
    if not s:
        raise GuardrailViolation("no phone number given")
    if re.search(r"[*#]", s):
        raise GuardrailViolation("numbers containing * or # (USSD/MMI codes) cannot be dialed")
    if not re.fullmatch(r"\+?[\d\s().\-]+", s):
        raise GuardrailViolation(f"not a phone number: {raw!r} (digits, spaces, dashes and a leading + only; "
                                 "use press_keys for extensions)")
    plus = s.startswith("+") or s.startswith("00")
    digits = re.sub(r"\D", "", s)
    if s.startswith("00"):
        digits = digits[2:]
    if not digits:
        raise GuardrailViolation(f"not a phone number: {raw!r}")
    if digits in EMERGENCY_NUMBERS:
        raise GuardrailViolation(f"{raw} is an emergency or public-service number; Keryx never calls those")
    if not plus:
        cc = re.sub(r"\D", "", default_country_code or "")
        if not cc:
            raise GuardrailViolation("number has no country code and no default_country_code is configured")
        national = digits.lstrip("0")  # trunk prefix
        if national in EMERGENCY_NUMBERS:
            raise GuardrailViolation(f"{raw} is an emergency or public-service number; Keryx never calls those")
        digits = cc + national
    if len(digits) > 15:
        raise GuardrailViolation(f"{raw} is longer than any valid phone number")
    return "+" + digits


def _national_part(e164: str) -> str:
    """Best-effort national significant number: strip a 1-3 digit country code (heuristic)."""
    d = e164.lstrip("+")
    if d.startswith(("1", "7")):
        return d[1:]
    # Common 2-digit codes; others assumed 3 digits.
    two = {"20", "27", "30", "31", "32", "33", "34", "36", "39", "40", "41", "43", "44", "45", "46", "47",
           "48", "49", "51", "52", "53", "54", "55", "56", "57", "58", "60", "61", "62", "63", "64", "65",
           "66", "81", "82", "84", "86", "90", "91", "92", "93", "94", "95", "98"}
    return d[2:] if d[:2] in two else d[3:]


def parse_quiet_hours(spec: str) -> Optional[tuple[dtime, dtime]]:
    spec = (spec or "").strip()
    if not spec:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", spec)
    if not m:
        raise ValueError(f"quiet_hours must look like '21:00-09:00', got {spec!r}")
    h1, m1, h2, m2 = (int(x) for x in m.groups())
    return dtime(h1, m1), dtime(h2, m2)


def in_quiet_hours(now: datetime, spec: str) -> bool:
    window = parse_quiet_hours(spec)
    if window is None:
        return False
    start, end = window
    t = now.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end  # wraps past midnight


def local_now(tz_name: str = "") -> datetime:
    if tz_name:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_name))
    return datetime.now().astimezone()


@dataclass
class GuardrailResult:
    number: str = ""
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def check_call(raw_number: str, settings: Settings, *, now: Optional[datetime] = None,
               calls_last_hour: int = 0) -> GuardrailResult:
    """All pre-dial checks. Never raises; collects human-readable violations."""
    res = GuardrailResult()
    try:
        number = normalize_number(raw_number, settings.default_country_code)
    except GuardrailViolation as e:
        res.violations.append(str(e))
        return res
    res.number = number
    national = _national_part(number)
    if national in EMERGENCY_NUMBERS or national.lstrip("0") in EMERGENCY_NUMBERS:
        res.violations.append(f"{number} is an emergency or public-service number; Keryx never calls those")
    elif len(national) < MIN_NATIONAL_DIGITS:
        res.violations.append(f"{number} looks like a short/service code; only full phone numbers can be dialed")
    if any(number.startswith(p) for p in PREMIUM_PREFIXES):
        res.violations.append(f"{number} is a premium-rate number")
    for blocked in settings.blocked_numbers:
        b = str(blocked).strip()
        if b and number.startswith(b if b.startswith("+") else "+" + re.sub(r"\D", "", b)):
            res.violations.append(f"{number} is on your blocked list")
            break
    if settings.allowed_numbers:
        allowed = set()
        for a in settings.allowed_numbers:
            try:
                allowed.add(normalize_number(str(a), settings.default_country_code))
            except GuardrailViolation:
                continue
        if number not in allowed:
            res.violations.append(f"{number} is not in allowed_numbers")
    try:
        current = now or local_now(settings.timezone)
        if in_quiet_hours(current, settings.quiet_hours):
            res.violations.append(f"it is {current:%H:%M} — inside quiet hours ({settings.quiet_hours}); "
                                  "try again later")
    except Exception as e:  # bad config must fail closed
        res.violations.append(f"quiet-hours check failed ({e}); fix the keryx quiet_hours/timezone setting")
    if calls_last_hour >= settings.max_calls_per_hour:
        res.violations.append(f"rate limit: {calls_last_hour} calls in the last hour "
                              f"(max {settings.max_calls_per_hour})")
    return res
