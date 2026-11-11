"""Brief compiler: turns Hermes's place_call arguments into a validated Brief and the voice
model's instructions + function tools."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

OUTCOME_STATUSES = ("achieved", "partial", "failed", "declined", "voicemail", "wrong_number")


class BriefError(ValueError):
    pass


def _str_list(value: Any, name: str, max_items: int = 20) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise BriefError(f"{name} must be a list of strings")
    out = [str(v).strip() for v in value if str(v).strip()]
    if len(out) > max_items:
        raise BriefError(f"{name} has more than {max_items} items")
    return out


@dataclass
class Brief:
    to: str                      # as given by Hermes; normalized number lives in Guardrails
    goal: str
    callee_name: str = ""
    context: str = ""
    limits: list[str] = field(default_factory=list)
    may_share: list[str] = field(default_factory=list)
    must_not_share: list[str] = field(default_factory=list)
    language: str = "English"
    success_criteria: str = ""
    voicemail_message: str = ""
    principal: str = "my principal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def callee(self) -> str:
        return self.callee_name or self.to


def compile_brief(args: dict[str, Any], principal: str = "my principal") -> Brief:
    """Validate place_call arguments. Raises BriefError with a message Hermes can act on."""
    if not isinstance(args, dict):
        raise BriefError("arguments must be an object")
    to = str(args.get("to") or "").strip()
    goal = str(args.get("goal") or "").strip()
    if not to:
        raise BriefError("'to' (phone number) is required")
    if not goal:
        raise BriefError("'goal' is required: what should this call achieve?")
    if len(goal) > 1000:
        raise BriefError("'goal' is too long (max 1000 characters); keep the brief focused")
    context = str(args.get("context") or "").strip()
    if len(context) > 4000:
        raise BriefError("'context' is too long (max 4000 characters)")
    return Brief(
        to=to, goal=goal,
        callee_name=str(args.get("callee_name") or "").strip()[:120],
        context=context,
        limits=_str_list(args.get("limits"), "limits"),
        may_share=_str_list(args.get("may_share"), "may_share", 40),
        must_not_share=_str_list(args.get("must_not_share"), "must_not_share", 40),
        language=str(args.get("language") or "English").strip()[:60] or "English",
        success_criteria=str(args.get("success_criteria") or "").strip()[:1000],
        voicemail_message=str(args.get("voicemail_message") or "").strip()[:500],
        principal=principal,
    )


def _bullets(items: list[str], empty: str) -> str:
    return "\n".join(f"- {i}" for i in items) if items else f"- {empty}"


def build_instructions(brief: Brief, disclosure: str) -> str:
    """System instructions for the realtime voice model."""
    voicemail = (f'If you reach voicemail or an answering machine, leave exactly this message, then call '
                 f'end_call with status "voicemail": "{brief.voicemail_message}"'
                 if brief.voicemail_message else
                 'If you reach voicemail or an answering machine, do not leave a message; call end_call '
                 'with status "voicemail".')
    return f"""You are a phone assistant on a live outbound call, calling on behalf of {brief.principal}.
You are speaking with: {brief.callee}.

The call has ALREADY started with this disclosure, played to them before you speak: "{disclosure}"
Do not repeat the disclosure word for word, but if anyone asks, always confirm truthfully that you are an AI assistant.

# Goal
{brief.goal}
{f"Success looks like: {brief.success_criteria}" if brief.success_criteria else ""}

# Background (for you; share only what is allowed below)
{brief.context or "(none)"}

# Hard limits: never exceed or agree to anything beyond these
{_bullets(brief.limits, "No commitments of money, time or personal details beyond the goal.")}

# You may share
{_bullets(brief.may_share, f"Only {brief.principal}'s first name and the purpose of the call.")}

# Never share
{_bullets(brief.must_not_share, "Anything not listed under 'You may share'.")}
- Payment card numbers, OTPs, passwords, bank details or government ID numbers, ever.

# How to talk
- Speak {brief.language}. Match the other person's language if they switch.
- Sound natural and polite. Keep each turn to one or two short sentences, then let them speak.
- Open by briefly saying why you are calling.
- If they ask something you cannot answer from this brief, or ask you to agree to anything outside the limits,
  say "One moment please, let me check." and call ask_principal with the exact question. Use its answer.
  If ask_principal returns no answer, say you will confirm later and do not guess.
- If an automated menu asks you to press keys, call press_keys with the digits.
- Never invent facts, prices, names, dates or commitments.
- {voicemail}

# Ending the call
When the goal is achieved, clearly refused, or cannot progress: thank them, say goodbye, and THEN call end_call
with an honest status and summary. Record concrete facts (prices, dates, times, names, reference numbers) in details.
Report what the other person actually said, not what you hoped: if they said they could not hear you, could not
understand you, or the line was bad, say so in the summary and do not use "achieved" for that part of the goal.
If their words reach you garbled, ask them to repeat instead of guessing.
You have a strict time limit; if told time is nearly up, wrap up immediately.
"""


VOICE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "ask_principal",
        "description": ("Ask the person you are calling for (your principal) a question you cannot answer from "
                        "the brief. Say 'One moment please' first. Returns their answer, or no answer."),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The exact question, self-contained."}},
            "required": ["question"],
        },
    },
    {
        "type": "function",
        "name": "press_keys",
        "description": "Send touch-tone (DTMF) digits for an automated phone menu.",
        "parameters": {
            "type": "object",
            "properties": {"digits": {"type": "string", "description": "Digits 0-9, * or #."}},
            "required": ["digits"],
        },
    },
    {
        "type": "function",
        "name": "end_call",
        "description": "Report the outcome and hang up. Call this only AFTER saying goodbye.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": list(OUTCOME_STATUSES)},
                "summary": {"type": "string", "description": "One or two sentences: what happened."},
                "details": {"type": "object", "description": "Concrete facts agreed or learned (key: value).",
                            "additionalProperties": True},
                "follow_ups": {"type": "array", "items": {"type": "string"},
                               "description": "Things the principal should do next (e.g. calendar event)."},
            },
            "required": ["status", "summary"],
        },
    },
]


def normalize_outcome(raw: Any) -> dict[str, Any]:
    """Coerce the model's end_call arguments into a well-formed outcome dict."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {"summary": raw}
    if not isinstance(raw, dict):
        raw = {}
    status = str(raw.get("status") or "partial")
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    follow = raw.get("follow_ups") if isinstance(raw.get("follow_ups"), list) else []
    return {
        "status": status if status in OUTCOME_STATUSES else "partial",
        "summary": str(raw.get("summary") or "").strip()[:2000],
        "details": details,
        "follow_ups": [str(f)[:500] for f in follow][:20],
    }
