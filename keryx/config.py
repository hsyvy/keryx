"""Plugin settings: defaults merged with ``plugins.entries.keryx.settings`` from Hermes config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional

HARD_MAX_CALL_SECONDS = 300  # the 5-minute cap is a safety rule; config may only lower it


@dataclass
class Settings:
    transport: str = "bluetooth-linux"
    voice_engine: str = "openai-realtime"
    openai_api_key: str = ""
    realtime_model: str = "gpt-realtime"
    live_model: str = "gpt-live-1"
    live_backend_model: str = "gpt-5.4-mini"
    voice: str = "marin"
    principal_name: str = ""
    default_country_code: str = "+91"
    disclosure_text: str = "Hi, this is an AI assistant calling on behalf of {principal}."
    disclosure_audio_file: str = ""
    max_call_seconds: int = HARD_MAX_CALL_SECONDS
    ring_timeout_seconds: int = 45
    quiet_hours: str = "21:00-09:00"
    timezone: str = ""
    max_calls_per_hour: int = 6
    allowed_numbers: list = field(default_factory=list)
    blocked_numbers: list = field(default_factory=list)
    record: bool = False
    readonly_tools: list = field(default_factory=list)
    principal_timeout_seconds: int = 30
    bluetooth_device: str = ""
    telephony_bus: str = "user"

    def __post_init__(self) -> None:
        self.max_call_seconds = max(30, min(int(self.max_call_seconds or HARD_MAX_CALL_SECONDS),
                                            HARD_MAX_CALL_SECONDS))
        self.ring_timeout_seconds = max(10, min(int(self.ring_timeout_seconds or 45), 120))
        self.principal_timeout_seconds = max(5, min(int(self.principal_timeout_seconds or 30), 90))
        self.max_calls_per_hour = max(1, int(self.max_calls_per_hour or 6))
        for name in ("allowed_numbers", "blocked_numbers", "readonly_tools"):
            value = getattr(self, name)
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            setattr(self, name, list(value or []))

    @property
    def principal(self) -> str:
        return self.principal_name.strip() or "my principal"

    @property
    def disclosure(self) -> str:
        return self.disclosure_text.replace("{principal}", self.principal).strip()


def load_settings(get_config: Optional[Callable[[str, Any], Any]] = None,
                  get_secret: Optional[Callable[[str], Optional[str]]] = None) -> Settings:
    """Build Settings from a Hermes ``ctx.get_config``-style getter (missing keys keep defaults)."""
    values: dict[str, Any] = {}
    if get_config is not None:
        sentinel = object()
        for f in fields(Settings):
            try:
                v = get_config(f.name, sentinel)
            except Exception:
                v = sentinel
            if v is not sentinel and v is not None:
                values[f.name] = v
    s = Settings(**values)
    if not s.openai_api_key:
        if get_secret is not None:  # inside Hermes: profile-scoped secrets only, never os.environ
            try:
                key = get_secret("OPENAI_API_KEY")
            except Exception:
                key = None
        else:  # standalone (tests, simulator)
            key = os.environ.get("OPENAI_API_KEY")
        s.openai_api_key = key or ""
    return s
