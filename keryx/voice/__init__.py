"""Voice engine adapters: OpenAI Realtime and OpenAI GPT-Live. Later: Gemini Live, Zavi stack (Indic tuning)."""

from __future__ import annotations

from .base import VoiceEngine, VoiceEvent

__all__ = ["VoiceEngine", "VoiceEvent", "make_engine"]


def make_engine(settings) -> VoiceEngine:
    if settings.voice_engine == "openai-realtime":
        from .openai_realtime import OpenAIRealtimeEngine
        return OpenAIRealtimeEngine(api_key=settings.openai_api_key, model=settings.realtime_model,
                                    voice=settings.voice)
    if settings.voice_engine == "openai-live":
        from .openai_live import OpenAILiveEngine
        return OpenAILiveEngine(api_key=settings.openai_api_key, model=settings.live_model,
                                voice=settings.voice, backend_model=settings.live_backend_model)
    raise RuntimeError(f"unknown voice_engine {settings.voice_engine!r} (available: openai-realtime, openai-live)")
