"""OpenAI Realtime API adapter (speech-to-speech over WebSocket, server VAD, function calling).

Speaks the GA event schema and also accepts the older beta event names.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Optional

from .base import (AUDIO, CLOSED, ERROR, RESPONSE_DONE, RESPONSE_STARTED, SPEECH_STARTED, TOOL_CALL,
                   TRANSCRIPT, VoiceEngine, VoiceEvent)

logger = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime"
_IGNORED_ERRORS = {"response_cancel_not_active", "conversation_already_has_active_response"}


class OpenAIRealtimeEngine(VoiceEngine):
    name = "openai-realtime"
    input_rate = 24000
    output_rate = 24000

    def __init__(self, api_key: str, model: str = "gpt-realtime", voice: str = "marin",
                 transcription_model: str = "gpt-4o-mini-transcribe", vad_silence_ms: int = 450) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (keryx needs it for the realtime voice model)")
        self.api_key, self.model, self.voice = api_key, model, voice
        self.transcription_model = transcription_model
        self.vad_silence_ms = vad_silence_ms
        self._ws: Any = None
        self._queue: asyncio.Queue[VoiceEvent] = asyncio.Queue()
        self._reader: Optional[asyncio.Task] = None
        self._audio_item_id: Optional[str] = None
        self._send_lock = asyncio.Lock()

    async def connect(self, instructions: str, tools: list[dict[str, Any]]) -> None:
        try:
            from websockets.asyncio.client import connect as ws_connect  # websockets >= 13
            kw = {"additional_headers": {"Authorization": f"Bearer {self.api_key}"}}
        except ImportError:  # pragma: no cover
            from websockets import connect as ws_connect  # type: ignore
            kw = {"extra_headers": {"Authorization": f"Bearer {self.api_key}"}}
        self._ws = await ws_connect(f"{REALTIME_URL}?model={self.model}", max_size=None, **kw)
        await self._send({"type": "session.update", "session": {
            "type": "realtime",
            "instructions": instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.input_rate},
                    "transcription": {"model": self.transcription_model},
                    "turn_detection": {"type": "server_vad", "silence_duration_ms": self.vad_silence_ms,
                                       "prefix_padding_ms": 300, "create_response": True,
                                       "interrupt_response": True},
                },
                "output": {"format": {"type": "audio/pcm", "rate": self.output_rate}, "voice": self.voice},
            },
            "tools": tools,
            "tool_choice": "auto",
        }})
        self._reader = asyncio.create_task(self._read_loop())

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            return
        async with self._send_lock:
            await self._ws.send(json.dumps(payload))

    async def _read_loop(self) -> None:
        q = self._queue
        try:
            async for raw in self._ws:
                try:
                    ev = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                t = ev.get("type", "")
                if t in ("response.output_audio.delta", "response.audio.delta"):
                    self._audio_item_id = ev.get("item_id") or self._audio_item_id
                    try:
                        q.put_nowait(VoiceEvent(AUDIO, data=base64.b64decode(ev.get("delta") or "")))
                    except (ValueError, TypeError):
                        pass
                elif t in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
                    if ev.get("transcript"):
                        q.put_nowait(VoiceEvent(TRANSCRIPT, role="agent", text=ev["transcript"]))
                elif t == "conversation.item.input_audio_transcription.completed":
                    if (ev.get("transcript") or "").strip():
                        q.put_nowait(VoiceEvent(TRANSCRIPT, role="callee", text=ev["transcript"].strip()))
                elif t == "input_audio_buffer.speech_started":
                    q.put_nowait(VoiceEvent(SPEECH_STARTED))
                elif t == "response.function_call_arguments.done":
                    q.put_nowait(VoiceEvent(TOOL_CALL, call_id=ev.get("call_id", ""), name=ev.get("name", ""),
                                            arguments=ev.get("arguments") or "{}"))
                elif t == "response.created":
                    q.put_nowait(VoiceEvent(RESPONSE_STARTED))
                elif t == "response.done":
                    q.put_nowait(VoiceEvent(RESPONSE_DONE))
                elif t == "error":
                    err = ev.get("error") or {}
                    if err.get("code") in _IGNORED_ERRORS:
                        continue
                    q.put_nowait(VoiceEvent(ERROR, text=str(err.get("message") or err)))
        except Exception as e:  # connection dropped
            q.put_nowait(VoiceEvent(ERROR, text=f"realtime connection lost: {e}", fatal=True))
        finally:
            q.put_nowait(VoiceEvent(CLOSED))

    async def events(self) -> AsyncIterator[VoiceEvent]:
        while True:
            ev = await self._queue.get()
            yield ev
            if ev.kind == CLOSED:
                return

    async def send_audio(self, pcm: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def send_tool_result(self, call_id: str, output: str, respond: bool = True) -> None:
        await self._send({"type": "conversation.item.create",
                          "item": {"type": "function_call_output", "call_id": call_id, "output": output}})
        if respond:
            await self._send({"type": "response.create"})

    async def system_message(self, text: str, respond: bool = True) -> None:
        await self._send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "system", "content": [{"type": "input_text", "text": text}]}})
        if respond:
            await self._send({"type": "response.create"})

    async def start_response(self) -> None:
        await self._send({"type": "response.create"})

    async def interrupted(self, audio_heard_ms: int) -> None:
        await self._send({"type": "response.cancel"})
        if self._audio_item_id:
            await self._send({"type": "conversation.item.truncate", "item_id": self._audio_item_id,
                              "content_index": 0, "audio_end_ms": max(0, int(audio_heard_ms))})
            self._audio_item_id = None

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
