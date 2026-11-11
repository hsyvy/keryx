"""OpenAI Realtime adapter: event mapping (GA + beta names) and outgoing messages, with a fake socket."""

import asyncio
import base64
import json

from keryx.voice import base as V
from keryx.voice.openai_realtime import OpenAIRealtimeEngine


class FakeWS:
    def __init__(self, incoming):
        self.incoming = [json.dumps(m) for m in incoming]
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.incoming:
            raise StopAsyncIteration
        return self.incoming.pop(0)

    async def close(self):
        pass


def test_event_mapping_and_barge_in():
    audio = base64.b64encode(b"\x01\x00" * 10).decode()
    ws = FakeWS([
        {"type": "response.created"},
        {"type": "response.output_audio.delta", "item_id": "item_1", "delta": audio},
        {"type": "response.audio.delta", "item_id": "item_1", "delta": audio},
        {"type": "response.output_audio_transcript.done", "transcript": "Hello there"},
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": " Who is this? "},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "response.function_call_arguments.done", "call_id": "c1", "name": "end_call",
         "arguments": "{\"status\": \"achieved\", \"summary\": \"ok\"}"},
        {"type": "error", "error": {"code": "response_cancel_not_active", "message": "ignore me"}},
        {"type": "error", "error": {"message": "real problem"}},
        {"type": "response.done"},
    ])

    async def go():
        eng = OpenAIRealtimeEngine(api_key="sk-test")
        eng._ws = ws
        eng._reader = asyncio.create_task(eng._read_loop())
        events = [ev async for ev in eng.events()]
        await eng.interrupted(320)
        await eng.send_tool_result("c1", "{}", respond=False)
        await eng.system_message("Time is nearly up.")
        return events

    events = asyncio.run(go())
    kinds = [e.kind for e in events]
    assert kinds == [V.RESPONSE_STARTED, V.AUDIO, V.AUDIO, V.TRANSCRIPT, V.TRANSCRIPT, V.SPEECH_STARTED,
                     V.TOOL_CALL, V.ERROR, V.RESPONSE_DONE, V.CLOSED]
    assert events[1].data == b"\x01\x00" * 10
    assert (events[3].role, events[3].text) == ("agent", "Hello there")
    assert (events[4].role, events[4].text) == ("callee", "Who is this?")
    assert events[6].name == "end_call" and json.loads(events[6].arguments)["status"] == "achieved"
    assert events[7].text == "real problem"
    sent_types = [m["type"] for m in ws.sent]
    assert sent_types == ["response.cancel", "conversation.item.truncate", "conversation.item.create",
                          "conversation.item.create", "response.create"]
    assert ws.sent[1] == {"type": "conversation.item.truncate", "item_id": "item_1", "content_index": 0,
                          "audio_end_ms": 320}
    assert ws.sent[2]["item"]["type"] == "function_call_output"
    assert ws.sent[3]["item"]["role"] == "system"


def test_requires_key():
    try:
        OpenAIRealtimeEngine(api_key="")
    except RuntimeError as e:
        assert "OPENAI_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
