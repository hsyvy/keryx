"""bluetooth-linux transport against a fake oFono-compatible D-Bus (no real phone)."""

import asyncio

from keryx.transport.base import ANSWERED, BUSY, NO_ANSWER, TransportError
from keryx.transport.linux_bluetooth import BluetoothLinuxTransport, _pairs
from keryx.transport.pipewire import find_bluez_call_nodes, find_phone_address

MODEM = "/org/pipewire/Telephony/ag0"
CALL = MODEM + "/call0"


class FakeBus:
    """PipeWire-style by default: Dial(s) returns nothing and the call appears in GetCalls.
    bus="system" mimics oFono: Dial(ss) returns the call path."""

    def __init__(self, states, modems=None, root_missing=True, bus="user", busy_line=False, call_appears=True):
        self.bus = bus
        self.service = {"user": "org.pipewire.Telephony", "system": "org.ofono"}[bus]
        self.states = list(states)
        self.calls = []
        self.modems = modems if modems is not None else {MODEM: {"Interfaces": {"type": "as", "data": [
            "org.ofono.VoiceCallManager"]}, "Online": {"type": "b", "data": True}}}
        self.root_missing = root_missing
        self.dialed = busy_line
        self.call_appears = call_appears

    async def call(self, path, iface, method, sig="", *args, timeout=10.0):
        self.calls.append((path, iface, method, args))
        if method == "GetModems":
            if path == "/" and self.root_missing:
                raise TransportError("no such object")
            return self.modems
        if method == "Dial":
            assert sig == ("s" if self.bus == "user" else "ss") and len(args) == len(sig)
            self.dialed = True
            return CALL if self.bus == "system" else None
        if method == "GetCalls":
            if not self.dialed or not self.call_appears:
                return {}
            state = self.states.pop(0) if len(self.states) > 1 else (self.states[0] if self.states else "")
            return {CALL: {"State": {"type": "s", "data": state}}} if state else {}
        return None


def run(c):
    return asyncio.run(c)


def test_pairs_handles_both_shapes():
    assert _pairs([["/a", {"x": {"type": "s", "data": "1"}}]]) == [("/a", {"x": "1"})]
    assert _pairs({"/b": {"y": {"type": "b", "data": True}}}) == [("/b", {"y": True})]


def test_preflight_dial_and_answer():
    bus = FakeBus(["dialing", "dialing", "alerting", "active"])
    t = BluetoothLinuxTransport(client=bus, poll_s=0.01)
    assert run(t.preflight()) == [] and t.modem == MODEM
    run(t.dial("+919812345678"))
    assert ("Dial", ("+919812345678",)) in [c[2:] for c in bus.calls] and t.call_path == CALL
    assert run(t.wait_answered(5)) == ANSWERED
    run(t.hangup())
    assert bus.calls[-1][1:3] == ("org.ofono.VoiceCall", "Hangup") and t.remote_ended()


def test_ofono_dial_returns_path():
    bus = FakeBus(["dialing", "active"], bus="system")
    t = BluetoothLinuxTransport(client=bus, poll_s=0.01)
    run(t.dial("+919812345678"))
    assert t.call_path == CALL and ("Dial", ("+919812345678", "")) in [c[2:] for c in bus.calls]
    assert run(t.wait_answered(5)) == ANSWERED


def test_lost_call_object_still_hangs_up_everything():
    bus = FakeBus(["dialing"], call_appears=False)
    t = BluetoothLinuxTransport(client=bus, poll_s=0.01)
    import keryx.transport.linux_bluetooth as lb
    real = lb.time.monotonic
    start = real()
    lb.time.monotonic = lambda: real() + (0 if real() - start < 0.05 else 60)
    try:
        try:
            run(t.dial("+919812345678"))
        except TransportError as e:
            assert "no call appeared" in str(e)
        else:
            raise AssertionError("expected TransportError")
    finally:
        lb.time.monotonic = real
    run(t.hangup())
    assert bus.calls[-1][2] == "HangupAll" and t.remote_ended()


def test_start_audio_links_call_streams(monkeypatch):
    import keryx.transport.linux_bluetooth as lb

    async def fake_dump():
        return PW_DUMP_CALL_STREAMS

    class FakeStream:
        def __init__(self, kind, rate, **kw):
            self.node_name = "keryx." + kind
            self.kw = kw

        async def start(self):
            pass

        async def stop(self):
            pass
    linked = []

    async def fake_links(source, sink, cap, play):
        linked.append((source, sink, cap, play))
        return []
    monkeypatch.setattr(lb, "pw_dump", fake_dump)
    monkeypatch.setattr(lb, "PwStream", FakeStream)
    monkeypatch.setattr(lb, "ensure_call_links", fake_links)

    class FakeVirtual:
        def __init__(self, rate):
            self.started = self.stopped = False

        async def start(self):
            self.started = True

        async def stop(self):
            self.stopped = True
    monkeypatch.setattr(lb, "VirtualNodes", FakeVirtual)
    bus = FakeBus(["active"])
    t = BluetoothLinuxTransport(device="A0:B1:C2:D3:E4:F5", client=bus, poll_s=0.01)
    run(t.preflight())
    run(t.dial("+919812345678"))

    async def go():
        await t.start_audio()
        assert t._capture.kw == {"autoconnect": False}
        assert t._playback.kw == {"latency": "60ms", "autoconnect": False}
        virtual = t._virtual
        assert virtual.started
        await t.close()
        assert virtual.stopped and t._virtual is None
    run(go())
    assert linked[0] == ("bluez_input.A0_B1_C2_D3_E4_F5.0", "bluez_output.A0_B1_C2_D3_E4_F5.1",
                         "keryx.capture", "keryx.playback")
    assert (lb.AG_TRANSPORT, "Activate") not in [c[1:3] for c in bus.calls]  # streams were already there


def test_rejected_and_unanswered():
    t = BluetoothLinuxTransport(client=FakeBus(["dialing", "disconnected"]), poll_s=0.01)
    run(t.dial("+919812345678"))
    assert run(t.wait_answered(5)) == BUSY
    t2 = BluetoothLinuxTransport(client=FakeBus(["dialing", "alerting", ""]), poll_s=0.01)
    run(t2.dial("+919812345678"))
    assert run(t2.wait_answered(5)) == NO_ANSWER


def test_ring_timeout_hangs_up():
    bus = FakeBus(["alerting"])
    t = BluetoothLinuxTransport(client=bus, poll_s=0.01)
    run(t.dial("+919812345678"))
    assert run(t.wait_answered(0.1)) == NO_ANSWER
    assert any(c[2] == "Hangup" for c in bus.calls)


def test_preflight_reports_missing_phone_and_busy_line():
    t = BluetoothLinuxTransport(client=FakeBus([""], modems={}), poll_s=0.01)
    problems = run(t.preflight())
    assert problems and "no phone connected" in problems[0]
    t2 = BluetoothLinuxTransport(client=FakeBus(["active"], busy_line=True), poll_s=0.01)
    assert "already on a call" in run(t2.preflight())[0]


def test_dtmf():
    bus = FakeBus(["active"], busy_line=True)
    t = BluetoothLinuxTransport(client=bus)
    run(t.preflight())
    run(t.send_dtmf("12#"))
    assert bus.calls[-1][2:] == ("SendTones", ("12#",))


PW_DUMP = [
    {"type": "PipeWire:Interface:Device", "info": {"props": {
        "device.api": "bluez5", "api.bluez5.address": "60:55:56:BE:F3:C3", "api.bluez5.class": "0x240404",
        "device.icon-name": "audio-headset-bluetooth"}}},
    {"type": "PipeWire:Interface:Device", "info": {"props": {
        "device.api": "bluez5", "api.bluez5.address": "AA:BB:CC:DD:EE:FF", "api.bluez5.class": "0x5a020c",
        "device.icon-name": "phone"}}},
    {"type": "PipeWire:Interface:Node", "info": {"props": {
        "node.name": "bluez_input.60:55:56:BE:F3:C3", "media.class": "Audio/Source"}}},
    {"type": "PipeWire:Interface:Node", "info": {"props": {
        "node.name": "bluez_capture_internal.AA:BB:CC:DD:EE:FF", "media.class": "Stream/Input/Audio/Internal"}}},
    {"type": "PipeWire:Interface:Node", "info": {"props": {
        "node.name": "bluez_input.AA:BB:CC:DD:EE:FF", "media.class": "Audio/Source"}}},
    {"type": "PipeWire:Interface:Node", "info": {"props": {
        "node.name": "bluez_output.AA_BB_CC_DD_EE_FF.1", "media.class": "Audio/Sink",
        "api.bluez5.address": "AA:BB:CC:DD:EE:FF"}}},
]


PW_DUMP_CALL_STREAMS = [  # what PipeWire 1.6 showed during a real call (laptop = hands-free unit)
    {"type": "PipeWire:Interface:Device", "info": {"props": {
        "device.api": "bluez5", "api.bluez5.address": "A0:B1:C2:D3:E4:F5", "api.bluez5.class": "0x5a020c"}}},
    {"type": "PipeWire:Interface:Node", "info": {"props": {
        "node.name": "bluez_input.A0_B1_C2_D3_E4_F5.0", "media.class": "Stream/Output/Audio"}}},
    {"type": "PipeWire:Interface:Node", "info": {"props": {
        "node.name": "bluez_output.A0_B1_C2_D3_E4_F5.1", "media.class": "Stream/Input/Audio"}}},
    {"type": "PipeWire:Interface:Node", "info": {"props": {
        "node.name": "bluez_input.60_55_56_BE_F3_C3.0", "media.class": "Audio/Source"}}},
]


def test_call_streams_are_found():
    addr = find_phone_address(PW_DUMP_CALL_STREAMS)
    assert addr == "A0:B1:C2:D3:E4:F5"
    assert find_bluez_call_nodes(PW_DUMP_CALL_STREAMS, addr) == (
        "bluez_input.A0_B1_C2_D3_E4_F5.0", "bluez_output.A0_B1_C2_D3_E4_F5.1")


def test_parse_links_and_mic_is_cut_from_the_call(monkeypatch):
    import keryx.transport.pipewire as pw
    text = """bluez_input.A0_B1_C2_D3_E4_F5.0:output_MONO
  |-> alsa_output.speaker:playback_FL
alsa_input.mic:capture_FL
  |-> bluez_output.A0_B1_C2_D3_E4_F5.1:input_MONO
bluez_output.A0_B1_C2_D3_E4_F5.1:input_MONO
  |<- alsa_input.mic:capture_FL
  |<- keryx.mic:capture_MONO
"""
    assert pw.parse_links(text) == [
        ("alsa_input.mic:capture_FL", "bluez_output.A0_B1_C2_D3_E4_F5.1:input_MONO"),
        ("bluez_input.A0_B1_C2_D3_E4_F5.0:output_MONO", "alsa_output.speaker:playback_FL"),
        ("keryx.mic:capture_MONO", "bluez_output.A0_B1_C2_D3_E4_F5.1:input_MONO"),
    ]
    ran = []
    SRC, SINK = "bluez_input.A0_B1_C2_D3_E4_F5.0", "bluez_output.A0_B1_C2_D3_E4_F5.1"

    async def fake_run(*argv, timeout=5.0):
        ran.append(argv)
        if argv[:2] == ("pw-link", "-l"):
            return 0, text
        if argv[:2] == ("pw-link", "-i"):
            return 0, f"{SINK}:input_FL\n{SINK}:input_FR\nkeryx.ear:input_MONO\n"
        if argv[:2] == ("pw-link", "-o"):
            return 0, f"keryx.mic:capture_MONO\n{SRC}:output_MONO\n"
        return 0, ""
    monkeypatch.setattr(pw, "_run", fake_run)
    problems = asyncio.run(pw.ensure_call_links(SRC, SINK, "keryx.capture", "keryx.playback"))
    assert problems == []
    for pair in ((SRC, "keryx.ear"), ("keryx.ear", "keryx.capture"),
                 ("keryx.playback", "keryx.mic"), ("keryx.mic", SINK)):
        assert ("pw-link", *pair) in ran
    assert ("pw-link", "keryx.mic:capture_MONO", f"{SINK}:input_FR") in ran  # mono fanned out to stereo
    unlinks = [a for a in ran if a[:2] == ("pw-link", "-d")]
    assert unlinks == [("pw-link", "-d", "alsa_input.mic:capture_FL", f"{SINK}:input_MONO")]


def test_pipewire_node_lookup_picks_the_phone_not_the_earbuds():
    addr = find_phone_address(PW_DUMP)
    assert addr == "AA:BB:CC:DD:EE:FF"
    assert find_bluez_call_nodes(PW_DUMP, addr) == ("bluez_input.AA:BB:CC:DD:EE:FF", "bluez_output.AA_BB_CC_DD_EE_FF.1")
    assert find_bluez_call_nodes(PW_DUMP, "") == (None, None)
