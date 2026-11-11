"""bluetooth-linux: the laptop is a Bluetooth hands-free unit (car kit) for the user's phone.

Dialing and call state use the oFono D-Bus API (org.ofono.VoiceCallManager / VoiceCall), which both
real oFono (system bus, ``org.ofono``) and PipeWire's built-in telephony (user bus,
``org.pipewire.Telephony``, PipeWire >= 1.4 with WirePlumber ``bluez5.hfphsp-backend = native`` and the
``hfp_hf`` role) provide. We talk to it with ``busctl --json`` so there is no D-Bus Python dependency.
Call audio appears as normal PipeWire nodes for the phone, read/written with pw-record / pw-play.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import time
from typing import Any, Callable, Optional

from .base import ANSWERED, BUSY, FAILED, NO_ANSWER, CallTransport, TransportError
from .pipewire import PwStream, VirtualNodes, ensure_call_links, find_bluez_call_nodes, find_phone_address, pw_dump

SERVICES = {"user": "org.pipewire.Telephony", "system": "org.ofono"}
MANAGER_PATHS = ("/", "/org/pipewire/Telephony")
VCM = "org.ofono.VoiceCallManager"
VOICECALL = "org.ofono.VoiceCall"
AG_TRANSPORT = "org.pipewire.Telephony.AudioGatewayTransport1"


def _unwrap(v: Any) -> Any:
    """busctl --json variants arrive as {"type": "s", "data": ...}."""
    if isinstance(v, dict) and set(v) >= {"type", "data"}:
        return _unwrap(v["data"])
    return v


def _pairs(data: Any) -> list[tuple[str, dict[str, Any]]]:
    """Normalize a(oa{sv}) (oFono) or a{oa{sv}} (PipeWire) into [(path, props)]."""
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [tuple(x) for x in data if isinstance(x, (list, tuple)) and len(x) == 2]
    else:
        items = []
    for path, props in items:
        out.append((str(path), {k: _unwrap(v) for k, v in (props or {}).items()}))
    return out


class BusctlClient:
    def __init__(self, bus: str = "user") -> None:
        if bus not in SERVICES:
            raise TransportError(f"telephony_bus must be 'user' or 'system', got {bus!r}")
        self.bus = bus
        self.service = SERVICES[bus]

    async def call(self, path: str, iface: str, method: str, sig: str = "", *args: str,
                   timeout: float = 10.0) -> Any:
        if not shutil.which("busctl"):
            raise TransportError("busctl not found (systemd)")
        argv = ["busctl", f"--{self.bus}", "--json=short", f"--timeout={int(timeout)}", "call",
                self.service, path, iface, method]
        if sig:
            argv += [sig, *args]
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), timeout + 2)
        if proc.returncode != 0:
            raise TransportError(f"{iface}.{method} failed: {(err or b'').decode().strip() or proc.returncode}")
        if not out.strip():
            return None
        try:
            payload = json.loads(out)
        except ValueError:
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        return data[0] if isinstance(data, list) and len(data) == 1 else data


class BluetoothLinuxTransport(CallTransport):
    name = "bluetooth-linux"
    sample_rate = 24000  # PipeWire resamples from the SCO link (8/16 kHz) for us
    # pw-play pulls ~21.3 ms per graph cycle while we pace 20 ms frames; without a head start it
    # periodically finds the pipe short and drops a 20 ms slice (~1 every 0.6 s, measured).
    playback_lead_ms = 80

    def __init__(self, device: str = "", bus: str = "user", notify: Optional[Callable[[str], None]] = None,
                 client: Optional[BusctlClient] = None, poll_s: float = 0.25) -> None:
        super().__init__(notify)
        self.device = device.strip()
        self.client = client or BusctlClient(bus)
        self.poll_s = poll_s
        self.modem: Optional[str] = None
        self.call_path: Optional[str] = None
        self._capture: Optional[PwStream] = None
        self._playback: Optional[PwStream] = None
        self._virtual: Optional[VirtualNodes] = None
        self._ended = False
        self._dialed = False
        self._last_state = ""

    # --- discovery -------------------------------------------------------------------------
    async def _modems(self) -> list[tuple[str, dict[str, Any]]]:
        last: Optional[Exception] = None
        for path in MANAGER_PATHS:
            try:
                return _pairs(await self.client.call(path, "org.ofono.Manager", "GetModems"))
            except TransportError as e:
                last = e
        raise TransportError(f"no oFono-compatible telephony service on the {self.client.bus} bus "
                             f"({self.client.service}): {last}")

    async def _pick_modem(self) -> str:
        modems = await self._modems()
        candidates = []
        for path, props in modems:
            ifaces = props.get("Interfaces") or []
            if ifaces and VCM not in ifaces:
                continue
            if props.get("Online") is False or props.get("Powered") is False:
                continue
            candidates.append((path, props))
        if self.device:
            want = self.device.upper()
            for path, props in candidates:
                hay = " ".join([path, str(props.get("Serial", "")), str(props.get("Name", ""))]).upper()
                if want in hay or want.replace(":", "_") in hay:
                    return path
            if len(candidates) == 1:
                return candidates[0][0]
            raise TransportError(f"no connected phone matches bluetooth_device={self.device}")
        if not candidates:
            raise TransportError("no phone connected as a Bluetooth audio gateway. Pair your phone, connect it, "
                                 "and allow 'phone calls' for this computer in the phone's Bluetooth settings")
        if len(candidates) > 1:
            raise TransportError("several phones are connected; set keryx bluetooth_device to one address")
        return candidates[0][0]

    async def preflight(self) -> list[str]:
        problems = []
        for exe in ("busctl", "pw-record", "pw-play", "pw-dump"):
            if not shutil.which(exe):
                problems.append(f"{exe} not found")
        if problems:
            return problems
        try:
            self.modem = await self._pick_modem()
        except TransportError as e:
            problems.append(str(e))
            return problems
        try:
            calls = _pairs(await self.client.call(self.modem, VCM, "GetCalls"))
        except TransportError as e:
            problems.append(f"cannot read call state: {e}")
            return problems
        if any(str(p.get("State")) not in ("disconnected", "") for _, p in calls):
            problems.append("your phone is already on a call")
        return problems

    # --- call control ----------------------------------------------------------------------
    async def _call_paths(self) -> list[str]:
        assert self.modem
        return [path for path, _ in _pairs(await self.client.call(self.modem, VCM, "GetCalls"))]

    async def dial(self, number: str) -> None:
        if self.modem is None:
            self.modem = await self._pick_modem()
        before = set(await self._call_paths())
        self._dialed = True  # from here on, hangup() must end whatever the phone placed
        self._ended = False
        if self.client.bus == "user":
            # PipeWire: Dial(s number) -> nothing; the call shows up in GetCalls / CallAdded.
            path = await self.client.call(self.modem, VCM, "Dial", "s", number, timeout=20)
        else:
            # oFono: Dial(s number, s hide_callerid) -> o call ("" = network default)
            path = await self.client.call(self.modem, VCM, "Dial", "ss", number, "", timeout=20)
        if path:
            self.call_path = str(_unwrap(path))
            return
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            new = [p for p in await self._call_paths() if p not in before]
            if new:
                self.call_path = new[0]
                return
            await asyncio.sleep(self.poll_s)
        raise TransportError("the phone accepted Dial but no call appeared")

    async def _state(self) -> str:
        """Current state of our call ('' once it is gone)."""
        assert self.modem
        calls = _pairs(await self.client.call(self.modem, VCM, "GetCalls"))
        for path, props in calls:
            if path == self.call_path:
                return str(props.get("State") or "")
        return ""

    async def wait_answered(self, timeout: float) -> str:
        start = time.monotonic()
        deadline = start + timeout
        saw_alerting = seen = False
        while time.monotonic() < deadline:
            try:
                state = await self._state()
            except TransportError:
                state = self._last_state or "dialing"
            self._last_state = state
            seen = seen or bool(state)
            if state == "active":
                return ANSWERED
            if state == "alerting":
                saw_alerting = True
            # The call object can lag Dial() briefly; only treat "gone" as ended once seen (or after 5 s).
            if state == "disconnected" or (state == "" and (seen or time.monotonic() - start > 5)):
                self._ended = True
                if not seen:
                    return FAILED
                return NO_ANSWER if saw_alerting else BUSY
            await asyncio.sleep(self.poll_s)
        await self.hangup()
        return NO_ANSWER

    async def _watch(self) -> None:
        while not self._ended:
            await asyncio.sleep(0.5)
            try:
                state = await self._state()
            except TransportError:
                continue
            if state in ("", "disconnected"):
                self._ended = True
                return
            await self._relink()

    async def _relink(self) -> None:
        """The SCO link can renegotiate mid-call and PipeWire then recreates the call streams;
        find them again and re-wire (idempotent)."""
        if not (self._capture and self._playback):
            return
        with contextlib.suppress(Exception):
            objects = await pw_dump()
            source, sink = find_bluez_call_nodes(objects, self.device or find_phone_address(objects))
            if source and sink:
                self._nodes = (source, sink)
                await ensure_call_links(source, sink, self._capture.node_name, self._playback.node_name)

    async def start_audio(self) -> None:
        address = self.device
        source = sink = None
        start = time.monotonic()
        asked = False
        # The phone renegotiates the SCO link right after answer (observed: ~1.5 s with no call
        # streams), then PipeWire recreates bluez_input/bluez_output for the call.
        while time.monotonic() - start < 10:
            objects = await pw_dump()
            address = address or find_phone_address(objects)
            source, sink = find_bluez_call_nodes(objects, address)
            if source and sink:
                break
            if not asked and time.monotonic() - start > 4 and self.client.bus == "user" and self.modem:
                # Some phones wait for the hands-free side to ask for call audio.
                asked = True
                with contextlib.suppress(Exception):
                    await self.client.call(self.modem, AG_TRANSPORT, "Activate")
            await asyncio.sleep(0.2)
        if not (source and sink):
            raise TransportError("call connected but no Bluetooth call-audio nodes appeared in PipeWire "
                                 f"(phone address {address or 'unknown'}). Check WirePlumber's bluez5 roles "
                                 "include hfp_hf, or set keryx bluetooth_device")
        self._nodes = (source, sink)
        # Our streams don't auto-connect; we link them to the phone's call streams ourselves,
        # through two virtual devices so the links are clocked (see VirtualNodes).
        self._virtual = VirtualNodes(self.sample_rate)
        await self._virtual.start()
        self._capture = PwStream("capture", self.sample_rate, autoconnect=False)
        # A deeper playback buffer rides out scheduling jitter between our pacer and the SCO clock.
        self._playback = PwStream("playback", self.sample_rate, latency="60ms", autoconnect=False)
        await self._capture.start()
        await self._playback.start()
        problems: list[str] = []
        for _ in range(25):  # our nodes take a moment to appear
            await asyncio.sleep(0.1)
            problems = await ensure_call_links(source, sink, self._capture.node_name, self._playback.node_name)
            if not problems:
                break
        if problems:
            raise TransportError("; ".join(problems))
        self._watcher = asyncio.create_task(self._watch())

    async def read_frame(self) -> Optional[bytes]:
        if self._capture is None:
            return None
        return await self._capture.read(self.frame_bytes)

    async def write_frame(self, pcm: bytes) -> None:
        if self._playback is not None:
            await self._playback.write(pcm)

    async def send_dtmf(self, digits: str) -> None:
        if not self.modem:
            raise TransportError("no active modem")
        await self.client.call(self.modem, VCM, "SendTones", "s", digits)

    def remote_ended(self) -> bool:
        return self._ended

    async def hangup(self) -> None:
        done = False
        if self.call_path:
            with contextlib.suppress(Exception):
                await self.client.call(self.call_path, VOICECALL, "Hangup")
                done = True
        if not done and self._dialed and self.modem:
            # We dialed but lost (or never got) the call object: end every call on the line rather
            # than leave the callee connected to silence.
            with contextlib.suppress(Exception):
                await self.client.call(self.modem, VCM, "HangupAll")
        self._dialed = False
        self._ended = True

    async def close(self) -> None:
        self._ended = True
        watcher = getattr(self, "_watcher", None)
        if watcher:
            watcher.cancel()
        for s in (self._capture, self._playback, self._virtual):
            if s:
                with contextlib.suppress(Exception):
                    await s.stop()
        self._capture = self._playback = self._virtual = None
