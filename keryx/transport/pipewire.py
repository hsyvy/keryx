"""PipeWire audio streams via pw-record / pw-play subprocesses (raw s16le mono), plus node lookup."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from typing import Any, Optional


class PwStream:
    """One raw PCM stream: capture (pw-record -> stdout) or playback (stdin -> pw-play)."""

    def __init__(self, kind: str, rate: int, target: Optional[str] = None, latency: str = "20ms",
                 role: str = "Communication", autoconnect: bool = True) -> None:
        assert kind in ("capture", "playback")
        self.kind, self.rate, self.target, self.latency, self.role = kind, rate, target, latency, role
        self.autoconnect = autoconnect
        self.node_name = "keryx." + kind
        self.proc: Optional[asyncio.subprocess.Process] = None

    async def start(self) -> None:
        exe = "pw-record" if self.kind == "capture" else "pw-play"
        if not shutil.which(exe):
            raise RuntimeError(f"{exe} not found (install pipewire-bin / pipewire-utils)")
        argv = [exe, "--raw", "--rate", str(self.rate), "--channels", "1", "--format", "s16",
                "--latency", self.latency, "--media-role", self.role, "-P",
                "{ node.name = " + self.node_name + " node.description = Keryx"
                + ("" if self.autoconnect else " node.autoconnect = false node.dont-reconnect = true") + " }"]
        if self.target:
            argv += ["--target", self.target]
        argv.append("-")
        if self.kind == "capture":
            self.proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL)
        else:
            self.proc = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL)

    async def read(self, nbytes: int) -> Optional[bytes]:
        assert self.proc and self.proc.stdout
        try:
            return await self.proc.stdout.readexactly(nbytes)
        except (asyncio.IncompleteReadError, ConnectionError):
            return None

    async def write(self, data: bytes) -> None:
        assert self.proc and self.proc.stdin
        if self.proc.stdin.is_closing():
            return
        self.proc.stdin.write(data)
        with contextlib.suppress(ConnectionError):
            await self.proc.stdin.drain()

    async def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        with contextlib.suppress(Exception):
            if p.stdin:
                p.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            p.terminate()
        try:
            await asyncio.wait_for(p.wait(), 2)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                p.kill()


async def pw_dump() -> list[dict[str, Any]]:
    if not shutil.which("pw-dump"):
        return []
    proc = await asyncio.create_subprocess_exec("pw-dump", stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    try:
        data = json.loads(out or b"[]")
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def _props(obj: dict[str, Any]) -> dict[str, Any]:
    return ((obj.get("info") or {}).get("props") or {}) if isinstance(obj, dict) else {}


SOURCE_CLASSES = ("Stream/Output/Audio", "Audio/Source")  # phone -> laptop (callee's voice)
SINK_CLASSES = ("Stream/Input/Audio", "Audio/Sink")       # laptop -> phone (what the callee hears)


def find_bluez_call_nodes(objects: list[dict[str, Any]], address: str) -> tuple[Optional[str], Optional[str]]:
    """(source_node_name, sink_node_name) carrying call audio for the phone at *address*, or (None, None).

    With the laptop as hands-free unit (phone = audio gateway), PipeWire 1.6 exposes the call as two
    *streams*: ``bluez_input.<MAC>.N`` (Stream/Output/Audio, the callee's voice) and
    ``bluez_output.<MAC>.N`` (Stream/Input/Audio, what goes to the phone). WirePlumber links them to the
    default speaker and mic. Headset-style setups expose Audio/Source and Audio/Sink nodes instead. We
    accept both, preferring the stream form, and link to them explicitly with pw-link."""
    addr = address.upper().replace("_", ":")
    if not addr:
        return None, None
    forms = (addr, addr.replace(":", "_"))
    best: dict[str, tuple[int, str]] = {}
    for obj in objects:
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        p = _props(obj)
        name = str(p.get("node.name") or "")
        node_addr = str(p.get("api.bluez5.address") or "").upper()
        if not name.startswith("bluez_") or not (node_addr == addr or any(f in name.upper() for f in forms)):
            continue
        cls = str(p.get("media.class") or "")
        for role, classes in (("source", SOURCE_CLASSES), ("sink", SINK_CLASSES)):
            if cls in classes:
                rank = classes.index(cls)
                if role not in best or rank < best[role][0]:
                    best[role] = (rank, name)
    return (best["source"][1] if "source" in best else None, best["sink"][1] if "sink" in best else None)


async def _run(*argv: str, timeout: float = 5.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return -1, "timeout"
    return proc.returncode or 0, (out or b"").decode(errors="replace")


def parse_links(text: str) -> list[tuple[str, str]]:
    """`pw-link -l` output -> [(output_port, input_port)]."""
    links: set[tuple[str, str]] = set()
    current = ""
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            current = line.strip()
            continue
        item = line.strip()
        if item.startswith("|->"):
            links.add((current, item[3:].strip()))
        elif item.startswith("|<-"):
            links.add((item[3:].strip(), current))
    return sorted(links)


async def pw_links() -> list[tuple[str, str]]:
    code, out = await _run("pw-link", "-l")
    return parse_links(out) if code == 0 else []


async def pw_link(output_node: str, input_node: str) -> bool:
    """Link all matching ports of two nodes. Already linked counts as success."""
    code, out = await _run("pw-link", output_node, input_node)
    return code == 0 or "exist" in out.lower()


async def pw_unlink(output_port: str, input_port: str) -> None:
    await _run("pw-link", "-d", output_port, input_port)


def _node_of(port: str) -> str:
    return port.rsplit(":", 1)[0]


MIC_NODE = "keryx.mic"  # virtual source: our playback -> here -> phone uplink stream
EAR_NODE = "keryx.ear"  # virtual source: phone downlink stream -> here -> our capture


def _die_with_parent() -> None:  # pragma: no cover - runs in the child
    import ctypes
    import signal
    with contextlib.suppress(Exception):
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


class VirtualNodes:
    """Two null-audio virtual sources that sit between our pw-cat streams and the phone's call streams.

    Linking a pw-cat stream straight to the bluez call stream (stream -> stream) leaves no device
    clocking that link, and PipeWire then drops 10-20% of the audio. Going through a device node
    (stream -> device -> stream, which is how WirePlumber normally wires a mic) is clean. The nodes
    belong to one pw-cli process and vanish when it exits (or when we die)."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.proc: Optional[asyncio.subprocess.Process] = None

    def _spec(self, name: str, desc: str) -> str:
        return ("create-node adapter { factory.name = support.null-audio-sink node.name = " + name
                + ' node.description = "' + desc + '" media.class = Audio/Source/Virtual'
                + " audio.position = [ MONO ] audio.rate = " + str(self.rate)
                + " node.autoconnect = false priority.session = 0 priority.driver = 0 }\n")

    async def start(self) -> None:
        if not shutil.which("pw-cli"):
            raise RuntimeError("pw-cli not found")
        self.proc = await asyncio.create_subprocess_exec(
            "pw-cli", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL, preexec_fn=_die_with_parent)
        assert self.proc.stdin
        self.proc.stdin.write((self._spec(MIC_NODE, "Keryx call voice")
                               + self._spec(EAR_NODE, "Keryx call listener")).encode())
        await self.proc.stdin.drain()

    async def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        with contextlib.suppress(ProcessLookupError):
            p.terminate()
        try:
            await asyncio.wait_for(p.wait(), 3)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                p.kill()


async def _ports(direction: str, node: str) -> list[str]:
    code, out = await _run("pw-link", direction)
    return [l.strip() for l in out.splitlines() if _node_of(l.strip()) == node] if code == 0 else []


async def _fan_out(out_node: str, in_node: str, links: list[tuple[str, str]]) -> None:
    """pw-link pairs ports by channel, so a MONO output only reaches FL of a stereo input:
    feed a single output port into every input channel."""
    outs = await _ports("-o", out_node)
    if len(outs) != 1:
        return
    fed = {i for o, i in links if o == outs[0]}
    for port in await _ports("-i", in_node):
        if port not in fed:
            await _run("pw-link", outs[0], port)


async def ensure_call_links(source: str, sink: str, capture_node: str, playback_node: str) -> list[str]:
    """Wire the call (idempotent):
        phone downlink stream -> keryx.ear -> our capture
        our playback -> keryx.mic -> phone uplink stream
    and nothing else into the phone uplink (WirePlumber connects the laptop mic to it by default).
    Returns problems, if any."""
    problems = []
    for out_node, in_node in ((source, EAR_NODE), (EAR_NODE, capture_node),
                              (playback_node, MIC_NODE), (MIC_NODE, sink)):
        if not await pw_link(out_node, in_node):
            problems.append(f"could not link {out_node} -> {in_node}")
    links = await pw_links()
    for out_port, in_port in links:
        if _node_of(in_port) == sink and _node_of(out_port) != MIC_NODE:
            await pw_unlink(out_port, in_port)
    await _fan_out(MIC_NODE, sink, links)
    await _fan_out(source, EAR_NODE, links)
    return problems


def _is_phone_class(cod: Any) -> bool:
    """Bluetooth Class of Device: major device class 0x02 is Phone."""
    try:
        value = int(str(cod), 16) if isinstance(cod, str) else int(cod)
    except (TypeError, ValueError):
        return False
    return (value >> 8) & 0x1F == 0x02


def find_phone_address(objects: list[dict[str, Any]]) -> str:
    """Address of the single connected bluez device that is a phone, else ''."""
    found = []
    for obj in objects:
        p = _props(obj)
        if obj.get("type") != "PipeWire:Interface:Device" or str(p.get("device.api") or "bluez5") != "bluez5":
            continue
        if not p.get("api.bluez5.address"):
            continue
        icon = str(p.get("device.icon-name") or p.get("api.bluez5.icon") or "") + str(p.get("device.form-factor") or "")
        if _is_phone_class(p.get("api.bluez5.class")) or "phone" in icon and "headphone" not in icon:
            found.append(str(p["api.bluez5.address"]))
    return found[0] if len(found) == 1 else ""
