"""`hermes keryx ...`: doctor (is the phone line ready?), calls, show, simulate (fake call)."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess


def register_cli(parser) -> None:
    sub = parser.add_subparsers(dest="keryx_cmd")
    sub.add_parser("doctor", help="Check Bluetooth/PipeWire telephony, voice key and settings")
    calls = sub.add_parser("calls", help="List recent calls")
    calls.add_argument("--limit", type=int, default=10)
    show = sub.add_parser("show", help="Show one call with transcript")
    show.add_argument("call_id")
    sim = sub.add_parser("simulate", help="Run a fake call end to end (fake phone + scripted voice model)")
    sim.add_argument("--goal", default="Book a haircut for Saturday 11am, max ₹600")


def keryx_command(args, runtime=None) -> int:
    from .tools import Keryx
    rt = runtime or Keryx(None)
    cmd = getattr(args, "keryx_cmd", None) or "doctor"
    if cmd == "doctor":
        print(asyncio.run(doctor(rt)))
    elif cmd == "calls":
        print(rt.calls_command(""))
    elif cmd == "show":
        print(rt.calls_command(args.call_id))
    elif cmd == "simulate":
        from .testing import simulate_call
        print(json.dumps(asyncio.run(simulate_call(goal=args.goal)), indent=2, ensure_ascii=False))
    return 0


async def doctor(rt) -> str:
    s = rt.settings()
    model = (f"{s.live_model} + backend {s.live_backend_model}" if s.voice_engine == "openai-live"
             else s.realtime_model)
    out = [f"transport: {s.transport}   voice: {s.voice_engine} ({model}, {s.voice})"]
    ok = lambda b: "✅" if b else "❌"  # noqa: E731
    out.append(f"{ok(bool(s.openai_api_key))} OPENAI_API_KEY {'set' if s.openai_api_key else 'missing'}")
    out.append(f"{ok(bool(s.principal_name))} principal_name: {s.principal_name or '(unset — callee hears “my principal”)'}")
    out.append(f"   disclosure: \"{s.disclosure}\"")
    out.append(f"   limits: {s.max_call_seconds}s/call, {s.max_calls_per_hour}/hour, quiet hours {s.quiet_hours or 'off'}")
    for exe in ("busctl", "pw-record", "pw-play", "pw-dump", "wpctl"):
        out.append(f"{ok(bool(shutil.which(exe)))} {exe}")
    if s.transport == "bluetooth-linux":
        try:
            names = subprocess.run(["busctl", f"--{s.telephony_bus}", "list", "--no-pager"], capture_output=True,
                                   text=True, timeout=5).stdout
        except Exception:
            names = ""
        svc = "org.pipewire.Telephony" if s.telephony_bus == "user" else "org.ofono"
        out.append(f"{ok(svc in names)} {svc} on the {s.telephony_bus} bus")
        if svc not in names and s.telephony_bus == "user":
            out.append("   → PipeWire ≥ 1.4 with WirePlumber: enable `bluez5.hfphsp-backend = native`, add `hfp_hf` to "
                       "`bluez5.roles`, and keep the telephony service on (see README).")
        from .transport import make_transport
        t = make_transport(s.transport, settings=s)
        problems = await t.preflight()
        out.append(f"{ok(not problems)} phone line: " + ("ready (" + str(getattr(t, 'modem', '')) + ")"
                                                      if not problems else "; ".join(problems)))
    return "\n".join(out)
