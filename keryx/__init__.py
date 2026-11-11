"""Keryx: Hermes plans the call, a realtime voice model talks, your own phone dials.

    Hermes ──place_call──► [approval hook] ─► guardrails ─► brief compiler
                                                    │
                     voice engine ◄─► audio bridge ◄─► CallTransport ◄─► your phone (Bluetooth HFP)
                          │
                    ask_principal ─► principal resolver (Hermes model + read-only tools, else asks you)

register(ctx) wires the tool, the approval hook, /calls and `hermes keryx`.
"""

from __future__ import annotations

import logging
import platform

logger = logging.getLogger(__name__)

_runtime = None


def register(ctx) -> None:
    global _runtime
    from .cli import keryx_command, register_cli
    from .tools import PLACE_CALL_SCHEMA, Keryx

    if platform.system().lower() != "linux":
        logger.info("keryx: %s transports are not built yet (linux only for now)", platform.system())
        return
    _runtime = Keryx(ctx)
    ctx.register_tool(name="place_call", toolset="keryx", schema=PLACE_CALL_SCHEMA,
                      handler=_runtime.handle_place_call, emoji="📞",
                      description="Outbound phone call from the user's phone via a realtime voice model")
    ctx.register_hook("pre_tool_call", _runtime.pre_tool_call)
    ctx.register_hook("agent_loop_stopped", _runtime.on_agent_loop_stopped)
    ctx.register_command("calls", _runtime.calls_command, description="Recent Keryx calls; /calls <id> for one",
                         args_hint="[call id]")
    ctx.register_cli_command(name="keryx", help="Keryx phone calls (doctor, calls, simulate)",
                             setup_fn=register_cli, handler_fn=lambda args: keryx_command(args, _runtime),
                             description="Check the phone line, list calls, or run a fake call end to end.")
