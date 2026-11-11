"""Phone-line transports: bluetooth-linux (your phone over Bluetooth HFP) and speakerphone.
Planned: phonelink-windows (WinRT PhoneLine + VB-Cable) and continuity-mac (tel:// + BlackHole)."""

from __future__ import annotations

from typing import Callable, Optional

from .base import ANSWERED, BUSY, FAILED, NO_ANSWER, CallTransport, TransportError

__all__ = ["ANSWERED", "BUSY", "FAILED", "NO_ANSWER", "CallTransport", "TransportError", "make_transport"]


def make_transport(name: str, *, settings=None, notify: Optional[Callable[[str], None]] = None) -> CallTransport:
    if name == "bluetooth-linux":
        from .linux_bluetooth import BluetoothLinuxTransport
        return BluetoothLinuxTransport(device=getattr(settings, "bluetooth_device", ""),
                                       bus=getattr(settings, "telephony_bus", "user"), notify=notify)
    if name == "speakerphone":
        from .speakerphone import SpeakerphoneTransport
        return SpeakerphoneTransport(notify=notify)
    if name in ("phonelink-windows", "continuity-mac"):
        raise TransportError(f"transport {name!r} is planned but not built yet; use bluetooth-linux or speakerphone")
    raise TransportError(f"unknown transport {name!r}")
