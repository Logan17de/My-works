from __future__ import annotations

from typing import Protocol
from ..models import Signal


class SignalSink(Protocol):
    def write_signal(self, signal: Signal) -> None: ...
