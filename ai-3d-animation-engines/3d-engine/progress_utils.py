#!/usr/bin/env python3
"""Tiny progress/heartbeat helper shared by Colab engine scripts."""
from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field


def _fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class Progress:
    name: str
    total: int
    started: float = field(default_factory=time.monotonic)
    current: int = 0

    def _prefix(self) -> str:
        return f"[{self.name}][{self.current}/{self.total}][{_fmt(time.monotonic()-self.started)}]"

    def step(self, message: str) -> None:
        self.current = min(self.current + 1, self.total)
        print(f"\n{self._prefix()} ▶ {message}", flush=True)

    def info(self, message: str) -> None:
        print(f"{self._prefix()}   {message}", flush=True)

    def ok(self, message: str) -> None:
        print(f"{self._prefix()} ✓ {message}", flush=True)

    def warn(self, message: str) -> None:
        print(f"{self._prefix()} ⚠ {message}", flush=True)

    def done(self, message: str = "Complete") -> None:
        self.current = self.total
        print(f"\n{self._prefix()} ✅ {message}", flush=True)

    @contextlib.contextmanager
    def heartbeat(self, label: str, every: float = 30.0):
        """Print a quiet heartbeat while a long blocking operation runs."""
        stop = threading.Event()
        started = time.monotonic()

        def worker() -> None:
            while not stop.wait(every):
                elapsed = _fmt(time.monotonic() - started)
                print(f"{self._prefix()}   … {label} still running ({elapsed} in this stage)", flush=True)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1.0)
