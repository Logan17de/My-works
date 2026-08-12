from __future__ import annotations

from pathlib import Path
from ..models import Signal
from ..serialization import dumps


class JsonLineSignalSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_signal(self, signal: Signal) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(dumps(signal) + "\n")
