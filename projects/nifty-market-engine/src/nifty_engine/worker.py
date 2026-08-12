from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time
from typing import Protocol

from .engine import SignalEngine
from .models import MarketSnapshot, Signal, SupportResistanceLevel
from .risk import RiskState

logger = logging.getLogger(__name__)


class SnapshotSource(Protocol):
    def next_snapshot(self) -> MarketSnapshot: ...


class LevelSource(Protocol):
    def current_levels(self) -> tuple[SupportResistanceLevel, ...]: ...


class SignalWriter(Protocol):
    def write_signal(self, signal: Signal) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    evaluation_interval_seconds: float = 2.0
    failure_backoff_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.evaluation_interval_seconds <= 0 or self.failure_backoff_seconds <= 0:
            raise ValueError("worker intervals must be positive")


class MarketWorker:
    """Long-running signal worker intentionally separate from Vercel request functions."""
    def __init__(
        self, *, engine: SignalEngine, snapshot_source: SnapshotSource,
        level_source: LevelSource, signal_writer: SignalWriter,
        risk_state: RiskState, config: WorkerConfig | None = None,
    ) -> None:
        self.engine = engine
        self.snapshot_source = snapshot_source
        self.level_source = level_source
        self.signal_writer = signal_writer
        self.risk_state = risk_state
        self.config = config or WorkerConfig()

    def run_once(self) -> Signal:
        snapshot = self.snapshot_source.next_snapshot()
        age = max(
            (datetime.now(timezone.utc) - snapshot.timestamp.astimezone(timezone.utc)).total_seconds(),
            0.0,
        )
        signal = self.engine.evaluate(
            snapshot, self.level_source.current_levels(), self.risk_state,
            data_age_seconds=age,
        )
        self.signal_writer.write_signal(signal)
        return signal

    def run_forever(self) -> None:
        while True:
            started = time.monotonic()
            try:
                self.run_once()
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("market worker iteration failed")
                time.sleep(self.config.failure_backoff_seconds)
                continue
            time.sleep(max(self.config.evaluation_interval_seconds - (time.monotonic() - started), 0.0))
