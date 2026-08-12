from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .engine import SignalEngine
from .models import EventKind, MarketSnapshot, Signal, SupportResistanceLevel
from .risk import RiskState


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    snapshot: MarketSnapshot
    levels: tuple[SupportResistanceLevel, ...]
    data_age_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    frames: int
    breakouts: int
    reversals: int
    uncertain: int
    no_level: int
    risk_approved: int


@dataclass(frozen=True, slots=True)
class ReplayResult:
    signals: tuple[Signal, ...]
    summary: ReplaySummary


class ReplayRunner:
    """Deterministic replay that never invents option fills or historical P&L."""
    def __init__(self, engine: SignalEngine) -> None:
        self.engine = engine

    def run(self, frames: Iterable[ReplayFrame], risk_state: RiskState) -> ReplayResult:
        signals: list[Signal] = []
        counts = {event: 0 for event in EventKind}
        risk_approved = 0
        for frame in frames:
            signal = self.engine.evaluate(
                frame.snapshot, frame.levels, risk_state,
                data_age_seconds=frame.data_age_seconds,
            )
            signals.append(signal)
            counts[signal.event] += 1
            if signal.risk.allowed:
                risk_approved += 1
        return ReplayResult(
            tuple(signals),
            ReplaySummary(
                len(signals), counts[EventKind.BREAKOUT], counts[EventKind.REVERSAL],
                counts[EventKind.UNCERTAIN], counts[EventKind.NO_LEVEL], risk_approved,
            ),
        )
