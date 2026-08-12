from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Iterable


@dataclass(frozen=True, slots=True)
class VolumeRateSample:
    symbol: str
    session_date: date
    minute_bucket: int
    volume_rate: float


def build_intraday_volume_baselines(
    samples: Iterable[VolumeRateSample], *, min_sessions: int = 5,
) -> dict[tuple[str, int], float]:
    """Build leakage-safe time-of-day volume-rate baselines from prior sessions."""
    if min_sessions <= 0:
        raise ValueError("min_sessions must be positive")
    grouped: dict[tuple[str, int], dict[date, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        if sample.volume_rate < 0:
            raise ValueError("volume_rate cannot be negative")
        if not 0 <= sample.minute_bucket < 24 * 60:
            raise ValueError("minute_bucket must be in [0, 1439]")
        grouped[(sample.symbol, sample.minute_bucket)][sample.session_date].append(sample.volume_rate)
    output: dict[tuple[str, int], float] = {}
    for key, sessions in grouped.items():
        if len(sessions) < min_sessions:
            continue
        session_rates = [median(values) for values in sessions.values()]
        output[key] = float(median(session_rates))
    return output
