from __future__ import annotations

import math


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def bps_change(current: float, previous: float) -> float:
    if previous <= 0:
        return 0.0
    return (current / previous - 1.0) * 10_000.0


def pct_change(current: float, previous: float) -> float:
    if previous <= 0:
        return 0.0
    return (current / previous - 1.0) * 100.0


def squash(value: float, scale: float) -> float:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return math.tanh(value / scale)


def activity_from_rvol(rvol: float, cap: float) -> float:
    if cap <= 1.0:
        raise ValueError("rvol cap must be > 1")
    bounded = clamp(rvol, 0.0, cap)
    return clamp(math.log1p(bounded) / math.log1p(cap), 0.0, 1.0)
