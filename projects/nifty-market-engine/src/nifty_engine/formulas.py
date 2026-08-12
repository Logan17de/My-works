from __future__ import annotations

import math
from typing import Iterable

from .math_utils import activity_from_rvol, bps_change, clamp, pct_change, safe_div, squash
from .models import CashMetrics, ConstituentTick, FuturesMetrics
from .params import StrategyParams


def constituent_metrics(
    ticks: Iterable[ConstituentTick], params: StrategyParams
) -> CashMetrics:
    rows = tuple(ticks)
    if not rows:
        raise ValueError("at least one constituent is required")

    weights = [max(row.index_weight, 0.0) for row in rows]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0 for _ in rows]
        total_weight = float(len(rows))
    weights = [w / total_weight for w in weights]

    weighted_signed_activity = 0.0
    weighted_activity = 0.0
    signed_acceleration = 0.0
    advancers = 0
    decliners = 0
    active_shares: list[float] = []

    for row, weight in zip(rows, weights, strict=True):
        dt = max(row.seconds_elapsed, 1e-6)
        delta_volume = max(row.cumulative_volume - row.previous_cumulative_volume, 0)
        volume_rate = delta_volume / dt
        baseline = max(row.baseline_volume_rate, 1e-6)
        rvol = volume_rate / baseline
        activity = activity_from_rvol(rvol, params.rvol_cap)

        move_bps = bps_change(row.price, row.previous_price)
        direction = squash(move_bps, params.direction_scale_bps)
        if move_bps > 0:
            advancers += 1
        elif move_bps < 0:
            decliners += 1

        previous_rate = max(row.previous_volume_rate, 1e-6)
        accel_ratio = (volume_rate - previous_rate) / previous_rate
        accel = math.tanh(accel_ratio)

        signed = weight * activity * direction
        weighted_signed_activity += signed
        weighted_activity += weight * activity
        signed_acceleration += weight * accel * direction
        active_shares.append(weight * activity)

    pressure = safe_div(weighted_signed_activity, weighted_activity)
    breadth = (advancers - decliners) / len(rows)

    # Participation uses normalized 1-HHI. It is high when activity is spread broadly and
    # low when only one or two names dominate the weighted activity.
    if weighted_activity <= 1e-12:
        participation = 0.0
    else:
        shares = [share / weighted_activity for share in active_shares if share > 0]
        hhi = sum(share * share for share in shares)
        n = len(rows)
        participation = clamp((1.0 - hhi) / (1.0 - 1.0 / n), 0.0, 1.0) if n > 1 else 1.0

    participation_factor = params.participation_floor + (
        1.0 - params.participation_floor
    ) * participation
    raw_score = params.cash_pressure_weight * pressure + params.breadth_weight * breadth
    score = clamp(raw_score * participation_factor)

    return CashMetrics(
        pressure=clamp(pressure),
        breadth=clamp(breadth),
        participation=participation,
        signed_volume_acceleration=clamp(signed_acceleration),
        score=score,
        active_count=len(rows),
        advancers=advancers,
        decliners=decliners,
    )


def futures_metrics(tick: FuturesTick, params: StrategyParams) -> FuturesMetrics:
    dt = max(tick.seconds_elapsed, 1e-6)
    price_move_bps = bps_change(tick.price, tick.previous_price)
    price_direction = squash(price_move_bps, params.futures_direction_scale_bps)

    volume_rate = max(tick.volume - tick.previous_volume, 0) / dt
    rvol = volume_rate / max(tick.baseline_volume_rate, 1e-6)
    volume_activity = activity_from_rvol(rvol, params.rvol_cap)

    oi_change_pct = pct_change(tick.open_interest, tick.previous_open_interest)
    # Positive OI change confirms the direction; falling OI reduces confidence in it.
    oi_confirmation = price_direction * squash(oi_change_pct, params.futures_oi_scale_pct)

    current_basis = tick.price - tick.spot_price
    previous_basis = tick.previous_price - tick.previous_spot_price
    basis_change_bps = safe_div(current_basis - previous_basis, tick.spot_price) * 10_000.0
    basis_change = squash(basis_change_bps, params.futures_basis_scale_bps)

    directional_with_activity = price_direction * (0.50 + 0.50 * volume_activity)
    score = clamp(
        params.futures_price_weight * directional_with_activity
        + params.futures_oi_weight * oi_confirmation
        + params.futures_basis_weight * basis_change
    )

    return FuturesMetrics(
        price_direction=clamp(price_direction),
        volume_activity=volume_activity,
        oi_confirmation=clamp(oi_confirmation),
        basis_change=clamp(basis_change),
        score=score,
    )


def combined_direction_score(
    cash: CashMetrics, futures: FuturesMetrics, params: StrategyParams
) -> float:
    total = params.combined_cash_weight + params.combined_futures_weight
    if total <= 0:
        raise ValueError("combined weights must be positive")
    score = (
        params.combined_cash_weight * cash.score
        + params.combined_futures_weight * futures.score
    ) / total
    return clamp(score)
