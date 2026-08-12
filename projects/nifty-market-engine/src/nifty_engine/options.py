from __future__ import annotations

import math

from .math_utils import clamp, safe_div
from .models import ContractSelection, OptionContract, OptionType
from .params import StrategyParams


def _minmax(value: float, values: list[float], invert: bool = False) -> float:
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    score = 0.5 if math.isclose(lo, hi) else (value - lo) / (hi - lo)
    return 1.0 - score if invert else score


def select_option(
    contracts: tuple[OptionContract, ...],
    desired_type: OptionType,
    params: StrategyParams,
) -> ContractSelection:
    candidates = [
        contract for contract in contracts
        if contract.option_type is desired_type
        and params.min_abs_delta <= abs(contract.greeks.delta) <= params.max_abs_delta
        and contract.ltp > 0 and contract.lot_size > 0
    ]
    if not candidates:
        return ContractSelection(None, 0.0, "no contract passed delta/price/lot filters")

    volumes = [math.log1p(max(c.volume, 0)) for c in candidates]
    ois = [math.log1p(max(c.open_interest, 0)) for c in candidates]
    theta_ratios = [safe_div(abs(c.greeks.theta), c.ltp) for c in candidates]
    ivs = [c.greeks.iv for c in candidates]
    gammas = [max(c.greeks.gamma, 0.0) for c in candidates]

    best = None
    best_score = -1.0
    rejection_reason = ""
    for contract in candidates:
        if contract.bid_price is not None and contract.ask_price is not None:
            mid = (contract.bid_price + contract.ask_price) / 2.0
            spread_pct = safe_div(contract.ask_price - contract.bid_price, mid)
            if spread_pct > params.max_spread_pct:
                rejection_reason = "candidate spread exceeded configured maximum"
                continue

        delta_span = max(params.max_abs_delta - params.min_abs_delta, 1e-6)
        delta_fit = clamp(
            1.0 - abs(abs(contract.greeks.delta) - params.target_abs_delta) / delta_span,
            0.0, 1.0,
        )
        volume_score = _minmax(math.log1p(max(contract.volume, 0)), volumes)
        oi_score = _minmax(math.log1p(max(contract.open_interest, 0)), ois)
        liquidity = (
            params.option_volume_liquidity_weight * volume_score
            + params.option_oi_liquidity_weight * oi_score
        )
        theta_score = _minmax(safe_div(abs(contract.greeks.theta), contract.ltp), theta_ratios, True)
        iv_score = _minmax(contract.greeks.iv, ivs, True)
        gamma_score = _minmax(max(contract.greeks.gamma, 0.0), gammas)
        score = clamp(
            params.option_delta_weight * delta_fit
            + params.option_liquidity_weight * liquidity
            + params.option_theta_weight * theta_score
            + params.option_iv_weight * iv_score
            + params.option_gamma_weight * gamma_score,
            0.0, 1.0,
        )
        if score > best_score:
            best = contract
            best_score = score

    if best is None:
        return ContractSelection(None, 0.0, rejection_reason or "no liquid contract")
    return ContractSelection(best, best_score, "best delta/liquidity/theta/IV/gamma fit")
