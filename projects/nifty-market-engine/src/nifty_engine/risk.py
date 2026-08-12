from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from .models import ContractSelection, RiskDecision
from .params import StrategyParams


@dataclass(slots=True)
class RiskState:
    account_equity: float
    realized_pnl_today: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    last_trade_at: datetime | None = None
    open_position: bool = False


class RiskEngine:
    def __init__(self, params: StrategyParams) -> None:
        self.params = params

    def evaluate(
        self, *, now: datetime, confidence: float, contract: ContractSelection,
        state: RiskState, data_age_seconds: float = 0.0,
        constituent_count: int | None = None,
    ) -> RiskDecision:
        if state.account_equity <= 0:
            return RiskDecision(False, 0, 0.0, "account equity must be positive")
        if confidence < self.params.min_signal_confidence:
            return RiskDecision(False, 0, 0.0, "signal confidence below minimum")
        if constituent_count is not None and constituent_count < self.params.min_constituents:
            return RiskDecision(False, 0, 0.0, "insufficient NIFTY constituent coverage")
        if data_age_seconds > self.params.max_data_age_seconds:
            return RiskDecision(False, 0, 0.0, "market data is stale")
        if state.open_position:
            return RiskDecision(False, 0, 0.0, "one-position rule: an option position is already open")
        if state.trades_today >= self.params.max_trades_per_day:
            return RiskDecision(False, 0, 0.0, "daily trade-count limit reached")
        if state.consecutive_losses >= self.params.max_consecutive_losses:
            return RiskDecision(False, 0, 0.0, "consecutive-loss circuit breaker reached")
        if state.realized_pnl_today <= -state.account_equity * self.params.daily_loss_limit_pct:
            return RiskDecision(False, 0, 0.0, "daily loss circuit breaker reached")
        if state.last_trade_at is not None:
            if (now - state.last_trade_at).total_seconds() < self.params.cooldown_seconds:
                return RiskDecision(False, 0, 0.0, "cooldown active")
        if contract.contract is None:
            return RiskDecision(False, 0, 0.0, "no eligible option contract")

        max_risk = state.account_equity * self.params.risk_per_trade_pct
        cost_per_lot = contract.contract.ltp * contract.contract.lot_size
        if cost_per_lot <= 0:
            return RiskDecision(False, 0, max_risk, "invalid option premium or lot size")
        lots = math.floor(max_risk / cost_per_lot)
        if lots < 1:
            return RiskDecision(False, 0, max_risk, "risk budget cannot fund one option lot")
        return RiskDecision(True, lots * contract.contract.lot_size, max_risk, "risk checks passed")
