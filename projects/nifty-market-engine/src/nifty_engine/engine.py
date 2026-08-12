from __future__ import annotations

from .formulas import combined_direction_score, constituent_metrics, futures_metrics
from .levels import LevelClassifier
from .models import (
    ContractSelection, Direction, EventKind, LevelKind, MarketSnapshot, OptionType,
    RiskDecision, Signal, SupportResistanceLevel,
)
from .options import select_option
from .params import StrategyParams
from .risk import RiskEngine, RiskState


class SignalEngine:
    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params = params or StrategyParams()
        self.level_classifier = LevelClassifier(self.params)
        self.risk_engine = RiskEngine(self.params)
        self._previous_combined_score = 0.0

    @staticmethod
    def _trade_direction(event: EventKind, level: SupportResistanceLevel) -> Direction:
        if event is EventKind.BREAKOUT:
            return Direction.BULLISH if level.kind is LevelKind.RESISTANCE else Direction.BEARISH
        if event is EventKind.REVERSAL:
            return Direction.BEARISH if level.kind is LevelKind.RESISTANCE else Direction.BULLISH
        return Direction.FLAT

    def evaluate(
        self, snapshot: MarketSnapshot, levels: tuple[SupportResistanceLevel, ...],
        risk_state: RiskState, *, data_age_seconds: float = 0.0,
    ) -> Signal:
        cash = constituent_metrics(snapshot.constituents, self.params)
        future = futures_metrics(snapshot.futures, self.params)
        combined = combined_direction_score(cash, future, self.params)
        nearest = self.level_classifier.nearest_level(snapshot.spot_price, levels)
        level_metrics = self.level_classifier.classify(
            now=snapshot.timestamp, spot=snapshot.spot_price,
            previous_spot=snapshot.previous_spot_price, level=nearest,
            combined_score=combined, previous_combined_score=self._previous_combined_score,
            participation=cash.participation,
            signed_volume_acceleration=cash.signed_volume_acceleration,
        )
        self._previous_combined_score = combined

        direction = Direction.FLAT
        contract = ContractSelection(None, 0.0, "no actionable level event")
        risk = RiskDecision(False, 0, 0.0, "no actionable level event")
        reasons: list[str] = []

        if nearest is None:
            reasons.append("spot is not close enough to an enabled support/resistance level")
        elif level_metrics.event not in {EventKind.BREAKOUT, EventKind.REVERSAL}:
            reasons.append("level interaction is still uncertain; no-trade state wins")
        else:
            direction = self._trade_direction(level_metrics.event, nearest)
            desired_option = OptionType.CE if direction is Direction.BULLISH else OptionType.PE
            contract = select_option(snapshot.options, desired_option, self.params)
            risk = self.risk_engine.evaluate(
                now=snapshot.timestamp, confidence=level_metrics.event_score,
                contract=contract, state=risk_state, data_age_seconds=data_age_seconds,
                constituent_count=cash.active_count,
            )
            reasons.extend([
                f"{level_metrics.event.value} at {nearest.kind.value} {nearest.price:.2f}",
                f"cash={cash.score:+.3f}, futures={future.score:+.3f}, combined={combined:+.3f}",
                contract.reason, risk.reason,
            ])

        confidence = (
            level_metrics.event_score
            if level_metrics.event in {EventKind.BREAKOUT, EventKind.REVERSAL} else 0.0
        )
        return Signal(
            snapshot.timestamp, level_metrics.event, direction, confidence, combined,
            cash, future, level_metrics, contract, risk, tuple(reasons)
        )
