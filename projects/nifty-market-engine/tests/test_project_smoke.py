from datetime import date, datetime, timedelta, timezone

import pytest

from nifty_engine.baselines import VolumeRateSample, build_intraday_volume_baselines
from nifty_engine.brokers.groww_data import SlidingWindowRateLimiter
from nifty_engine.engine import SignalEngine
from nifty_engine.formulas import constituent_metrics
from nifty_engine.models import (
    ConstituentTick, Direction, EventKind, FuturesTick, LevelKind, MarketSnapshot,
    OptionContract, OptionGreeks, OptionType, SupportResistanceLevel,
)
from nifty_engine.params import StrategyParams
from nifty_engine.risk import RiskState


def snapshot(spot: float, previous: float, when: datetime) -> MarketSnapshot:
    rows = tuple(
        ConstituentTick(f"S{i}", 100.9, 100.0, 160_000+i*100, 100_000, 1500, 1200, 15, 1)
        for i in range(50)
    )
    future = FuturesTick("NIFTY-FUT", spot+12, previous+2, 1_300_000, 1_200_000, 2000, 15,
                         12_120_000, 12_000_000, spot, previous)
    option = OptionContract("NIFTY-X-CE", OptionType.CE, 25000, "2099-01-01", 150, 1_000_000,
                            1_200_000, 75, OptionGreeks(.58,.0018,-7,11,2,13),149.8,150.2)
    return MarketSnapshot(when, spot, previous, rows, future, (option,))


def test_cash_pressure_is_bullish_for_broad_positive_move() -> None:
    metric = constituent_metrics(snapshot(25_000, 24_990, datetime.now(timezone.utc)).constituents, StrategyParams())
    assert metric.score > 0.5
    assert metric.advancers == 50


def test_breakout_classifies_and_selects_call() -> None:
    params = StrategyParams(breakout_threshold=.58, reversal_threshold=.58, decision_margin=.05,
                            persistence_target_seconds=10, min_signal_confidence=.58,
                            risk_per_trade_pct=.02)
    engine = SignalEngine(params)
    state = RiskState(account_equity=2_000_000)
    level = (SupportResistanceLevel("R1", LevelKind.RESISTANCE, 25_000),)
    t0 = datetime.now(timezone.utc)
    engine.evaluate(snapshot(25_000,24_990,t0), level, state)
    engine.evaluate(snapshot(25_018,25_000,t0+timedelta(seconds=1)), level, state)
    signal = engine.evaluate(snapshot(25_025,25_018,t0+timedelta(seconds=20)), level, state)
    assert signal.event is EventKind.BREAKOUT
    assert signal.direction is Direction.BULLISH
    assert signal.contract.contract is not None
    assert signal.risk.allowed


def test_prior_session_baseline_and_config_validation() -> None:
    start = date(2026,1,1)
    rows = [VolumeRateSample("AAA", start+timedelta(days=i), 600, 100+i) for i in range(5)]
    assert ("AAA",600) in build_intraday_volume_baselines(rows, min_sessions=5)
    with pytest.raises(ValueError): StrategyParams(cash_pressure_weight=.8)
    with pytest.raises(ValueError): SlidingWindowRateLimiter(max_per_second=10, max_per_minute=5)
