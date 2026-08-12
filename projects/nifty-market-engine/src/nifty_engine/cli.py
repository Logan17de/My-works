from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .engine import SignalEngine
from .models import (
    ConstituentTick, FuturesTick, LevelKind, MarketSnapshot, OptionContract,
    OptionGreeks, OptionType, SupportResistanceLevel,
)
from .risk import RiskState
from .serialization import dumps


def _demo_snapshot() -> MarketSnapshot:
    constituents = tuple(
        ConstituentTick(
            f"STOCK{i}", 100.8 + i * 0.01, 100.0, 150_000 + i * 1000,
            100_000, 1500.0, 1200.0, 15.0, 1.0,
        ) for i in range(50)
    )
    future = FuturesTick(
        "NIFTY-FUT", 25035.0, 24998.0, 1_250_000, 1_200_000, 1600.0,
        15.0, 12_100_000, 12_000_000, 25025.0, 24995.0,
    )
    options = (
        OptionContract(
            "NIFTY-DEMO-25000-CE", OptionType.CE, 25000, "2099-01-01", 145.0,
            900_000, 1_100_000, 75,
            OptionGreeks(0.58, 0.0018, -7.0, 11.2, 2.1, 13.5), 144.8, 145.2,
        ),
    )
    return MarketSnapshot(
        datetime.now(timezone.utc), 25025.0, 24995.0, constituents, future, options,
    )


def paper_demo() -> None:
    engine = SignalEngine()
    levels = (SupportResistanceLevel("R1", LevelKind.RESISTANCE, 25000.0),)
    touch = _demo_snapshot()
    touch = MarketSnapshot(
        touch.timestamp, 25000.0, 24995.0, touch.constituents, touch.futures, touch.options,
    )
    state = RiskState(account_equity=2_000_000)
    engine.evaluate(touch, levels, state)
    print(dumps(engine.evaluate(_demo_snapshot(), levels, state)))


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY market event engine")
    parser.add_argument("command", choices=["paper-demo"])
    args = parser.parse_args()
    if args.command == "paper-demo":
        paper_demo()


if __name__ == "__main__":
    main()
