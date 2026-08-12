from __future__ import annotations

from pathlib import Path
import json

from nifty_engine.params import StrategyParams


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    params = StrategyParams.from_json(root / "config" / "strategy.example.json")
    symbols = json.loads((root / "config" / "nifty50.symbols.json").read_text())
    assert len(symbols["symbols"]) == 50
    print(f"strategy loaded; {len(symbols['symbols'])} NIFTY symbols; breakout={params.breakout_threshold}")
