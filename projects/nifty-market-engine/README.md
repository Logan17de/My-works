# NIFTY Market Engine

A production-oriented rewrite of the NIFTY options idea around one testable hypothesis:

> When NIFTY interacts with a known support/resistance level, classify the interaction from **constituent participation + futures confirmation**, then use **option Greeks/liquidity** to choose the contract. Trade only when the event is confidently a breakout or reversal; otherwise do nothing.

This repository does **not** assume that higher volume proves institutional activity, and the default parameters are not presented as profitable. The project is structured to collect evidence first.

## What is implemented

- NIFTY-50 constituent relative-volume pressure, breadth, activity concentration, and volume acceleration.
- NIFTY futures price/volume/OI/basis confirmation.
- Stateful support/resistance touch tracking.
- Three-way level classifier: `BREAKOUT`, `REVERSAL`, `UNCERTAIN`.
- CE/PE selection using delta, volume/OI liquidity, theta cost, IV, gamma, and optional spread filtering.
- Premium-budget sizing and hard risk vetoes.
- Fully functional paper broker.
- Groww market-data adapter plus a paper broker; real-money order placement is intentionally outside this repository.
- Supabase schema + Python signal sink.
- Vercel-ready Next.js dashboard skeleton under `apps/web`.
- Deterministic unit tests, synthetic event tests, and a leakage-safe replay harness.
- Prior-session intraday volume-baseline builder.
- Data-quality veto for stale snapshots or incomplete NIFTY-50 coverage.

## Groww fit

The implementation matches the current Groww API shape:

- `get_quote` supplies price, cumulative `volume`, `open_interest`, OI change, and implied volatility.
- `get_option_chain` supplies NIFTY strikes, LTP, OI, volume, and Greeks.
- the feed supports up to 1,000 subscribed instruments, which is enough for NIFTY 50 + index/futures/options LTPs.
- Groww's feed documentation exposes LTP for equity/F&O; therefore V1 treats quote snapshots as the source for volume/OI features rather than pretending the feed contains fields it does not document.

## Safe defaults

Execution in this repository is **paper-only**. Groww credentials are used only by the market-data adapter. The broker interface exists so paper fills can be tested consistently, but there is no real-money `place_order` implementation in this project.

Never commit `.env`, API secrets, TOTP secrets, Supabase service-role keys, or access tokens.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest
python -m nifty_engine.cli paper-demo
```

## Project layout

```text
src/nifty_engine/        core engine, formulas, broker/storage adapters
config/                  strategy and constituent examples
docs/                    formulas and architecture
supabase/migrations/     database schema/RLS
apps/web/                 Vercel-ready Next.js dashboard
tests/                    deterministic tests
```

## Deployment target

The intended production split is:

```text
Groww <-- Python worker --> Supabase <-- Next.js on Vercel
```

The worker owns broker credentials and market collection. The Vercel app only reads sanitized signal state. This keeps trading credentials out of the browser and avoids coupling a market-hours collector to a request-scoped serverless function.

## NIFTY constituents

`config/nifty50.symbols.json` is a dated bootstrap list. Constituents and index weights are time-varying and **must be refreshed before live use**. The engine accepts an explicit `index_weight`; equal weights are appropriate only for early smoke tests.

## Formula reference

See [`docs/FORMULAS.md`](docs/FORMULAS.md). All coefficients live in `StrategyParams`, so every formula can be changed, versioned, and backtested without rewriting the engine. See [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) for leakage controls, chronological splits, ablations, and the paper-to-live gate.

## Evidence path before live trading

1. Unit/synthetic tests.
2. Historical replay with fixed parameters.
3. Out-of-sample replay.
4. Live paper mode with real Groww data and exact option prices/spreads.
5. Compare expected vs realized paper fills/slippage.
6. Only then consider enabling live execution.
