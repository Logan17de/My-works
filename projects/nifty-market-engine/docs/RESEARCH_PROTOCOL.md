# Research and validation protocol

The strategy is a hypothesis until it survives out-of-sample testing with realistic option prices and costs.

## 1. No future leakage

Every feature at decision time `t` must be computable from data timestamped `<= t`.

- A support/resistance level used in a replay must have been known before the interaction. Do not derive a level from candles that occur after the trade.
- Intraday volume baselines must use prior sessions only. Never use the current session's completed volume to normalize an earlier point in the same session.
- Futures and constituent snapshots must be aligned to a common decision timestamp with an explicit maximum age.
- Historical option P&L requires actual historical option prices. Do not substitute the later spot move and call it option profit.

## 2. Volume baseline

For each constituent and minute-of-day bucket, build a baseline from prior sessions:

```text
baseline_rate(symbol, minute) = median(session_volume_rate over prior sessions)
```

Use at least five prior sessions for smoke testing and substantially more for research. Expiry/news sessions should not be deleted merely because they are inconvenient; robust statistics are used to limit their influence.

## 3. Split design

Use chronological splits, not random row splits.

```text
train/calibration window -> validation window -> sealed test window
```

Tune thresholds/weights only on the calibration/validation windows. The sealed test window is evaluated once for the reported result. For a longer study, use walk-forward evaluation.

## 4. Event labels

The engine itself does not need a supervised label to run. For research, define an outcome horizon before testing, for example 1, 3, 5, and 15 minutes after the event.

Record:

- future NIFTY return at each horizon;
- maximum favorable excursion (MFE);
- maximum adverse excursion (MAE);
- whether the crossed level held or failed;
- option return using the exact selected contract when historical option data exists.

Do not change the outcome horizon after seeing which one looks best without treating that as a new experiment.

## 5. Metrics

At minimum report:

- number of level interactions;
- breakout / reversal / uncertain counts;
- signal precision conditional on a predeclared outcome rule;
- coverage: fraction of interactions that become actionable;
- average and median future return by event class;
- MFE and MAE distributions;
- paper/live slippage versus observed quote;
- option expectancy after spread, fees, and slippage;
- maximum drawdown and worst loss streak;
- results by time of day, expiry proximity, volatility regime, and support vs resistance.

A high win rate alone is not sufficient. Expectancy and drawdown matter.

## 6. Ablations

The first useful research question is whether each input adds value. Run the same sealed periods with:

1. level + price only;
2. + constituent pressure/breadth;
3. + participation/volume acceleration;
4. + futures price/volume;
5. + futures OI/basis;
6. full direction engine;
7. full engine + option contract selector.

If a component does not improve out-of-sample results, remove it rather than keeping it because it sounds sophisticated.

## 7. Paper-to-live gate

Live trading should remain disabled until all of these are true:

- deterministic/unit tests pass;
- historical replay is reproducible;
- a sealed out-of-sample period is positive after realistic costs;
- real Groww paper observations validate data mapping and timestamps;
- paper execution has enough samples to estimate slippage;
- any future live-order integration has idempotency and circuit breakers tested with mocks;
- secrets are stored outside Git and the worker uses least-privilege infrastructure.

The repository is intentionally paper-only. A future live-order service should be a separate, explicitly reviewed integration after the evidence gates above are satisfied.
