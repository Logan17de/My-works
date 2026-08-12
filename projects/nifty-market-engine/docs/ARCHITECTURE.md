# Architecture

## Runtime split

```text
                        GitHub
                           |
                    CI / versioning
                           |
          +----------------+----------------+
          |                                 |
   Long-running worker                 Vercel / Next.js
   Python 3.11+                        dashboard + API
          |                                 |
          | Groww API / feed                | read signals
          v                                 v
      Groww market data  --->  Supabase  <--- browser
          |
          v
   signal / risk / paper execution
```

The Python engine is deliberately framework-independent. It can run locally, on a VPS, in a container, or in another long-lived runtime. The web application under `apps/web` is independently deployable to Vercel.

A request-driven Python API can also be deployed to Vercel, but the market-hours collector should not depend on a short-lived function invocation. Long-running data collection and broker connectivity belong in the worker.

## Data flow

1. **LTP feed:** subscribe to constituent, NIFTY spot, futures, and selected option LTPs for responsive prices.
2. **Volume/OI snapshots:** poll `get_quote` at a rate-limited cadence because the full quote contains cumulative volume and OI.
3. **Option chain:** refresh less frequently than LTP and use it for Greeks, volume, OI, strikes, and contract selection.
4. **Signal engine:** produce `BREAKOUT`, `REVERSAL`, or `UNCERTAIN` at configured support/resistance levels.
5. **Risk engine:** veto unsafe or over-budget trades.
6. **Execution adapter:** paper broker only in this repository; any future live-order service is a separate integration boundary.
7. **Storage:** write sanitized signals to Supabase; keep private order/trade data under RLS.

## Support/resistance input

V1 accepts levels as data rather than coupling the strategy to a charting UI. Levels can later come from:

- manual entry from Groww Terminal,
- an approved upstream API if Groww exposes one,
- or our own deterministic support/resistance calculator from historical candles.

That makes the core engine testable even if the source of the levels changes.

## Testing modes

- **Unit tests:** deterministic formula and risk tests.
- **Synthetic replay:** controlled breakout/reversal scenarios.
- **Historical replay:** feed time-ordered candles/snapshots without order placement.
- **Paper live:** consume live Groww data but route orders to `PaperBroker`.
- **Live candidate:** only after a separately reviewed broker service is implemented and the research gates pass.
