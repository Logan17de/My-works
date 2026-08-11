# NIFTY Market Engine

A test-first market-analysis and paper-execution project for studying NIFTY support/resistance interactions using NIFTY-50 constituent participation, NIFTY futures confirmation, and option Greeks/liquidity for contract selection.

The project separates a long-running Python market worker from a Vercel-ready Next.js dashboard and Supabase persistence. It is designed for deterministic replay and paper testing before any real-money integration is considered.

Default parameters are research hypotheses, not claims of profitability. High aggregate volume is treated as strong market participation, not proof of institutional activity.

See the formulas, architecture, and research protocol in `docs/`.