# Packaged proof verification

Recorded against the included source on `2026-10-03T05:19:51.783925+00:00`.

- Python: 3.12.14 (CPython, linux)
- Deterministic controls: 53 passed; 0 failed; 0 skipped
- Scripted workflows: 10 passed; 0 failed
- External transport calls: 0
- Unapproved or duplicate fake submissions: 0
- Automatic send retries: 0
- Model invoked: false
- Production Mail MCP tested: false
- Fixture: `zetbros-synthetic-v1`
- Source content revision: `65c3c6c62292d1987c4900bf3f88d51d9fbdb9487a2e37d963b124387158d03b`

The README replay command and interactive EOF/cancellation path were also run
successfully from the packaged directory. Every source/fixture/README hash in
`source-manifest.json` was matched to the files in this package.

All nine pricing scenarios and the price-floor/hour-cap figures were recalculated
from `../../commercial/unit_economics.json` using decimal arithmetic and matched
to the published rounding precision. Those inputs remain planning assumptions.

Passing results apply only to the independent offline, single-process, in-memory
harness. They establish no live Mail MCP control, authenticated approval, model
quality, delivery guarantee, durable state, production SLA, or website acceptance.
See `../README.md` and `../../technical/README.md` for the complete scope.
