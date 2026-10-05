# Zetbros revenue work package

An offline approval-control proof, a bounded first-customer offer, and dated
buyer research. Prepared on 3 October 2026.

## Start here

- [Offline proof and commands](proof/README.md)
- [Commercial execution pack](commercial/commercial_execution_pack.md)
- [Labor-inclusive pricing scenarios](commercial/unit_economics.json)
- [Buyer research snapshot](validation/buyer-validation-2026-10-03.md)
- [Unsent discovery drafts](validation/discovery-drafts-2026-10-03.txt)
- [Technical scope and limitations](technical/README.md)
- [Executed control matrix](technical/test-matrix.md)
- [Verification summary](proof/evidence/verification-summary.md)
- [Implemented, unpublished service-page patch](website-preview/SERVICE_PAGE_HANDOFF.md)

## Run the proof

Python 3.10+ is required. No dependencies, account, credentials, API calls, model
subscription, or installation are needed.

From the repository root:

```sh
cd zetbros-revenue/proof
python3 run_tests.py
python3 demo.py --scripted-replay --output evidence
```

For a local interactive decision, run `python3 demo.py` instead. The console
prints the exact synthetic action and accepts `approve <displayed full digest>`,
`deny`, or `cancel`. Scripted replay approvals are fixture decisions and are
never evidence of a human sign-off.

The aggregate runner executes deterministic control tests and ten scripted
workflow cases, then writes source hashes, results, minimized synthetic traces,
and a captured replay to `proof/evidence`. Rerunning changes evidence timestamps
and durations, so those files can appear modified in Git.

## What is complete

- Standalone, standard-library Python harness with fictional `.invalid` addresses
- Exact-payload proposals, one-use local decisions, expiry/context checks,
  in-memory duplicate suppression, and no automatic resend on uncertainty
- Deterministic tests and ten explicitly scripted workflow evaluations
- Proposed USD 300 audit and qualified USD 1,500 read-only pilot, with discovery,
  scope, acceptance, handoff, and labor-inclusive economics
- Dated public buyer research and two unsent discovery drafts
- A narrow `/agents` website patch with recorded automated results; unpublished

The prices are planning hypotheses. Buyer-posted budgets are not earned revenue,
accepted customer terms, or proof that a buyer wants Zetbros' service. Check live
availability and actual scope before acting on any listing.

## Boundaries and pending work

This harness does not run or validate any existing live Mail MCP implementation.
It is single-process and in-memory, with a fixed fixture clock and a locally
trusted operator. It is not authenticated production authorization, durable
idempotency, a multi-user system, an actual model evaluation, a delivery service,
a compliance certification, or a production SLA.

All included mail data, decisions, and transport results are synthetic. The
Python guard blocks the tested socket/network/subprocess paths, but it is not an
OS sandbox. Developer code inside the process remains trusted. Read the proof's
full limitations before drawing conclusions from a passing test.

The website service page is implemented and unpublished. `website-preview/`
contains only the narrow three-file patch and its review handoff. Final build and
TypeScript checks passed, along with 6 Worker tests and 19 static tests. Browser
UI verification is blocked and incomplete, so no visual or interaction pass is
claimed. The full website checkout and build output are not included, and this
GitHub publication does not deploy the website. No prospect demo, outreach,
customer contract, or live-mail action was performed by this package.

Existing repository content is preserved. This package is self-contained under
`zetbros-revenue/` and does not change the repository's other projects.
