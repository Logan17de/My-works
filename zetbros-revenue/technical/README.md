# Technical scope and claim boundaries

This summary describes only the standalone offline proof in `../proof`.

## Architecture

The fixture loader validates a versioned JSON schema before constructing the
fake backend. Every address is both `.invalid` and allowlisted. `DemoTools`
provides a bounded workflow interface. `send_email` and `reply_email` prepare
proposals rather than executing a send.

An immutable action contains the exact envelope, message reference, fixture
fingerprint, and context digest. A local decision channel accepts an exact
displayed action digest. The engine checks decision identity, issuance, expiry,
payload, fixture, and original-message consistency before execution. The local
identity is a test fixture, not authenticated production identity.

Before touching fake transport, the engine records the start event, consumes
the decision, and reserves the key/action. A known terminal result is returned
on replay. A new proposal cannot silently repeat an identical uncertain action.
Fake acceptance, partial acceptance, uncertainty, rejection, and Sent-copy
failure remain distinct outcomes. None establishes real delivery.

## Observable evidence

`../proof/evidence/test-results.json` separates deterministic controls from the
ten fixture-driven workflows. Each control records its matrix ID, assertion,
runtime, source revision, fake transport counts, and event references. The
source manifest lets a reviewer compare the tested files to the Git checkout.

`control-traces.jsonl` and `synthetic-trace.jsonl` omit message subjects/bodies,
raw adapter exceptions, and credential canaries. Workflow-results and replay
files additionally contain the exact fictional previews. Every recorded actor,
recipient, decision ID, and proposal is synthetic. Identifiers such as
`demo_approval_0001` are in-memory fixture references, not secrets or production
approval credentials.

## Test families

- I01-I09: startup/fixture validation, network guards, address restrictions,
  read semantics, references, input limits/types, draft isolation, headers,
  sender restrictions, and Reply-To resolution
- P01-P10: pending/valid decisions, denial/cancel, forgery, issuance/expiry,
  payload/context mutation, replay/reentrancy/key conflicts, one injection
  fixture, tool restrictions, and pre/post-submission audit failures
- D01-D08: fake rejection/acceptance, Sent-copy failure, uncertainty, partial
  outcomes, bounded read retry, event minimization, and result-claim checks
- E01-E10: ten explicitly scripted workflow rubrics; no model/client evaluations

The detailed generated matrix is in [test-matrix.md](test-matrix.md).

## What passing does not establish

- Any existing live Mail MCP implementation or its security controls
- Real provider connectivity, accepted delivery, or operational reliability
- Authenticated human decisions, OAuth, RBAC, or tenant isolation
- Durable idempotency, restart/crash recovery, or parallel/distributed concurrency
- General model reasoning, tool selection, or prompt-injection immunity
- Tamper-proof logging, regulatory compliance, production readiness, or an SLA

The Python guard is a tested process-level aid, not an OS sandbox. Developer
code is trusted, state is in-memory, the clock is fixed, and intentional exact
repeat-send behavior is absent. Full limitations are in the proof README.

## Proposed live acceptance work

A future deployment needs its own source and revision, verified client/server
and upstream authentication boundaries, least-privilege access, durable decision
and retry state, crash/concurrency tests, provider-specific uncertainty handling,
redaction/retention/access checks, client compatibility, and operating ownership.
Use synthetic inputs until real data processing is explicitly agreed.

These are future acceptance requirements, not completed features. The current
commercial entry pilot is bounded to one documented API, one account/operator,
one selected client, and at most three read-only tools. Live writes need their
own qualification and scope.

The website service-page patch is implemented and unpublished; see
`../website-preview/SERVICE_PAGE_HANDOFF.md`. Its recorded final checks passed
(build, TypeScript, 6 Worker tests and 19 static tests). Browser UI verification
is blocked and incomplete. Those results are separate from the offline proof.
No website or production deployment is performed by this package.
