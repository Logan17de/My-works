# Zetbros offline synthetic approval proof

OFFLINE SYNTHETIC TEST ONLY. This independent Python harness uses fictional mail
and an in-memory fake transport. It is not an existing Mail MCP implementation,
live connector, production approval system, or customer deployment.

## Run

Requires Python 3.10+ and only the standard library.

From this directory:

```sh
python3 run_tests.py
python3 demo.py --scripted-replay --output evidence
python3 demo.py
```

- `run_tests.py` runs the controls and ten separate scripted workflows, records
  source hashes and results, and regenerates the trace and replay transcript
- `--scripted-replay` uses labeled fixture decisions, not a human sign-off
- The interactive command displays the exact payload and requires the operator
  to type `approve <full displayed action digest>`, `deny`, or `cancel`
- Invalid input, EOF, or interruption cancels the current proposal
- Invalid fixtures and any mode other than `synthetic` stop before backend creation
- The clock is fixed; tests advance it to cover not-before and exact-expiry
  boundaries. CLI wall-clock expiry is not claimed

These commands write local evidence. They do not send mail or contact a provider.

## Demonstrated controls

1. Synthetic list/search/read without changing read flags
2. Resolve Reply-To or From before preview, without reply-all
3. Immutable typed action data covering sender, recipients, subject, body,
   original message/folder/version, fixture/context, and canonical SHA-256 digest
4. A separate local-owner decision channel; the exposed tool surface has no
   approve or execute tool and rejects approval booleans or source strings
5. One decision bound to proposal, action, fixture, run, actor, issuance, and expiry
6. Rejection of missing/forged approval, denial/cancel, expiry, payload/context
   changes, replayed permissions, and conflicting keys or proposals
7. In-memory reservations before fake transport and stored receipts on retries;
   newly proposed exact actions are also suppressed after a terminal outcome
8. Separate simulated submission and Sent-copy status; partial or uncertain
   outcomes and failed copies never trigger an automatic resend
9. Required proposal/decision/start audit failures block execution; a failed
   post-submission result record preserves the known result and consumed decision
10. Minimized synthetic events without body, subject, raw exception text, or
    credential canaries

Addresses must end in `.invalid` and be on the fixture allowlist. There is no
live backend or fallback. Fake drafts, transport attempts, and accepted outbox
entries are distinct stores.

## Files and evidence

- `proof.py`: fixture backend, bounded tools, proposals, decisions, in-memory
  duplicate controls, fake outcomes, events, and deterministic workflow
- `safety.py`: Python audit guard for tested network and subprocess escape paths
- `demo.py`: exact-payload console and clearly labeled scripted replay
- `fixtures/mailbox-v1.json`: versioned fictional mailbox and context
- `tests/test_proof.py`: deterministic control and transport-contract tests
- `evaluate_workflows.py`: ten separate fixture-driven workflows, E01-E10
- `run_tests.py`: aggregate runner and content-hash provenance
- `evidence/test-results.json`: assertions, runtime, source hashes, case results,
  counters, trace references, and scope exclusions
- `evidence/workflow-results.json`: scripted rubrics, tool inputs, outcomes, and traces
- `evidence/source-manifest.json`: SHA-256 of all included proof source/fixtures/README
- `evidence/control-traces.jsonl`: minimized per-unit-case events
- `evidence/synthetic-trace.jsonl`: minimized final replay events
- `evidence/scripted-replay.txt` and `evidence/replay-results.json`: captured replay
- `evidence/unit-test-output.txt`: actual unittest output
- `evidence/verification-summary.md`: human-readable results and limits

Source hashes identify the exact tested contents. A commit is intentionally not
embedded in the generated evidence because including that evidence in the same
commit would create circular provenance. Compare the manifest with the files at
the GitHub commit to verify the executed source.

## Coverage

Control families: I01-I09, P01-P10, and D01-D08. Additional cases separately
exercise each canonical field, issuance/expiry boundaries, reentrant execution,
cross-proposal/key bindings, Reply-To/context drift, preview isolation, malformed
types, provider-constructor spies, adapter exceptions, and post-send audit failure.
See `../technical/test-matrix.md` and the generated JSON for the exact case list.

I10 attachments and I11 move/archive/delete are excluded because those tools are
not exposed. Website checks, live Mail MCP/provider tests, actual model/client
evaluations, restart/crash recovery, multi-user identity, and distributed or
parallel concurrency are outside this proof.

## Limitations

State is lost on restart. Exact-envelope suppression prevents an intentional
repeat-send of the same action in a session; that workflow is not implemented.
The in-flight test is single-process reentrancy, not distributed concurrency or
durable atomicity. No crash recovery has been tested.

The console trusts its local operator. It provides no login, OAuth, verified
production identity, RBAC, or tenant isolation. Arbitrary developer Python in
the process can access private objects and is outside the trust boundary. The
demonstrated boundary is the bounded workflow tool surface.

The audit hook denies the tested Python socket/DNS/HTTP/SMTP/IMAP/subprocess
events before external transport. Constructor spies cover the scripted workflow.
This is not an OS firewall or a claim that adversarial native/FFI code cannot
make a connection. The harness has no credential lookup, environment loading,
network client, or live transport path.

Fixture metadata explicitly selects the workflow branches. No model is invoked.
The injection fixture therefore tests one deterministic branch and action
boundary, not a general classifier, model tool-selection quality, or immunity to
prompt injection. The completion scorer likewise checks observed synthetic
status, rather than model output.

JSONL events are observability, not tamper-proof auditing, retention controls,
compliance evidence, or incident monitoring. Required pre-execution records are
fail-closed; post-execution record failures are disclosed. Direct developer calls
to private backend APIs are outside trace-completeness claims. Passing synthetic
canaries does not establish protection of real customer data.

Supported claim: an offline synthetic harness demonstrates exact-action local
decisions, tested denial/expiry/mutation/replay rejection, minimized events, and
no automatic resend on uncertain fake outcomes.

Live use needs a separately scoped implementation and review covering verified
identity, least privilege, durable state, crash/concurrency behavior, secrets,
provider uncertainty, audit access/retention, and an accountable operating owner.
