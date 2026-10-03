# Zetbros support-reply control service

A runnable, persistent release-candidate backend for one customer-owned
single-host deployment and one support-mail reply workflow. This is separate
from the existing offline proof. It is not a universal gateway or a hosted
multi-tenant product.

**Live outbound delivery is disabled in code. Buyer production release is
blocked until the actual Mail source/adapter, existing customer identity
configuration, approved deployment, and release acceptance are available.**
No environment switch enables sending. No Mail source, real mailbox, provider
credential, customer deployment, model/client integration, or real delivery was
accessed or tested for this change.

## Run and verify

Python 3.12 is the tested runtime. From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q zetbros_service tests
```

`requirements-lock.txt` additionally pins the complete runtime dependency tree.
The implementation pins the top-level packages to versions installed and tested
in the build environment. The complete tested dependency/runtime inventory is
recorded in `evidence/test-results.json`. These pins are not a vulnerability
clearance or a future dependency-update promise. No software was installed or
system configuration changed in a customer environment.

For a customer-controlled, private evaluation, review `deploy/config.example.json`
and supply the existing issuer's trusted **public** JWKS and explicit existing
agent/reviewer subject/client mappings. The placeholders are not usable login
credentials. This service has no token issuer, password login, default key,
fixture secret, OAuth-grant creation, or provider-credential lookup. Missing or
invalid config/JWKS/source directory prevents startup.

Create an existing approved local durable-volume directory for the database and
mount the customer-staged source directory read-only. With approved data and
configuration in those locations:

```sh
export ZETBROS_CONFIG_FILE=/absolute/path/to/customer-config.json
python3 -m zetbros_service.maintenance init  # Genuinely new deployment only
python3 -m zetbros_service.maintenance check
python3 -m uvicorn zetbros_service.api:configured_app --factory \
  --host 127.0.0.1 --port 8081 --workers 1 --no-access-log \
  --timeout-keep-alive 5 --limit-concurrency 32 --timeout-graceful-shutdown 30
```

The service binds loopback. Remote access requires an existing customer-approved
HTTPS reverse proxy with strict header/body/read timeouts, a 32 KiB request
limit and traffic limiting. Deploying that proxy or the included systemd example
is a separate operator action, not performed here. There is no public deployment.

## The one workflow

1. An existing authenticated agent reads `GET /v1/source/{source_id}`. The
   response contains a bounded customer-staged record and its canonical source
   fingerprint. This is **not authoritative live Mail state**.
2. The agent posts `POST /v1/proposals` with only operation_key, source_id,
   source_version, source_fingerprint and exact plain-text body. Sender, single
   reply destination, subject, identity, operation, policy and times are resolved
   server-side. Unknown fields, arbitrary To/CC, send/reply-all, attachments,
   deletion, bulk writes and approval booleans are rejected.
3. A separately authenticated human reviewer fetches
   `GET /v1/proposals/{proposal_id}` and inspects the exact stored payload and
   complete digest. `POST /v1/reviews/{proposal_id}/decision` accepts only that
   digest and approve, deny or revoke. Agent tokens cannot call reviewer endpoints,
   even if they assert a reviewer role/scope/subject in request fields or headers.
4. Approval, decision record, durable execution reservation and required audit
   event commit in one transaction. The worker rechecks configured identities,
   policy, current source fingerprint, issuance, expiry and revocation. A required
   preflight failure makes zero adapter calls and does not consume approval.
5. This release records `blocked / delivery_disabled` durably. In controlled
   adapter tests only, a successful claim atomically consumes approval, fences
   the execution and records its audit before invoking the test adapter. No DB
   transaction stays open during invocation. Results survive restart.

Same logical operation key plus unchanged request returns the stored proposal;
changed data returns a conflict. Keys are scoped to deployment customer/connector,
with account/source/revision in the bound payload. They are not global body hashes.
An intentional second reply uses a new explicit operation key and requires a
new human approval. A consumed approval never permits another attempt.

Accepted submission is not delivery. Failed Sent copy does not cause a resend.
A crash after claim, a possible submission, timeout, invalid adapter result or
unexpected exception becomes uncertain and is never automatically reclaimed or
retried. Late completions are fenced. Revocation after claim is refused because
it cannot reliably retract a possible submission. Human reconciliation records
an append-only observation and evidence reference; it never reopens an uncertain
operation or triggers a send. Reviewers can fetch the preserved observations at
`GET /v1/reviews/{proposal_id}/reconciliations`.

## Identity boundary

RS256 access tokens are verified against deployment-owned pinned public JWKS,
fixed HTTPS issuer, one exact audience, configured typ, required subject/client,
expiration, issued-at and not-before claims, and a bounded token lifetime.
Unknown kid, other algorithms, private/symmetric keys, remote jku/jwk headers,
wrong issuer/audience and unconfigured subject/client pairs fail closed.
Authorization comes from the server mapping, not token scopes/role/tenant fields.
Agent and reviewer client mappings are distinct. JWKS rotation requires the
customer operator to verify the issuer's public keys, update the approved local
key file, and restart; no automatic fetching from token-controlled URLs occurs.
Existing tokens for removed mappings lose access after that reviewed restart.
Agent and reviewer client ID sets must be disjoint; the same human subject may
use distinct separately configured clients without upgrading the agent client.

This is a Bearer-only JSON API, with no browser review UI, cookies or CORS flow.
Browser Origin requests and tokens in URLs are refused. No MCP interface or
multi-client compatibility is claimed. A future remote MCP adapter needs its
own OAuth resource/audience and client acceptance tests.

## Customer-staged source contract

`source_directory/{source_id}.json` is a customer-owned read-only record. IDs are
bounded identifiers, never paths or URLs. Symlinks, nonregular/oversized files,
unknown fields, wrong customer/connector/account, bad address/header values and
stale version/fingerprint are refused. Source opens are nonblocking before
regular-file checks, so FIFOs cannot hang the service. The maximum record size is 64 KiB; source
and reply bodies are at most 16,000 characters. The strict `SourceMessage`
schema is in `zetbros_service/models.py`. All records used in automated tests
are temporary fictional `.invalid` data. No source examples are shipped as
production customer data, and there is no public snapshot-upload endpoint.

A snapshot may be stale despite a matching file fingerprint. Live release must
replace this with the actual owning Mail adapter, fetch/revalidate real original
message state immediately before submission, and independently test that
adapter's provider-specific acceptance/uncertainty behavior. Transport
credentials must remain in an isolated adapter/service credential boundary;
the agent/model and reviewer interface must never receive them.

## Storage, audits and operations

Normal startup and all database reconnects require an existing initialized
ledger and open SQLite with `mode=rw`. A missing/deleted database, empty schema
or missing identity/quarantine metadata fails closed. An operator-only `init`
command exclusively creates a new database for a genuinely new deployment.
Never use init to recover a lost volume, failed restore or existing deployment;
restore its complete ledger instead. Pending private restore candidates also
prevent fresh initialization.

SQLite uses WAL, foreign keys and synchronous=FULL, with versioned migration 1,
`BEGIN IMMEDIATE` transactions, unique business-operation constraints and
conditional claims. Approval/claim/expiry/result time checks sample the clock
after acquiring the transaction, so lock waits cannot authorize stale expiry
or late completion. The database records deployment identity and rejects being
reused for a different customer/account/sender/issuer/audience. This profile is
only a local durable volume on one host. Network filesystems, serverless,
ephemeral containers, horizontal multi-host deployment, HA and power-loss
hardware guarantees are not supported or verified.

Full proposal/source-derived payloads and decision/execution metadata are
private data. Restrict the DB/backups to the approved service user and named
operator; use approved volume encryption and retention policies. Installing
permissions/encryption or granting access was not performed. This service has
no payload retention deletion job. Before real data, the customer must sign off
payload/audit/backup retention and a purge process that preserves the operation
ledger; deleting deduplication history could permit repeats.

Audit records contain IDs, tenant/connector, trusted actor/client, digest,
policy, times, event/outcome and sanitized error class. They omit bodies,
subjects, addresses, tokens and raw provider exceptions. App-level append-only
triggers prevent routine mutation. A DB owner can still alter the DB, so this
is not tamper-proof or compliance-certified auditing. Rate limits persist
across service restart. Storage/audit failure before claim blocks execution;
post-invocation result-record failure leaves a consumed, fenced claim and later
records uncertainty instead of resending.

`GET /healthz` checks process liveness. `GET /readyz` independently discloses
control readiness, disabled delivery, staged source and unverified live source.
A 200 control-ready response is never a live-delivery readiness claim. Graceful
shutdown stops new work and lets an in-flight adapter complete within its
configured bound. Abrupt shutdown follows the no-reclaim uncertainty path.

## Backup and restore

```sh
python3 -m zetbros_service.maintenance backup --destination /approved/new-backup.sqlite3
```

The SQLite backup API captures proposals, decisions, the execution ledger,
audit, configured identity and rate state together. Do not copy only the main
WAL-mode file while the service runs. Verify and protect backups under the same
customer data policy. Before restore, stop the service and preserve the current
DB/WAL/SHM and newer operation evidence. Point the reviewed configuration at a
**new absent** database path, then:

```sh
python3 -m zetbros_service.maintenance restore --source /approved/backup.sqlite3
```

Restore prepares an exclusive private same-directory candidate, validates it,
commits quarantine plus ledger conversions/audit, checkpoints and fsyncs the
self-contained file, then atomically links it to the absent final path without
overwriting a racing destination. An interrupted preparation leaves no final
database; normal startup refuses the missing ledger. Do not restart after a
failed restore until a complete reviewed restore succeeds. Hidden interrupted
candidates are never auto-discovered and must be inspected by the operator.

Every restore sets a persistent execution quarantine, converts historical
claimed actions to uncertain and blocks queued actions. No API or CLI releases
that quarantine. An older backup can omit later submissions; blindly unblocking
it could duplicate mail. Release needs a separately reviewed version, comparison
with the current operation ledger/provider evidence, and human reconciliation.
Restore is not a mechanism to retry an uncertain operation.

## Verification and release gates

`evidence/test-results.json` contains exact test names, outcomes, source hashes,
base PR revision and runtime. `evidence/verification-summary.md` separates passed
control tests from blocked/unrun buyer-release gates. Independent-process claim
and abrupt-crash cases execute actual OS child processes; transport is always a
clearly identified controlled test adapter. No real model, SMTP, IMAP or Mail
provider was invoked. Finite tests do not establish prompt-injection immunity.

Official implementation references:
- [SQLite transaction semantics](https://www.sqlite.org/lang_transaction.html)
- [SQLite WAL and FULL durability](https://www.sqlite.org/wal.html)
- [PyJWT claim and key validation](https://pyjwt.readthedocs.io/en/stable/usage.html)
- [FastAPI security primitives](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)
