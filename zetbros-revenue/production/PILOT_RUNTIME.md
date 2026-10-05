# Reviewed pilot runtime: runnable code, dormant shipped setup

This checkpoint adds our own live-capable session factory, runtime assembly and
minimal human review interface. The implementation is complete enough for an
operator-configured isolated pilot; the user does not have to implement a factory.
All shipped configuration examples still select disabled outbound delivery.
No real provider connection, credential, issuer grant, host modification, ledger
migration, deployment or mail send was performed in this implementation stage.

## Disabled-install correction

The current supported default is now the explicit pending-identity health-only
stage. Its init/check/start walkthrough creates no ledger and has no credential
dependency. After actual identity setup, a disabled exact-wire MailStore workflow
is supported without live flags. See [DISABLED_STAGING.md](DISABLED_STAGING.md).
The original runtime details below describe the identity-configured modes.

## Runtime assembly and live safety boundary

`configured_app` and `configured_pilot_app` use the same explicit runtime builder.
Default `customer_staged_snapshot` + `disabled` settings return the original
non-sending service. An operator-configured pilot requires both adapters to be
`private_spacemail`, an exact trusted reviewer origin, and a matching private
profile whose outbound value is `owner_approved_live`. Unknown/partial/mismatched
configuration fails closed. These are local operator configuration fields, never
agent/model tool inputs. No token issuer, default key, developer authentication,
OAuth grant or credential enrollment endpoint exists.

The live factory opens fresh isolated sessions to `mail.spacemail.com` with
verified implicit TLS, IMAP 993 and SMTP 465. Account, sender, aliases, tenant,
connector and policy stay pinned to the per-customer profile. A single bounded
DNS worker resolves only that fixed hostname; a stalled resolver cannot spawn
an unlimited replacement pool, and callers exit without connecting. A guard
registers sockets before connect and TLS handshake and interrupts them at the
absolute deadline through greeting/authentication and cleanup. SMTP uses a
fixed EHLO address literal, avoiding unbounded local-hostname DNS.

Incoming SMTP lines/multiline totals and IMAP response/literal/untagged totals
are bounded before materialization. The IMAP command debug cache is suppressed
because standard imaplib otherwise caches LOGIN text even with debug=0. Sessions
never log provider text or credentials. Source reads execute off the event loop;
a blocked provider read does not block human review or process-health handling.

The exact-wire bridge still gates after DATA 354 and before body, preserves
final-250 acceptance through cancellation/Sent cleanup, never automatically
retries uncertain/claimed operations, and records delivery as unverified. Human
approval is not provider acceptance or delivery evidence.

## Protected credential mechanism

`SystemdCredentialSource` reads only the fixed `spacemail-password` entry inside
systemd's `CREDENTIALS_DIRECTORY`, when an explicitly enabled pilot actually
opens a session. It rejects missing/nonregular/symlinked, exposed, oversized or
malformed files and returns only sanitized errors. It does not read archived
configuration, arbitrary environment password values, agent inputs or chat.

The separately proposed live unit uses `LoadCredential` with a source path but
no secret value. The default disabled staging unit has no credential dependency.
The owner/operator must securely supply or reuse the approved protected source;
creating the service account, configuring credential access and enabling the
unit are separate consequential actions requiring approval. No real secret
was read, entered, copied or saved for this checkpoint. Python strings cannot
promise cryptographic zeroization; this is a protected service-process boundary,
not a credential-isolation hardware claim.

The mechanism follows [systemd's credentials documentation](https://github.com/systemd/systemd/blob/main/docs/CREDENTIALS.md).
The session interfaces are based on the installed Python 3.12
[SMTP](https://docs.python.org/3.12/library/smtplib.html) and
[IMAP](https://docs.python.org/3.12/library/imaplib.html) implementations, tested
here only with fake I/O. Real authentication/TLS/provider behavior remains unrun.

## Human review

`/review` serves self-hosted HTML/CSS/JavaScript. The reviewer supplies an existing
properly scoped reviewer token in a password field; it lives in page memory only,
never storage, cookies, URLs or logs. The API verifies the pinned issuer/audience/
JWKS and the existing server-side reviewer subject/client mapping. Agent tokens
cannot approve, even if their token or request asserts a reviewer role.

The UI pins one canonical origin, rejects foreign origins/CORS/query tokens and
cookie authority, renders mail with text-only DOM operations, and uses a strict
CSP. It displays full exact wire preview, action, source binding and composite
review digest before approve/reject. Action-only snapshots are not actionable
as exact-wire approvals. In enabled pilot mode, its notice explicitly says
approval authorizes attempted submission of that exact reply. In dormant mode,
it explicitly describes recorded decisions without delivery.

Repeated/stale decisions, newer-preview races, logout/cancel and uncertain POST
responses do not trigger silent retries. Review state must be fetched again
when a decision response is inconclusive. The browser does not create identity
grants or expose an automatic sign-in bypass. Real reviewer identity/client
acceptance and browser pixel/host behavior remain part of live setup acceptance.

## New ledger only

`python -m zetbros_service.pilot init` stages without a ledger while identity is
pending. With actual identity configuration, it verifies public JWKS and profile
before explicitly creating an absent new private ledger, including in disabled
mode. It performs no Mail I/O or credential lookup. Existing paths fail through O_EXCL. It never migrates, replaces
or repairs an existing ledger; lost/restored operation history must retain the
existing quarantine/reconciliation policy.

`python -m zetbros_service.pilot check` validates local control assembly without
provider operations. Live readiness stays false until provider/deployment/customer
acceptance exists; the status distinguishes an operator-configured unverified
pilot from disabled defaults.

## Proposed single-customer deployment and resource envelope

The proposed layout is `/srv/zetbros/pilot`, one locked non-login `zetbros-pilot`
service identity, one account, one process/worker and local durable ledger. The
HTTP API binds loopback 8081. No existing Mail service/tunnel is modified. Remote
exposure must use an explicitly approved boundary and exact reviewer origin;
HTTP loopback alone is not approval to expose a public unauthenticated service.

`deploy/zetbros-pilot.service.example` is a proposal, not an installed unit. It
includes MemoryHigh 96 MiB, MemoryMax 128 MiB, no swap, CPUQuota 25%, TasksMax 32,
eight-request concurrency and four blocking API thread tokens. Source/provider
buffers and resolver/watchdog growth are bounded. Account creation, directory
ownership/permissions, credential access and unit installation were not executed.

Two offline samples measured 55.8–56.5 MiB peak RSS and 21–23 sampled threads, including
benchmark callers, below the proposed 96/128-MiB and 32-task limits. This is one
bounded cloud sample, not a general memory ceiling. The sample is recorded separately. It uses a fresh cloud process,
eight concurrent ASGI proposal requests and eight fake-provider operations,
including retained verified TLS contexts. It is not measurement of the target
host, cgroup, real TLS traffic or CPU-throttled latency. Do not enable if target
startup/controlled acceptance violates the approved resource envelope.

## Verification scope

The final aggregate excludes the legacy loopback HTTP fixture and the Chromium
pixel-QA launch. ASGI authentication/preview/decision tests and a socket-free Node
DOM/state harness cover repeated/racing/stale loads, cancel/logout, uncertain
responses, safe text and malformed-response redaction. Chromium could not start
under this workspace's socket restriction; no permission escalation was attempted.
Thus actual browser rendering/pixel and deployed-browser behavior are unrun.
Systemd unit verification could not run in this cloud workspace because the
verifier attempted its read-only /run/systemd runtime path. No escalation or
host/unit change was attempted; target systemd/unit acceptance remains unrun.
Python asyncio uses local wake-up socketpairs internally; no provider or listening
network socket was opened by these acceptance tests.

## Specific remaining owner setup

The next step is to approve the concrete pilot setup and supply nonsecret
configuration, not to write implementation code:

1. Confirm the proposed pilot layout/account/resource limits and whether this is
   a genuinely new ledger, with permission to install the reviewed code/unit
2. Identify the intended mailbox and approved sender/aliases; securely authorize
   the existing protected credential source for systemd LoadCredential
3. Provide the existing trusted public JWKS/issuer/audience and distinct agent and
   human-reviewer subject/client mappings, plus the exact reviewer access origin
4. Approve one controlled recipient/workflow for live acceptance before any real
   submission. No broad approval identifies that recipient automatically

No new identity grant is inherently required if suitable existing identities
cover this API; missing or mismatched identities must be resolved explicitly.
Protected credential entry and security/access changes cannot be replaced by
general project approval. These are actual setup gates, not requests for the
user to implement another adapter.
