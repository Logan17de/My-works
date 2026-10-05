# Private exact-wire bridge: dormant protocol implementation

This is the historical protocol checkpoint. The later
[PILOT_RUNTIME.md](PILOT_RUNTIME.md) adds our own live-capable factory and human
review interface, with dormant shipped defaults and live setup still unperformed.

This increment adds our own stdlib-shaped SMTP/IMAP phase bridge on top of the
approved-byte ledger candidate. It neither imports nor redistributes the
supplied Mail implementation. It is disabled by default, has no real session
factory, contains no credential lookup/login/network constructor, and remains
unreachable from `configured_app`. Tests use fake objects and actual local
threads, never actual sockets. No deployment, grant, existing-ledger migration,
provider connection or external send occurred.

Base checkpoint: `71348ad197969f121776a46444a6ac20dd05f47b`.

## Implemented protocol sequence

`PrivatePreparedWireTransport` accepts one strictly bound profile and factory.
Only an explicitly fictional `offline_only` factory can exercise its test mode.
The public profile fixes SpaceMail verified implicit TLS endpoints, one customer,
connector, account, mailbox, approved sender/aliases, INBOX, Sent and short budgets.
Every opened session must return a freshly revalidated matching session identity.
This identity is trusted-factory metadata, not proof of actual authentication or
TLS. Live credential and endpoint ownership still need operator verification.

The prepared record now carries its full action. Before opening any factory,
the bridge checks tenant/account/connector/policy, approved sender, exact preview,
action digest, deterministic Message-ID, wire hash, bounded MIME/CRLF framing
and exact visible headers/body. It never regenerates a message or uses
`sendmail`, `send_message`, `data` or a convenience reply tool.

The transaction is MAIL -> one RCPT -> DATA command -> 354 -> current durable
approval gate -> exact wire write -> final reply. The gate is called after 354
and before body may start. The only wire transformation is SMTP dot transparency
and its terminator; the receiver reconstructs the exact approved MIME bytes.
Final DATA 250 establishes submission acceptance. Envelope/command acceptance
alone does not. Body/write/ack faults are uncertain; known prebody refusal means
non-submission. There is no retry loop.

A per-operation watchdog shutdown/closes the supplied isolated socket at the
absolute budget, and each phase sets a remaining inactivity timeout. Cancellation
also interrupts that session and fences later body calls. The worker shields
transaction completion to preserve a known final 250 through QUIT/Sent-copy
cleanup; its ledger records that result and then honors cancellation before
claiming another operation. Unknown completion remains uncertain and never
reclaimable. These behaviors are verified with fake sockets. Real socket shutdown,
connect/auth timeout enforcement and provider behavior are still unrun; the
private factory must enforce the deadline during creation/authentication before
it yields a session. A timeout argument or watchdog cannot repair a nonconforming
factory that ignores its deadline or shares a connection between operations.

## Sent-copy semantics

Sent filing occurs only after known SMTP acceptance. It searches the deterministic
Message-ID once. A single existing match must pass matching UIDVALIDITY, bounded
UID/size/PEEK fetch and exact wire-byte comparison before it is called stored.
An ambiguous, malformed or different-byte match fails without APPEND. Existing
copy verification is limited to 64 KiB by the established read parser; larger
matches need human reconciliation and cannot be used to trigger a new copy.

If no copy exists, one APPEND uses the exact approved bytes. Positive APPEND
acknowledgment means stored; explicit failure means failed; lost acknowledgment
means unknown. Cleanup failure cannot erase a known successful copy or SMTP
acceptance. Copy failure/uncertainty never permits resending mail. Delivery is
always unverified here.

## Single-customer isolation assessment

The supported candidate profile is one customer/account per process and local
ledger, with a pinned identity configuration and dedicated owner factory.
Read-port, source, transport, action and returned session bindings must agree;
same-sender cross-tenant/account substitutions are refused. The ledger also
refuses deployment identity rebinding and action-only contract downgrade.

This can support an isolated single-customer deployment after its operator
verifies the actual process, storage, factory and identity boundary. It does not
prove multi-tenant hosting, shared credential pools, tenant routing, shared user
identities, cross-host concurrency, HA or isolation from the host/DB owner. Do not
put multiple customer configurations/accounts into one global factory or ledger.
Tests of data bindings cannot replace OS/process/storage isolation.

## Nonsecret setup packet

`deploy/private-bridge-profile.example.json` is a validated disabled profile.
`deploy/private-bridge-staging.example.json` is a review-only layout, not an
installer or enabled service manifest. Existing `deploy/config.example.json`
continues to select staged source and disabled outbound delivery.

The concrete next input is the intended existing Zetbros host/deployment and
an authorized task environment for inspecting it. No password or private key
belongs in chat. On that host, an operator must identify:

- The intended mailbox address, approved sender/aliases and existing protected
  credential-store mechanism, without revealing the credential to the model
- Existing issuer, access-token audience, trusted public JWKS and distinct
  agent/human reviewer subject-client mappings; whether these existing tokens
  actually cover the backend API
- A dedicated per-customer service identity, approved local durable-volume and
  backup/retention locations, and the existing HTTPS/access boundary
- Whether its ledger is genuinely new or already exists; this code provides no
  safe existing-ledger migration and must never initialize over it

A private factory then supplies authenticated stdlib-shaped sessions with fixed
verified endpoints, hard creation/auth deadlines, one isolated socket per
operation, trustworthy matching identity metadata and safe cleanup. Its interface
never accepts model/user proposal credentials. Wiring that factory and enabling
a reviewed runtime are still explicit owner-controlled work; the current code
cannot activate them from configuration.

The existing Mail MCP grant cannot substitute for this factory: its tools lack
the required raw/thread state and exact-wire gate. A new grant is not inherently
necessary if the approved bridge stays inside the existing authorized private
boundary with existing credentials and identity; that must be verified on the
chosen host. New persistent access/security changes require separate approval.
After deployment readiness, live acceptance also needs one specifically approved
test recipient/workflow. Generic “approval for all” does not identify that target.

## Verification

Run `python run_private_bridge_checks.py`, then
`python -m compileall -q zetbros_service tests`.

Evidence is separate under `evidence/private-wire-bridge/`. The runner excludes
the legacy loopback HTTP fixture and records full source/runtime provenance.
It covers phase ordering, exact bytes, gate-after-354, negative responses, partial
writes/lost ack, watchdog/cancel interruption, final-250 preservation, Sent copy,
strict bindings, forged profiles and real-ledger no-retry behavior. Earlier
checkpoint evidence remains unchanged.
