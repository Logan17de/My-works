# Mail owner integration: offline exact-preview candidate

This increment inspects the supplied Mail 0.1.2 interface and integrates our own
control code with its read-only selected-mailbox shape. **Configured production
still uses `SnapshotSource` and `DisabledTransport`. No setting activates this
candidate. No Mail/SMTP/IMAP socket, login, credential, deployed identity,
Oracle configuration, external email, merge or deployment was used.** A prior combined regression run included the existing temporary loopback-only
HTTP fixture; no provider socket was used. The final runner excludes that one
fixture to keep this increment's final verification socket-free. Mail tests use
only fictional injected objects without network or login methods.

Base remote commit: `7fb50e5138fe05ad4e002eab3182f9c50be4a8c9` on
`zetbros-spacemail-contract-wip-2026-10-03`. New evidence is kept separately under
`evidence/mail-integration/`; earlier evidence remains a historical checkpoint.

## Actual interface findings

The supplied package declares Python >=3.11, version 0.1.2 and `mcp==2.2.0`.
Its tools expose one globally configured mailbox, with administrator-approved
sender aliases. Its canonical message locator is an unpadded base64url encoding
of JSON `[folder, UIDVALIDITY-as-string, UID-as-string]` prefixed by `v1_`.

`mailbox('INBOX', readonly=True)` yields an already selected imaplib-shaped
session. Our `OwnerMailSource` uses that interface with the stricter existing
raw UID contract: matching UIDVALIDITY, exact UID/size preflight, then bounded
BODY.PEEK. It maps opaque IDs to bounded `imap-v1` IDs and preserves full raw
message hash, Reply-To and thread evidence. Alias discovery cannot change the
backend's configured sender or create approval.

The existing `read_email` response omits raw bytes, Reply-To, References and
In-Reply-To, so it is insufficient as an authoritative source for this workflow.
The existing `reply_email` builds a new Message-ID, Date and MIME wire and sends
immediately without a backend claim/preview fence. **We never invoke that tool,
`send_email`, `_send`, drafts, attachments or mailbox mutations.** Client tool
approval is insufficient for the backend contract.

A pure result normalizer documents its actual accepted/rejected/unknown and
Sent-copy result shape. `all_recipients_refused` legitimately has no refusal list
in this version. Contradictory fields, partial acceptance, mismatched wire
identity or malformed results become uncertain. A failed or unknown Sent copy
never becomes permission to resend. SMTP acceptance always leaves delivery
unverified. This normalizer performs no submission.

The supplied implementation and archive artifacts are not vendored or
republished. No license was supplied; reuse/publication rights for that
implementation need clarification before distributing it. Only our own
interface adapter, ledger integration, fictional tests and documentation are
part of this commit. Private mailbox files, screenshots, host configuration,
cache contents and nested archives were not opened or copied into this work.

## Implemented backend boundary

`MailService` is an explicitly injected offline candidate using the same pinned
JWT verifier, configured agent/reviewer mapping and existing authenticated API.
It is separate from `configured_app` and cannot be selected from settings.

- `MailStore.initialize_offline` explicitly extends a genuinely new temporary
  ledger; startup never migrates an existing ledger
- A proposal stores the action, exact deterministic MIME bytes, full visible
  `WirePreview` and its digest atomically with the required proposal audit
- The human reviewer sees the action and full preview through the existing GET
  endpoint and approves a composite digest binding action digest plus preview
  digest. Neither the action-only nor preview-only digest can approve it
- Immutable preview triggers prohibit routine update/delete. Decisions,
  execution reservations and audit events use the composite review digest
- Before claim, the worker validates fresh source identity/fingerprint through
  the original atomic approval/claim logic. Required blocks do not consume it
- After claim, it re-fetches the original, re-prepares, and compares the complete
  exact preview and stored wire bytes. Changed routing/thread/raw state cannot
  reach the prepared-wire port; consumed claims remain terminal
- Pre-submit and test-port before-body gates recheck token/deadline, current
  proposer/reviewer mappings, approval expiry, action expiry, policy, quarantine,
  both digests and exact bytes without keeping a DB transaction across provider
  work. Callback success is one-shot; duplicate or failed callback attempts can
  never establish accepted submission
- Results are revalidated even if a port returns a forged Pydantic instance.
  Exceptions/timeouts and possible submissions remain uncertain and fenced;
  no automatic retry or reclaim exists
- Base action-only startup refuses a mail-review ledger, preventing contract
  downgrade and mixed action-only proposals. Explicit restore staging has a
  maintenance-only handle that cannot create/decide/claim. Backup/restore keep
  the preview and wire with the ledger and preserve restore quarantine

The prepared-wire port has **no live implementation**. Its callback ordering is
an offline orchestration contract, not proof of an actual SMTP before-DATA fence,
network deadlines, authentication or provider behavior. The owner's existing
30-second connection timeout also does not prove bounded total protocol work.

## Verification

Run `python run_mail_integration_checks.py` from this directory, then
`python -m compileall -q zetbros_service tests`.

The new tests cover canonical owner IDs, read-only UID/PEEK operations,
UIDVALIDITY rollover, size/wrong-UID refusal, sanitized failures, exact preview
persistence, reviewer role separation and composite-digest rejection, immutable
storage, atomic audit rollback, exact claim/gates, post-claim and preclaim drift,
short approval expiry, current identity/quarantine changes, missing/duplicate
gates, malformed/forged results, uncertain SMTP, independent Sent-copy failures,
idempotency/restart, missing ledger extension, downgrade refusal and quarantined
restore. All source/wire/recipient data are fictional `.invalid` fixtures.
The final socket-free runner passes 109 tests: 29 new integration tests and
80 existing control/parser/process-crash regression tests. An earlier combined
110-test run passed with the one existing loopback HTTP test; that HTTP fixture
was not rerun after the last small review fixes. Both evidence snapshots remain
separate, with source hashes. Compile checks and whitespace checks also pass.

## Later private-bridge checkpoint

Our own stdlib-shaped phase bridge now implements exact-wire SMTP sequencing,
prebody approval fencing, watchdog/cancellation and Sent-copy handling with fake
sessions. See [PRIVATE_WIRE_BRIDGE.md](PRIVATE_WIRE_BRIDGE.md). It remains dormant;
no real authenticated session factory or deployment is present. The 109-test
owner-integration evidence above is preserved as its historical checkpoint.

## Remaining live release gates

1. Owner-controlled credential/account/endpoint attestation and a secure private
   bridge for the intended single mailbox and approved sender; aliases alone
   do not prove the authenticated account. Arbitrary owner environment endpoint
   overrides must not bypass the fixed verified SpaceMail transport policy
2. A reviewed prepared-wire SMTP bridge that submits these exact stored bytes,
   honors fresh source validation and the gate immediately before body, enforces
   real per-phase deadlines and preserves acceptance/uncertainty/Sent-copy
   distinctions. The existing convenience send tools cannot be substituted
3. Existing separate agent and human reviewer subject/client mappings, approved
   issuer/audience and public JWKS, plus the real human review/client acceptance
4. An explicit reviewed deployment/data/retention/backup plan on the intended
   customer host; safe existing-ledger migration is not implemented here.
   Oracle deployment, private access and security changes remain separate gates
5. Explicit approval for one controlled live recipient/workflow and secure owner
   authentication, followed by real TLS/login/read/submission/Sent-copy/delivery
   evidence and customer release acceptance
6. Rights clarification before distributing the supplied Mail implementation

This increment closes the source-inspection gap and the offline exact-preview
integration gap. It does not close any of these live gates or claim buyer readiness.
