# SpaceMail adapter contract: dormant provider implementation

This addition is **not live verified and not buyer ready**. It supplies reusable
SpaceMail parsing, exact reply preparation and SMTP result handling. It does not
open connections, authenticate, read credentials, send email, install services,
change Oracle configuration or activate delivery. Production still constructs
`DisabledTransport`; no configuration value can enable this module.

Base: `Logan17de/My-works` commit
`d19b6d4228e295296ae043a4efc47845cb50daa8`. All 23 base files, including its
reviewed authentication, approval ledger, runtime and previous evidence, remain
byte for byte unchanged. The new source is `zetbros_service/spacemail_contract.py`.
The additional tests and evidence describe a separate dormant candidate.

## Provider compatibility and implemented behavior

SpaceMail's [official client configuration](https://www.spaceship.com/knowledgebase/connect-spacemail-to-email-client/)
supports `mail.spacemail.com`, IMAP 993 and SMTP 465 with implicit TLS. The
provider describes mailbox password authentication. This implementation contains
no password input, discovery or storage. It makes no claim about provider OAuth.

`SpaceMailEndpoints` fixes the host, ports, verified implicit TLS, one INBOX and
bounded operation timeout. `verified_tls_context` builds a CA and hostname
verifying context with TLS 1.2 minimum; it opens no socket. Actual TLS negotiation,
certificate validation against SpaceMail and mailbox authentication are UNRUN.

`ImapReadPlan` accepts only `imap-v1:<UIDVALIDITY>:<UID>`, with both values in
the positive 32-bit range. IDs cannot select a folder, arbitrary path, sequence
number, message range or provider URL. A bridge must use this sequence:

1. Open an independently bounded, authenticated IMAP session for the deployment's
   account using verified implicit TLS at the fixed provider endpoint
2. EXAMINE INBOX, or the equivalent `select('INBOX', readonly=True)`
3. Verify exactly one matching UIDVALIDITY value
4. UID FETCH the requested UID with `(UID RFC822.SIZE)`
5. Refuse missing, ambiguous or larger than 65,536-byte messages before fetching
6. UID FETCH with `(UID BODY.PEEK[]<0.65537>)`; never use ordinary BODY or RFC822
7. Validate the returned UID, single literal, declared length and complete
   RFC822.SIZE match, then parse the original bytes

There are no STORE, Seen, MOVE, DELETE, APPEND, mailbox creation or reply-all
operations in the read plan. `parse_size_response` and `parse_body_response`
validate the corresponding imaplib-shaped responses. Sequence-number changes
have no identity meaning; UIDVALIDITY rollover refuses the old locator.

`parse_source` accepts one bounded plain-text message with an ASCII header block
(UTF-8 encoded-word subjects are supported) and UTF-8 or US-ASCII text content.
It supports strict 7bit, 8bit, base64 and quoted-printable decoding. Multipart,
HTML, attachments, ambiguous routing/thread headers, groups/multiple From or
Reply-To addresses, unsupported charsets, malformed encodings and unbounded
references are refused. A missing original Message-ID is refused in this narrow
workflow. Broader MIME handling requires a separate implementation and review.

`SpaceMailRecord` extends the existing source contract with the fixed provider,
INBOX, UIDVALIDITY, UID, Message-ID and References. The version is a SHA-256 of
the entire original RFC822 record. Its canonical source fingerprint additionally
binds the deployment/account identity and visible threading metadata. Mail text
never grants approval or changes policy.

`prepare_reply` revalidates a `ReplyAction` against a freshly parsed source and
the fixed deployment binding. Sender, single Reply-To-or-From destination,
subject, source version/fingerprint, tenant/account/connector, policy and expiry
must match. Arbitrary destination, CC, body controls, self reply and altered
bindings are refused. It emits deterministic MIME bytes containing exactly the
action's From, To, subject and UTF-8 body, plus source-bound In-Reply-To and
References. Base64 preserves the approved body bytes, including Unicode and
newline choices. Message-ID derives from the full action digest; Date derives
from its creation time. It emits no attachment, CC or BCC. Wire size and line
length are bounded.

The visible `WirePreview` includes sender, destination, subject, exact body,
Message-ID, In-Reply-To, References, Date, source version/fingerprint,
action digest and final wire SHA-256. `PreparedReply.preview_digest` binds every
preview field. Changing a visible thread header or wire hash changes this digest.

`SmtpOutcome` reduces one transaction's results without I/O. MAIL 250, RCPT
250/251 and DATA-command 354 permit advancing, but do not establish submission.
Only a positive final DATA 250 establishes `accepted`. Explicit negative SMTP
responses establish `rejected`; invalid/duplicate/out-of-order events cannot
establish acceptance. Once the body may have started, timeout, cancellation or
exception leaves `uncertain`. Before any body starts, interruption means known
non-submission. Terminal results cannot be reopened for retry. Every result says
delivery `unverified`. A failed or uncertain Sent copy preserves accepted
submission, and cannot trigger a resend. Raw provider strings are not accepted
into result fields or logs.

## Required live bridge and approval integration

The existing Mail connector's owning source remains unavailable. This addition
does not duplicate or replace that server and does not bypass its denied source
bundle. There is no production adapter factory, remote MCP gateway or send API.

Before activating any live bridge, the owning connector must be inspected and
the following integration implemented and reviewed:

- Map its authenticated, account-bound source reads into this bounded protocol
  contract; enforce the actual fixed endpoint, CA/hostname verification and
  deadlines at socket/read/write boundaries. A timeout parameter or async wait
  alone does not bound blocking synchronous network work
- Keep provider authentication inside the connector's protected credential
  boundary. Do not accept mailbox credentials or provider tokens in proposals,
  approval requests, URLs, audit events or agent tools. Existing credential
  configuration needs the owner's secure, explicitly approved route
- Extend the authenticated human preview and durable immutable payload to
  include `WirePreview` and its digest. The original service currently approves
  the existing action digest; the new preview is not wired into its public API
  or database. The fictional bridge tests do not claim otherwise
- Persist both action and wire-preview digests with reviewer identity, decision,
  source binding and expiry. Reserve their approved operation and required audit
  atomically before network using the existing no-retry claim semantics
- Re-fetch authoritative source inside the reserved worker before SMTP; refuse
  changed UIDVALIDITY, UID, raw message, recipient or thread headers. Rebuild the
  preview and compare its exact approved digest and bytes
- Recheck current approval expiry, revocation, policy and claim fencing immediately
  before DATA/body. Do not hold the database transaction across network I/O. The
  current fixed action expiry alone is insufficient for the full live integration
- Bound every protocol phase and gracefully stop before new claims. Never retry
  a claimed or potentially submitted action on connection loss, worker restart,
  restore or missing final acknowledgment. A confirmed accepted transaction and
  Sent-copy storage are independent outcomes
- Preserve the operation ledger, restore quarantine, source retention and
  sanitized append-only audit. Restrict full mail and wire preview access to the
  deployment's authorized principals; never copy actual messages into public tests

The minimum owner inputs remain: the existing Mail connector's repository/path
or an authorized Oracle task environment; the intended mailbox/approved sender
and actual deployment path/availability policy; existing separate agent and human
review identity mappings; and one explicitly approved recipient/workflow for a
controlled delivery acceptance test. No private key or password belongs in chat
or the public repository. Oracle private-key handoff/access is not inferred from
this local work.

## Verification and release gates

Run from this production directory:

`python run_spacemail_checks.py`

The runner leaves previous base evidence unchanged. It records the combined
original and new tests in `evidence/spacemail/test-results.json`, along with file
hashes, source revision and the original 23 Git blob identities. Test-only IMAP
and SMTP sessions contain fictional messages and no socket/login methods.

PASS, subject to the recorded candidate revision: strict provider/parser/wire
primitives; controlled protocol result faults; actual durable-worker approval,
claim-before-protocol, source drift, idempotency/restart and uncertainty tests;
unchanged configured production delivery disabled; original suite regression.

BLOCKED: owning Mail source integration; authoritative live-source bridge;
human wire-preview/digest persistence; existing deployed identity mapping;
approved Oracle/local-volume deployment; authorized delivery acceptance.

UNRUN: SpaceMail authentication/TLS/network behavior, real IMAP read and SMTP
submission, real Sent-copy behavior, delivery, deployed host lifecycle, live
identity provider and buyer acceptance. Fictional protocol success proves none
of those provider/deployment outcomes.

Protocol references: [Python IMAP client](https://docs.python.org/3.12/library/imaplib.html),
[Python SMTP client](https://docs.python.org/3.12/library/smtplib.html),
[SMTP transaction semantics](https://www.rfc-editor.org/rfc/rfc5321.html),
[reply message threading](https://www.rfc-editor.org/rfc/rfc5322.html).
