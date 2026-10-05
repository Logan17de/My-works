# Obtain the owner's Google identity without granting reviewer access

The standalone identity-proof app closes the initial-enrollment dependency.
The owner does not need to find or guess their Google subject. This app verifies
one Google sign-in and shows that browser its immutable `sub`. **Proof is not
permission:** it never enrolls the account, issues a reviewer/agent session,
creates/opens a ledger, constructs a Mail service/worker, reads credentials,
writes configuration or calls a mail provider. Existing reviewer allowlisting
remains unchanged and fail-closed.

All shipped proof configuration is disabled with null client/origin. The normal
pending-identity staging app remains unchanged. This release is reviewed source
and offline acceptance only; the installed pilot is not activated or changed.

## Smallest practical sequence

1. Approve a bounded identity-proof deployment window at the already selected
   HTTPS review origin. Install the new pinned code separately and start only
   `google_identity_proof:configured_identity_proof_app` with the public client ID
   and exact HTTPS origin. No unknown subject, agent issuer, mailbox identity,
   ledger, signing secret or Google client secret is needed for this step.
2. The owner opens `/identity` in their browser and deliberately uses the Google
   sign-in button with the intended account. The browser submits the ID token to
   this same approved HTTPS origin. The server checks Google signature, exact
   audience/issuer, issuance/expiry and the single-use nonce/CSRF challenge.
   Only the verified immutable subject is displayed; the ID token is not shown,
   logged, retained, exported or passed to a decoder/third-party debugging site.
3. The owner confirms that the displayed proof came from their intended account.
   Then request the specific approval to place **that verified subject and the
   dedicated reviewer client ID** in the private reviewer allowlist. A first
   proof never self-enrolls. Proofs from other Google accounts can show those
   accounts their own identity but provide no access to any application data.
4. Stop the temporary proof service/window. Configure the separate actual agent
   identity resource before a ledger-backed reviewer pilot is enabled. That is a
   later setup step, not a prerequisite for the identity proof. Keep outbound
   Mail disabled through authentication acceptance.

Google's [audience guidance](https://support.google.com/cloud/answer/15549945?hl=en)
states that Sign in with Google/basic identity-only requests are exempt from the
Testing trusted-test-user-list requirement. Do not add a test user merely to
obtain this identity proof, publish the app, or request Gmail/Drive scopes. If a
real sign-in is blocked, inspect the actual error and configured scopes/account
restrictions before requesting a specific Google setting change.
The [ID-token verification guide](https://developers.google.com/identity/gsi/web/guides/verify-google-id-token)
explains signed audience/issuer/expiry checks and the immutable subject identifier.
The browser action proves the chosen Google account, not project ownership or
reviewer permission. Email, domains and claimed roles are not authorization.

## Exact proposed hosting footprint, not deployed

- Files: `/srv/zetbros/identity-proof/releases/{approved-commit}/production`, a
  `current` link to that reviewed release, separate venv and public-only
  `/srv/zetbros/identity-proof/config/proof.json`
- Service: proposed `zetbros-identity-proof.service`, using an isolated dynamic
  non-login `zetbros-proof` service identity; no persistent account/password is
  created, and known pilot/credential paths are explicitly inaccessible. Preflight
  must verify that no static user/group already owns this proposed name, because
  systemd would otherwise reuse it
- Listener: `127.0.0.1:8082`, one worker; no public port or new DNS/TLS service
- Proxy: only exact `/identity` and `/identity/*` locations in the approved HTTPS
  server point to 8082. The holding default, `/review`, `/v1`, installed pilot at
  8081 and existing Mail service/tunnel remain untouched
- Budget: MemoryHigh 64 MiB, MemoryMax 80 MiB, no swap, CPUQuota 25%, TasksMax 16,
  four HTTP requests and two blocking route threads. A 600-second app window and
  systemd RuntimeMaxSec enforce closure; Restart=no prevents reopening. No writable ledger path or
  LoadCredential. This is a proposed additional temporary process, so verify
  current host free memory and the combined existing-service budget before start
- Public profile: `runtime_state: identity_proof_only`, `enabled: true`, actual
  public web client ID, exact canonical HTTPS origin. No subject/role/agent/mail/
  database fields are accepted. The shipped disabled example cannot start it

The [systemd 255 execution documentation](https://github.com/systemd/systemd/blob/v255/man/systemd.exec.xml)
describes dynamic UID allocation; its [service documentation](https://github.com/systemd/systemd/blob/v255/man/systemd.service.xml)
describes the runtime limit. Target unit/sandbox acceptance remains unrun.

`deploy/zetbros-identity-proof.service.example` and
`deploy/identity-proof.proxy-location.example` are staging proposals only. The
proxy fragment is not a complete host configuration: use verified existing
rate/connection limits, overwrite forwarding headers, trust only the loopback
proxy and test canonical scheme/host/client-IP behavior. Request bodies and
proof subjects must not enter logs. Bound header/read/body/time/resource limits
and a short owner-attended window are required before public exposure. Neither
application challenge bounds nor a proxy fragment promises DDoS resistance.

The app accepts no bearer authority or query parameters. Its challenge cookie
is distinct, Secure/HttpOnly/SameSite=Lax and scoped to `/identity`, disappears
on proof/cancel, and is not an authenticated session. Challenges expire within
two minutes by default; replacement, successful/failed verification and cancel
consume them. Verification rechecks expiry after a slow key check. Restart
forgets outstanding challenges. Pools and peer bootstrap requests are bounded;
raw forwarding headers cannot bypass the app's transport-peer throttle.

The UI safely renders only the verified subject and `authority: none`. Cancel,
restart, late bootstrap/GIS/proof responses and inconclusive responses must not
render stale proof or silently retry token submission. A completed request
cannot be un-sent; cancelling drops its display and remaining challenge, with
no granted authority to revoke. No recovery of a subject from logs/raw tokens
is offered.

## Exact approvals and remaining acceptance

Offline implementation/review/checkpoint does not perform these actions:

1. Approve installing the identified proof-only release, public profile and
   separate temporary unit/layout and dynamic-user isolation, the
   narrowly scoped proxy change and start/stop window under the stated budget
2. Owner performs Google sign-in/consent in their browser for identity data to
   Google and the exact approved review origin; no model handles raw credentials
3. Approve the resulting exact subject/client reviewer allowlist grant after
   confirming the intended account. This is separate from proof capture
4. Choose an existing agent issuer/API audience/client/subject and verified public
   JWKS, or approve creation of a suitably scoped separate agent credential.
   Google reviewer ID tokens cannot replace the API's agent access credentials

A bounded cloud sample of mocked identity proof and four concurrent shell/health
callers peaked at 50.5 MiB RSS and ten sampled threads, below the proposed 64/80-MiB
and 16-task limits. It is not a target-host or worst-case resource ceiling.

Real Google key fetching/login, HTTPS cookies/CSP/popup/FedCM/proxy/browser pixels,
server unit/resource behavior and host deployment remain unrun. The normal Mail
credential and controlled-send approvals remain later; identity proof cannot
activate them. Review the [existing login boundary](GOOGLE_REVIEWER_LOGIN.md) and
[disabled staging workflow](DISABLED_STAGING.md) when moving to an approved pilot.
