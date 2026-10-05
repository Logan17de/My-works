# Dormant Google reviewer sign-in and separate agent credentials

This release adds our own Google Identity Services (GIS) button and dedicated
ID-token login exchange to the exact-wire human review surface. It does not
provision a Google client, grant access, log in, connect to Google/Mail, deploy,
or send a message. The shipped pending-identity install still has no ledger,
authentication service, provider factory or worker. The new example
`deploy/google-reviewer.disabled.example.json` has `enabled: false`, null client
and origin, and an empty subject list; it cannot enable sign-in.

## Exact owner configuration needed

1. Choose the exact canonical HTTPS reviewer origin and its approved deployment
   boundary. Configure a dedicated Google **Web application** client with that
   authorized JavaScript origin. This implementation uses a JavaScript callback,
   not an OAuth code/redirect exchange. Configure required branding/audience/test
   users in the selected owner project. Identity-only sign-in requires no Gmail
   or Drive scope. Project IDs/numbers are not OAuth client IDs.
2. Supply the resulting public web client ID and the intended reviewers' verified
   immutable Google `sub` values. Authorization never uses an email address,
   project owner, email domain, `email_verified`, scopes or caller metadata.
   A first Google sign-in cannot self-enroll a reviewer. Obtaining/verifying the
   owner's actual subject and creating/configuring the client remain separately
   approved setup actions; no default or guessed identity is supplied here.
3. Identify the separate agent access-token issuer/resource and existing client/
   subject mapping. Provide its verified **public** JWKS file, HTTPS issuer,
   exact API audience and allowed agent pairs. The issuer cannot be Google's
   login issuer and the audience/client cannot be the reviewer web client ID.
   If suitable existing agent credentials do not exist, creating them requires
   a separately approved access setup. A GIS ID token is not an agent credential.
4. In a private, owner-controlled configuration, select `reviewer_auth:
   google_oidc`, set `google_reviewer_profile_file` to an absolute regular JSON
   file, and make `review_origin` match its origin exactly. The enabled profile
   must contain the actual client ID and 1–32 actual reviewer subjects. Its
   `reviewer_subjects` must exactly match `principals` entries with role reviewer
   and that client ID; agent client IDs must remain disjoint. Keep outbound
   delivery disabled during authentication acceptance.

The other mailbox/tenant/policy/public-agent-identity fields must already be real
and consistent with the private Mail profile before a new ledger is initialized.
Follow [DISABLED_STAGING.md](DISABLED_STAGING.md). A missing/disabled/mismatched
Google profile fails **before** new-ledger creation. Existing paths are refused;
there is no migration or manual live-flag/schema bypass. This release needs no
Google client secret, API access/refresh token, reviewer session signing key,
Supabase service role or password. Never put credentials in chat/public config.

Google's [setup guide](https://developers.google.com/identity/gsi/web/guides/get-google-api-clientid)
describes the web client, authorized origins, branding and GIS CSP directives.
The [server verification guide](https://developers.google.com/identity/gsi/web/guides/verify-google-id-token)
describes signature, audience, issuer, expiry and immutable subject validation;
the [nonce reference](https://developers.google.com/identity/gsi/web/reference/html-reference#data-nonce)
describes replay protection. Our strict checks and server-side mapping below are
additional implementation requirements, not Google account grants.

## Runtime and endpoint boundary

- `/review/auth/bootstrap` issues a bounded single-use challenge with random
  nonce/login-CSRF and a Secure, HttpOnly, SameSite=Lax `/review` challenge cookie.
  The GIS library loads from its official URL only when Google mode is selected.
- The GIS callback posts its ID token **only** to `/review/auth/google`, with the
  matching challenge cookie, login-CSRF and exact origin. The verifier accepts
  RS256 public Google JWKS from one fixed HTTPS URL, exact client audience,
  Google's documented issuer, current expiry/issuance, matching nonce and an
  allowlisted subject. Token-controlled key URLs and other algorithms fail.
  Public keys cache for at most five minutes; unknown keys fail closed.
- The exchange returns an opaque, hashed-in-memory reviewer session cookie with
  a maximum five-minute lifetime. It does not retain the Google ID token or
  return API tokens. Sessions are scoped to one mapped reviewer and disappear
  on restart; new login replaces that reviewer's older session. Logout requires
  session-CSRF, clears it and provides no auto-login behavior.
- `/review/auth/session` restores only that short reviewer session. Review reads
  require the cookie; decisions, revoke and reconciliation also require exact
  origin and one `X-Review-CSRF`. Query parameters and bearer authority are
  forbidden throughout the Google `/review` surface.
- The existing own-audience RS256 agent bearer interface is wrapped agent-only.
  Google ID tokens and cookies never authenticate `/v1`. Agent credentials cannot
  access any Google `/review`, `/v1/reviews/*`, or `/v1/audit` path. Trusted roles
  come only from the server's subject/client map.

The UI displays the immutable full exact-wire preview and composite digest.
Approval/rejection requires that digest and a still-pending proposal. Repeated,
stale, interrupted or inconclusive responses never silently retry a decision.
Cancel/logout, late sign-in callbacks/cookies and back-forward races have offline
state coverage. The existing source drift/UIDVALIDITY fences, one-time claim,
pre-DATA exact-byte approval, SMTP uncertainty and Sent-copy distinctions remain.

Cookie-authenticated operator endpoints preserve audit/revoke/reconciliation:
`POST /review/api/proposals/{id}/revoke`,
`POST /review/api/proposals/{id}/reconciliation`,
`GET /review/api/proposals/{id}/reconciliations`, and
`GET /review/api/audit`. Later audit pages use the nonsecret sequence path
`GET /review/api/audit/after/{sequence}` (100 events per page, integer 0..2^63−1).
Mutation bodies retain the existing strict digest/reconciliation schemas.
These operator APIs are available to the authenticated reviewer; the minimal
visual page presently offers preview/approve/reject. Reconciliation records an
observation and never reopens uncertainty or retries a send.

## Public-origin and resource acceptance still required

This remains one customer, mailbox/account, process/worker and local ledger.
There is no shared multi-tenant session store or distributed service profile.
The challenge/session pools are bounded. Bootstrap replaces the browser's prior
challenge and is limited to eight requests per minute per transport peer by
default (maximum 128 active peer buckets). Raw `X-Forwarded-For` is not trusted.
A distributed attacker can still exhaust the small challenge pool or resource
budget: before public activation the approved proxy must supply trustworthy
client-IP/scheme/host handling and enforce verified global/per-IP request and
body/time limits. An unconfigured proxy can collapse all users into one peer or
accept spoofed forwarding headers; application bounds alone are no DDoS claim.

The Google key-fetch transport has fixed DNS/TLS host, bounded wait/response and
no redirects, authorization header or provider credentials. Its actual network
behavior has not been exercised. Mock tests use fictional keys/tokens/DNS/TLS;
no provider/listening sockets, real tokens, credentials or login were used.
ASGI and Node state checks are not browser pixels or actual Google/FedCM/popup
compatibility. Actual HTTPS/cookie/CSP/COOP/referrer/proxy behavior must be checked
in the approved browser/deployment, within the proposed 96/128-MiB and 32-task
budget, before owner-approved login activation. A fresh cloud sample of mocked login
and eight concurrent cookie-review/health callers peaked at 54.8 MiB RSS and
16 sampled threads; this is not a target-host or worst-case ceiling. Evidence
is in `evidence/google-reviewer-auth/memory-sample.json`. No currently installed service
was changed by this source checkpoint, and mail delivery stays disabled.
