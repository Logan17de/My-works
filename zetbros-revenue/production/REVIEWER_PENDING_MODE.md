# Reviewer login before workflow configuration

The existing pilot entrypoint supports `runtime_state:
reviewer_enabled_workflow_pending`. This is an explicit owner-selected mode,
not the shipped pending-identity default. It reuses the existing Google reviewer
profile, cryptographic verifier, short opaque cookies, login/logout endpoints,
and review shell. Enrollment requires an approved exact Google subject/client
mapping; email addresses never authorize access.

Its private configuration contains only the runtime state, HTTPS review origin,
absolute `google_reviewer_profile_file`, reviewer-only `principals`,
`reviewer_auth: google_oidc`, and `outbound_adapter: disabled`. The enabled Google
profile must match the origin and every reviewer subject/client pair exactly.
Agent issuer, audience, JWKS, Mail profile, and database fields are rejected.

The mode constructs no agent verifier, MailService, provider factory, worker or
ledger. `/review` supports fresh Google login, session checks and logout.
Proposal controls are hidden and disabled; all workflow API reads and mutations,
including approve, revoke and reconciliation, return an unavailable response.
The agent API is disabled even for an authenticated reviewer. Health distinguishes
reviewer-login availability from workflow readiness. This is reviewer enrollment
and authentication readiness, not an operational action-review workflow.

Use the same dedicated pilot service and approved resource envelope. Do not
activate delivery or add credentials to enable this mode. Preserve an existing
proof service and unrelated host services when changing the review proxy.

Later, after real full Settings, separate agent identity and mailbox/policy
configuration are approved, initialize a genuinely new ledger and restart the
same pilot in its supported full configuration. Preserve the approved Google
client, subject and origin. Process-local sessions are discarded at restart;
the user must sign in again. Cookies cannot carry authority across the transition.
Never migrate, rebind or initialize over an existing ledger. This mode provides
no automatic transition or configuration mutation endpoint.
