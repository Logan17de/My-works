# Supported disabled staging walkthrough

The deployment-blocking initialization defect is fixed. The default pilot can
now be installed and started while owner identity configuration is pending,
without a ledger, Mail credential dependency or live adapter flags. It does not
invent temporary identities or ask the user to implement a workaround.

## Phase 1: pending identity, no ledger

Install the reviewed code and copy `deploy/pilot-runtime.disabled.example.json`
to the approved service configuration path. This schema contains only the
explicit `disabled_pending_identity` state, disabled adapters and proposed
ledger location. Issuer, JWKS, principals, mailbox and tenant fields are absent
and are rejected if added to this pending schema.

The default `deploy/zetbros-pilot.service.example` has no LoadCredential entry.
It starts the health-only staging app at loopback 8081 with the proposed resource
limits. Installing that unit/account/layout remains a separately authorized
host action; no host change was performed for this fix.

With the configuration file selected by ZETBROS_CONFIG_FILE:

1. `python -m zetbros_service.pilot init` succeeds with `ledger_created: false`.
   It validates staging only and creates no directory, database or identity pin.
   An existing or dangling-symlink ledger path is refused without modification
2. `python -m zetbros_service.pilot check` succeeds with identity pending,
   delivery disabled, no open ledger and control/live readiness false
3. Start `zetbros_service.pilot_runtime:configured_pilot_app` through the proposed
   unit. GET /healthz returns 200. GET /readyz returns 503 with the explicit
   pending-identity reason. This 503 is the correct readiness result, not a
   failed install. Review, proposal and mutation routes return 503 and perform
   no action; no reviewer token is requested in this stage

The staging app constructs no verifier, source, factory, worker or store. The
legacy maintenance CLI also rejects this pending configuration before any
init/restore/backup operation, so it cannot create an accidental empty ledger.

## Phase 2: actual identity configured, still disabled

Only after the owner supplies the actual existing issuer/audience/public JWKS,
separate agent/reviewer mappings, account/sender and exact review origin, prepare
normal Settings with the private mail source contract and outbound still
`disabled`. The matching private profile must also have outbound `disabled`.
No live flags or manual schema changes are required.

`pilot init` first rejects obvious REPLACE/example-domain placeholders, verifies
the public JWKS/principal configuration and matching profile, and then creates
only an absent new exact-wire MailStore using O_EXCL. It never constructs a live
session factory or reads credentials. Missing/private JWKS, mismatched profiles,
invalid identities and existing paths fail before initialization.

`pilot check` and startup now use that same MailStore, rather than falling back
to the action-only service. Private source reads remain unavailable and delivery
remains disabled; no credential or provider operation occurs. Identity cannot
be rebound. Existing-ledger migration is not offered.

## Phase 3: separately approved live setup

After specific credential/access and live-workflow approvals, the operator may
use the separate `deploy/zetbros-pilot.live.service.example` and matching explicit
live configuration. The same configured identity and exact-wire ledger are
preserved. Switching to live is not another initialization or migration and
cannot change the identity pin. The current fix does not enable or deploy it.

## Verification

The exact shipped pending config -> init -> check -> ASGI startup walkthrough is
covered in `tests/test_disabled_pilot_walkthrough.py`, including maintenance
entrypoints, existing paths/symlinks, disabled profile/JWKS validation, no-I/O
startup and identity-preserving later activation with fake clients only.
`measure_disabled_staging_memory.py` exercises the same pending init/check/app
path with eight concurrent in-process callers. It creates no ledger and reads
no identity or credential. The cloud sample is not target-host/cgroup acceptance.

The least-risk pending install deliberately has no ledger. Actual identity
values are a real setup requirement before permanent pinning; placeholder
identities are not a supported substitute. No provider connections, real
credentials, remote changes or sends were performed for this correction.
