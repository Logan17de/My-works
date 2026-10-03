# Release-candidate verification

Prepared 3 October 2026 UTC. The exact test names, source SHA-256 inventory,
content revision, dependency versions and SQLite/Python runtime are in
[test-results.json](test-results.json). Run `python3 run_checks.py` to regenerate
that evidence against the checkout. Existing package base is My-works draft PR 4
at `7b207534173da9ed7a46ca3ce894b84b7adb3569`.

## PASS

- 55 control/runtime cases passed, 0 failed, 0 errors, 0 skipped
- Compilation and installed-package consistency checks passed
- Actual uvicorn loopback HTTP workflow with externally signed test tokens,
  strict proposal, exact reviewer preview/digest, persistent disabled-delivery
  result, graceful application shutdown and restart
- Four independent OS processes competed for the same queued operation; exactly
  one durable claim committed and three were refused
- Abrupt OS child exits after committed claim and after a possible-submission
  marker recovered to uncertainty, with no automatic invocation/reclaim
- WAL plus synchronous=FULL, migration 1, immutable deployment identity,
  operation-key constraints, required decision/claim/result audit transactions,
  restart/replay preservation and late-result fencing
- Required preflight/audit/storage failure, expiry, policy/source drift,
  revocation and disabled/missing-source conditions made zero adapter calls
- Post-invocation result/audit failure retained consumed claim and recovered to
  uncertainty without retry
- Submission/Sent-copy/delivery distinction, exceptions, cancellation, timeout,
  no automatic resend, and newly approved intentional second operation
- Pinned RSA public JWKS, issuer/audience/time/subject/client/type validation,
  unknown kid/algorithm/remote-key refusal, server-configured role binding,
  spoofed reviewer role/body/header refusal and reviewer-only audit access
- Strict schemas, unknown-field/type/header/path/size checks, bounded body read,
  duplicate JSON/NaN refusal, no browser Origin or URL tokens
- Clock boundary regressions cross actual SQLite transaction contention; stale
  approval/claim expiry and late-result times are refused after the lock
- Nonblocking FIFO source refusal executes in a bounded child process
- Missing/deleted/uninitialized ledgers and metadata fail closed on startup and
  per-request reconnects; fresh initialization is explicit and exclusive
- Backup/restore preserves operation history and identity, marks possible
  submissions uncertain and enters permanent send quarantine. An actual process
  exit immediately after copying backup leaves the final path absent; startup
  and fresh init refuse that interrupted restore. Atomic publication refuses a
  racing destination without overwriting it
- Reconciliation observations/evidence references are immutable append-only rows
- Minimized audit omits exact fictional bodies, subjects, addresses, credentials
  and raw adapter errors; append-only application triggers refuse routine edits

All message/transport tests are fictional. The HTTP runtime uses disabled
transport. Ephemeral test signing keys exist only in test memory; only their
public JWKS is written to temporary test storage, then removed. The test runner
does not create a customer credential or persistent access.

## BLOCKED

- Buyer live-delivery release: actual owning Mail source and adapter unavailable
- Authoritative original-message revalidation and live transport contract tests
- Existing customer identity-provider/client mapping and approved deployment
- Customer agreement on actual workflow, data processing, named review owner,
  payload/audit/backup retention, incident handling and release acceptance

## UNRUN

- Real SMTP/IMAP/provider connectivity, actual acceptance/delivery and Sent copy
- Real model, remote MCP, customer AI client and human-identity provider flows
- Customer host installation, TLS/reverse-proxy, secrets/permissions/encryption
- Container/systemd installation or service hardening on a customer host
- Real power-loss, filesystem hardware durability, multi-host/HA/network-filesystem
- Dependency vulnerability clearance and external penetration/security assessment
- Browser review UI (none is provided)

No public deployment, live mail, demo outreach, account/credential creation,
persistent-access changes, purchases or merge was performed. Passing controls
make this a reviewable release-candidate service, not buyer production acceptance,
a delivery guarantee, security certification or SLA.
