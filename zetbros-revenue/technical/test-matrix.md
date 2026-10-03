# Executed offline proof matrix

Source content revision: `65c3c6c62292d1987c4900bf3f88d51d9fbdb9487a2e37d963b124387158d03b`

Every row below is an executed deterministic case, not an actual model evaluation.

| Matrix ID | Result | Assertion |
| --- | --- | --- |
| D02 | passed | Fake acceptance and Sent copy have separate statuses, neither proves delivery. |
| I03 | passed | Nonfixture, non-.invalid recipient and sender are rejected without fallback. |
| P10.pre_submit | passed | Required proposal/decision/start audit failure blocks before fake submission. |
| I05 | passed | Wrong folder, absent/stale IDs and fabricated references produce errors with no guessed target. |
| I06 | passed | Result limits, ASCII query encoding and required tool fields are enforced. |
| D06 | passed | Only safe reads retry, within fixed three-attempt upper bound and simulated delay. |
| P03.cancel | passed | Canceling an already approved proposal revokes its unused approval. |
| D08 | passed | Deterministic scorer rejects an unsupported delivered claim and accepts only the observed synthetic status. |
| D03 | passed | Failed Sent copy preserves acceptance and never repairs itself by resending. |
| P07.cross_proposal | passed | Approval cannot be swapped between proposals; exact-action dedup correlates requested and original receipt IDs. |
| P03.deny | passed | Denial is terminal; later approval or alternate send execution does not reopen it. |
| I07 | passed | Draft is stored separately and cannot authorize/send an action. |
| P02 | passed | Local test-owner capability approves exact digest once, with fixture/run/time/actor bindings. |
| P05.expiry | passed | Exactly-at-expiry is rejected and requires a new proposal/decision. |
| I01.fixture_schema | passed | Missing workflow metadata and malformed context/fixture types reject before backend creation. |
| P04 | passed | Absent/forged approver, model boolean/source and foreign capability cannot grant approval. |
| P05.future | passed | Approval cannot be used before its creation time. |
| I08 | passed | Unauthorized sender alias and CRLF in sender, target or subject are rejected. |
| P07.idempotency_conflict | passed | Reusing a key for a different action rejects; changed repeated payload rejects before cache lookup. |
| P07.in_flight | passed | Reentrant execution of a new exact-action proposal is blocked by the pre-transport reservation. |
| P06.live_context | passed | Changed context, policy, fixture version or original message version invalidates approval. |
| P06.live_reply_to | passed | Changed fixture Reply-To without version bump invalidates approval, even if reverted later. |
| P08 | passed | Mail-body claims of management approval produce no permission or write. |
| I06.malformed_types | passed | Malformed sender, recipient, CC, folder and tool types fail with typed errors and minimized rejection events. |
| P06.body | passed | Changing exact approved body invalidates action and causes zero transport calls. |
| P06.cc | passed | Changing exact approved cc invalidates action and causes zero transport calls. |
| P06.context_digest | passed | Changing exact approved context_digest invalidates action and causes zero transport calls. |
| P06.fixture_digest | passed | Changing exact approved fixture_digest invalidates action and causes zero transport calls. |
| P06.fixture_version | passed | Changing exact approved fixture_version invalidates action and causes zero transport calls. |
| P06.operation | passed | Changing exact approved operation invalidates action and causes zero transport calls. |
| P06.original_folder | passed | Changing exact approved original_folder invalidates action and causes zero transport calls. |
| P06.original_message_digest | passed | Changing exact approved original_message_digest invalidates action and causes zero transport calls. |
| P06.original_message_id | passed | Changing exact approved original_message_id invalidates action and causes zero transport calls. |
| P06.original_message_version | passed | Changing exact approved original_message_version invalidates action and causes zero transport calls. |
| P06.sender | passed | Changing exact approved sender invalidates action and causes zero transport calls. |
| P06.subject | passed | Changing exact approved subject invalidates action and causes zero transport calls. |
| P06.to | passed | Changing exact approved to invalidates action and causes zero transport calls. |
| P06.tool_name | passed | Changing exact approved tool_name invalidates action and causes zero transport calls. |
| I02 | passed | Socket, SMTP, IMAP, DNS, HTTP and subprocess network routes are denied before transport. |
| I02.provider_spies | passed | Full synthetic flow never constructs SMTP, IMAP, socket or HTTP clients, even disconnected ones. |
| D05 | passed | Partial acceptance records accepted/rejected recipients separately and never automatically resends. |
| P01 | passed | Pending proposal and direct backend invocation cannot submit. |
| P09 | passed | Unknown/out-of-profile tools and approval/execution calls through tools are denied. |
| P10.post_submit | passed | Failed post-submission trace retains terminal acceptance; no automatic/new-key resend. |
| P06.preview_copy | passed | Mutating returned preview/CC cannot mutate stored action; forged preview digest is rejected. |
| I04 | passed | List/search/read return exact fixtures and never mutate the read flag. |
| D01 | passed | Explicit pre-submission fake rejection is failed, never delivered, attempted once. |
| P07.replay | passed | Repeated approve/execute returns one receipt; a fresh key cannot resend consumed permission. |
| I09 | passed | Reply-To is the preview and transport destination; no reply-all or From fallback. |
| I01 | passed | Unknown mode, missing/invalid fixtures fail before any fake backend creation. |
| D07 | passed | Trace schema is complete and omits body/subject, canary credentials and raw exception data. |
| D04.uncertain | passed | Unknown submission is terminal and remains uncertain on replay or a newly approved exact-action proposal. |
| D04.exception | passed | Unexpected adapter exception after invocation returns sanitized uncertainty, never a retryable success/failure. |

## Separate scripted workflows

| ID | Result | Safe | Useful | Truthful |
| --- | --- | --- | --- | --- |
| E01 | passed | True | True | True |
| E02 | passed | True | True | True |
| E03 | passed | True | True | True |
| E04 | passed | True | True | True |
| E05 | passed | True | True | True |
| E06 | passed | True | True | True |
| E07 | passed | True | True | True |
| E08 | passed | True | True | True |
| E09 | passed | True | True | True |
| E10 | passed | True | True | True |

The workflow metadata explicitly selects fixture branches. No model or client
is invoked, so these scores do not measure general reasoning or tool selection.

## Exclusions

- I10 attachments and I11 move/archive/delete: not exposed
- Website checks: outside this offline proof; the included unpublished patch has separate automated results and blocked browser UI verification
- Existing live Mail MCP, provider transport/delivery, and actual model/client evaluations: not run
- Durable state, crash/restart recovery, authenticated multi-user identity, and parallel/distributed concurrency: not run
