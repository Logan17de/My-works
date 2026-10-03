# Zetbros first customer commercial pack

Prepared 3 October 2026. This review draft defines a bounded service offer, discovery process, proposal, delivery checks, and pricing model for a first paid customer. All prices and terms are hypotheses for review, not accepted customer terms or evidence of sales. Publication of this work package does not establish a customer agreement or a website launch.

## Decisions to make before selling

1. Lead with a USD 300 Integration Readiness Audit. It can conclude that the integration should not be built. A paid audit establishes a paying relationship; it does not establish successful implementation or recurring demand.
2. Quote a USD 1,500 Agent Integration Pilot only after confirming a small, feasible scope. This price is an economic hypothesis, not validated willingness to pay. A USD 1,250 introductory price needs an explicit decision to accept lower contribution. Retire the broad USD 950 production promise.
3. Limit the pilot to one documented API, one existing customer account or tenant, one named operator, one agreed AI client, and up to three read-only tools. Measure one useful workflow, such as looking up open service requests and preparing a source-linked status brief. The brief stays in the selected client and is reviewed by the operator.
4. Preserve the approval and audit differentiator through an internal synthetic write-control test using fake transport. Real writes, production authorization work, and ongoing operation require separate qualification and scope.
5. Qualify through written scope and discovery. The synthetic proof is not a customer deployment, and no prospect demo or outreach has been performed as part of this work package.

## What changes from the original plan

The plan's one-system scope still bundles potentially independent projects: authentication, five tools, server-enforced approvals, logs, deployment, evaluation, and production readiness. The tool count does not bound OAuth complexity, customer permissions, API quality, or the consequences of a mistaken action. A working email send establishes connectivity; it does not establish tenant isolation, approval enforcement, safe retry behavior, or operational readiness.

The proposed first pilot therefore delivers a testable lookup outcome with explicit access limits. The next paid scope can add a single controlled write after its risks and approval mechanism are understood. A reusable gateway, shared dashboard, multi-client compatibility, and a connector catalog remain outside the first-customer work.

Technical status: the independently implemented offline harness in ../proof has recorded tests. It does not validate any existing live Mail MCP implementation, authenticated approval service, or production audit system. A tool description requiring permission is not evidence that the server enforces it.

# Offer and qualification

## Proposed customer facing offer

### Zetbros Integration Readiness Audit

We map one repetitive workflow and check whether a small AI integration is practical. You receive the proposed tool list, access and data-flow map, acceptance checks, major risks, and a go or no-go recommendation. If the scope fits, we also provide a fixed-scope implementation proposal.

Proposed fee: USD 300 for one workflow and one system. Includes one 45-minute discovery session, review of the supplied API documentation, and one 30-minute handoff. Target turnaround is two business days after the required inputs arrive and a start date is agreed. The audit does not include implementation, a security certification, or a promise that the API will support the workflow. The fee is not automatically credited against later work.

Please bring a short description of the task, the API documentation, the person who owns access, and two redacted or synthetic examples. Do not send passwords or API keys in email or chat.

### Zetbros Agent Integration Pilot

We connect one agreed lookup workflow to your chosen AI client and test it against representative examples. You receive the connector source, a fixed tool list, access configuration, test results, and an operating guide so your team can decide whether to expand it.

Proposed fee: USD 1,500 after technical qualification. The pilot covers one documented API, one customer account or tenant, one named operator, one agreed client and version, and up to three read-only tools. It includes one installation in an existing customer-approved environment, one handoff session, and one consolidated review of the agreed deliverables. Target delivery is five business days after the start conditions are satisfied; the actual start date and elapsed schedule are agreed in the proposal.

The pilot uses an existing approved authentication pattern. New OAuth infrastructure, multi-user authorization, live writes, a custom interface, hosting subscriptions, and ongoing monitoring are separately scoped. Your team supplies the required account, model, and hosting access and approves where its data can be processed. The pilot is a supervised evaluation with documented limits. Production use requires a separate release decision.

### Short positioning copy

Zetbros helps teams connect one repeatable business workflow to an AI client, with a defined tool set, access limits, and evidence of how it behaves on agreed tests.

### Discovery invitation copy

Which task does your team repeat in the same system each week? Send a short description and the API documentation, and we can assess whether it fits a small paid integration audit.

This is prepared copy for later approval. It has not been sent to any recipient. It makes no demo offer and claims no existing certifications or customer results.

## Customer fit

Prioritize a small B2B SaaS, agency, or operations team with an accessible process owner, a documented API, existing account access, and a manual lookup that recurs several times a week. Examples include a service-request status brief or an account-support summary sourced from a single system. These are starting hypotheses, not evidence of a chosen niche.

The first pilot should avoid medical records, regulated financial decisions, legal advice, recruiting selection, children's information, payment execution, and irreversible changes. A regulated organization is not automatically disqualified, but this first scope must be limited to a clearly permitted, low-risk workflow with an accountable data owner. Sensitive or regulated data triggers a separate review rather than an automatic promise to support it.

Use these four mandatory checks before a fixed pilot quote:

- A buyer owns the workflow and can approve both the commercial terms and the data use.
- API access, relevant endpoints, limits, and the chosen client transport are confirmed from documentation and an authorized test.
- The expected scope fits the labor budget at the founder's selected hourly cost.
- The customer accepts a read-only supervised pilot and its explicit exclusions.

If one is unresolved, offer the audit or pause. A free exploratory call should be capped at 15 minutes; do not give away a complete architecture review or build a bespoke proof to win the deal.

## Discovery questionnaire

Capture the answer, its owner, and whether it is verified or assumed. A credential value is never an answer field.

1. What exact task starts the workflow, and what output marks it complete? Provide one example from start to finish.
2. How many times did the team do it last week? How long does a typical case take, including checking and correcting it? Who can measure five baseline examples?
3. Which single system would make the first version useful? What other systems can be excluded for now?
4. Which records and fields are needed? Which must never leave the source system? Are the examples synthetic, redacted, or approved for processing?
5. What API documentation, sandbox, API plan, and rate limits are available? Who owns the vendor account and approves access?
6. Which AI client, plan, version, and deployment environment will be used? Has that exact client successfully connected to a similar transport and authentication method?
7. Does the workflow need one shared service identity or separate end-user permissions? If separate permissions are necessary, stop treating this as the entry pilot.
8. Can the source credential be genuinely restricted to reading the agreed records? Does a nominal read endpoint cause side effects, mark records read, trigger exports, or consume billable quotas?
9. What personal, confidential, or regulated information can appear in results? Who approves its processing by the selected model provider and any host?
10. What would a wrong answer cause? Which missing, stale, ambiguous, or conflicting records must cause the integration to stop or ask?
11. What source links or record identifiers must accompany the result so a human can verify it?
12. Does the customer expect sends, edits, deletion, payments, approvals, unattended execution, or scheduling? Record each separately and exclude it from the read-only quote.
13. Who can review the fixed test set, report defects, accept deliverables, disable the integration, and handle an incident?
14. What would make the pilot worth its total cost? What time saving is required, after review time and model or hosting costs?
15. What procurement, confidentiality, security-review, IP, contract, or insurance requirements apply? Who can resolve them before any start date is promised?
16. What budget is approved, in which currency, and when can the required inputs be supplied? Are there deadlines that would be harmed by treating a pilot as a production service?

## Audit deliverables and acceptance

The audit produces a one-page workflow map; the proposed tool and field inventory; a data and access map; a risks and dependencies list; and either a scoped pilot proposal or a written no-go conclusion with reasons. Each unknown has an owner and a suggested resolution. Acceptance is based on delivery of these items and their discussion at handoff, not on the recommendation being to proceed.

The audit's three-hour delivery budget is 0.75 hours for discovery, 1.0 for document and API review, 0.5 for the recommendation, 0.5 for handoff, and 0.25 for administration. It excludes implementation and privileged access to live accounts. Allocate an additional hour of sales effort in the base economics.

# Pilot scope and acceptance

## Scope schedule to complete for each customer

- Customer and legal contracting identity: [complete before an offer]
- Process owner and acceptance owner: [name and business contact]
- Desired output: [one sentence]
- Source system, account or tenant, and permitted record subset: [exact values]
- AI client, version, account owner, and model processing destination: [exact values]
- Deployment location and operator: [one existing environment]
- Tool 1: [name, allowed endpoint, input fields, output fields]
- Tool 2: [name, allowed endpoint, input fields, output fields]
- Tool 3: [optional; omit if unnecessary]
- Inbound authentication boundary and upstream credential method: [document separately]
- Agreed timeout, retry, result-count, and quota limits: [complete before build]
- Allowed log fields, access owner, location, and retention: [complete before build]
- Included fixtures, client scenarios, and business baseline: [versioned test-set reference]

"One system" means one documented API and one account or tenant. Two products from the same vendor count as two systems if they require distinct APIs or authorization flows. A read-only tool means the actual permitted upstream operations do not mutate business records; its name, HTTP verb, or MCP annotation alone is insufficient. If the API cannot provide a suitably restricted boundary, re-scope or decline the entry pilot.

## Included deliverables

- A customer-specific connector with at most three typed, allowlisted tools for the signed-off lookup workflow.
- Source code and configuration examples with no embedded credentials, plus a pinned dependency list and the recorded test environment.
- Read-only restrictions, input validation, bounded results, timeouts, and handling for permission, missing-record, rate-limit, and upstream-failure conditions.
- Redacted operational events containing request ID, timestamp, tool, outcome, duration, and error class. User identity is logged only if verified and authorized. This is an operational log, not a tamper-proof compliance audit trail.
- One test report and one client compatibility result for the named version. No promise of compatibility with every MCP client.
- One installation and smoke test in the agreed existing environment, an operating guide, access-revocation instructions, and a 30-minute handoff.
- One consolidated revision round for alignment with the agreed scope. Added endpoints, changed workflows, new clients, and new permissions require a new estimate.

## Explicit exclusions

No email sends, database or CRM updates, refunds, deletes, payments, account administration, bulk exports, autonomous schedules, or external notifications. No new authorization server, custom OAuth consent system, SSO, role matrix, tenant isolation service, secret manager, gateway, dashboard, web UI, RAG ingestion, migration, custom model training, or second system. No penetration test, compliance certification, guaranteed uptime, guaranteed time saving, or 24-hour support. New vendor accounts, paid subscriptions, API usage, taxes, and procurement fees are not silently absorbed by Zetbros.

## Acceptance checks

Agree on fixtures and expected outputs before implementation. These are proposed requirements to test, not claims that the current Mail MCP already passes them.

1. Tool inventory: the agreed client discovers only the named tools. Calls to unknown tools or out-of-scope endpoints fail. The source account cannot perform disallowed writes using the pilot credential; retain evidence of the actual permission boundary.
2. Data boundary: requests outside the permitted account or record subset are denied without revealing the restricted record. Client-supplied identity fields cannot substitute for verified authorization. If the platform cannot provide this boundary, the entry scope is ineligible.
3. Input and output: malformed inputs and missing required fields fail predictably. Valid results conform to the documented schema and include source identifiers. Result limits are enforced and any truncation is explicit.
4. Failure handling: fixtures for a missing record, denied or revoked credential, timeout, rate limit, and upstream failure produce understandable errors. The documented retry and time limits are respected. No raw credential or sensitive response is exposed through the error.
5. Logs and secrets: inspect successful and failed test events. Required metadata is present; credential canaries and excluded record content are absent from source, logs, and tool outputs. These are bounded tests, not a guarantee of universal secret isolation.
6. Adapter tests: run at least 20 agreed deterministic cases covering normal reads and the boundaries above. Every mandatory assertion passes. Record test version, code revision, environment, result, and any limitation.
7. Agent behavior: run 12 agreed scenarios in the named client, including normal requests, ambiguity, missing information, out-of-scope requests, and instruction-like text in retrieved data. Inspect each result against the agreed rubric. Required outputs use the correct records and source identifiers; disallowed actions remain impossible at the integration boundary. Any failure is recorded and blocks acceptance until corrected or explicitly re-scoped. Passing this finite set does not prove immunity to prompt injection.
8. Handoff: the customer operator can install or start the connector using the guide, perform the agreed lookup, find a request in the log, and revoke or disable access. The installation test does not create an uptime or hosting obligation.

Business measurement is separate from technical acceptance. Record the time spent on five baseline cases and five pilot-assisted cases, including review and corrections. Report observed median minutes, error count, and the small sample size. Do not promise a saving before measurement or call a faster incorrect answer successful.

## Internal approval and audit proof

Maintain a clearly labeled OFFLINE SYNTHETIC TEST ONLY exercise separately from customer delivery. It uses fake records and a fake transport that cannot send email or update a live system. It is evidence of the tested harness only, not of the current Mail MCP server or any customer deployment.

The proposed write operation has a canonical preview containing the action, target, exact payload, and risk. An authorization decision must be associated with the exact action, an approver identity in the simulated environment, an expiry, and a unique operation ID. Test that no approval, denial, expiry, replay, a changed payload, and an incorrect approver produce zero fake executions. A valid approval produces one fake execution and a correlated event. Simulated uncertain outcomes must be reported without silently repeating the write.

Record the code revision and pass or fail result for each case. When a live write is eventually scoped, independently verify its real authenticated approver identity, storage, concurrency, retry behavior, access control, and production transport. A prompt asking permission and a boolean tool annotation are not substitutes for enforcement.

# Proposal and statement of work draft

## Proposal details

Status: commercial draft for review. This is not an accepted agreement, an invoice, or authorization to perform live work. No signature or acceptance has been obtained. The final contracting entity, governing terms, data terms, IP terms, cancellation treatment, and any tax handling require resolution before issue.

Prepared for: [customer legal name and verified contact]

Prepared by: [Zetbros contracting entity and authorized representative]

Project: [one workflow] read-only Agent Integration Pilot

## Proposed objective

Connect [system and permitted records] to [client and version] so [named operator] can produce [specific source-linked output] through the agreed tools. The operator remains responsible for reviewing the result. The pilot will provide enough evidence to decide whether a further implementation is justified.

## Proposed work

The scope schedule, included deliverables, exclusions, and acceptance checks in this pack become the project schedule only after the customer-specific fields have been completed and approved. The customer will provide existing authorized API access, approved examples, the agreed model and deployment environment, a process owner, and a review contact. Credential setup will use an approved secure process; credentials will not be solicited in ordinary email or chat.

Start conditions are a final authorized proposal, agreed commercial terms, required payment under those terms, complete inputs, confirmed access, and the founder's readiness checks. Target duration is five business days from the agreed start, with customer delays and third-party failures surfaced promptly. If a dependency changes the estimate, pause the dependent work and agree on the revised scope or schedule before continuing.

## Proposed fee and milestones

Pilot fee: USD 1,500, subject to scope qualification. Proposed payment structure for founder review is USD 750 before the agreed start and USD 750 after written acceptance. No payment request is being made by this document. A separately commissioned audit is USD 300 and is not automatically credited, so audit plus pilot totals USD 1,800 before any separately stated taxes or approved third-party costs.

The customer pays its own existing SaaS, model, and hosting providers. Any new charge, channel fee, or procurement requirement must be identified in the final commercial summary before agreement. No automatic maintenance subscription is included.

Suggested milestones are scope and fixture approval; implementation and test evidence; and installation, handoff, and acceptance. Acceptance is an explicit written decision based on the agreed tests. Silence is not acceptance. A proposed five-business-day customer review window is for planning and must be agreed; it does not automatically trigger an invoice or bind either party.

## Changes and defects

One consolidated scope-alignment review is included. A defect is a reproducible failure against an agreed acceptance check, not an added capability. Proposed post-handoff defect window is seven calendar days, with two human hours reserved in the delivery budget. These hours are an internal economic reserve, not a blanket limit on obligations. The founder must agree how unresolved defects, cancellation, and refunds will be handled in the final terms before selling. Do not improvise a legal remedy or offer unlimited support.

New write actions, endpoints, clients, permissions, UI, or schedules need a written change estimate and approval. No additional work or charge begins merely because a customer asks for it informally.

## Data ownership and handoff questions to resolve

Confirm which source code and materials the customer receives, the permitted reuse of non-customer-specific components, and the licenses of dependencies. Do not assume that customer work can be reused or that proprietary information can enter a common library. Confirm confidentiality, approved processing destinations, log retention, deletion obligations, and who operates the system after handoff. These are items for the parties to resolve in their agreement, not settled terms in this draft.

# Delivery checklist

## Before a proposal is issued

- [ ] Confirm the founder can undertake and invoice this work, including contracting identity, IP ownership, and invoicing requirements. Seek qualified advice where needed; do not speculate about legal eligibility.
- [ ] Approve the price, labor-cost assumption, total-hour budget, commercial terms, and customer-specific scope.
- [ ] Verify evidence for every statement about existing capabilities. Remove security, compliance, performance, and production claims that lack evidence.
- [ ] Confirm buyer authority, exact system, API access, client, transport, and read-only permissions.
- [ ] Resolve sensitive-data processing, account access, procurement, and any agreement requirements.
- [ ] Obtain authorization for the actual recipient and communication before sending. Do not offer a demo.

## Before customer work begins

- [ ] Agree on the outcome, test set, exclusions, review owner, operating owner, and dependency checklist.
- [ ] Complete a data-flow and access map distinguishing client-to-server authentication from server-to-upstream authentication.
- [ ] Use synthetic or redacted fixtures until real-data processing is approved. Establish the permitted model and hosting destinations.
- [ ] Confirm a restricted account and test access through the permitted secure process. Do not create persistent credentials without required approval.
- [ ] Record recurring costs and agree any new charges before work begins.
- [ ] Record manual baseline cases and begin tracking human hours, including sales and rework.

## During implementation and review

- [ ] Keep the adapter and customer-specific configuration separate. Do not build a generic platform.
- [ ] Check actual upstream endpoint behavior; enforce the tool and record allowlists at the boundary.
- [ ] Implement and test input limits, result limits, timeouts, revocation, error redaction, and bounded read retries.
- [ ] Run the adapter suite and named-client scenarios; save failures and corrections as well as passing results.
- [ ] Check the logs for unnecessary content and any secret canaries. Document the retention and access owner.
- [ ] Review estimated hours at 50% of the budget. If remaining work will exceed the priced scope, reduce scope by agreement or obtain a change approval before overrunning.

## Handoff and close

- [ ] Save the revision, configuration template, dependency record, test report, and limitations in the agreed customer destination.
- [ ] Customer operator verifies the setup, lookup, log search, and revocation steps.
- [ ] Obtain explicit acceptance or record the precise failing criterion. Do not infer acceptance from silence.
- [ ] Compare manual and assisted cases without promising general ROI from a small sample.
- [ ] Confirm the operating owner, support boundary, final payment status, and access cleanup obligations.
- [ ] Record actual labor, fees, cash costs, rework, and reasons the customer bought. Seek separate permission before any testimonial or case study.

# Labor inclusive unit economics

## Assumptions and definitions

Every number here is a planning assumption in USD. The USD 50 per human hour rate is a placeholder for the founder's labor and review time, not the founder's known wage or a market rate. Agent-generated work still requires human review. Allocate sales time across wins, including unsuccessful opportunities; do not count only the final customer's discovery call. Existing subscriptions have zero assumed incremental cash cost only while the work stays within their authorized use and limits.

The base pilot uses 15 delivery hours and 4 allocated acquisition hours. Delivery hours comprise 1.0 for scope and administration, 4.5 for the adapter, 2.0 for access and failure controls, 2.0 for tests, 2.5 for installation and documentation, 1.0 for review and handoff, and 2.0 reserved for rework or defects. This is a feasibility budget, not evidence that a first custom build will fit it.

Allow USD 25 of project cash costs in the pricing model as contingency. That allowance authorizes no purchase; actual incremental spending in this preparation phase remains USD 0. The illustrative collection or channel fee is 3% of revenue, with a 15% sensitivity. These are scenario inputs, not asserted Stripe or Upwork fees. Verify actual terms before issuing a quote.

Project contribution = collected revenue minus collection or channel fees, incremental cash costs, and all allocated human labor. Contribution margin = project contribution divided by revenue. This excludes tax, fixed overhead, insurance, future liabilities, and general product development, so it is not accounting profit. Cash contribution before labor is shown separately and must never be called founder profit.

## Base scenarios

| Scenario | Revenue | Total human hours | Cash cost | Fee | Labor at USD 50 per hour | Contribution | Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Audit with 3 delivery and 1 sales hour | 300 | 4 | 0 | 9 | 200 | 91 | 30.3% |
| Narrow USD 950 exception | 950 | 10 | 25 | 28.50 | 500 | 396.50 | 41.7% |
| Introductory pilot | 1,250 | 19 | 25 | 37.50 | 950 | 237.50 | 19.0% |
| Recommended quote hypothesis | 1,500 | 19 | 25 | 45 | 950 | 480 | 32.0% |
| Pilot with 10 additional hours | 1,500 | 29 | 25 | 45 | 1,450 | -20 | -1.3% |
| Pilot with 15% channel fee | 1,500 | 19 | 25 | 225 | 950 | 300 | 20.0% |
| Pilot with USD 75 human hour cost | 1,500 | 19 | 25 | 45 | 1,425 | 5 | 0.3% |

For the default USD 1,500 scenario, cash after fees and cash costs is USD 1,430 before labor. Dividing it by 19 human hours gives USD 75.26 per hour before overhead and tax. The labor-inclusive contribution is USD 480, not USD 1,430.

## Quoting rules

The minimum price for a target contribution margin is (total human hours times hourly labor cost plus cash costs) divided by (1 minus fee rate minus target margin). The denominator must be positive. At 19 hours, USD 50 per hour, USD 25 cash costs, a 3% fee, and a 30% target margin, the minimum is USD 1,455.22. At USD 75 per hour it rises to USD 2,164.18. The founder must choose the actual labor-cost assumption before fixing the price.

A USD 950 offer can support at most 12.23 total human hours at that same 30% target. Use a stricter 10-hour all-in cap because uncertain work needs room for error. With four sales hours allocated, only six delivery hours remain; this is appropriate only for a well-understood, already-tested adapter pattern. It cannot carry the original broad production scope. At the default USD 1,500 price the corresponding maximum is 19.6 total hours, so the 19-hour budget has very little room for overrun.

Do not discount by crediting the full audit automatically. With the base inputs, a USD 300 audit plus USD 1,500 pilot yields USD 1,800 of revenue, 23 total human hours, USD 54 fees, USD 25 cash costs, and USD 571 contribution at USD 50 per hour. Crediting all USD 300 reduces contribution to USD 280 on USD 1,500 of combined revenue. Any credit must be a deliberate acquisition cost or reflect genuinely avoided work.

Before a quote, replace estimated hours, the hourly labor cost, channel fees, and any taxes or required cash costs with the actual known inputs. If the price buyers will accept does not support the scope, narrow the scope or decline. Do not hide the difference in unpaid founder time or future support.

# First week execution priorities

The first week begins with internal preparation. Customer-dependent actions are listed as gated work and remain blocked until the founder authorizes the channel, recipient, and relevant commercial terms. Elapsed days below are priorities, not bookings or promises.

## Day 1 Finalize a sellable boundary

Founder: choose the pilot price, labor-cost assumption, and allowed initial customer profile. Confirm capacity and the business-eligibility checklist. Review this pack and the technical evidence. Deliverable: one approved offer and one explicit list of claims that can be substantiated. Do not treat draft prices or target dates as accepted customer terms.

## Day 2 Establish internal evidence

Technical owner: verify the actual Mail MCP repository and deployment before describing existing capabilities. Build or run only the approved synthetic approval and audit checks with fake transport, recording their limits. Commercial owner: map every intended claim to that evidence. Deliverable: a claim ledger with verified, unverified, and out-of-scope entries. Stop after evidence needed for the first offer; do not start the gateway.

## Day 3 Select the first sales test

Commercial research owner: identify up to five relevant, current opportunities with a named workflow, API access signal, plausible buyer authority, budget evidence where available, and the permitted contact route. Check whether each remains open. Prepare a short written fit note and a specific discovery invitation for the founder to review. No mass list, paid tooling, application, contact, or demo is authorized by this preparation task.

## Day 4 Seek the smallest customer commitment

After communication authorization, send a specific written invitation to one qualified prospect or respond through an approved channel. Qualify with a short call or written exchange. Seek one paid USD 300 audit or, when scope is already evidenced, a qualified pilot proposal. Record actual customer language and objections. If there is no authorized outreach yet, stop at the ready-to-send material rather than manufacturing activity.

## Day 5 Review revenue evidence

If an audit is purchased under agreed terms, deliver the audit and a go or no-go decision. If a pilot is bought, begin only after its start conditions pass. If no buyer has paid, review the observed objections and choose one change to the offer, niche, or channel. Do not build more infrastructure as a substitute for customer evidence.

## Weekly scorecard and stopping conditions

Track qualified opportunities, authorized contacts, replies, discovery conversations, paid audits, scoped proposals, pilot cash collected, hours by category, blockers, and the next owner. Start with zeros or unknowns; targets are not achieved results. The first cash milestone is one paying customer, while implementation validation requires one paid pilot delivered and accepted.

Pause the dependent work when permission, account access, data approval, contracting eligibility, or technical feasibility is unresolved. Continue safe preparation only. Stop building the commercial pack when the offer, questionnaire, scope, economics, and next decision are reviewable. Product work remains deferred until repeated paying customers show a shared need.

# Evidence and claim discipline

The market research is a dated snapshot in ../validation. This pack does not establish a market median, validate every listing, or treat listed budgets as booked revenue. The proposed prices come from the explicit labor model and still need a real sales test.

The official MCP tools specification treats tool annotations as untrusted unless supplied by a trusted server and leaves the user interaction model to implementations. The commercial consequence is to describe the tested enforcement mechanism instead of implying that an MCP tool automatically has a reliable approval gate. Source: https://modelcontextprotocol.io/specification/2026-07-28/server/tools (checked 3 October 2026).

MCP security guidance identifies token-audience mistakes and token passthrough as authorization risks, and separately discusses local-server risks and access controls. The commercial consequence is to verify the selected transport and its authorization boundaries rather than advertise generic "auth included" as a completed security property. Source: https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices (checked 3 October 2026).

Use "tested against the agreed cases" when the test report supports it. Use "restricted to the agreed tool and record set" only after boundary tests and configuration evidence. Use "redacted operational events" only after inspection. Do not use "secure," "production-ready," "enterprise-grade," "compliant," "tamper-proof," "zero risk," "full audit trail," or "works with every client" as unqualified claims. A source link, code revision, and passing test must identify the exact boundary of each claim.

The remaining decisions are the founder's labor-cost assumption and price, technical proof status, the permitted sales channel and first recipient, and the final commercial and data terms. Nothing in this pack resolves those decisions by implication.
