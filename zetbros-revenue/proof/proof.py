"""Independent OFFLINE SYNTHETIC harness. NOT the existing Mail MCP server."""
from safety import guard_is_installed, install_offline_guard

install_offline_guard()

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import re


class ProofError(RuntimeError):
    """Fixed error codes, never provider exception text or user payload content."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


class FixtureClock:
    def __init__(self, start):
        self.now = datetime.fromisoformat(start)
        if self.now.utcoffset() != timedelta(0):
            raise ProofError("invalid_fixture_clock")

    def advance(self, seconds):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
            raise ProofError("invalid_clock_advance")
        self.now += timedelta(seconds=seconds)


class Fixtures:
    def __init__(self, data):
        self.data = deepcopy(data)
        self._validate()

    @classmethod
    def load(cls, path):
        if path is None or not Path(path).is_file():
            raise ProofError("fixtures_missing")
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ProofError("fixtures_invalid") from None
        return cls(data)

    def _validate(self):
        try:
            d = self.data
            if not isinstance(d, dict) or set(d) != {"synthetic", "version", "clock", "human_actor", "senders",
                                                     "allowed_addresses", "context", "messages"}:
                raise ValueError
            if d["synthetic"] is not True or not re.fullmatch(r"zetbros-synthetic-v[0-9]+", d["version"]):
                raise ValueError
            if not re.fullmatch(r"demo_[a-z_]+", d["human_actor"]):
                raise ValueError
            FixtureClock(d["clock"])
            addresses = d["allowed_addresses"]
            if not isinstance(addresses, list) or not addresses or len(set(addresses)) != len(addresses):
                raise ValueError
            if any(not synthetic_address(a) for a in addresses):
                raise ValueError
            if (not isinstance(d["senders"], list) or not d["senders"]
                    or any(not isinstance(a, str) or a not in addresses for a in d["senders"])):
                raise ValueError
            context = d["context"]
            if (not isinstance(context, dict) or set(context) != {"guide_v1", "support_policy_v1", "customer_lookup"}
                    or any(not isinstance(context[k], str) or not context[k] or len(context[k].encode("utf-8")) > 16000
                           for k in ("guide_v1", "support_policy_v1"))
                    or not isinstance(context["customer_lookup"], dict)):
                raise ValueError
            for account_id, account in context["customer_lookup"].items():
                if (not re.fullmatch(r"demo_account_[0-9]{2}", account_id) or not isinstance(account, dict)
                        or set(account) != {"label"} or not isinstance(account["label"], str)):
                    raise ValueError
            if not isinstance(d["messages"], list) or not d["messages"]:
                raise ValueError
            seen = set()
            for m in d["messages"]:
                if not isinstance(m, dict) or set(m) != {"id", "folder", "version", "from", "reply_to",
                                                        "subject", "body", "read", "workflow_case"}:
                    raise ValueError
                if (not isinstance(m["folder"], str) or not isinstance(m["id"], str)
                        or not isinstance(m["workflow_case"], str)
                        or m["workflow_case"] not in {"guide", "refund", "injection", "missing_identifier", "no_action"}):
                    raise ValueError
                key = (m["folder"], m["id"])
                if key in seen or not re.fullmatch(r"demo_msg_[0-9]{2}", m["id"]):
                    raise ValueError
                seen.add(key)
                if m["folder"] not in {"INBOX", "Archive"} or type(m["version"]) is not int or m["version"] < 1:
                    raise ValueError
                if m["from"] not in addresses or (m["reply_to"] is not None and m["reply_to"] not in addresses):
                    raise ValueError
                if not isinstance(m["subject"], str) or "\r" in m["subject"] or "\n" in m["subject"]:
                    raise ValueError
                if not isinstance(m["body"], str) or type(m["read"]) is not bool:
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ProofError("fixtures_invalid") from None

    @property
    def version(self):
        return self.data["version"]

    @property
    def fingerprint(self):
        # Includes messages, Reply-To, context, policy, identity and allowlists.
        return digest(self.data)

    def message(self, folder, message_id):
        for m in self.data["messages"]:
            if m["folder"] == folder and m["id"] == message_id:
                return m
        raise ProofError("message_not_found_or_stale")


def synthetic_address(value):
    return (isinstance(value, str) and len(value) <= 254
            and re.fullmatch(r"[a-z0-9._+-]+@[a-z0-9.-]+\.invalid", value) is not None)


@dataclass(frozen=True)
class Action:
    tool_name: str
    operation: str
    sender: str
    to: tuple
    cc: tuple
    subject: str
    body: str
    original_folder: str | None
    original_message_id: str | None
    original_message_version: int | None
    original_message_digest: str | None
    fixture_version: str
    fixture_digest: str
    context_digest: str

    @property
    def action_digest(self):
        return digest(asdict(self))


@dataclass
class Proposal:
    proposal_id: str
    action: Action
    action_digest: str
    state: str = "awaiting_approval"
    approval_id: str | None = None


@dataclass
class Approval:
    approval_id: str
    proposal_id: str
    action_digest: str
    fixture_version: str
    fixture_digest: str
    run_id: str
    approver: str
    decision_source: str
    issued_at: datetime
    expires_at: datetime
    consumed: bool = False


@dataclass(frozen=True)
class Outcome:
    proposal_id: str
    action_digest: str
    status: str
    simulated_submission_status: str
    sent_copy_status: str
    accepted_recipients: tuple = ()
    rejected_recipients: tuple = ()
    unknown_recipients: tuple = ()
    error_class: str | None = None
    retry_count: int = 0
    delivered: bool = False
    synthetic: bool = True
    duplicate: bool = False
    audit_status: str = "recorded"
    receipt_proposal_id: str | None = None


class SyntheticAudit:
    """In-memory minimized events; fail_event is deliberate test fault injection."""
    def __init__(self, run_id, clock, fixture_version):
        self.run_id = run_id
        self.clock = clock
        self.fixture_version = fixture_version
        self.events = []
        self.fail_event = None

    def emit(self, event_type, proposal=None, approval=None, outcome=None,
             error_class=None, tool_name=None, decision_actor=None, decision_source=None):
        if self.fail_event in {"*", event_type}:
            raise ProofError("required_audit_unavailable")
        a = proposal.action if proposal else None
        event = {
            "run_id": self.run_id,
            "event_id": f"demo_event_{len(self.events) + 1:04d}",
            "timestamp": self.clock.now.isoformat(),
            "event_type": event_type,
            "request_id": proposal.proposal_id if proposal else None,
            "proposal_id": proposal.proposal_id if proposal else None,
            "fixture_version": self.fixture_version,
            "environment": "synthetic",
            "actor_id": approval.approver if approval else decision_actor or "demo_workflow",
            "agent_id": "deterministic_fixture_workflow",
            "tool_name": a.tool_name if a else tool_name,
            "risk_class": "simulated_external_write" if a else "synthetic_read_or_draft",
            "action_digest": proposal.action_digest if proposal else None,
            "target_summary": {
                "recipients": [v if synthetic_address(v) else "[redacted]" for v in a.to + a.cc],
                "original_folder": a.original_folder,
                "original_message_id": a.original_message_id,
            } if a else {},
            "decision_source": approval.decision_source if approval else decision_source,
            "approval_id": approval.approval_id if approval else (proposal.approval_id if proposal else None),
            "approval_state": ("consumed" if approval and approval.consumed else
                               "approved" if approval and event_type == "approval_recorded" else
                               proposal.state if proposal else None),
            "execution_result": outcome.status if outcome else None,
            "receipt_proposal_id": outcome.receipt_proposal_id if outcome else None,
            "error_class": error_class or (outcome.error_class if outcome else None),
            "latency_ms": 0,
            "latency_kind": "fixed_simulated",
            "retry_count": outcome.retry_count if outcome else 0,
            "simulated_submission_status": outcome.simulated_submission_status if outcome else None,
            "sent_copy_status": outcome.sent_copy_status if outcome else None,
            "accepted_recipients": list(outcome.accepted_recipients) if outcome else [],
            "rejected_recipients": list(outcome.rejected_recipients) if outcome else [],
            "unknown_recipients": list(outcome.unknown_recipients) if outcome else [],
        }
        # Bodies, subjects, credential values, provider errors and incoming instructions are omitted.
        self.events.append(event)
        return event["event_id"]

    def save(self, path):
        Path(path).write_text("".join(canonical(e) + "\n" for e in self.events), encoding="utf-8")


class FakeTransport:
    """Only in-memory outcomes. Cannot construct a live backend or fallback."""
    def __init__(self, capability):
        if not guard_is_installed():
            raise ProofError("offline_guard_missing")
        self.__capability = capability
        self.calls = []
        self.outbox = []
        self.scenarios = {("INBOX", "demo_msg_06"): "uncertain"}
        self.raise_after_submission = False

    def submit(self, action, proposal_id, capability):
        if capability is not self.__capability:
            raise ProofError("unapproved_transport_invocation")
        scenario = self.scenarios.get((action.original_folder, action.original_message_id), "accept")
        if scenario not in {"accept", "reject_before", "uncertain", "copy_failed", "partial"}:
            raise ProofError("invalid_fake_transport_scenario")
        self.calls.append({"proposal_id": proposal_id, "digest": action.action_digest, "synthetic": True})
        recipients = action.to + action.cc
        if self.raise_after_submission:
            raise RuntimeError("RAW_PROVIDER_EXCEPTION_CANARY")
        if scenario == "reject_before":
            return Outcome(proposal_id, action.action_digest, "failed", "rejected_before_submission",
                           "not_attempted", rejected_recipients=recipients, error_class="fake_rejection")
        if scenario == "uncertain":
            return Outcome(proposal_id, action.action_digest, "uncertain", "unknown", "not_attempted",
                           unknown_recipients=recipients, error_class="fake_interruption")
        if scenario == "partial":
            if len(recipients) < 2:
                raise ProofError("partial_scenario_requires_multiple_recipients")
            accepted, rejected = recipients[:1], recipients[1:]
            self.outbox.append({"proposal_id": proposal_id, "action": action, "accepted": accepted})
            return Outcome(proposal_id, action.action_digest, "partial", "partial_acceptance", "not_attempted",
                           accepted_recipients=accepted, rejected_recipients=rejected,
                           error_class="fake_partial_rejection")
        self.outbox.append({"proposal_id": proposal_id, "action": action, "accepted": recipients})
        return Outcome(proposal_id, action.action_digest, "simulated_accepted", "accepted",
                       "copy_failed" if scenario == "copy_failed" else "copied",
                       accepted_recipients=recipients,
                       error_class="fake_sent_copy_failure" if scenario == "copy_failed" else None)


class Engine:
    """Single-process control boundary. No persistent state or production identity."""
    def __init__(self, fixtures, run_id="demo_run_001"):
        if not re.fullmatch(r"demo_run_[a-z0-9_]+", run_id):
            raise ProofError("invalid_run_id")
        fixtures._validate()
        self.fixtures = fixtures
        self.run_id = run_id
        self.clock = FixtureClock(fixtures.data["clock"])
        self.audit = SyntheticAudit(run_id, self.clock, fixtures.version)
        self.__execution_capability = object()
        self.__decision_capability = object()
        self.transport = FakeTransport(self.__execution_capability)
        self._proposals = {}
        self._approvals = {}
        self._receipts = {}  # idempotency key -> (proposal, digest, approval, outcome)
        self._action_receipts = {}  # exact action -> known result, including uncertainty
        self._inflight = {}  # reserve before invocation, including reentrant requests
        self._inflight_actions = set()
        self.drafts = []
        self.read_calls = 0
        self.read_failures = []
        self.tools = DemoTools(self)

    def human_channel(self, *, scripted=False):
        return HumanDecisionChannel(self, self.__decision_capability,
                                    "scripted_test_fixture" if scripted else "human_console")

    def validate_action(self, action, *, live=True):
        if not isinstance(action, Action):
            raise ProofError("invalid_action_schema")
        if (action.operation, action.tool_name) not in {("reply", "reply_email"), ("send", "send_email")}:
            raise ProofError("tool_not_allowed")
        if type(action.to) is not tuple or type(action.cc) is not tuple or not action.to or len(action.to + action.cc) > 4:
            raise ProofError("invalid_recipients")
        addresses = (action.sender,) + action.to + action.cc
        if any(not synthetic_address(a) or a not in self.fixtures.data["allowed_addresses"] for a in addresses):
            raise ProofError("address_outside_synthetic_allowlist")
        if action.sender not in self.fixtures.data["senders"]:
            raise ProofError("unapproved_sender")
        if len(set(action.to + action.cc)) != len(action.to + action.cc):
            raise ProofError("duplicate_recipient")
        if not isinstance(action.subject, str) or len(action.subject) > 256 or "\r" in action.subject or "\n" in action.subject:
            raise ProofError("invalid_header")
        if not isinstance(action.body, str) or not action.body:
            raise ProofError("invalid_body")
        try:
            if len(action.body.encode("utf-8")) > 16384:
                raise ProofError("invalid_body")
        except UnicodeError:
            raise ProofError("invalid_body") from None
        if not live:
            return
        if (action.fixture_version != self.fixtures.version or action.fixture_digest != self.fixtures.fingerprint
                or action.context_digest != digest(self.fixtures.data["context"])):
            raise ProofError("fixture_or_context_changed")
        if action.operation == "reply":
            message = self.fixtures.message(action.original_folder, action.original_message_id)
            if (action.original_message_version != message["version"]
                    or action.original_message_digest != digest(message)
                    or action.to != (message["reply_to"] or message["from"],)):
                raise ProofError("original_message_or_destination_changed")
        elif any(v is not None for v in (action.original_folder, action.original_message_id,
                                        action.original_message_version, action.original_message_digest)):
            raise ProofError("invalid_send_reference")

    def prepare(self, tool_name, *, body, subject=None, sender=None, to=None, cc=(),
                folder=None, message_id=None):
        if not isinstance(tool_name, str) or tool_name not in {"reply_email", "send_email"}:
            raise ProofError("tool_not_allowed")
        if type(cc) not in {tuple, list} or (to is not None and type(to) not in {tuple, list}):
            raise ProofError("invalid_recipients")
        if tool_name == "reply_email":
            m = self.fixtures.message(folder, message_id)
            recipient = (m["reply_to"] or m["from"],)
            if to is not None and tuple(to) != recipient:
                raise ProofError("reply_destination_override_not_allowed")
            ref = (folder, message_id, m["version"], digest(m))
            subject = "Re: " + m["subject"] if subject is None else subject
        else:
            recipient = tuple(to) if to is not None else ()
            ref = (None, None, None, None)
        a = Action(tool_name, "reply" if tool_name == "reply_email" else "send",
                   self.fixtures.data["senders"][0] if sender is None else sender, recipient, tuple(cc), subject, body,
                   *ref, self.fixtures.version, self.fixtures.fingerprint,
                   digest(self.fixtures.data["context"]))
        self.validate_action(a)
        pid = f"demo_proposal_{len(self._proposals) + 1:04d}"
        p = Proposal(pid, a, a.action_digest)
        self.audit.emit("proposal_created", proposal=p)
        self._proposals[pid] = p
        return self.preview(pid)

    def proposal(self, proposal_id):
        if not isinstance(proposal_id, str) or proposal_id not in self._proposals:
            raise ProofError("proposal_not_found")
        return self._proposals[proposal_id]

    def preview(self, proposal_id):
        p = self.proposal(proposal_id)
        return {"proposal_id": p.proposal_id, "action_digest": p.action_digest,
                "state": p.state, "synthetic": True, "payload": deepcopy(asdict(p.action))}

    def _invalidate(self, p, code):
        p.state = "invalidated"
        self.audit.emit("execution_blocked", proposal=p, error_class=code)
        raise ProofError(code)

    def _record_decision(self, capability, proposal_id, decision, displayed_digest, actor, source, ttl_seconds):
        if capability is not self.__decision_capability:
            self.audit.emit("decision_rejected", error_class="untrusted_decision_channel", tool_name="local_decision")
            raise ProofError("untrusted_decision_channel")
        if actor != self.fixtures.data["human_actor"] or source not in {"human_console", "scripted_test_fixture"}:
            self.audit.emit("decision_rejected", error_class="unauthorized_approver", tool_name="local_decision")
            raise ProofError("unauthorized_approver")
        p = self.proposal(proposal_id)
        if not isinstance(decision, str) or decision not in {"approve", "deny", "cancel"}:
            raise ProofError("invalid_decision")
        if p.state not in {"awaiting_approval", "approved"}:
            raise ProofError("proposal_terminal")
        if displayed_digest != p.action_digest:
            self._invalidate(p, "preview_digest_mismatch")
        try:
            self.validate_action(p.action)
        except ProofError:
            self._invalidate(p, "proposal_context_changed")
        if decision in {"deny", "cancel"}:
            previous = p.state
            p.state = "denied" if decision == "deny" else "canceled"
            try:
                self.audit.emit("decision_recorded", proposal=p, decision_actor=actor, decision_source=source)
            except ProofError:
                # Denial/cancellation remains terminal even if the audit is broken.
                raise
            if p.approval_id:
                self._approvals[p.approval_id].consumed = True
            return {"proposal_id": p.proposal_id, "state": p.state,
                    "decision_source": source, "previous_state": previous}
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 300:
            raise ProofError("invalid_approval_ttl")
        if p.approval_id:
            old = self._approvals[p.approval_id]
            if self.clock.now >= old.expires_at or old.consumed:
                p.state = "expired" if not old.consumed else "invalidated"
                raise ProofError("new_proposal_and_decision_required")
            return old.approval_id  # Repeated approve click cannot mint a new use.
        aid = f"demo_approval_{len(self._approvals) + 1:04d}"
        approval = Approval(aid, p.proposal_id, p.action_digest, p.action.fixture_version,
                            p.action.fixture_digest, self.run_id, actor, source,
                            self.clock.now, self.clock.now + timedelta(seconds=ttl_seconds))
        # Required decision event must succeed before approval can be granted.
        self.audit.emit("approval_recorded", proposal=p, approval=approval)
        self._approvals[aid] = approval
        p.approval_id, p.state = aid, "approved"
        return aid

    def execute(self, proposal_id, approval_id=None, *, action=None, idempotency_key=None):
        p = self.proposal(proposal_id)
        requested = p.action if action is None else action
        if not isinstance(requested, Action) or requested.action_digest != p.action_digest:
            self._invalidate(p, "approved_payload_changed")
        key = p.proposal_id if idempotency_key is None else idempotency_key
        if not isinstance(key, str) or not re.fullmatch(r"demo_[a-z0-9_]{1,64}", key):
            raise ProofError("invalid_idempotency_key")
        if key in self._inflight or p.action_digest in self._inflight_actions:
            self.audit.emit("execution_blocked", proposal=p, error_class="action_in_flight")
            raise ProofError("action_in_flight")
        if key in self._receipts:
            old_pid, old_digest, old_aid, outcome = self._receipts[key]
            if (old_pid, old_digest, old_aid) != (p.proposal_id, p.action_digest, approval_id):
                raise ProofError("idempotency_conflict")
            cached = replace(outcome, proposal_id=p.proposal_id, duplicate=True,
                             receipt_proposal_id=outcome.receipt_proposal_id or outcome.proposal_id)
            self.audit.emit("duplicate_suppressed", proposal=p, outcome=cached)
            return cached
        if p.state != "approved" or approval_id != p.approval_id or approval_id not in self._approvals:
            self.audit.emit("execution_blocked", proposal=p, error_class="valid_approval_required")
            raise ProofError("valid_approval_required")
        approval = self._approvals[approval_id]
        if (approval.consumed or approval.proposal_id != p.proposal_id
                or approval.action_digest != p.action_digest or approval.run_id != self.run_id
                or approval.fixture_version != p.action.fixture_version
                or approval.fixture_digest != p.action.fixture_digest
                or approval.approver != self.fixtures.data["human_actor"]):
            self._invalidate(p, "approval_invalid_or_consumed")
        if self.clock.now < approval.issued_at:
            self._invalidate(p, "approval_not_yet_valid")
        if self.clock.now >= approval.expires_at:
            p.state = "expired"
            self.audit.emit("execution_blocked", proposal=p, approval=approval, error_class="approval_expired")
            raise ProofError("approval_expired")
        try:
            self.validate_action(requested)
        except ProofError:
            self._invalidate(p, "approved_context_changed")
        known = self._action_receipts.get(p.action_digest)
        if known:
            # An uncertain outcome cannot be retried by minting another proposal/key.
            cached = replace(known, proposal_id=p.proposal_id, duplicate=True,
                             receipt_proposal_id=known.receipt_proposal_id or known.proposal_id)
            approval.consumed, p.state = True, "duplicate_suppressed"
            self.audit.emit("duplicate_suppressed", proposal=p, approval=approval, outcome=cached)
            self._receipts[key] = (p.proposal_id, p.action_digest, approval_id, cached)
            return cached
        # Audit the start before consuming permission or touching fake transport.
        self.audit.emit("execution_started", proposal=p, approval=approval)
        approval.consumed, p.state = True, "executing"
        self._inflight[key] = (p.proposal_id, p.action_digest, approval_id)
        self._inflight_actions.add(p.action_digest)
        try:
            outcome = self.transport.submit(requested, p.proposal_id, self.__execution_capability)
        except Exception:
            # An unexpected adapter error may have happened after submission.
            # Preserve one-use and return uncertainty, never leak or resend.
            outcome = Outcome(p.proposal_id, p.action_digest, "uncertain", "unknown", "not_attempted",
                              unknown_recipients=requested.to + requested.cc,
                              error_class="unexpected_fake_transport_error")
        p.state = outcome.status
        self._action_receipts[p.action_digest] = outcome
        self._receipts[key] = (p.proposal_id, p.action_digest, approval_id, outcome)
        del self._inflight[key]
        self._inflight_actions.remove(p.action_digest)
        try:
            self.audit.emit("execution_result", proposal=p, approval=approval, outcome=outcome)
        except ProofError:
            # Submission status remains known; an audit failure does not undo it.
            outcome = replace(outcome, audit_status="record_failed")
            self._action_receipts[p.action_digest] = outcome
            self._receipts[key] = (p.proposal_id, p.action_digest, approval_id, outcome)
        return outcome


class HumanDecisionChannel:
    """Only the local CLI/test owner gets this channel; it is never a demo tool."""
    def __init__(self, engine, capability, source):
        self.__engine, self.__capability, self.source = engine, capability, source

    def decide(self, proposal_id, decision, displayed_digest, *, ttl_seconds=120, actor=None):
        return self.__engine._record_decision(
            self.__capability, proposal_id, decision, displayed_digest,
            self.__engine.fixtures.data["human_actor"] if actor is None else actor,
            self.source, ttl_seconds)


class DemoTools:
    """Bounded untrusted-workflow interface. Sending tools PROPOSE, never send."""
    ALLOWED = frozenset({"list_folders", "list_senders", "list_emails", "search_emails",
                         "read_email", "create_draft", "reply_email", "send_email"})
    PARAMETERS = {
        "list_folders": set(), "list_senders": set(),
        "list_emails": {"folder", "limit"}, "search_emails": {"folder", "query", "limit"},
        "read_email": {"folder", "message_id"},
        "create_draft": {"folder", "message_id", "body", "subject", "sender", "cc"},
        "reply_email": {"folder", "message_id", "body", "subject", "sender", "cc"},
        "send_email": {"to", "body", "subject", "sender", "cc"},
    }
    REQUIRED = {
        "list_emails": {"folder"}, "search_emails": {"folder", "query"},
        "read_email": {"folder", "message_id"},
        "create_draft": {"folder", "message_id", "body"},
        "reply_email": {"folder", "message_id", "body"},
        "send_email": {"to", "body", "subject"},
    }

    def __init__(self, engine):
        self.__engine = engine

    def call(self, name, **parameters):
        try:
            return self._call(name, **parameters)
        except ProofError as error:
            safe_name = name if isinstance(name, str) and name in self.ALLOWED else "outside_demo_profile"
            self.__engine.audit.emit("tool_rejected", tool_name=safe_name, error_class=error.code)
            raise

    def _call(self, name, **parameters):
        if not isinstance(name, str) or name not in self.ALLOWED:
            raise ProofError("tool_not_allowed")
        if set(parameters) - self.PARAMETERS[name]:
            raise ProofError("unexpected_tool_parameter")
        if self.REQUIRED.get(name, set()) - set(parameters):
            raise ProofError("missing_tool_parameter")
        e = self.__engine
        if name in {"reply_email", "send_email"}:
            return e.prepare(name, **parameters)
        if name == "create_draft":
            # Reuse proposal validation without generating approval or a send.
            p = e.prepare("reply_email", **parameters)
            a = e.proposal(p["proposal_id"]).action
            e.audit.emit("draft_created", tool_name="create_draft")
            e.proposal(p["proposal_id"]).state = "draft_only"
            draft = {"draft_id": f"demo_draft_{len(e.drafts) + 1:04d}", "payload": asdict(a), "synthetic": True}
            e.drafts.append(draft)
            return deepcopy(draft)
        if name == "list_folders":
            return sorted({m["folder"] for m in e.fixtures.data["messages"]})
        if name == "list_senders":
            return deepcopy(e.fixtures.data["senders"])
        e.read_calls += 1
        if e.read_failures:
            fault = e.read_failures.pop(0)
            if fault not in {"fake_read_timeout", "fake_read_rate_limit"}:
                raise ProofError("invalid_fake_read_fault")
            raise ProofError(fault)
        folder = parameters.get("folder")
        if not isinstance(folder, str) or folder not in {m["folder"] for m in e.fixtures.data["messages"]}:
            raise ProofError("folder_not_found")
        if name == "read_email":
            answer = deepcopy(e.fixtures.message(folder, parameters.get("message_id")))
        else:
            limit = parameters.get("limit", 10)
            if type(limit) is not int or not 1 <= limit <= 50:
                raise ProofError("invalid_result_limit")
            messages = [m for m in e.fixtures.data["messages"] if m["folder"] == folder]
            if name == "search_emails":
                query = parameters.get("query")
                if not isinstance(query, str) or not query or not query.isascii() or len(query) > 128:
                    raise ProofError("invalid_ascii_query")
                messages = [m for m in messages if query.lower() in (m["subject"] + " " + m["body"]).lower()]
            answer = deepcopy(messages[:limit])
        e.audit.emit("read_result", tool_name=name)
        return answer

    def safe_read(self, name, *, max_attempts=2, **parameters):
        if not isinstance(name, str) or name not in {"read_email", "list_emails", "search_emails"}:
            raise ProofError("read_retry_tool_not_allowed")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ProofError("invalid_read_retry_limit")
        for attempt in range(max_attempts):
            try:
                return self.call(name, **parameters)
            except ProofError as error:
                if error.code not in {"fake_read_timeout", "fake_read_rate_limit"} or attempt + 1 == max_attempts:
                    raise
                self.__engine.clock.advance(1)
                self.__engine.audit.emit("safe_read_retry", error_class=error.code, tool_name=name)


class FixtureWorkflow:
    """Scripted fixture cases, not a model or a general prompt-injection detector."""
    def __init__(self, tools, context):
        self.tools = tools
        self.context = deepcopy(context)

    def plan(self, message_id, folder="INBOX"):
        m = self.tools.call("read_email", folder=folder, message_id=message_id)
        case = m["workflow_case"]  # Explicit fixed fixture rubric, not inferred intelligence.
        if case == "guide":
            body = self.context["guide_v1"] + "\nSource: guide_v1 (local fictional guide)"
            p = self.tools.call("reply_email", folder=folder, message_id=message_id, body=body)
            return {"kind": "approval_required", "proposal": p, "sources": ["guide_v1"],
                    "summary": "Exact synthetic reply prepared; approval required before any simulation."}
        if case == "refund":
            d = self.tools.call("create_draft", folder=folder, message_id=message_id,
                                body="Your refund request needs review by the human support owner. No refund has been executed or promised.")
            return {"kind": "draft_only", "draft": d, "sources": ["support_policy_v1"],
                    "summary": "Synthetic escalation draft only; no refund or send."}
        if case == "missing_identifier":
            d = self.tools.call("create_draft", folder=folder, message_id=message_id,
                                body="Please provide your Example Product account reference so the support owner can review your account.")
            return {"kind": "draft_only", "draft": d, "sources": [],
                    "summary": "Synthetic clarification draft only; account identifier is missing."}
        if case == "injection":
            return {"kind": "blocked_untrusted_request", "sources": [],
                    "summary": "Mail instructions cannot authorize sends, exfiltration or permission changes."}
        if case == "no_action":
            return {"kind": "summary_only", "sources": [], "summary": "Resolved fixture; no action needed."}
        raise ProofError("unsupported_fixture_case")


def completion_claim_is_supported(outcome, claimed_status):
    """Deterministic result-interpretation check, not an evaluation of model quality."""
    return (claimed_status == outcome.status and claimed_status != "delivered"
            and outcome.synthetic is True and outcome.delivered is False)


DEFAULT_FIXTURES = Path(__file__).parent / "fixtures" / "mailbox-v1.json"


def start(*, mode="synthetic", fixtures_path=DEFAULT_FIXTURES, run_id="demo_run_001"):
    # Validate mode and fixture availability before any backend is constructed.
    if mode != "synthetic":
        raise ProofError("only_offline_synthetic_mode_allowed")
    fixtures = Fixtures.load(fixtures_path)
    return Engine(fixtures, run_id=run_id)
