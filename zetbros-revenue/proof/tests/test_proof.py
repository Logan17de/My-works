"""Deterministic contract/control tests; no model calls or production source."""
from safety import blocked_events, install_offline_guard, NetworkAccessDenied

install_offline_guard()

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
import imaplib
import json
from pathlib import Path
import smtplib
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from proof import (Action, DEFAULT_FIXTURES, Engine, FakeTransport, FixtureWorkflow,
                   Fixtures, ProofError, canonical, completion_claim_is_supported, start)


def matrix(case_id, assertion):
    def decorator(fn):
        fn.case_id = case_id
        fn.assertion_description = assertion
        return fn
    return decorator


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.engines = []
        self.blocked_events_before = len(blocked_events)

    def engine(self):
        e = start(run_id=f"demo_run_{self._testMethodName}_{len(self.engines) + 1}")
        self.engines.append(e)
        return e

    def proposal(self, e, message_id="demo_msg_01", **kwargs):
        return e.tools.call("reply_email", folder="INBOX", message_id=message_id,
                            body="Fictional guide response.\nSource: guide_v1", **kwargs)

    def approve(self, e, p, **kwargs):
        return e.human_channel(scripted=True).decide(p["proposal_id"], "approve", p["action_digest"], **kwargs)

    def blocked(self, fn, code=None):
        with self.assertRaises(ProofError) as context:
            fn()
        if code is not None:
            self.assertEqual(context.exception.code, code)
        return context.exception.code

    def no_send(self, e):
        self.assertEqual(len(e.transport.calls), 0)
        self.assertEqual(len(e.transport.outbox), 0)

    @matrix("I01", "Unknown mode, missing/invalid fixtures fail before any fake backend creation.")
    def test_start_fail_closed(self):
        with patch("proof.FakeTransport") as constructor:
            self.blocked(lambda: start(mode="live"), "only_offline_synthetic_mode_allowed")
            self.blocked(lambda: start(fixtures_path=None), "fixtures_missing")
            with tempfile.TemporaryDirectory() as root:
                p = Path(root) / "invalid.json"
                p.write_text('{"synthetic":false}')
                self.blocked(lambda: start(fixtures_path=p), "fixtures_invalid")
            constructor.assert_not_called()

    @matrix("I01.fixture_schema", "Missing workflow metadata and malformed context/fixture types reject before backend creation.")
    def test_fixture_schema_before_backend(self):
        base = Fixtures.load(DEFAULT_FIXTURES).data
        mutations = (lambda d: d["messages"][0].pop("workflow_case"),
                     lambda d: d["context"].update(guide_v1=["truthy but not text"]),
                     lambda d: d["context"].update(support_policy_v1=17),
                     lambda d: d["context"].update(customer_lookup=[]),
                     lambda d: d["messages"][0].update(workflow_case="unsupported_case"),
                     lambda d: d["messages"][0].update(folder=[]))
        with tempfile.TemporaryDirectory() as root, patch("proof.FakeTransport") as constructor:
            p = Path(root) / "bad-fixtures.json"
            for mutate in mutations:
                data = deepcopy(base)
                mutate(data)
                p.write_text(json.dumps(data), encoding="utf-8")
                self.blocked(lambda: start(fixtures_path=p), "fixtures_invalid")
            constructor.assert_not_called()

    @matrix("I02", "Socket, SMTP, IMAP, DNS, HTTP and subprocess network routes are denied before transport.")
    def test_network_guard(self):
        from urllib.request import urlopen
        attempts = [lambda: socket.socket(), lambda: socket.create_connection(("mail.invalid", 25)),
                    lambda: socket.getaddrinfo("mail.invalid", 25),
                    lambda: smtplib.SMTP("mail.invalid", 25),
                    lambda: imaplib.IMAP4("mail.invalid"),
                    lambda: urlopen("https://mail.invalid", timeout=1),
                    lambda: subprocess.run(["python3", "-c", "pass"])]
        for attempt in attempts:
            with self.assertRaises(NetworkAccessDenied):
                attempt()
        self.assertEqual(len(blocked_events) - self.blocked_events_before, len(attempts))

    @matrix("I02.provider_spies", "Full synthetic flow never constructs SMTP, IMAP, socket or HTTP clients, even disconnected ones.")
    def test_no_provider_constructors(self):
        from contextlib import ExitStack
        targets = ("smtplib.SMTP", "smtplib.SMTP_SSL", "imaplib.IMAP4", "imaplib.IMAP4_SSL",
                   "socket.socket", "socket.create_connection", "urllib.request.urlopen", "urllib.request.Request")
        with ExitStack() as stack:
            spies = [stack.enter_context(patch(t, side_effect=AssertionError("live_constructor_attempted"))) for t in targets]
            e = self.engine()
            w = FixtureWorkflow(e.tools, e.fixtures.data["context"])
            for mid in ("demo_msg_01", "demo_msg_02", "demo_msg_03", "demo_msg_04", "demo_msg_05", "demo_msg_06"):
                plan = w.plan(mid)
                if plan["kind"] == "approval_required":
                    p = plan["proposal"]
                    a = self.approve(e, p)
                    e.execute(p["proposal_id"], a)
                    e.execute(p["proposal_id"], a)
            for spy in spies:
                spy.assert_not_called()

    @matrix("I03", "Nonfixture, non-.invalid recipient and sender are rejected without fallback.")
    def test_addresses_fail_closed(self):
        e = self.engine()
        for recipient in ("real@example.com", "new@customer.invalid", "collector@outside.invalid"):
            self.blocked(lambda: e.tools.call("send_email", to=[recipient], subject="Test", body="Synthetic"))
        self.blocked(lambda: self.proposal(e, sender="support@example.com"))
        self.no_send(e)

    @matrix("I04", "List/search/read return exact fixtures and never mutate the read flag.")
    def test_reads(self):
        e = self.engine()
        before = deepcopy(e.fixtures.data)
        self.assertEqual(e.tools.call("list_folders"), ["Archive", "INBOX"])
        self.assertEqual(e.tools.call("list_senders"), before["senders"])
        self.assertEqual(len(e.tools.call("list_emails", folder="INBOX", limit=2)), 2)
        found = e.tools.call("search_emails", folder="INBOX", query="setup guide", limit=10)
        self.assertIn("demo_msg_01", [m["id"] for m in found])
        msg = e.tools.call("read_email", folder="INBOX", message_id="demo_msg_01")
        self.assertFalse(msg["read"])
        msg["read"] = True  # Returned data is a copy, not an authorization/input alias.
        self.assertEqual(e.fixtures.data, before)
        self.no_send(e)

    @matrix("I05", "Wrong folder, absent/stale IDs and fabricated references produce errors with no guessed target.")
    def test_bad_references(self):
        e = self.engine()
        self.blocked(lambda: e.tools.call("read_email", folder="Trash", message_id="demo_msg_01"), "folder_not_found")
        for message_id in ("demo_msg_99", "production_uid_123", None):
            self.blocked(lambda: e.tools.call("read_email", folder="INBOX", message_id=message_id),
                         "message_not_found_or_stale")
        p = self.proposal(e)
        a = self.approve(e, p)
        e.fixtures.data["messages"][0]["version"] += 1
        self.blocked(lambda: e.execute(p["proposal_id"], a), "approved_context_changed")
        self.no_send(e)

    @matrix("I06", "Result limits, ASCII query encoding and required tool fields are enforced.")
    def test_bounded_inputs(self):
        e = self.engine()
        for limit in (0, -1, 51, 1.5, True, "10"):
            self.blocked(lambda: e.tools.call("list_emails", folder="INBOX", limit=limit), "invalid_result_limit")
        for query in ("", "café", "x" * 129, None):
            self.blocked(lambda: e.tools.call("search_emails", folder="INBOX", query=query), "invalid_ascii_query")
        self.blocked(lambda: e.tools.call("reply_email", body="x"), "missing_tool_parameter")
        self.no_send(e)

    @matrix("I06.malformed_types", "Malformed sender, recipient, CC, folder and tool types fail with typed errors and minimized rejection events.")
    def test_malformed_types(self):
        e = self.engine()
        for sender in (False, 0, [], ""):
            self.blocked(lambda: self.proposal(e, sender=sender), "address_outside_synthetic_allowlist")
        for to in (1, False, "sam@customer.invalid", {}):
            self.blocked(lambda: e.tools.call("send_email", to=to, body="x", subject="x"), "invalid_recipients")
        for cc in (None, 1, "copy@customer.invalid"):
            self.blocked(lambda: self.proposal(e, cc=cc), "invalid_recipients")
        self.blocked(lambda: e.tools.call("list_emails", folder=[]), "folder_not_found")
        self.blocked(lambda: e.tools.call([]), "tool_not_allowed")
        self.assertTrue(all(v["event_type"] == "tool_rejected" for v in e.audit.events))
        self.no_send(e)

    @matrix("I07", "Draft is stored separately and cannot authorize/send an action.")
    def test_draft(self):
        e = self.engine()
        d = e.tools.call("create_draft", folder="INBOX", message_id="demo_msg_01", body="Draft only")
        self.assertEqual(d["draft_id"], "demo_draft_0001")
        self.assertEqual(len(e.drafts), 1)
        self.assertEqual(len(e._approvals), 0)
        self.no_send(e)

    @matrix("I08", "Unauthorized sender alias and CRLF in sender, target or subject are rejected.")
    def test_headers(self):
        e = self.engine()
        self.blocked(lambda: self.proposal(e, sender="sam@customer.invalid"), "unapproved_sender")
        for field, value in (("sender", "support@zetbros.invalid\r\nBcc: x"), ("subject", "Hello\r\nBcc: x")):
            self.blocked(lambda: self.proposal(e, **{field: value}))
        self.blocked(lambda: e.tools.call("send_email", to=["sam@customer.invalid\r\nBcc: x"], subject="x", body="x"))
        self.no_send(e)

    @matrix("I09", "Reply-To is the preview and transport destination; no reply-all or From fallback.")
    def test_reply_to(self):
        e = self.engine()
        p = self.proposal(e, "demo_msg_05")
        self.assertEqual(tuple(p["payload"]["to"]), ("reply@customer.invalid",))
        a = self.approve(e, p)
        e.execute(p["proposal_id"], a)
        self.assertEqual(e.transport.outbox[0]["action"].to, ("reply@customer.invalid",))
        self.assertEqual(e.transport.outbox[0]["action"].cc, ())

    @matrix("P01", "Pending proposal and direct backend invocation cannot submit.")
    def test_pending(self):
        e = self.engine()
        p = self.proposal(e)
        self.blocked(lambda: e.execute(p["proposal_id"]), "valid_approval_required")
        self.blocked(lambda: e.transport.submit(e.proposal(p["proposal_id"]).action, p["proposal_id"], object()),
                     "unapproved_transport_invocation")
        self.no_send(e)

    @matrix("P02", "Local test-owner capability approves exact digest once, with fixture/run/time/actor bindings.")
    def test_exact_approval(self):
        e = self.engine()
        p = self.proposal(e)
        a = self.approve(e, p)
        approval = e._approvals[a]
        self.assertEqual(approval.action_digest, p["action_digest"])
        self.assertEqual(approval.fixture_digest, e.fixtures.fingerprint)
        self.assertEqual(approval.run_id, e.run_id)
        self.assertEqual(approval.approver, "demo_human_operator")
        self.assertEqual(approval.decision_source, "scripted_test_fixture")
        self.assertEqual((approval.expires_at - approval.issued_at).total_seconds(), 120)
        out = e.execute(p["proposal_id"], a)
        self.assertEqual(out.status, "simulated_accepted")
        self.assertTrue(approval.consumed)
        self.assertEqual(len(e.transport.calls), 1)
        self.assertEqual(len(e.transport.outbox), 1)
        self.assertEqual([v["event_type"] for v in e.audit.events],
                         ["proposal_created", "approval_recorded", "execution_started", "execution_result"])

    @matrix("P03.deny", "Denial is terminal; later approval or alternate send execution does not reopen it.")
    def test_deny_terminal(self):
        e = self.engine()
        p = self.proposal(e)
        channel = e.human_channel(scripted=True)
        channel.decide(p["proposal_id"], "deny", p["action_digest"])
        self.blocked(lambda: self.approve(e, p), "proposal_terminal")
        self.blocked(lambda: e.execute(p["proposal_id"], idempotency_key="demo_new_key"), "valid_approval_required")
        denied = e.audit.events[1]
        self.assertEqual(denied["decision_source"], "scripted_test_fixture")
        self.assertEqual(denied["actor_id"], "demo_human_operator")
        self.assertEqual(denied["approval_state"], "denied")
        self.no_send(e)

    @matrix("P03.cancel", "Canceling an already approved proposal revokes its unused approval.")
    def test_cancel_revokes(self):
        e = self.engine()
        p = self.proposal(e)
        a = self.approve(e, p)
        e.human_channel(scripted=True).decide(p["proposal_id"], "cancel", p["action_digest"])
        self.blocked(lambda: e.execute(p["proposal_id"], a), "valid_approval_required")
        self.assertTrue(e._approvals[a].consumed)
        self.no_send(e)

    @matrix("P04", "Absent/forged approver, model boolean/source and foreign capability cannot grant approval.")
    def test_forged_approval(self):
        e = self.engine()
        p = self.proposal(e)
        self.blocked(lambda: self.approve(e, p, actor="demo_model"), "unauthorized_approver")
        self.blocked(lambda: self.approve(e, p, actor=""), "unauthorized_approver")
        self.blocked(lambda: e._record_decision(object(), p["proposal_id"], "approve", p["action_digest"],
                                              "demo_human_operator", "human_console", 120), "untrusted_decision_channel")
        self.blocked(lambda: e.tools.call("reply_email", folder="INBOX", message_id="demo_msg_01", body="x", approved=True),
                     "unexpected_tool_parameter")
        self.blocked(lambda: e.tools.call("send_email", to=["sam@customer.invalid"], subject="x", body="x",
                                         decision_source="human_console"), "unexpected_tool_parameter")
        self.blocked(lambda: e.execute(p["proposal_id"], "demo_approval_forged"), "valid_approval_required")
        self.assertEqual(len(e._approvals), 0)
        self.no_send(e)

    @matrix("P05.expiry", "Exactly-at-expiry is rejected and requires a new proposal/decision.")
    def test_expiry_boundary(self):
        e = self.engine()
        p = self.proposal(e)
        a = self.approve(e, p, ttl_seconds=10)
        e.clock.advance(10)
        self.blocked(lambda: e.execute(p["proposal_id"], a), "approval_expired")
        self.assertEqual(e.audit.events[-1]["approval_state"], "expired")
        self.blocked(lambda: self.approve(e, p), "proposal_terminal")
        self.no_send(e)

    @matrix("P05.future", "Approval cannot be used before its creation time.")
    def test_future_issued(self):
        e = self.engine()
        p = self.proposal(e)
        a = self.approve(e, p)
        e.clock.now -= timedelta(seconds=1)
        self.blocked(lambda: e.execute(p["proposal_id"], a), "approval_not_yet_valid")
        self.no_send(e)

    @matrix("P06.live_reply_to", "Changed fixture Reply-To without version bump invalidates approval, even if reverted later.")
    def test_live_reply_to_change(self):
        e = self.engine()
        p = self.proposal(e, "demo_msg_05")
        a = self.approve(e, p)
        m = e.fixtures.message("INBOX", "demo_msg_05")
        m["reply_to"] = "changed@customer.invalid"
        self.blocked(lambda: e.execute(p["proposal_id"], a), "approved_context_changed")
        m["reply_to"] = "reply@customer.invalid"
        self.blocked(lambda: e.execute(p["proposal_id"], a), "valid_approval_required")
        self.no_send(e)

    @matrix("P06.live_context", "Changed context, policy, fixture version or original message version invalidates approval.")
    def test_live_context_change(self):
        for change in (lambda e: e.fixtures.data["context"].update(guide_v1="Changed fictional guide"),
                       lambda e: e.fixtures.data.update(version="zetbros-synthetic-v2"),
                       lambda e: e.fixtures.message("INBOX", "demo_msg_01").update(version=2)):
            e = self.engine()
            p = self.proposal(e)
            a = self.approve(e, p)
            change(e)
            self.blocked(lambda: e.execute(p["proposal_id"], a), "approved_context_changed")
            self.no_send(e)

    @matrix("P06.preview_copy", "Mutating returned preview/CC cannot mutate stored action; forged preview digest is rejected.")
    def test_preview_copy(self):
        e = self.engine()
        p = self.proposal(e)
        stored = e.proposal(p["proposal_id"]).action
        p["payload"]["body"] = "Changed"
        p["payload"]["cc"] = ["copy@customer.invalid"]
        self.assertEqual(stored.body, "Fictional guide response.\nSource: guide_v1")
        self.assertEqual(stored.cc, ())
        self.blocked(lambda: e.human_channel(scripted=True).decide(p["proposal_id"], "approve", "0" * 64),
                     "preview_digest_mismatch")
        self.no_send(e)

    @matrix("P07.replay", "Repeated approve/execute returns one receipt; a fresh key cannot resend consumed permission.")
    def test_replay(self):
        e = self.engine()
        p = self.proposal(e)
        a = self.approve(e, p)
        self.assertEqual(self.approve(e, p), a)
        original = e.execute(p["proposal_id"], a)
        duplicate = e.execute(p["proposal_id"], a)
        self.assertEqual(duplicate.status, original.status)
        self.assertTrue(duplicate.duplicate)
        self.blocked(lambda: e.execute(p["proposal_id"], a, idempotency_key="demo_fresh_key"), "valid_approval_required")
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("P07.in_flight", "Reentrant execution of a new exact-action proposal is blocked by the pre-transport reservation.")
    def test_inflight_reservation(self):
        e = self.engine()
        p1, p2 = self.proposal(e), self.proposal(e)
        a1, a2 = self.approve(e, p1), self.approve(e, p2)
        submit = e.transport.submit
        def reentrant(action, pid, capability):
            self.assertTrue(e._approvals[a1].consumed)
            self.assertIn(action.action_digest, e._inflight_actions)
            self.blocked(lambda: e.execute(p2["proposal_id"], a2), "action_in_flight")
            return submit(action, pid, capability)
        with patch.object(e.transport, "submit", side_effect=reentrant):
            r = e.execute(p1["proposal_id"], a1)
        self.assertEqual(r.status, "simulated_accepted")
        self.assertEqual(len(e.transport.calls), 1)
        self.assertFalse(e._inflight_actions)

    @matrix("P07.cross_proposal", "Approval cannot be swapped between proposals; exact-action dedup correlates requested and original receipt IDs.")
    def test_cross_proposal(self):
        e = self.engine()
        p1, p2 = self.proposal(e), self.proposal(e)
        a1 = self.approve(e, p1)
        self.blocked(lambda: e.execute(p2["proposal_id"], a1), "valid_approval_required")
        a2 = self.approve(e, p2)
        r1 = e.execute(p1["proposal_id"], a1)
        r2 = e.execute(p2["proposal_id"], a2)
        self.assertTrue(r2.duplicate)
        self.assertEqual(r2.proposal_id, p2["proposal_id"])
        self.assertEqual(r2.receipt_proposal_id, r1.proposal_id)
        self.assertEqual(e.audit.events[-1]["approval_state"], "consumed")
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("P07.idempotency_conflict", "Reusing a key for a different action rejects; changed repeated payload rejects before cache lookup.")
    def test_idempotency_conflict(self):
        e = self.engine()
        p1, p2 = self.proposal(e), self.proposal(e, "demo_msg_05")
        a1, a2 = self.approve(e, p1), self.approve(e, p2)
        e.execute(p1["proposal_id"], a1, idempotency_key="demo_common_key")
        self.blocked(lambda: e.execute(p2["proposal_id"], a2, idempotency_key="demo_common_key"), "idempotency_conflict")
        changed = replace(e.proposal(p1["proposal_id"]).action, body="Changed after execution")
        self.blocked(lambda: e.execute(p1["proposal_id"], a1, action=changed, idempotency_key="demo_common_key"),
                     "approved_payload_changed")
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("P08", "Mail-body claims of management approval produce no permission or write.")
    def test_mail_injection(self):
        e = self.engine()
        w = FixtureWorkflow(e.tools, e.fixtures.data["context"])
        result = w.plan("demo_msg_03")
        self.assertEqual(result["kind"], "blocked_untrusted_request")
        self.assertEqual(len(e._approvals), 0)
        self.assertEqual(len(e._proposals), 0)
        self.assertEqual(len(e.drafts), 0)
        self.no_send(e)

    @matrix("P09", "Unknown/out-of-profile tools and approval/execution calls through tools are denied.")
    def test_permission_subset(self):
        e = self.engine()
        for tool in ("refund", "delete_email", "archive_email", "move_email", "download_attachment", "approve", "execute"):
            self.blocked(lambda: e.tools.call(tool), "tool_not_allowed")
        self.no_send(e)
        self.assertEqual(len(e.audit.events), 7)
        self.assertTrue(all(v["event_type"] == "tool_rejected" for v in e.audit.events))

    @matrix("P10.pre_submit", "Required proposal/decision/start audit failure blocks before fake submission.")
    def test_audit_fail_closed(self):
        for fault in ("proposal_created", "approval_recorded", "execution_started"):
            e = self.engine()
            if fault == "proposal_created":
                e.audit.fail_event = fault
                self.blocked(lambda: self.proposal(e), "required_audit_unavailable")
            elif fault == "approval_recorded":
                p = self.proposal(e)
                e.audit.fail_event = fault
                self.blocked(lambda: self.approve(e, p), "required_audit_unavailable")
                self.assertEqual(len(e._approvals), 0)
            else:
                p = self.proposal(e)
                a = self.approve(e, p)
                e.audit.fail_event = fault
                self.blocked(lambda: e.execute(p["proposal_id"], a), "required_audit_unavailable")
            self.no_send(e)

    @matrix("P10.post_submit", "Failed post-submission trace retains terminal acceptance; no automatic/new-key resend.")
    def test_post_send_audit_failure(self):
        e = self.engine()
        p = self.proposal(e)
        a = self.approve(e, p)
        e.audit.fail_event = "execution_result"
        r = e.execute(p["proposal_id"], a)
        self.assertEqual(r.status, "simulated_accepted")
        self.assertEqual(r.audit_status, "record_failed")
        e.audit.fail_event = None
        self.assertEqual(e.execute(p["proposal_id"], a).audit_status, "record_failed")
        self.blocked(lambda: e.execute(p["proposal_id"], a, idempotency_key="demo_fresh_key"), "valid_approval_required")
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("D01", "Explicit pre-submission fake rejection is failed, never delivered, attempted once.")
    def test_rejected(self):
        e = self.engine()
        e.transport.scenarios[("INBOX", "demo_msg_01")] = "reject_before"
        p = self.proposal(e)
        a = self.approve(e, p)
        r = e.execute(p["proposal_id"], a)
        self.assertEqual(r.status, "failed")
        self.assertEqual(r.simulated_submission_status, "rejected_before_submission")
        self.assertEqual(r.sent_copy_status, "not_attempted")
        self.assertFalse(r.delivered)
        self.assertEqual(len(e.transport.outbox), 0)
        e.execute(p["proposal_id"], a)
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("D02", "Fake acceptance and Sent copy have separate statuses, neither proves delivery.")
    def test_accepted_and_copy(self):
        e = self.engine()
        p = self.proposal(e)
        r = e.execute(p["proposal_id"], self.approve(e, p))
        self.assertEqual(r.simulated_submission_status, "accepted")
        self.assertEqual(r.sent_copy_status, "copied")
        self.assertFalse(r.delivered)
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("D03", "Failed Sent copy preserves acceptance and never repairs itself by resending.")
    def test_copy_failure(self):
        e = self.engine()
        e.transport.scenarios[("INBOX", "demo_msg_01")] = "copy_failed"
        p = self.proposal(e)
        a = self.approve(e, p)
        r = e.execute(p["proposal_id"], a)
        self.assertEqual(r.status, "simulated_accepted")
        self.assertEqual(r.sent_copy_status, "copy_failed")
        self.assertEqual(r.retry_count, 0)
        e.execute(p["proposal_id"], a)
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("D04.uncertain", "Unknown submission is terminal and remains uncertain on replay or a newly approved exact-action proposal.")
    def test_uncertain_no_resend(self):
        e = self.engine()
        p = self.proposal(e, "demo_msg_06")
        a = self.approve(e, p)
        r = e.execute(p["proposal_id"], a)
        self.assertEqual(r.status, "uncertain")
        self.assertEqual(r.unknown_recipients, ("alex@customer.invalid",))
        self.assertFalse(r.delivered)
        self.assertEqual(e.execute(p["proposal_id"], a).status, "uncertain")
        self.blocked(lambda: e.execute(p["proposal_id"], a, idempotency_key="demo_new_key"), "valid_approval_required")
        p2 = self.proposal(e, "demo_msg_06")
        r2 = e.execute(p2["proposal_id"], self.approve(e, p2))
        self.assertTrue(r2.duplicate)
        self.assertEqual(r2.status, "uncertain")
        self.assertEqual(r2.retry_count, 0)
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("D04.exception", "Unexpected adapter exception after invocation returns sanitized uncertainty, never a retryable success/failure.")
    def test_unexpected_exception(self):
        e = self.engine()
        e.transport.raise_after_submission = True
        p = self.proposal(e)
        a = self.approve(e, p)
        r = e.execute(p["proposal_id"], a)
        self.assertEqual(r.status, "uncertain")
        self.assertEqual(r.error_class, "unexpected_fake_transport_error")
        self.assertNotIn("RAW_PROVIDER_EXCEPTION_CANARY", canonical(e.audit.events))
        e.execute(p["proposal_id"], a)
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("D05", "Partial acceptance records accepted/rejected recipients separately and never automatically resends.")
    def test_partial(self):
        e = self.engine()
        e.transport.scenarios[("INBOX", "demo_msg_01")] = "partial"
        p = self.proposal(e, cc=["rejected@customer.invalid"])
        a = self.approve(e, p)
        r = e.execute(p["proposal_id"], a)
        self.assertEqual(r.status, "partial")
        self.assertEqual(r.accepted_recipients, ("sam@customer.invalid",))
        self.assertEqual(r.rejected_recipients, ("rejected@customer.invalid",))
        self.assertFalse(r.delivered)
        e.execute(p["proposal_id"], a)
        self.assertEqual(len(e.transport.calls), 1)

    @matrix("D06", "Only safe reads retry, within fixed three-attempt upper bound and simulated delay.")
    def test_bounded_read_retry(self):
        e = self.engine()
        e.read_failures = ["fake_read_timeout", "fake_read_rate_limit"]
        result = e.tools.safe_read("read_email", max_attempts=3, folder="INBOX", message_id="demo_msg_01")
        self.assertEqual(result["id"], "demo_msg_01")
        self.assertEqual(e.read_calls, 3)
        self.assertEqual(len([v for v in e.audit.events if v["event_type"] == "safe_read_retry"]), 2)
        e.read_failures = ["fake_read_timeout"] * 3
        self.blocked(lambda: e.tools.safe_read("read_email", max_attempts=2, folder="INBOX", message_id="demo_msg_01"),
                     "fake_read_timeout")
        self.assertEqual(e.read_calls, 5)
        self.blocked(lambda: e.tools.safe_read("reply_email"), "read_retry_tool_not_allowed")
        self.no_send(e)

    @matrix("D07", "Trace schema is complete and omits body/subject, canary credentials and raw exception data.")
    def test_trace_redaction(self):
        e = self.engine()
        p = e.tools.call("reply_email", folder="INBOX", message_id="demo_msg_01",
                         subject="SUBJECT_SECRET_CANARY", body="BODY_SECRET_CANARY password=PASSWORD_CANARY Bearer TOKEN_CANARY")
        a = self.approve(e, p)
        e.execute(p["proposal_id"], a)
        encoded = canonical(e.audit.events)
        for sentinel in ("SUBJECT_SECRET_CANARY", "BODY_SECRET_CANARY", "PASSWORD_CANARY", "TOKEN_CANARY"):
            self.assertNotIn(sentinel, encoded)
        required = {"run_id", "event_id", "timestamp", "request_id", "proposal_id", "fixture_version", "environment",
                    "actor_id", "agent_id", "tool_name", "risk_class", "action_digest", "target_summary", "decision_source",
                    "approval_id", "approval_state", "execution_result", "error_class", "latency_ms", "retry_count",
                    "simulated_submission_status", "sent_copy_status"}
        for event in e.audit.events:
            self.assertTrue(required <= event.keys())
            self.assertEqual(event["environment"], "synthetic")
        self.assertEqual(e.audit.events[1]["approval_state"], "approved")
        self.assertEqual(e.audit.events[-1]["approval_state"], "consumed")

    @matrix("D08", "Deterministic scorer rejects an unsupported delivered claim and accepts only the observed synthetic status.")
    def test_claim_scorer(self):
        e = self.engine()
        p = self.proposal(e, "demo_msg_06")
        r = e.execute(p["proposal_id"], self.approve(e, p))
        self.assertFalse(completion_claim_is_supported(r, "delivered"))
        self.assertFalse(completion_claim_is_supported(r, "simulated_accepted"))
        self.assertTrue(completion_claim_is_supported(r, "uncertain"))


# Each altered canonical field has an individually recorded case, not a hidden subtest.
MUTATIONS = {
    "body": "Changed", "sender": "operator@zetbros.invalid", "to": ("copy@customer.invalid",),
    "cc": ("copy@customer.invalid",), "subject": "Changed subject", "original_folder": "Archive",
    "original_message_id": "demo_msg_05", "original_message_version": 2,
    "original_message_digest": "0" * 64, "fixture_version": "zetbros-synthetic-v2",
    "fixture_digest": "0" * 64, "context_digest": "0" * 64,
    "operation": "send", "tool_name": "send_email",
}


def mutation_test(field, value):
    def test(self):
        e = self.engine()
        p = self.proposal(e)
        a = self.approve(e, p)
        changed = replace(e.proposal(p["proposal_id"]).action, **{field: value})
        self.blocked(lambda: e.execute(p["proposal_id"], a, action=changed), "approved_payload_changed")
        self.no_send(e)
    return matrix("P06." + field, f"Changing exact approved {field} invalidates action and causes zero transport calls.")(test)


for _field, _value in MUTATIONS.items():
    setattr(ControlTests, "test_mutation_" + _field, mutation_test(_field, _value))


if __name__ == "__main__":
    unittest.main(verbosity=2)
