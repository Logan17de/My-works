"""Owner-shaped fictional ports only. No supplied code, sockets or credentials."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

import test_service as base
from test_spacemail_contract import BINDING, PLAN, source_wire
from zetbros_service.adapters import DisabledTransport, SourceUnavailable
from zetbros_service.api import configured_app, create_app
from zetbros_service.auth import Principal
from zetbros_service.mail_integration import MailService, MailStore, review_digest
from zetbros_service.mail_owner import (OwnerMailSource, decode_owner_message_id,
                                      normalize_owner_result, owner_message_id)
from zetbros_service.models import DecisionInput, ProposalInput, ReplyAction, TransportResult, canonical, digest
from zetbros_service.spacemail_contract import ContractError, SmtpOutcome, prepare_reply
from zetbros_service.store import Store, StoreError


class FictionalOwner:
    """The exact inspected read interface, intentionally without convenience tools."""
    def __init__(self):
        self.raw = source_wire()
        self.untagged_responses = {"UIDVALIDITY": [b"123"]}
        self.calls = []
        self.size_override = None
        self.uid_override = None
        self.allowed = [BINDING.sender_address]
        self.on_open = None

    def list_senders(self):
        return {"default_sender": self.allowed[0], "allowed_senders": list(self.allowed)}

    @contextmanager
    def mailbox(self, folder, readonly=True):
        if self.on_open:
            self.on_open()
        self.calls.append(("mailbox", folder, readonly))
        yield self

    def uid(self, command, uid, items):
        self.calls.append((command, uid, items))
        returned_uid = self.uid_override or uid
        if items == "(UID RFC822.SIZE)":
            size = self.size_override if self.size_override is not None else len(self.raw)
            return "OK", [f"7 (UID {returned_uid} RFC822.SIZE {size})".encode()]
        if items == "(UID BODY.PEEK[]<0.65537>)":
            return "OK", [(f"7 (UID {returned_uid} BODY[]<0> {{{len(self.raw)}}}".encode(), self.raw), b")"]
        raise AssertionError("Unexpected mailbox operation")


class FictionalPreparedPort:
    enabled = True
    def __init__(self):
        self.wires = []
        self.calls = 0
        self.sent_copy = "stored"
        self.drop_ack = False
        self.before_gate = None
        self.skip_gate = False

    async def submit_prepared(self, prepared, before_body):
        self.calls += 1
        state = SmtpOutcome()
        state.response("mail", 250); state.response("rcpt", 250); state.response("data", 354)
        if self.before_gate:
            self.before_gate()
        if not self.skip_gate:
            try:
                before_body()
            except StoreError:
                return state.interrupted()
        state.begin_body()
        self.wires.append(prepared.wire)
        if self.drop_ack:
            return state.interrupted("transport_timeout")
        state.response("final", 250)
        state.sent_copy(self.sent_copy)
        return state.result()


class OwnerInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.owner = FictionalOwner()
        self.source = OwnerMailSource(BINDING, self.owner)

    def test_opaque_ids_roundtrip_with_exact_inspected_format(self):
        expected = "v1_" + base64.urlsafe_b64encode(b'["INBOX","123","42"]').decode().rstrip("=")
        self.assertEqual(owner_message_id(PLAN), expected)
        self.assertEqual(decode_owner_message_id(expected), PLAN)
        self.assertEqual(self.source.get_owner(expected).source_id, PLAN.source_id)

    def test_opaque_ids_reject_padding_alternate_folder_values_and_ranges(self):
        def encode(value):
            return "v1_" + base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
        invalid = [owner_message_id(PLAN) + "=", "v1_../", "v1_" + "A" * 100,
                   *[encode(v) for v in (["Sent", "123", "42"], ["INBOX", 123, "42"],
                     ["INBOX", "123", "01"], ["INBOX", "0", "42"], ["INBOX", "123", "4294967296"],
                     ["INBOX", "123", "42:*"], ["INBOX", "１２３", "42"], ["INBOX", "123", "42", "extra"])]]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ContractError):
                decode_owner_message_id(value)
        self.assertEqual(self.owner.calls, [])

    def test_owner_mailbox_read_is_exact_peek_plan_without_convenience_tools(self):
        with patch("socket.create_connection", side_effect=AssertionError("network attempted")):
            record = self.source.get(PLAN.source_id)
        self.assertEqual(self.owner.calls, [("mailbox", "INBOX", True),
            ("FETCH", "42", "(UID RFC822.SIZE)"), ("FETCH", "42", "(UID BODY.PEEK[]<0.65537>)")])
        self.assertEqual(record.version, "rfc822:" + hashlib.sha256(self.owner.raw).hexdigest())
        self.assertEqual(record.reply_to, "replyto@example.invalid")
        self.assertEqual(record.rfc_references, ("<earlier@example.invalid>",))

    def test_uidvalidity_rollover_refuses_before_fetch(self):
        self.owner.untagged_responses["UIDVALIDITY"] = [b"124"]
        with self.assertRaises(SourceUnavailable): self.source.get(PLAN.source_id)
        self.assertEqual(self.owner.calls, [("mailbox", "INBOX", True)])

    def test_wrong_uid_and_oversize_refuse_before_body_fetch(self):
        for change in ({"uid_override": "41"}, {"size_override": 65537}):
            with self.subTest(change=change):
                self.owner = FictionalOwner(); self.source = OwnerMailSource(BINDING, self.owner)
                for key, value in change.items(): setattr(self.owner, key, value)
                with self.assertRaises(SourceUnavailable): self.source.get(PLAN.source_id)
                self.assertEqual(len(self.owner.calls), 2)

    def test_sender_allowlist_cannot_choose_an_unconfigured_alias(self):
        self.owner.allowed = ["other@example.invalid"]
        with self.assertRaises(SourceUnavailable): self.source.get(PLAN.source_id)
        self.assertEqual(self.owner.calls, [])

    def test_raw_provider_errors_are_discarded(self):
        def fail(): raise RuntimeError("PASSWORD_CANARY BODY_CANARY")
        self.owner.on_open = fail
        with self.assertRaises(SourceUnavailable) as raised: self.source.get(PLAN.source_id)
        self.assertNotIn("CANARY", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)


class MailIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.ServiceTests.setUpClass()

    def setUp(self):
        self.fixture = base.ServiceTests("test_accepted_sent_copy_failure_is_never_resent")
        self.fixture.jwk = base.ServiceTests.jwk
        self.fixture.key = base.ServiceTests.key
        self.fixture.setUp()
        self.owner = FictionalOwner()
        self.source = OwnerMailSource(BINDING, self.owner)
        # Explicitly recreate only this new temporary test ledger. No deployment
        # data or supplied archive artifacts are accessed by these tests.
        self.fixture.client.close()
        for suffix in ("", "-wal", "-shm"):
            from pathlib import Path
            path = Path(str(self.fixture.service.store.path) + suffix)
            if path.exists(): path.unlink()
        MailStore.initialize_offline(self.fixture.settings, self.fixture.clock, source=self.source)
        self.port = FictionalPreparedPort()
        self.service = MailService(self.fixture.settings, source=self.source, prepared_transport=self.port, clock=self.fixture.clock)
        self.client = TestClient(create_app(self.service, run_worker=False))

    def tearDown(self):
        self.client.close(); self.fixture.tearDown()

    def input(self, operation_key="ticket-owner-reply-0001"):
        record = self.source.get(PLAN.source_id)
        return ProposalInput(operation_key=operation_key, source_id=record.source_id,
                             source_version=record.version, source_fingerprint=digest(record), body="Exact reply\nwithout final newline")

    def propose(self, operation_key="ticket-owner-reply-0001"):
        return self.service.propose(self.input(operation_key), base.AGENT, str(uuid.uuid4()))

    def approve(self, proposal):
        return self.service.store.decide(proposal["id"], proposal["digest"], "approve", base.REVIEWER, str(uuid.uuid4()))

    def worker(self):
        asyncio.run(self.service.worker_once())

    def test_exact_action_wire_preview_and_composite_digest_persist_for_review(self):
        proposal = self.propose()
        action = ReplyAction.model_validate_json(canonical(proposal["payload"]))
        fresh = self.source.get(PLAN.source_id)
        prepared = prepare_reply(BINDING, action, fresh, now=int(self.fixture.clock()))
        self.assertEqual(proposal["wire_preview"], prepared.preview.model_dump(mode="json"))
        self.assertEqual(proposal["wire_preview_digest"], prepared.preview_digest)
        self.assertEqual(proposal["digest"], review_digest(action, prepared.preview))
        self.assertNotEqual(proposal["digest"], digest(action))
        restarted = MailService(self.fixture.settings, source=self.source, clock=self.fixture.clock)
        self.assertEqual(restarted.store.get(proposal["id"])["wire_preview"], proposal["wire_preview"])
        with self.service.store.connection() as conn:
            self.assertEqual(self.service.store.prepared(conn, proposal["id"], action).wire, prepared.wire)

    def test_action_only_preview_only_or_changed_digest_cannot_approve(self):
        proposal = self.propose()
        for shown in (proposal["action_digest"], proposal["wire_preview_digest"], "0" * 64):
            with self.subTest(shown=shown), self.assertRaises(StoreError):
                self.service.store.decide(proposal["id"], shown, "approve", base.REVIEWER, "review-test")
        self.assertIsNone(self.service.store.get(proposal["id"])["execution"])
        self.assertEqual(self.port.calls, 0)

    def test_authenticated_api_shows_full_wire_preview_and_agent_cannot_decide(self):
        # Reuse baseline token helper against the same existing configured issuer.
        token = self.fixture.token()
        proposal = self.propose()
        response = self.client.get("/v1/proposals/" + proposal["id"], headers={"Authorization": "Bearer " + token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["wire_preview"], proposal["wire_preview"])
        body = {"digest": proposal["digest"], "decision": "approve"}
        denied = self.client.post("/v1/reviews/" + proposal["id"] + "/decision", json=body,
                                 headers={"Authorization": "Bearer " + token})
        self.assertEqual(denied.status_code, 403)
        reviewer = self.fixture.token(subject=base.REVIEWER.subject,client=base.REVIEWER.client_id)
        accepted = self.client.post("/v1/reviews/" + proposal["id"] + "/decision", json=body,
                                   headers={"Authorization": "Bearer " + reviewer})
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json()["execution"]["digest"], proposal["digest"])

    def test_wire_preview_is_immutable_in_ledger(self):
        proposal = self.propose()
        for statement in ("UPDATE mail_previews SET preview='{}' WHERE proposal_id=?", "DELETE FROM mail_previews WHERE proposal_id=?"):
            with self.subTest(statement=statement), self.service.store.connection() as conn, self.assertRaises(sqlite3.IntegrityError):
                conn.execute(statement, (proposal["id"],))

    def test_exact_claim_gate_and_wire_no_retry_survive_restart(self):
        proposal = self.propose(); self.approve(proposal); self.worker(); self.worker()
        final = self.service.store.get(proposal["id"])
        self.assertEqual(final["execution"]["state"], "accepted")
        self.assertEqual(hashlib.sha256(self.port.wires[0]).hexdigest(), proposal["wire_preview"]["wire_sha256"])
        events = [x for x in self.service.store.audit() if x["action_id"] == proposal["id"]]
        self.assertEqual([x["event_type"] for x in events], ["proposal", "decision", "claim", "submission_gate", "submission_gate", "result"])
        self.assertTrue(all(x["action_digest"] == proposal["digest"] for x in events))
        restarted = MailService(self.fixture.settings, source=self.source, prepared_transport=self.port, clock=self.fixture.clock)
        asyncio.run(restarted.worker_once())
        self.assertEqual(self.port.calls, 1)

    def test_preclaim_thread_source_drift_invalidates_without_consuming(self):
        proposal = self.propose(); self.approve(proposal)
        self.owner.raw = source_wire(message_id="<changed@example.invalid>")
        self.worker()
        final = self.service.store.get(proposal["id"])
        self.assertEqual(final["execution"]["state"], "invalidated")
        self.assertIsNone(final["consumed_at"])
        self.assertEqual(self.port.calls, 0)

    def test_source_changes_after_claim_cannot_reach_prepared_port(self):
        proposal = self.propose(); self.approve(proposal)
        original = self.service.store.claim
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            self.owner.raw = source_wire(reply_to="changed@example.invalid")
            return result
        with patch.object(self.service.store, "claim", side_effect=changed): self.worker()
        final = self.service.store.get(proposal["id"])
        self.assertEqual(final["execution"]["state"], "rejected")
        self.assertIsNotNone(final["consumed_at"])
        self.assertEqual(final["execution"]["result"]["error_class"], "source_revalidation_failed")
        self.assertEqual(self.port.calls, 0)
        self.owner.raw = source_wire(); self.worker()
        self.assertEqual(self.port.calls, 0)

    def test_shorter_approval_expiry_is_rechecked_before_body(self):
        proposal = self.propose(); approved = self.approve(proposal)
        self.port.before_gate = lambda: setattr(self.fixture.clock, "now", approved["approval_expires_at"])
        # Default approval TTL exceeds the 60-second claim bound, so use a
        # reviewed short approval configuration to isolate the approval fence.
        short = self.fixture.settings.model_copy(update={"approval_ttl_seconds": 30})
        self.service.settings = short; self.service.store.settings = short
        with self.service.store.connection() as conn:
            conn.execute("UPDATE proposals SET approval_expires_at=? WHERE id=?", (self.fixture.clock.now + 30, proposal["id"]))
        self.port.before_gate = lambda: setattr(self.fixture.clock, "now", self.fixture.clock.now + 30)
        self.worker()
        self.assertEqual(self.port.wires, [])
        self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"], "rejected")
        self.worker(); self.assertEqual(self.port.calls, 1)

    def test_current_identity_removal_and_quarantine_gate_before_body(self):
        for mode in ("reviewer", "agent", "quarantine"):
            with self.subTest(mode=mode):
                proposal = self.propose("ticket-gate-mode-" + mode); self.approve(proposal)
                def change():
                    if mode == "quarantine":
                        with self.service.store.connection() as conn:
                            conn.execute("UPDATE metadata SET value='true' WHERE key='execution_quarantine'")
                    else:
                        keep = tuple(g for g in self.fixture.settings.principals if g.role != mode)
                        self.service.store.settings = self.fixture.settings.model_copy(update={"principals": keep})
                self.port.before_gate = change
                self.worker()
                self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"], "rejected")
                self.service.store.settings = self.fixture.settings
                with self.service.store.connection() as conn:
                    conn.execute("UPDATE metadata SET value='false' WHERE key='execution_quarantine'")
        self.assertEqual(self.port.wires, [])

    def test_missing_gate_acceptance_is_uncertain_and_never_retried(self):
        proposal = self.propose(); self.approve(proposal); self.port.skip_gate = True
        self.worker(); self.worker()
        self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"], "uncertain")
        self.assertEqual(self.port.calls, 1)

    def test_smtp_uncertain_and_sent_copy_failure_stay_distinct_without_retry(self):
        first = self.propose(); self.approve(first); self.port.drop_ack = True
        self.worker(); self.worker()
        self.assertEqual(self.service.store.get(first["id"])["execution"]["state"], "uncertain")
        self.port.drop_ack = False; self.port.sent_copy = "failed"
        second = self.propose("ticket-owner-copy-failed"); self.approve(second)
        self.worker(); self.worker()
        result = self.service.store.get(second["id"])["execution"]["result"]
        self.assertEqual((result["submission"], result["sent_copy"], result["delivery"]), ("accepted", "failed", "unverified"))
        self.assertEqual(self.port.calls, 2)

    def test_disabled_candidate_consumes_no_approval_and_configured_app_stays_base(self):
        disabled = MailService(self.fixture.settings, source=self.source, clock=self.fixture.clock)
        proposal = disabled.propose(self.input(), base.AGENT, "proposal-test"); self.approve(proposal)
        asyncio.run(disabled.worker_once())
        self.assertIsNone(disabled.store.get(proposal["id"])["consumed_at"])
        self.assertFalse(disabled.readiness()["live_delivery_ready"])
        self.assertIsInstance(disabled.transport, DisabledTransport)
        with patch("zetbros_service.api.load_settings", return_value=self.fixture.settings), self.assertRaises(RuntimeError):
            configured_app()
        with self.assertRaises(RuntimeError): Store(self.fixture.settings)
        # Production configured_app does not import/instantiate MailService.
        import inspect
        self.assertNotIn("MailService", inspect.getsource(configured_app))

    def test_idempotency_restart_conflict_and_other_agent_view(self):
        proposal = self.propose()
        self.fixture.clock.now += 1
        self.assertEqual(self.propose()["id"], proposal["id"])
        changed = self.input().model_copy(update={"body": "Changed exact text"})
        with self.assertRaises(StoreError):
            self.service.propose(changed, base.AGENT, "changed-proposal")
        with self.assertRaises(StoreError):
            self.service.view(proposal["id"], Principal("other-agent", "other-client", "agent"))
        self.assertEqual(self.service.store.get(proposal["id"])["wire_preview"], proposal["wire_preview"])

    def test_preview_audit_failure_rolls_back_both_records(self):
        with patch.object(self.service.store, "event", side_effect=StoreError("audit_failure", 503)), self.assertRaises(StoreError):
            self.propose()
        with self.service.store.connection() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM proposals").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM mail_previews").fetchone()[0], 0)

    def test_forged_claim_token_changed_wire_and_expired_claim_gate_refuse(self):
        proposal = self.propose(); self.approve(proposal)
        action, token = self.service.store.claim(proposal["id"], self.input().source_fingerprint, True, "claim-test")
        with self.service.store.connection() as conn:
            prepared = self.service.store.prepared(conn, proposal["id"], action)
        from zetbros_service.spacemail_contract import PreparedReply
        altered = PreparedReply(prepared.preview, prepared.wire + b"changed")
        for supplied_token, supplied_wire in (("forged-token", prepared), (token, altered)):
            with self.subTest(token=supplied_token), self.assertRaises(StoreError):
                self.service.store.submission_gate(proposal["id"], supplied_token, supplied_wire, "gate-test", "before_body")
        self.fixture.clock.now = self.service.store.get(proposal["id"])["execution"]["claim_deadline"]
        with self.assertRaises(StoreError):
            self.service.store.submission_gate(proposal["id"], token, prepared, "gate-test", "before_body")
        self.worker()
        self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"], "uncertain")
        self.assertEqual(self.port.calls, 0)

    def test_missing_preview_table_or_marker_never_falls_back_to_action_digest(self):
        proposal = self.propose()
        with self.service.store.connection() as conn:
            conn.execute("DELETE FROM metadata WHERE key='mail_review_schema'")
        with self.assertRaises(RuntimeError):
            MailService(self.fixture.settings, source=self.source, clock=self.fixture.clock)
        with self.assertRaises(RuntimeError): Store(self.fixture.settings)
        with patch("zetbros_service.api.load_settings", return_value=self.fixture.settings), self.assertRaises(RuntimeError):
            configured_app()
        with self.service.store.connection() as conn:
            conn.execute("INSERT INTO metadata VALUES ('mail_review_schema','1')")
            conn.execute("DROP TABLE mail_previews")
        with self.assertRaises(RuntimeError):
            MailService(self.fixture.settings, source=self.source, clock=self.fixture.clock)
        with self.assertRaises(StoreError):
            self.service.store.decide(proposal["id"], proposal["action_digest"], "approve", base.REVIEWER, "review-test")

    def test_restore_preserves_exact_preview_and_quarantines_unfinished_action(self):
        from zetbros_service.maintenance import backup, restore
        proposal = self.propose(); self.approve(proposal)
        destination = self.fixture.root / "mail-backup.sqlite3"
        backup(self.service.store, destination)
        settings = self.fixture.settings.model_copy(update={"database_path": str(self.fixture.root / "data/restored.sqlite3")})
        restore(settings, destination)
        restored = MailService(settings, source=self.source, prepared_transport=self.port, clock=self.fixture.clock)
        asyncio.run(restored.worker_once())
        view = restored.store.get(proposal["id"])
        self.assertEqual(view["wire_preview"], proposal["wire_preview"])
        self.assertEqual(view["digest"], proposal["digest"])
        self.assertEqual(view["execution"]["state"], "blocked")
        self.assertIsNone(view["consumed_at"])
        self.assertEqual(self.port.calls, 0)

    def test_duplicate_or_failed_body_gate_is_sticky_and_cannot_record_acceptance(self):
        proposal = self.propose(); self.approve(proposal)
        async def duplicate(prepared, gate):
            gate()
            try: gate()
            except StoreError: pass
            return TransportResult(submission="accepted", sent_copy="stored")
        self.port.submit_prepared = duplicate
        self.worker(); self.worker()
        final = self.service.store.get(proposal["id"])
        self.assertEqual(final["execution"]["state"], "uncertain")
        self.assertEqual(final["execution"]["result"]["error_class"], "invalid_adapter_result")

    def test_forged_pydantic_and_unknown_adapter_results_are_sanitized_uncertain(self):
        values = [TransportResult.model_construct(submission="accepted", sent_copy="RESULT_CANARY", error_class="none", delivery="unverified"),
                  TransportResult(submission="accepted", sent_copy="stored").model_copy(update={"submission": "RESULT_CANARY"}),
                  {"submission": "accepted", "sent_copy": "stored", "extra": "RESULT_CANARY"}]
        for index, value in enumerate(values):
            proposal = self.propose("ticket-forged-result-" + str(index)); self.approve(proposal)
            async def forged(prepared, gate, value=value):
                gate(); return value
            self.port.submit_prepared = forged
            self.worker()
            final = self.service.store.get(proposal["id"])
            self.assertEqual(final["execution"]["state"], "uncertain")
            self.assertNotIn("RESULT_CANARY", canonical(final["execution"]["result"]))

    def test_explicit_maintenance_handle_cannot_create_decide_or_claim(self):
        proposal = self.propose()
        maint = Store(self.fixture.settings, self.fixture.clock, maintenance_only=True)
        action = ReplyAction.model_validate_json(canonical(proposal["payload"]))
        for operation in (lambda: maint.create(action, "0" * 64, base.AGENT, "maintenance"),
                          lambda: maint.decide(proposal["id"], proposal["digest"], "approve", base.REVIEWER, "maintenance"),
                          lambda: maint.claim(proposal["id"], action.source_fingerprint, True, "maintenance")):
            with self.assertRaises(StoreError): operation()

    def test_invalid_locator_and_late_source_failures_are_sanitized_api_errors(self):
        headers = {"Authorization": "Bearer " + self.fixture.token()}
        response = self.client.get("/v1/source/message-001", headers=headers)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "source_unavailable")
        request = self.input()
        first = self.source.get(PLAN.source_id)
        changed_owner = FictionalOwner(); changed_owner.raw = source_wire(reply_to="changed@example.invalid")
        changed = OwnerMailSource(BINDING, changed_owner).get(PLAN.source_id)
        for late, status, code in ((SourceUnavailable("BODY_CANARY PASSWORD_CANARY"), 503, "source_unavailable"),
                                   (changed, 409, "source_or_preview_changed")):
            with self.subTest(status=status), patch.object(self.source, "get", side_effect=[first, late]):
                response = self.client.post("/v1/proposals", headers=headers, json=request.model_dump(mode="json"))
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["error"], code)
                self.assertNotIn("CANARY", response.text)
        with self.service.store.connection() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM proposals").fetchone()[0], 0)
        self.assertEqual(self.port.calls, 0)

    def test_owner_result_mapping_preserves_acceptance_uncertainty_and_sent_copy(self):
        proposal = self.propose()
        action = ReplyAction.model_validate_json(canonical(proposal["payload"]))
        with self.service.store.connection() as conn: prepared = self.service.store.prepared(conn, proposal["id"], action)
        common = {"from_email": BINDING.sender_address, "message_header_id": prepared.preview.message_id}
        for status, value, expected in (("saved", True, "stored"), ("already_present", True, "stored"),
                                       ("failed", False, "failed"), ("unknown", None, "unknown")):
            result = normalize_owner_result(common | {"sent": True, "status": "accepted_by_smtp", "refused_recipients": [],
                "sent_folder_copy_status": status, "sent_folder_copy": value, "detail": "PASSWORD_CANARY"}, prepared)
            self.assertEqual((result.submission, result.sent_copy), ("accepted", expected))
            self.assertNotIn("CANARY", canonical(result))
        for fields in ({"sent": None, "status": "delivery_unknown", "sent_folder_copy": False, "sent_folder_copy_status": "not_attempted"},
                       {"sent": True, "status": "partially_accepted", "refused_recipients": ["other@example.invalid"]},
                       {"sent": True, "status": "accepted_by_smtp", "refused_recipients": []},
                       {"sent": 1, "status": "accepted_by_smtp"}):
            self.assertEqual(normalize_owner_result(common | fields, prepared).submission, "uncertain")
        accepted = common | {"sent": True, "status": "accepted_by_smtp", "refused_recipients": [],
                             "sent_folder_copy_status": "saved", "sent_folder_copy": True}
        self.assertEqual(normalize_owner_result(accepted | {"smtp_code": 550}, prepared).submission, "uncertain")
        missing_copy = accepted | {"sent_folder_copy_status": "unknown"}
        del missing_copy["sent_folder_copy"]
        self.assertEqual(normalize_owner_result(missing_copy, prepared).submission, "uncertain")
        refused = common | {"sent": False, "status": "all_recipients_refused", "sent_folder_copy": False,
                           "sent_folder_copy_status": "not_attempted"}
        self.assertEqual(normalize_owner_result(refused, prepared).submission, "rejected")
        for refusal_list in ([], ["other@example.invalid"], [prepared.preview.to[0]]):
            self.assertEqual(normalize_owner_result(refused | {"refused_recipients": refusal_list}, prepared).submission, "uncertain")
        rejected = common | {"sent": False, "status": "rejected_by_smtp", "smtp_code": 550,
                             "sent_folder_copy": False, "sent_folder_copy_status": "not_attempted"}
        self.assertEqual(normalize_owner_result(rejected, prepared).submission, "rejected")
        self.assertEqual(normalize_owner_result(common | {"sent": True, "status": "accepted_by_smtp", "refused_recipients": [], "sent_folder_copy_status": []}, prepared).submission, "uncertain")
        for value in (True, 250, "550"):
            self.assertEqual(normalize_owner_result(rejected | {"smtp_code": value}, prepared).submission, "uncertain")
        self.assertEqual(normalize_owner_result(common | {"message_header_id": "<random-owner-id@example.invalid>"}, prepared).submission, "uncertain")
