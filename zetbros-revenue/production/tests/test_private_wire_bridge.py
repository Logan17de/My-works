"""Socket-free stdlib-shaped fake SMTP/IMAP sessions and real ledger integration."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

from pydantic import ValidationError

import test_mail_integration as integration
from test_spacemail_contract import BINDING, PLAN, action_for, source_wire
from zetbros_service.adapters import SourceUnavailable
from zetbros_service.mail_owner import OwnerMailSource
from zetbros_service.models import TransportResult, canonical
from zetbros_service.private_wire_bridge import (BoundSession, PrivateBridgeProfile, PrivatePreparedWireTransport,
    PrivateReadPort, SessionIdentity, smtp_data_frame, validate_prepared)
from zetbros_service.spacemail_contract import ContractError, parse_source, prepare_reply


PROFILE = PrivateBridgeProfile(binding=BINDING, mailbox_address=BINDING.sender_address)


class FakeSocket:
    def __init__(self):
        self.timeouts = []
        self.closed = threading.Event()
        self.shutdowns = []
    def settimeout(self, value): self.timeouts.append(value)
    def shutdown(self, value): self.shutdowns.append(value); self.closed.set()
    def close(self): self.closed.set()


class FakeSmtp:
    def __init__(self):
        self.sock = FakeSocket()
        self.calls = []
        self.codes = {"mail": 250, "rcpt": 250, "data": 354, "final": 250}
        self.raise_at = None
        self.after = None
        self.block_at = None
        self.blocked = threading.Event()
        self.bad_response = None
    def phase(self, name, *args):
        self.calls.append((name, *args))
        if name == self.block_at:
            self.blocked.set()
            if not self.sock.closed.wait(timeout=3): raise AssertionError("fake socket was not interrupted")
            raise OSError("PROVIDER_CANARY")
        if name == self.raise_at: raise OSError("PASSWORD_CANARY BODY_CANARY")
        if self.after: self.after(name)
        return self.bad_response if self.bad_response is not None else (self.codes.get(name, 250), b"provider response not logged")
    def mail(self, sender): return self.phase("mail", sender)
    def rcpt(self, recipient): return self.phase("rcpt", recipient)
    def docmd(self, command): return self.phase("data", command)
    def send(self, wire): self.phase("send", wire)
    def getreply(self): return self.phase("final")


class FakeImap(integration.FictionalOwner):
    def __init__(self):
        super().__init__()
        self.sock = FakeSocket()
        self.search = ("OK", [b""])
        self.append_status = "OK"
        self.raise_append = False
        self.block_append = False
        self.append_started = threading.Event()
    def uid(self, command, *args):
        if command == "SEARCH":
            self.calls.append((command, *args)); return self.search
        return super().uid(command, *args)
    def append(self, folder, flags, date, wire):
        self.calls.append(("APPEND", folder, flags, date, wire)); self.append_started.set()
        if self.block_append:
            if not self.sock.closed.wait(timeout=3): raise AssertionError("fake append not interrupted")
            raise OSError("APPEND_CANARY")
        if self.raise_append: raise OSError("APPEND_CANARY")
        return self.append_status, [b"ignored"]


class FakeFactory:
    offline_only = True
    def __init__(self):
        self.smtp_session = FakeSmtp(); self.imap_session = FakeImap()
        self.calls = []
        self.smtp_identity = None; self.imap_identity = None
        self.quit_error = False; self.imap_close_error = False
        self.quit_blocked = threading.Event(); self.block_quit = False
    def identity(self, profile, purpose):
        return SessionIdentity(binding=profile.binding, mailbox_address=profile.mailbox_address,
                               purpose=purpose, port=465 if purpose == "smtp" else 993)
    @contextmanager
    def smtp(self, profile, *, deadline, timeout):
        self.calls.append(("smtp", profile.binding.tenant_id, deadline, timeout))
        try: yield BoundSession(self.smtp_identity or self.identity(profile, "smtp"), self.smtp_session)
        finally:
            if self.block_quit:
                self.quit_blocked.set()
                if not self.smtp_session.sock.closed.wait(timeout=3): raise AssertionError("QUIT not interrupted")
                raise OSError("QUIT_CANARY")
            if self.quit_error: raise OSError("QUIT_CANARY")
    @contextmanager
    def imap(self, profile, *, folder, readonly, deadline, timeout):
        self.calls.append(("imap", folder, readonly, deadline, timeout))
        try: yield BoundSession(self.imap_identity or self.identity(profile, "imap"), self.imap_session)
        finally:
            if self.imap_close_error: raise OSError("LOGOUT_CANARY")


class PrivateWireTests(unittest.TestCase):
    def setUp(self):
        self.factory = FakeFactory()
        self.now = 1000
        self.monotonic_now = 100.0
        self.record = parse_source(BINDING, PLAN, source_wire())
        self.action = action_for(self.record)
        self.prepared = prepare_reply(BINDING, self.action, self.record, now=self.now)
        self.gates = 0
    def transport(self, **kwargs):
        return PrivatePreparedWireTransport(PROFILE, self.factory, offline_test_mode=True,
            clock=lambda: self.now, monotonic=lambda: self.monotonic_now, **kwargs)
    def gate(self): self.gates += 1
    def run_transport(self, **kwargs):
        with patch("socket.create_connection", side_effect=AssertionError("socket attempted")):
            return asyncio.run(self.transport(**kwargs).submit_prepared(self.prepared, self.gate))

    def test_default_disabled_and_nonfake_factory_cannot_activate(self):
        dormant = PrivatePreparedWireTransport(PROFILE, self.factory)
        self.assertFalse(dormant.enabled)
        with self.assertRaises(RuntimeError): asyncio.run(dormant.submit_prepared(self.prepared, self.gate))
        self.factory.offline_only = False
        with self.assertRaises(ValueError): self.transport()
        self.assertEqual(self.factory.calls, [])
        with self.assertRaises(ValidationError): PrivateBridgeProfile.model_validate(PROFILE.model_dump() | {"outbound": "enabled"})

    def test_explicit_stdlib_phases_gate_and_exact_wire_no_regeneration(self):
        result = self.run_transport()
        self.assertEqual((result.submission, result.sent_copy, result.delivery), ("accepted", "not_attempted", "unverified"))
        calls = self.factory.smtp_session.calls
        self.assertEqual([call[0] for call in calls], ["mail", "rcpt", "data", "send", "final"])
        self.assertEqual(calls[:3], [("mail", self.prepared.preview.sender), ("rcpt", self.prepared.preview.to[0]), ("data", "DATA")])
        self.assertEqual(calls[3][1], self.prepared.wire + b".\r\n")
        self.assertEqual(self.gates, 1)
        self.assertTrue(all(0 < timeout <= 3 for timeout in self.factory.smtp_session.sock.timeouts))

    def test_dot_stuffing_restores_exact_bytes_without_newline_changes(self):
        wire = b"Header: value\r\n\r\n.one\r\n..two\r\nlast\r\n"
        framed = smtp_data_frame(wire)
        self.assertEqual(framed, b"Header: value\r\n\r\n..one\r\n...two\r\nlast\r\n.\r\n")
        restored = b"\r\n".join(line[1:] if line.startswith(b"..") else line for line in framed[:-3].split(b"\r\n"))
        self.assertEqual(restored, wire)
        with self.assertRaises(ContractError): smtp_data_frame(b"bare LF\n")

    def test_each_negative_smtp_phase_is_known_rejected_and_stops(self):
        for phase in ("mail", "rcpt", "data", "final"):
            self.factory = FakeFactory(); self.factory.smtp_session.codes[phase] = 550
            result = self.run_transport()
            self.assertEqual(result.submission, "rejected")
            self.assertEqual(self.factory.smtp_session.calls[-1][0], phase)

    def test_missing_final_ack_and_partial_body_write_are_uncertain(self):
        for phase in ("send", "final"):
            self.factory = FakeFactory(); self.factory.smtp_session.raise_at = phase
            result = self.run_transport()
            self.assertEqual(result.submission, "uncertain")
            self.assertNotIn("CANARY", canonical(result))

    def test_prebody_connection_loss_and_gate_refusal_write_no_bytes(self):
        self.factory.smtp_session.raise_at = "data"
        self.assertEqual(self.run_transport().submission, "rejected")
        self.assertFalse(any(call[0] == "send" for call in self.factory.smtp_session.calls))
        self.factory = FakeFactory()
        def refused(): raise ContractError("gate_refused")
        self.gate = refused
        self.assertEqual(self.run_transport().submission, "rejected")
        self.assertFalse(any(call[0] == "send" for call in self.factory.smtp_session.calls))

    def test_deadline_or_action_expiry_after_354_fences_body(self):
        for clock in ("monotonic", "action"):
            self.factory = FakeFactory()
            def advance(phase):
                if phase == "data":
                    if clock == "monotonic": self.monotonic_now += PROFILE.total_timeout_seconds
                    else: self.now = self.action.expires_at
            self.factory.smtp_session.after = advance
            result = self.run_transport()
            self.assertEqual(result.submission, "rejected")
            self.assertFalse(any(call[0] == "send" for call in self.factory.smtp_session.calls))
            self.monotonic_now = 100.0; self.now = 1000

    def test_known_final_acceptance_survives_quit_and_postack_deadline(self):
        self.factory.quit_error = True
        self.assertEqual(self.run_transport().submission, "accepted")
        self.factory = FakeFactory()
        self.factory.smtp_session.after = lambda phase: setattr(self, "monotonic_now", 110.0) if phase == "final" else None
        result = self.run_transport(sent_copy=True)
        self.assertEqual((result.submission, result.sent_copy), ("accepted", "failed"))

    def test_cross_tenant_account_mailbox_and_forged_session_identity_refuse(self):
        changes = [{"binding": BINDING.model_copy(update={"tenant_id": "other-customer"})},
                   {"binding": BINDING.model_copy(update={"account_id": "other-account"})},
                   {"mailbox_address": "other@example.invalid"}, {"port": True}]
        for change in changes:
            self.factory = FakeFactory()
            self.factory.smtp_identity = self.factory.identity(PROFILE, "smtp").model_copy(update=change)
            result = self.run_transport()
            self.assertEqual(result.submission, "rejected")
            self.assertEqual(self.factory.smtp_session.calls, [])

    def test_cross_tenant_action_and_corrupt_wire_refuse_before_factory(self):
        for prepared in (replace(self.prepared, action=self.action.model_copy(update={"tenant_id": "other-customer"})),
                         replace(self.prepared, wire=self.prepared.wire + b"changed"),
                         replace(self.prepared, action=None),
                         replace(self.prepared, preview=self.prepared.preview.model_copy(update={"body": "changed"}))):
            self.factory = FakeFactory(); self.prepared = prepared
            self.assertEqual(self.run_transport().submission, "rejected")
            self.assertEqual(self.factory.calls, [])

    def test_malformed_or_unexpected_smtp_response_cannot_establish_acceptance(self):
        for value in ((True, b"ok"), (250, "private string"), (250, b"x" * 8193), (200, b"not accepted"), [250, b"ok"]):
            self.factory = FakeFactory(); self.factory.smtp_session.bad_response = value
            self.assertNotEqual(self.run_transport().submission, "accepted")

    def test_sent_copy_exact_append_existing_copy_and_copy_cleanup(self):
        result = self.run_transport(sent_copy=True)
        self.assertEqual((result.submission, result.sent_copy), ("accepted", "stored"))
        appended = [call for call in self.factory.imap_session.calls if call[0] == "APPEND"]
        self.assertEqual(appended, [("APPEND", '"Sent"', "\\Seen", None, self.prepared.wire)])
        self.factory = FakeFactory(); self.factory.imap_session.search = ("OK", [b"9"]); self.factory.imap_session.raw = self.prepared.wire; self.factory.imap_close_error = True
        result = self.run_transport(sent_copy=True)
        self.assertEqual((result.submission, result.sent_copy), ("accepted", "stored"))
        self.assertFalse(any(call[0] == "APPEND" for call in self.factory.imap_session.calls))

    def test_failed_and_unknown_sent_copy_preserve_acceptance(self):
        for mode, expected in (("search", "failed"), ("append-reject", "failed"), ("append-drop", "unknown")):
            self.factory = FakeFactory()
            if mode == "search": self.factory.imap_session.search = ("NO", [b"PRIVATE_CANARY"])
            if mode == "append-reject": self.factory.imap_session.append_status = "NO"
            if mode == "append-drop": self.factory.imap_session.raise_append = True
            result = self.run_transport(sent_copy=True)
            self.assertEqual((result.submission, result.sent_copy), ("accepted", expected))
            self.assertNotIn("CANARY", canonical(result))
            self.assertEqual(len([x for x in self.factory.imap_session.calls if x[0] == "APPEND"]), 0 if mode == "search" else 1)

    def test_private_read_port_is_account_bound_readonly_and_size_peek_only(self):
        port = PrivateReadPort(PROFILE, self.factory, monotonic=lambda: self.monotonic_now)
        source = OwnerMailSource(BINDING, port)
        record = source.get(PLAN.source_id)
        self.assertEqual(record, self.record)
        self.assertEqual(self.factory.calls[0][:3], ("imap", "INBOX", True))
        with self.assertRaises(SourceUnavailable):
            with port.mailbox("Sent", True): pass
        with self.assertRaises(SourceUnavailable):
            with port.mailbox("INBOX", False): pass
        with port.mailbox("INBOX", True) as session:
            with self.assertRaises(SourceUnavailable): session.uid("STORE", "42", "+FLAGS")
        self.factory.imap_identity = self.factory.identity(PROFILE, "imap").model_copy(update={"mailbox_address": "other@example.invalid"})
        with self.assertRaises(SourceUnavailable): source.get(PLAN.source_id)

    def test_existing_sent_header_collision_or_ambiguity_is_not_exact_copy(self):
        for search, raw in ((("OK", [b"9"]), source_wire()), (("OK", [b"9 10"]), self.prepared.wire)):
            self.factory = FakeFactory(); self.factory.imap_session.search = search; self.factory.imap_session.raw = raw
            result = self.run_transport(sent_copy=True)
            self.assertEqual((result.submission, result.sent_copy), ("accepted", "failed"))
            self.assertFalse(any(x[0] == "APPEND" for x in self.factory.imap_session.calls))

    def test_private_source_same_sender_other_tenant_or_account_cannot_be_assembled(self):
        for change in ({"tenant_id": "another-customer"}, {"account_id": "another-account"}):
            profile = PROFILE.model_copy(update={"binding": BINDING.model_copy(update=change)})
            port = PrivateReadPort(profile, self.factory)
            with self.assertRaises(ValueError): OwnerMailSource(BINDING, port)
            self.assertEqual(self.factory.calls, [])

    def test_forged_profile_overrides_fail_at_construction(self):
        for change in ({"outbound": "enabled"}, {"total_timeout_seconds": 1000},
                       {"endpoints": PROFILE.endpoints.model_copy(update={"host": "attacker.invalid"})}):
            profile = PROFILE.model_copy(update=change)
            with self.assertRaises(ValidationError): PrivateReadPort(profile, self.factory)
            with self.assertRaises(ValidationError): PrivatePreparedWireTransport(profile, self.factory)
        self.assertEqual(self.factory.calls, [])

    def test_disabled_profile_and_review_only_staging_templates_validate(self):
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        profile = PrivateBridgeProfile.model_validate_json((root / "deploy/private-bridge-profile.example.json").read_text())
        self.assertEqual(profile.outbound, "disabled")
        staging = json.loads((root / "deploy/private-bridge-staging.example.json").read_text())
        self.assertEqual(staging["workers"], 1)
        self.assertEqual(staging["outbound_factory"], "disabled")
        self.assertEqual(staging["status"], "review_only_not_an_executable_deployment")

    def test_watchdog_interrupts_blocked_final_reply_without_socket(self):
        profile = PROFILE.model_copy(update={"total_timeout_seconds": 1, "phase_timeout_seconds": 1})
        self.factory.smtp_session.block_at = "final"
        transport = PrivatePreparedWireTransport(profile, self.factory, offline_test_mode=True, clock=lambda: self.now)
        result = asyncio.run(transport.submit_prepared(self.prepared, self.gate))
        self.assertEqual(result.submission, "uncertain")
        self.assertEqual(self.factory.smtp_session.sock.shutdowns, [2])

    def test_cancellation_before_body_and_during_body_is_fenced(self):
        for phase, expected in (("data", "rejected"), ("final", "uncertain")):
            self.factory = FakeFactory(); self.factory.smtp_session.block_at = phase
            async def cancel():
                task = asyncio.create_task(self.transport().submit_prepared(self.prepared, self.gate))
                await asyncio.to_thread(self.factory.smtp_session.blocked.wait, 2)
                task.cancel(); return await task
            result = asyncio.run(cancel())
            self.assertEqual(result.submission, expected)
            if phase == "data": self.assertFalse(any(x[0] == "send" for x in self.factory.smtp_session.calls))

    def test_cancellation_after_final250_preserves_acceptance_and_copy_uncertainty(self):
        self.factory.imap_session.block_append = True
        async def cancel():
            task = asyncio.create_task(self.transport(sent_copy=True).submit_prepared(self.prepared, self.gate))
            self.assertTrue(await asyncio.to_thread(self.factory.imap_session.append_started.wait, 2))
            task.cancel(); return await task
        result = asyncio.run(cancel())
        self.assertEqual((result.submission, result.sent_copy), ("accepted", "unknown"))


class PrivateBridgeLedgerTests(unittest.TestCase):
    def setUp(self):
        integration.MailIntegrationTests.setUpClass()
        self.fixture = integration.MailIntegrationTests("test_exact_claim_gate_and_wire_no_retry_survive_restart")
        self.fixture.setUp()
        self.factory = FakeFactory()
        self.profile = PROFILE
        self.transport = PrivatePreparedWireTransport(PROFILE, self.factory, offline_test_mode=True,
            clock=self.fixture.fixture.clock, sent_copy=True)
        self.fixture.service.prepared_transport = self.transport
    def tearDown(self): self.fixture.tearDown()

    def test_real_ledger_gate_follows_data354_and_precedes_exact_bytes(self):
        proposal = self.fixture.propose(); self.fixture.approve(proposal)
        observed = []
        original = self.fixture.service.store.submission_gate
        def gate(*args, **kwargs):
            stage = args[-1]
            if stage == "before_body":
                observed.append(list(self.factory.smtp_session.calls))
            return original(*args, **kwargs)
        with patch.object(self.fixture.service.store, "submission_gate", side_effect=gate): self.fixture.worker()
        self.fixture.worker()
        self.assertEqual(observed[0][-1], ("data", "DATA"))
        self.assertFalse(any(x[0] == "send" for x in observed[0]))
        result = self.fixture.service.store.get(proposal["id"])
        self.assertEqual(result["execution"]["state"], "accepted")
        self.assertEqual(len([x for x in self.factory.smtp_session.calls if x[0] == "send"]), 1)
        framed = next(x[1] for x in self.factory.smtp_session.calls if x[0] == "send")
        self.assertEqual(hashlib.sha256(framed[:-3]).hexdigest(), proposal["wire_preview"]["wire_sha256"])

    def test_service_rejects_cross_binding_transport_before_factory(self):
        from zetbros_service.mail_integration import MailService
        other = PROFILE.model_copy(update={"binding": BINDING.model_copy(update={"tenant_id": "another-customer"})})
        transport = PrivatePreparedWireTransport(other, self.factory, offline_test_mode=True)
        with self.assertRaises(RuntimeError):
            MailService(self.fixture.fixture.settings, source=self.fixture.source, prepared_transport=transport)
        self.assertEqual(self.factory.calls, [])

    def test_real_ledger_identity_revocation_after354_writes_no_body(self):
        proposal = self.fixture.propose(); self.fixture.approve(proposal)
        def revoke(phase):
            if phase == "data":
                settings = self.fixture.service.store.settings
                self.fixture.service.store.settings = settings.model_copy(update={"principals": tuple(g for g in settings.principals if g.role != "reviewer")})
        self.factory.smtp_session.after = revoke
        self.fixture.worker(); self.fixture.worker()
        self.assertEqual(self.fixture.service.store.get(proposal["id"])["execution"]["state"], "rejected")
        self.assertFalse(any(x[0] == "send" for x in self.factory.smtp_session.calls))
