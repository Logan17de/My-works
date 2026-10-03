"""Fictional IMAP/SMTP transcripts only. No provider connection or credentials."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import ssl
import unittest
import uuid
from email import policy
from email.parser import BytesParser
from unittest.mock import patch

from pydantic import ValidationError

from zetbros_service.adapters import SourceNotFound, SourceUnavailable
from zetbros_service.models import ProposalInput, ReplyAction, digest
from zetbros_service.service import Service
from zetbros_service.spacemail_contract import (ContractError, ImapReadPlan, MAX_SOURCE_BYTES, ProviderBinding,
    SmtpOutcome, SpaceMailEndpoints, parse_body_response, parse_size_response, parse_source,
    prepare_reply, read_plan, validate_uidvalidity, verified_tls_context)
from zetbros_service.store import Store
import test_service as base_service

AGENT, REVIEWER = base_service.AGENT, base_service.REVIEWER


BINDING = ProviderBinding(tenant_id="test-customer", connector_id="support-mail", account_id="account-1",
    sender_address="support@example.invalid", policy_version="policy-1")
PLAN = read_plan("imap-v1:123:42")


def source_wire(body="A fictional support question\r\n", *, extra="", sender="Customer <sender@example.invalid>",
                reply_to="replyto@example.invalid", subject="Support question", message_id="<source-42@example.invalid>"):
    headers = (f"From: {sender}\r\nReply-To: {reply_to}\r\nTo: support@example.invalid\r\nSubject: {subject}\r\n"
               f"Message-ID: {message_id}\r\nReferences: <earlier@example.invalid>\r\n{extra}"
               "MIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: base64\r\n\r\n")
    return headers.encode("ascii") + base64.b64encode(body.encode("utf-8")) + b"\r\n"


def action_for(record, **changes):
    data = dict(tenant_id=BINDING.tenant_id, connector_id=BINDING.connector_id, account_id=BINDING.account_id,
        operation_key="ticket-42-reply-0001", source_id=record.source_id, source_version=record.version,
        source_fingerprint=digest(record), sender=BINDING.sender_address, to=(record.reply_to or record.sender,),
        subject="Re: "+record.subject, body="Exact reply\nno final newline", policy_version=BINDING.policy_version,
        proposer_subject=AGENT.subject, proposer_client=AGENT.client_id, created_at=1000, expires_at=1300)
    return ReplyAction(**(data | changes))


class FictionalImap:
    """Scripted read-only transcript, deliberately no socket/login/write methods."""
    def __init__(self, raw=None):
        self.raw = raw or source_wire()
        self.uidvalidity = 123
        self.calls = []

    def get(self, source_id):
        plan = read_plan(source_id)
        self.calls.append(("EXAMINE", plan.mailbox))
        validate_uidvalidity(plan, ("UIDVALIDITY", [str(self.uidvalidity).encode()]))
        self.calls.append(("UID FETCH", str(plan.uid), plan.size_items))
        size = parse_size_response(plan, "OK", [f"7 (UID {plan.uid} RFC822.SIZE {len(self.raw)})".encode()])
        self.calls.append(("UID FETCH", str(plan.uid), plan.body_items))
        raw = parse_body_response(plan, "OK", [(f"7 (UID {plan.uid} BODY[]<0> {{{len(self.raw)}}}".encode(), self.raw), b")"], size)
        return parse_source(BINDING, plan, raw)

    def ready(self):
        return True


class FictionalSmtp:
    """A test-only bridge for the actual durable worker, with no network methods."""
    enabled = True
    def __init__(self, fixture, source, *, final=250, drop_ack=False, sent_copy="stored", mutate_source=False):
        self.fixture, self.source = fixture, source
        self.final, self.drop_ack, self.copy_outcome = final, drop_ack, sent_copy
        self.mutate_source = mutate_source
        self.bodies = []
        self.envelopes = []
        self.claim_observations = []

    async def submit(self, action):
        # Prove the real control service already consumed approval and reserved the
        # bound operation in its durable DB before the fictional SMTP transcript.
        with self.fixture.service.store.connection() as connection:
            row = connection.execute("SELECT e.state,p.digest,p.consumed_at,e.claim_token FROM proposals p JOIN executions e ON p.id=e.proposal_id WHERE p.operation_key=?", (action.operation_key,)).fetchone()
        self.claim_observations.append(dict(row))
        if row["state"] != "claimed" or row["digest"] != digest(action) or row["consumed_at"] is None or not row["claim_token"]:
            raise RuntimeError("missing durable claim")
        if self.mutate_source:
            self.source.raw = source_wire(message_id="<changed-after-claim@example.invalid>")
        fresh = self.source.get(action.source_id)
        prepared = prepare_reply(BINDING, action, fresh, now=int(self.fixture.clock()))
        reducer = SmtpOutcome()
        self.envelopes.append((prepared.preview.sender, prepared.preview.to))
        reducer.response("mail", 250)
        reducer.response("rcpt", 250)
        reducer.response("data", 354)
        reducer.begin_body()
        self.bodies.append(prepared.wire)
        if self.drop_ack:
            return reducer.interrupted("transport_timeout")
        reducer.response("final", self.final)
        if self.final == 250:
            reducer.sent_copy(self.copy_outcome)
        return reducer.result()


class SpaceMailContractTests(unittest.TestCase):
    def test_fixed_official_tls_endpoints_and_no_activation_surface(self):
        endpoints = SpaceMailEndpoints()
        self.assertEqual((endpoints.host, endpoints.imap_port, endpoints.smtp_port), ("mail.spacemail.com", 993, 465))
        for change in ({"host": "attacker.example.invalid"}, {"imap_port": 143}, {"smtp_port": 587},
                       {"security": "unverified"}, {"inbox": "../Sent"}, {"password": "CANARY"}, {"operation_timeout_seconds": 0}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                SpaceMailEndpoints.model_validate(change)
        import zetbros_service.spacemail_contract as module
        self.assertFalse(any(hasattr(module, name) for name in ("connect", "login", "send", "submit", "SMTP", "IMAP4_SSL")))
        with patch("socket.create_connection", side_effect=AssertionError("network attempted")):
            context = verified_tls_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertGreaterEqual(context.minimum_version, ssl.TLSVersion.TLSv1_2)

    def test_uid_locator_bounds_no_sequence_ranges_paths_or_folders(self):
        self.assertEqual(read_plan("imap-v1:4294967295:4294967295").uid, 4294967295)
        for value in ("42", "imap-v1:0:42", "imap-v1:01:42", "imap-v1:1:4294967296", "imap-v1:1:1:*",
                      "INBOX/42", "../../message", "imap-v1:1:42\r\nDELETE INBOX", None):
            with self.subTest(value=value), self.assertRaises(ContractError): read_plan(value)
        for options in ({"mailbox":"Sent"}, {"readonly":False}, {"body_items":"(BODY[])"}, {"size_items":"ALL"}, {"uid":True}):
            with self.subTest(options=options), self.assertRaises(ContractError): ImapReadPlan(**({"uidvalidity":123,"uid":42}|options))

    def test_read_plan_and_transcript_never_mark_seen_or_mutate(self):
        source = FictionalImap()
        record = source.get(PLAN.source_id)
        self.assertEqual(record.source_id, "imap-v1:123:42")
        self.assertEqual(source.calls, [("EXAMINE","INBOX"),("UID FETCH","42","(UID RFC822.SIZE)"),
                                      ("UID FETCH","42","(UID BODY.PEEK[]<0.65537>)")])
        self.assertEqual(record.version, "rfc822:"+hashlib.sha256(source.raw).hexdigest())

    def test_uidvalidity_rollover_and_ambiguous_responses_refused(self):
        for response in (("UIDVALIDITY",[b"124"]),("UIDVALIDITY",[b"123",b"123"]),("OK",[b"123"]),None):
            with self.subTest(response=response), self.assertRaises(SourceUnavailable): validate_uidvalidity(PLAN,response)
        source=FictionalImap(); source.uidvalidity=124
        with self.assertRaises(SourceUnavailable): source.get(PLAN.source_id)
        self.assertEqual(source.calls,[("EXAMINE","INBOX")])

    def test_size_preflight_rejects_missing_wrong_uid_duplicates_and_oversize(self):
        self.assertEqual(parse_size_response(PLAN,"OK",[b"7 (RFC822.SIZE 500 UID 42)"]),500)
        for status,data in (("NO",[b"private BODY_CANARY"]),("OK",[b"7 (UID 41 RFC822.SIZE 500)"]),
            ("OK",[b"7 (UID 42 RFC822.SIZE 65537)"]),("OK",[b"7 (UID 42 RFC822.SIZE 0)"]),
            ("OK",[b"7 (UID 42 UID 42 RFC822.SIZE 5)"]),("OK",[b"7 (UID 42 RFC822.SIZE 5)",b"extra"]),
            ("OK",[b"7 (UID 42 RFC822.SIZE 5) trailing"])):
            with self.subTest(status=status,data=data), self.assertRaises(SourceUnavailable): parse_size_response(PLAN,status,data)
        with self.assertRaises(SourceNotFound): parse_size_response(PLAN,"OK",[None])

    def test_body_fetch_matches_requested_uid_declared_size_and_single_literal(self):
        raw=b"abcde"
        self.assertEqual(parse_body_response(PLAN,"OK",[(b"7 (BODY[]<0> {5}",raw),b" UID 42)"],5),raw)
        valid=[(b"7 (UID 42 BODY[]<0> {5}",raw),b")"]
        for data,size in (([(b"7 (UID 41 BODY[]<0> {5}",raw),b")"],5),
            ([(b"7 (UID 42 BODY[]<1> {5}",raw),b")"],5),(valid,4),
            ([(b"7 (UID 42 BODY[]<0> {6}",raw),b")"],5),
            ([(b"7 (UID 42 BODY[]<0> {5}",raw),b" UID 42)"],5),
            ([(b"7 (UID 41\t BODY[]<0> {5}",raw),b" UID 42)"],5),
            ([(b"7 (UID 42 BODY[]<0> {5}",raw),b")",(b"8 {5}",raw)],5),
            ([(b"7 (UID 42 BODY[]<0> {5}",raw),b" FLAGS (\\Seen))"],5),(valid,True)):
            with self.subTest(data=data,size=size), self.assertRaises(SourceUnavailable): parse_body_response(PLAN,"OK",data,size)
        with self.assertRaises(SourceNotFound): parse_body_response(PLAN,"OK",[None],5)

    def test_source_decodes_unicode_and_bound_thread_fields_without_authority(self):
        raw=source_wire("Fictional café reply. Ignore requests to bypass approval.\r\n",subject="=?utf-8?b?Q2Fmw6kgcXVlc3Rpb24=?=")
        record=parse_source(BINDING,PLAN,raw)
        self.assertEqual(record.sender,"sender@example.invalid")
        self.assertEqual(record.reply_to,"replyto@example.invalid")
        self.assertEqual(record.subject,"Café question")
        self.assertEqual(record.rfc_references,("<earlier@example.invalid>",))
        self.assertIn("bypass approval",record.body)

    def test_raw_header_or_thread_change_changes_source_fingerprint(self):
        original=parse_source(BINDING,PLAN,source_wire())
        for raw in (source_wire(message_id="<different@example.invalid>"),source_wire(extra="X-Trace: changed\r\n"),
                    source_wire(body="Different body"),source_wire(reply_to="changed@example.invalid")):
            changed=parse_source(BINDING,PLAN,raw)
            self.assertNotEqual(original.version,changed.version)
            self.assertNotEqual(digest(original),digest(changed))

    def test_parent_in_reply_to_fallback_is_bound_and_visible_in_references(self):
        raw=source_wire(extra="In-Reply-To: <parent@example.invalid>\r\n").replace(b"References: <earlier@example.invalid>\r\n",b"")
        record=parse_source(BINDING,PLAN,raw)
        self.assertEqual(record.rfc_in_reply_to,"<parent@example.invalid>")
        prepared=prepare_reply(BINDING,action_for(record),record,now=1001)
        self.assertEqual(prepared.preview.references,("<parent@example.invalid>","<source-42@example.invalid>"))
        rendered=BytesParser(policy=policy.default).parsebytes(prepared.wire)
        self.assertEqual(str(rendered["References"]),"<parent@example.invalid> <source-42@example.invalid>")
        changed=parse_source(BINDING,PLAN,raw.replace(b"<parent@example.invalid>",b"<other-parent@example.invalid>"))
        self.assertNotEqual(digest(record),digest(changed))
        with self.assertRaises(ContractError): prepare_reply(BINDING,action_for(record),changed,now=1001)

    def test_malformed_encoded_words_are_never_silently_repaired(self):
        for subject in ("=?utf-8?b?Zg?=","=?x-invalid?b?Zm9v?=","=?utf-8?b?/w==?=",
                        "=?utf-8?q?bad=XX?=","=?utf-8?b?Zg===?=","=?utf-8?q?line=0Abreak?=",
                        "=?utf-8?b?Zh==?=","=?utf-8?b?unfinished"):
            with self.subTest(subject=subject), self.assertRaises(SourceUnavailable):
                parse_source(BINDING,PLAN,source_wire(subject=subject))
        encoded_id=base64.b64encode(b"<source-42@example.invalid>").decode()
        with self.assertRaises(SourceUnavailable):
            parse_source(BINDING,PLAN,source_wire(message_id="=?utf-8?b?"+encoded_id+"?="))

    def test_nested_encoded_word_subject_cannot_change_approved_wire_preview(self):
        # The outer source header is valid; its decoded text looks like a second
        # encoded word. Python's wire serializer would otherwise change it to hi.
        record=parse_source(BINDING,PLAN,source_wire(subject="=?utf-8?b?PT91dGYtOD9iP2FHaz0/PQ==?="))
        self.assertEqual(record.subject,"=?utf-8?b?aGk=?=")
        with self.assertRaises(ContractError): prepare_reply(BINDING,action_for(record),record,now=1001)

    def test_duplicate_malformed_routing_and_thread_headers_refused(self):
        for raw in (source_wire(extra="From: another@example.invalid\r\n"),source_wire(extra="Subject: other\r\n"),
            source_wire(sender="one@example.invalid, two@example.invalid"),source_wire(reply_to="Team: a@example.invalid;"),
            source_wire(reply_to="a@example.invalid, b@example.invalid"),source_wire(message_id="<bad id@example.invalid>"),
            source_wire(extra="Message-ID: <second@example.invalid>\r\n"),source_wire(extra="Sender: agent@example.invalid\r\n"),
            source_wire(extra="Resent-To: other@example.invalid\r\n"),source_wire(extra="In-Reply-To: <one@example.invalid> <two@example.invalid>\r\n"),
            source_wire().replace(b"<earlier@example.invalid>",b"<earlier@example.invalid> <earlier@example.invalid>"),
            source_wire().replace(b"<earlier@example.invalid>",b"<source-42@example.invalid>")):
            with self.subTest(raw=raw[:80]), self.assertRaises(SourceUnavailable): parse_source(BINDING,PLAN,raw)

    def test_unsupported_mime_attachments_charset_and_encoding_refused(self):
        valid=source_wire()
        cases=(valid.replace(b"text/plain",b"text/html"),valid.replace(b"text/plain",b"multipart/mixed; boundary=x"),
            source_wire(extra='Content-Disposition: attachment; filename="private.txt"\r\n'),
            valid.replace(b"charset=utf-8",b"charset=iso-8859-1"),valid.replace(b"base64",b"binary"),
            valid.rsplit(b"\r\n\r\n",1)[0]+b"\r\n\r\n%%%INVALID%%%\r\n",
            source_wire("NUL \x00 body"),source_wire("Control \x01 body"),valid.replace(b"\r\n",b"\n"),
            b"X"*(MAX_SOURCE_BYTES+1),valid.replace(b"MIME-Version: 1.0",b"MIME-Version: 1.0\r\nBad header"))
        for raw in cases:
            with self.subTest(raw=raw[:50]), self.assertRaises(SourceUnavailable) as context: parse_source(BINDING,PLAN,raw)
            self.assertNotIn("body",str(context.exception))

    def test_ascii_and_quoted_printable_sources_have_exact_plain_body(self):
        for transfer,body,expected in ((b"7bit",b"Plain\r\n","Plain\r\n"),
                                      (b"quoted-printable",b"caf=C3=A9\r\n","café\r\n")):
            raw=source_wire().replace(b"base64",transfer).split(b"\r\n\r\n")[0]+b"\r\n\r\n"+body
            self.assertEqual(parse_source(BINDING,PLAN,raw).body,expected)
        raw=source_wire().replace(b"base64",b"quoted-printable").split(b"\r\n\r\n")[0]+b"\r\n\r\nBad=ZZ\r\n"
        with self.assertRaises(SourceUnavailable): parse_source(BINDING,PLAN,raw)

    def test_wire_preview_is_exact_deterministic_and_binds_visible_headers(self):
        record=parse_source(BINDING,PLAN,source_wire())
        for body in ("No final newline","CRLF\r\ntext\r\n","Unicode café ☀\n", ".\r\nQUIT\r\n"):
            action=action_for(record,body=body)
            prepared=prepare_reply(BINDING,action,record,now=1001)
            self.assertEqual(prepared,prepare_reply(BINDING,action,record,now=1002))
            message=BytesParser(policy=policy.default).parsebytes(prepared.wire)
            self.assertEqual(message.get_payload(decode=True).decode("utf-8"),body)
            self.assertEqual(str(message["From"]),action.sender)
            self.assertEqual(str(message["To"]),action.to[0])
            self.assertEqual(str(message["Subject"]),action.subject)
            self.assertEqual(str(message["In-Reply-To"]),record.rfc_message_id)
            self.assertEqual(str(message["References"]),"<earlier@example.invalid> <source-42@example.invalid>")
            self.assertEqual(prepared.preview.wire_sha256,hashlib.sha256(prepared.wire).hexdigest())
            self.assertEqual(prepared.preview.action_digest,digest(action))
            self.assertEqual(prepared.preview.source_fingerprint,digest(record))
            self.assertNotIn("Cc",message)
            self.assertNotIn("Bcc",message)
            self.assertFalse(message.is_multipart())
            self.assertNotEqual(prepared.preview_digest,digest(prepared.preview.model_copy(update={"message_id":"<different@example.invalid>"})))

    def test_source_recipient_subject_policy_and_identity_drift_refused(self):
        record=parse_source(BINDING,PLAN,source_wire())
        action=action_for(record)
        for changed in (action.model_copy(update={"to":("arbitrary@example.invalid",)}),
            action.model_copy(update={"sender":"other@example.invalid"}),action.model_copy(update={"subject":"Changed"}),
            action.model_copy(update={"policy_version":"policy-2"}),action.model_copy(update={"tenant_id":"another"}),
            action.model_copy(update={"body":"NUL\x00"}),action.model_copy(update={"operation_key":"short"}),
            action.model_copy(update={"created_at":True}),action.model_copy(update={"cc":("cc@example.invalid",)})):
            with self.subTest(changed=changed), self.assertRaises(ContractError): prepare_reply(BINDING,changed,record,now=1001)
        with self.assertRaises(ContractError): prepare_reply(BINDING,action,parse_source(BINDING,PLAN,source_wire(message_id="<new@example.invalid>")),now=1001)
        with self.assertRaises(ContractError): prepare_reply(BINDING,action,record.model_copy(update={"uid":43}),now=1001)
        for now in (999,1300,True):
            with self.subTest(now=now), self.assertRaises(ContractError): prepare_reply(BINDING,action,record,now=now)

    def test_smtp_acceptance_requires_final_data_ack_and_never_proves_delivery(self):
        reducer=SmtpOutcome()
        for stage,code in (("mail",250),("rcpt",251),("data",354)):
            self.assertIsNone(reducer.response(stage,code))
            with self.assertRaises(ContractError): reducer.result()
        reducer.begin_body()
        with self.assertRaises(ContractError): reducer.result()
        accepted=reducer.response("final",250)
        self.assertEqual((accepted.submission,accepted.delivery),("accepted","unverified"))
        failed_copy=reducer.sent_copy("failed")
        self.assertEqual((failed_copy.submission,failed_copy.sent_copy),("accepted","failed"))
        self.assertEqual(reducer.interrupted(),failed_copy)
        self.assertEqual(reducer.response("final",550),failed_copy)
        with self.assertRaises(ContractError): reducer.sent_copy("stored")

    def test_known_negative_smtp_response_terminal_without_internal_retry(self):
        for stage in ("mail","rcpt","data","final"):
            reducer=SmtpOutcome()
            for advance,code in (("mail",250),("rcpt",250),("data",354)):
                if advance==stage: break
                reducer.response(advance,code)
            if stage=="final": reducer.begin_body()
            rejected=reducer.response(stage,550)
            self.assertEqual(rejected.submission,"rejected")
            self.assertEqual(reducer.response("mail",250),rejected)
            with self.assertRaises(ContractError): reducer.sent_copy("stored")

    def test_lost_ack_timeout_and_cancellation_after_body_uncertain(self):
        for error in ("adapter_exception","transport_timeout"):
            before=SmtpOutcome()
            self.assertEqual(before.interrupted(error).submission,"rejected")
            after=SmtpOutcome()
            for stage,code in (("mail",250),("rcpt",250),("data",354)): after.response(stage,code)
            after.begin_body()
            result=after.interrupted(error)
            self.assertEqual((result.submission,result.error_class),("uncertain",error))
            self.assertEqual(after.response("final",250),result)
        with self.assertRaises(ContractError): SmtpOutcome().interrupted("RAW_PASSWORD_CANARY")

    def test_duplicate_out_of_order_and_malformed_codes_cannot_establish_acceptance(self):
        for stage,code in (("final",250),("data",354),("mail",True),("mail","250"),("mail",99),("mail",600),("mail",220)):
            with self.subTest(stage=stage,code=code):
                reducer=SmtpOutcome(); result=reducer.response(stage,code)
                self.assertEqual(result.submission,"uncertain")
                self.assertEqual(reducer.response("final",250),result)
        reducer=SmtpOutcome(); reducer.response("mail",250)
        self.assertEqual(reducer.response("mail",250).submission,"uncertain")
        self.assertEqual(SmtpOutcome().begin_body().submission,"uncertain")


class SpaceMailDurableBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_service.ServiceTests.setUpClass()

    def setUp(self):
        self.fixture=base_service.ServiceTests()
        self.fixture.setUp()
        self.source=FictionalImap()
        self.smtp=FictionalSmtp(self.fixture,self.source)
        self.fixture.service.source=self.source
        self.fixture.service.transport=self.smtp
        self.record=self.source.get(PLAN.source_id)

    def tearDown(self):
        self.fixture.tearDown()

    def proposal(self, operation_key="space-ticket-42-reply-1"):
        return self.fixture.service.propose(ProposalInput(operation_key=operation_key,source_id=self.record.source_id,
            source_version=self.record.version,source_fingerprint=digest(self.record),body="Exact approved fictional reply"),AGENT,str(uuid.uuid4()))

    def approve(self, proposal):
        return self.fixture.service.store.decide(proposal["id"],proposal["digest"],"approve",REVIEWER,str(uuid.uuid4()))

    def test_no_approval_means_no_protocol_exchange(self):
        proposal=self.proposal()
        asyncio.run(self.fixture.service.worker_once())
        self.assertEqual(self.smtp.bodies,[])
        self.assertEqual(self.smtp.claim_observations,[])
        self.assertEqual(self.fixture.service.store.get(proposal["id"])["state"],"pending")

    def test_durable_claim_before_exchange_restart_replay_and_failed_sent_copy(self):
        self.smtp.copy_outcome="failed"
        proposal=self.proposal(); self.approve(proposal)
        asyncio.run(self.fixture.service.worker_once())
        result=self.fixture.service.store.get(proposal["id"])
        self.assertEqual(result["execution"]["state"],"accepted")
        self.assertEqual(result["execution"]["result"]["sent_copy"],"failed")
        self.assertEqual(len(self.smtp.bodies),1)
        self.assertEqual(self.smtp.claim_observations[0]["state"],"claimed")
        self.fixture.service=Service(self.fixture.settings,source=self.source,transport=self.smtp,clock=self.fixture.clock)
        asyncio.run(self.fixture.service.worker_once())
        replay=self.proposal()
        self.assertEqual(replay["id"],proposal["id"])
        self.assertEqual(len(self.smtp.bodies),1)

    def test_uid_or_thread_drift_before_claim_blocks_without_consumption(self):
        proposal=self.proposal(); self.approve(proposal)
        self.source.raw=source_wire(message_id="<changed@example.invalid>")
        asyncio.run(self.fixture.service.worker_once())
        result=self.fixture.service.store.get(proposal["id"])
        self.assertEqual(result["execution"]["state"],"invalidated")
        self.assertIsNone(result["consumed_at"])
        self.assertEqual(self.smtp.bodies,[])

    def test_thread_drift_after_claim_refused_before_any_smtp_body(self):
        self.smtp.mutate_source=True
        proposal=self.proposal(); self.approve(proposal)
        asyncio.run(self.fixture.service.worker_once())
        self.assertEqual(self.fixture.service.store.get(proposal["id"])["execution"]["state"],"uncertain")
        self.assertEqual(len(self.smtp.claim_observations),1)
        self.assertEqual(self.smtp.envelopes,[])
        self.assertEqual(self.smtp.bodies,[])
        asyncio.run(self.fixture.service.worker_once())
        self.assertEqual(len(self.smtp.claim_observations),1)

    def test_data_ack_loss_terminal_uncertainty_across_restart(self):
        self.smtp.drop_ack=True
        proposal=self.proposal(); self.approve(proposal)
        asyncio.run(self.fixture.service.worker_once())
        self.assertEqual(self.fixture.service.store.get(proposal["id"])["execution"]["state"],"uncertain")
        self.fixture.service=Service(self.fixture.settings,source=self.source,transport=self.smtp,clock=self.fixture.clock)
        asyncio.run(self.fixture.service.worker_once())
        self.assertEqual(len(self.smtp.bodies),1)

    def test_configured_production_keeps_delivery_disabled_with_addition_present(self):
        service=Service(self.fixture.settings,clock=self.fixture.clock)
        self.assertFalse(service.transport.enabled)
        self.assertEqual(service.readiness()["delivery"],"disabled")
        self.assertFalse(service.readiness()["live_delivery_ready"])


if __name__=="__main__":
    unittest.main()
