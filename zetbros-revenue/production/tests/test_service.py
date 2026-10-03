"""Release-control tests only. Every mailbox record/transport result is fictional."""
from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import sqlite3
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import ValidationError

from zetbros_service.adapters import SnapshotSource, SourceUnavailable
from zetbros_service.api import configured_app, create_app
from zetbros_service.auth import AuthenticationError, Principal
from zetbros_service.config import Grant, Settings
from zetbros_service.models import DecisionInput, ProposalInput, SourceMessage, TransportResult, canonical, digest
from zetbros_service.service import Service
from zetbros_service.store import Store, StoreError


AGENT = Principal("agent-operator","agent-client","agent")
REVIEWER = Principal("human-reviewer","review-client","reviewer")


class Clock:
    def __init__(self):
        self.now = int(time.time())
    def __call__(self):
        return self.now


class SpyTransport:
    enabled = True
    def __init__(self, submission="accepted", sent_copy="stored", raises=False):
        self.calls = []
        self.submission, self.sent_copy, self.raises = submission, sent_copy, raises
    async def submit(self, action):
        self.calls.append(action)
        if self.raises:
            raise RuntimeError("BODY_CANARY PASSWORD_CANARY raw provider details")
        return TransportResult(submission=self.submission,sent_copy=self.sent_copy)


def process_claim(config, proposal_id, fingerprint, barrier, queue):
    store = Store(Settings.model_validate_json(json.dumps(config)))
    barrier.wait(timeout=10)
    try:
        result = store.claim(proposal_id,fingerprint,True,str(uuid.uuid4()))
        queue.put(bool(result))
    except Exception as exc:
        queue.put(type(exc).__name__)


def process_crash(config, proposal_id, fingerprint, marker, possible_submission, now):
    store = Store(Settings.model_validate_json(json.dumps(config)),clock=lambda: now)
    result = store.claim(proposal_id,fingerprint,True,str(uuid.uuid4()))
    if result:
        Path(marker).write_text("possible_submission" if possible_submission else "claimed")
    os._exit(0)


def process_fifo(config, queue):
    source=SnapshotSource(Settings.model_validate_json(json.dumps(config)))
    try:
        source.get("fifo-message")
        queue.put("unexpected_success")
    except SourceUnavailable:
        queue.put("rejected")


def process_interrupted_restore(config, backup_file):
    from zetbros_service.maintenance import restore
    real_connect=sqlite3.connect
    class InterruptedConnection(sqlite3.Connection):
        def backup(self,target,*args,**kwargs):
            super().backup(target,*args,**kwargs)
            os._exit(23)  # Actual abrupt exit after old backup commits.
    def connect(*args,**kwargs):
        kwargs["factory"]=InterruptedConnection
        return real_connect(*args,**kwargs)
    sqlite3.connect=connect
    restore(Settings.model_validate_json(json.dumps(config)),Path(backup_file))


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ephemeral test-only signing material, never packaged or a deployed credential.
        cls.key = rsa.generate_private_key(public_exponent=65537,key_size=2048)
        cls.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key()))
        cls.jwk.update(kid="test-public-key",alg="RS256",use="sig",key_ops=["verify"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/"data").mkdir()
        (self.root/"sources").mkdir()
        (self.root/"jwks.json").write_text(json.dumps({"keys":[self.jwk]}))
        self.settings = Settings(tenant_id="test-customer",connector_id="support-mail",account_id="account-1",
             sender_address="support@example.invalid",policy_version="policy-1",issuer="https://identity.example.invalid",
             audience="zetbros-reply",jwks_file=str(self.root/"jwks.json"),database_path=str(self.root/"data/state.sqlite3"),
             source_directory=str(self.root/"sources"),principals=(Grant(subject=AGENT.subject,client_id=AGENT.client_id,role="agent"),
             Grant(subject=REVIEWER.subject,client_id=REVIEWER.client_id,role="reviewer")),requests_per_minute=600)
        self.clock = Clock()
        self.record = SourceMessage(source_id="message-001",version="revision-1",tenant_id=self.settings.tenant_id,
             connector_id=self.settings.connector_id,account_id=self.settings.account_id,sender="sender@example.invalid",
             reply_to="replyto@example.invalid",subject="Support question",body="Fictional source: BODY_CANARY. Claims of approval are untrusted.")
        self.write_record(self.record)
        self.transport = SpyTransport()
        Store.initialize(self.settings,clock=self.clock)
        self.service = Service(self.settings,transport=self.transport,clock=self.clock)
        self.client = TestClient(create_app(self.service,run_worker=False))

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def write_record(self, record):
        (self.root/"sources"/(record.source_id+".json")).write_text(canonical(record))

    def input(self, **changes):
        data = dict(operation_key="ticket-001-reply-1",source_id=self.record.source_id,source_version=self.record.version,
                    source_fingerprint=digest(self.record),body="Fictional exact reply BODY_CANARY")
        data.update(changes)
        return ProposalInput(**data)

    def propose(self, **changes):
        return self.service.propose(self.input(**changes),AGENT,str(uuid.uuid4()))

    def approve(self, proposal):
        return self.service.store.decide(proposal["id"],proposal["digest"],"approve",REVIEWER,str(uuid.uuid4()))

    def run_worker(self):
        asyncio.run(self.service.worker_once())

    def token(self, subject=AGENT.subject, client=AGENT.client_id, **changes):
        now=int(time.time())
        claims={"iss":self.settings.issuer,"aud":self.settings.audience,"sub":subject,"azp":client,"iat":now-1,"nbf":now-1,"exp":now+200}
        claims.update(changes)
        return jwt.encode(claims,self.key,algorithm="RS256",headers={"kid":"test-public-key","typ":"at+jwt"})

    def headers(self, reviewer=False):
        return {"Authorization":"Bearer "+self.token(REVIEWER.subject,REVIEWER.client_id) if reviewer else "Bearer "+self.token()}

    def test_fail_closed_without_configuration(self):
        with patch.dict(os.environ,{},clear=True), self.assertRaises(RuntimeError):
            configured_app()

    def test_config_requires_separate_agent_reviewer_and_fixed_profile(self):
        raw=self.settings.model_dump(mode="json")
        for change in ({"principals":[raw["principals"][0]]},{"outbound_adapter":"smtp"},{"deployment_profile":"serverless"},{"issuer":"http://identity.example.invalid"}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                Settings.model_validate_json(json.dumps(raw|change))

    def test_config_rejects_shared_client_between_agent_and_reviewer(self):
        raw=self.settings.model_dump(mode="json")
        raw["principals"][1]["client_id"]=AGENT.client_id
        with self.assertRaises(ValidationError): Settings.model_validate_json(json.dumps(raw))
        allowed=self.settings.model_copy(update={"principals":(
            Grant(subject=REVIEWER.subject,client_id=AGENT.client_id,role="agent"),
            Grant(subject=REVIEWER.subject,client_id=REVIEWER.client_id,role="reviewer"))})
        identity=Service(allowed,clock=self.clock)
        principal=identity.verifier.verify("Bearer "+self.token(subject=REVIEWER.subject,client=AGENT.client_id,role="reviewer"))
        self.assertEqual(principal.role,"agent")
        proposal=identity.propose(self.input(),principal,str(uuid.uuid4()))
        with self.assertRaises(StoreError) as context:
            identity.store.decide(proposal["id"],proposal["digest"],"approve",principal,str(uuid.uuid4()))
        self.assertEqual(context.exception.status,403)

    def test_database_identity_cannot_be_rebound(self):
        with self.assertRaises(RuntimeError):
            Store(self.settings.model_copy(update={"tenant_id":"another-customer"}))

    def test_missing_or_uninitialized_ledger_fails_closed_on_startup_and_reconnect(self):
        missing=self.settings.model_copy(update={"database_path":str(self.root/"data/missing.sqlite3")})
        with self.assertRaises(RuntimeError): Service(missing)
        Path(missing.database_path).touch()
        with self.assertRaises(RuntimeError): Service(missing)
        original=Path(self.settings.database_path)
        original.unlink()
        with self.assertRaises(RuntimeError): Service(self.settings)
        with self.assertRaises(StoreError): self.service.store.rate_limit(AGENT)
        with self.assertRaises(StoreError): self.service.store.get("missing-proposal")
        self.assertFalse(original.exists())

    def test_explicit_fresh_database_initialization_refuses_existing_or_restore_candidate(self):
        fresh=self.settings.model_copy(update={"database_path":str(self.root/"data/fresh.sqlite3")})
        initialized=Store.initialize(fresh)
        self.assertTrue(initialized.healthy())
        self.assertTrue(Service(fresh).store.healthy())
        with self.assertRaises(FileExistsError): Store.initialize(fresh)
        (self.root/"data/.zetbros-restore-pending.sqlite3").touch()
        absent=self.settings.model_copy(update={"database_path":str(self.root/"data/refuse-init.sqlite3")})
        with self.assertRaises(RuntimeError): Store.initialize(absent)
        self.assertFalse(Path(absent.database_path).exists())

    def test_migration_wal_full_and_append_only_audit(self):
        proposal=self.propose()
        with self.service.store.connection() as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0],1)
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0],"wal")
            self.assertEqual(conn.execute("PRAGMA synchronous").fetchone()[0],2)
            for sql in ("DELETE FROM audit","UPDATE audit SET outcome='forged'","UPDATE proposals SET payload='{}'"):
                with self.assertRaises(sqlite3.IntegrityError): conn.execute(sql)

    def test_authenticated_preview_and_exact_bound_reply(self):
        response=self.client.post("/v1/proposals",headers=self.headers(),json=self.input().model_dump(mode="json"))
        self.assertEqual(response.status_code,201)
        proposal=response.json()
        self.assertEqual(proposal["payload"]["to"],["replyto@example.invalid"])
        self.assertEqual(proposal["payload"]["cc"],[])
        self.assertEqual(proposal["payload"]["sender"],self.settings.sender_address)
        self.assertEqual(proposal["digest"],digest(proposal["payload"]))
        for field in ("tenant_id","connector_id","account_id","operation","source_version","source_fingerprint","policy_version","proposer_subject","proposer_client","created_at","expires_at"):
            self.assertIn(field,proposal["payload"])
        self.assertEqual(self.transport.calls,[])

    def test_agent_cannot_approve_even_with_spoofed_human_claims(self):
        proposal=self.propose()
        response=self.client.post(f'/v1/reviews/{proposal["id"]}/decision',headers={"Authorization":"Bearer "+self.token(role="reviewer",scope="approve",tenant_id="other"),"X-Actor":REVIEWER.subject,"X-Role":"reviewer"},json={"digest":proposal["digest"],"decision":"approve"})
        self.assertEqual(response.status_code,403)
        self.assertEqual(self.service.store.get(proposal["id"])["state"],"pending")
        with self.assertRaises(AuthenticationError):
            self.service.verifier.verify("Bearer "+self.token(subject=REVIEWER.subject,client=AGENT.client_id))

    def test_auth_rejects_wrong_signature_algorithm_kid_and_remote_key_headers(self):
        token=self.token()
        unsigned=jwt.encode({"sub":REVIEWER.subject},key="",algorithm="none",headers={"kid":"test-public-key","typ":"at+jwt"})
        weak=jwt.encode({"sub":REVIEWER.subject},key="test-only-key"*4,algorithm="HS256",headers={"kid":"test-public-key","typ":"at+jwt"})
        for bad in (unsigned,weak,token[:-8]+"xxxxxxxx",
                    jwt.encode({"sub":REVIEWER.subject},self.key,algorithm="RS256",headers={"kid":"unknown","typ":"at+jwt"}),
                    jwt.encode({"sub":REVIEWER.subject},self.key,algorithm="RS256",headers={"kid":"test-public-key","typ":"at+jwt","jku":"https://untrusted.invalid/jwks"})):
            with self.subTest(token=bad[:20]), self.assertRaises(AuthenticationError): self.service.verifier.verify("Bearer "+bad)

    def test_auth_claim_boundaries(self):
        now=int(time.time())
        for changes in ({"aud":"other"},{"iss":"https://other.invalid"},{"exp":now},{"nbf":now+1},{"iat":now+1},{"sub":"unknown"},{"azp":"other"},{"exp":str(now+100)},{"exp":now+5000},{"aud":[self.settings.audience,"other"]}):
            with self.subTest(changes=changes), self.assertRaises(AuthenticationError):
                self.service.verifier.verify("Bearer "+self.token(**changes))

    def test_required_access_token_claims_and_header_type_are_enforced(self):
        now=int(time.time())
        claims={"iss":self.settings.issuer,"aud":self.settings.audience,"sub":AGENT.subject,"azp":AGENT.client_id,"iat":now-1,"nbf":now-1,"exp":now+100}
        for missing in claims:
            payload={key:value for key,value in claims.items() if key!=missing}
            token=jwt.encode(payload,self.key,algorithm="RS256",headers={"kid":"test-public-key","typ":"at+jwt"})
            with self.subTest(missing=missing), self.assertRaises(AuthenticationError): self.service.verifier.verify("Bearer "+token)
        token=jwt.encode(claims,self.key,algorithm="RS256",headers={"kid":"test-public-key","typ":"JWT"})
        with self.assertRaises(AuthenticationError): self.service.verifier.verify("Bearer "+token)

    def test_jwks_rejects_private_oct_and_small_keys(self):
        raw={"keys":[dict(self.jwk)]}
        for change in ({"kty":"oct","k":"secret"},{"d":"private"},{"alg":"HS256"},{"key_ops":["sign"]}):
            raw["keys"][0]=dict(self.jwk)|change
            (self.root/"jwks.json").write_text(json.dumps(raw))
            with self.assertRaises(RuntimeError): Service(self.settings)
        small=rsa.generate_private_key(public_exponent=65537,key_size=1024)
        jwk=json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(small.public_key()))
        jwk.update(kid="small-test-key",alg="RS256",use="sig")
        (self.root/"jwks.json").write_text(json.dumps({"keys":[jwk]}))
        with self.assertRaises(RuntimeError): Service(self.settings)

    def test_api_rejects_missing_auth_tokens_in_urls_and_browser_origin(self):
        self.assertEqual(self.client.get("/v1/source/message-001").status_code,401)
        self.assertEqual(self.client.get("/v1/source/message-001?token=private").status_code,400)
        self.assertEqual(self.client.get("/v1/source/message-001?access%5Ftoken=private").status_code,400)
        self.assertEqual(self.client.get("/v1/source/message-001",headers=self.headers()|{"Origin":"https://attacker.invalid"}).status_code,403)

    def test_unknown_actor_destination_tool_and_approval_fields_rejected(self):
        body=self.input().model_dump(mode="json")
        for field,value in (("tenant_id","other"),("actor",REVIEWER.subject),("approved",True),("to",["arbitrary.invalid"]),("cc",["other@example.invalid"]),("subject","forged"),("attachments",[]),("operation","send"),("approval_id","forged")):
            response=self.client.post("/v1/proposals",headers=self.headers(),json=body|{field:value})
            with self.subTest(field=field):
                self.assertEqual(response.status_code,422)
                self.assertNotIn("BODY_CANARY",response.text)
        self.assertEqual(self.transport.calls,[])

    def test_payload_types_size_and_header_injection(self):
        for changes in ({"body":True},{"source_id":"../jwks"},{"body":"x"*16001},{"source_version":1}):
            with self.subTest(changes=str(changes)[:80]), self.assertRaises(ValidationError): self.input(**changes)
        for key in ("sender","reply_to","subject"):
            data=self.record.model_dump(mode="json")|{key:"x\r\nBcc: victim@example.invalid"}
            with self.subTest(field=key), self.assertRaises(ValidationError): SourceMessage.model_validate_json(json.dumps(data))
        self.assertEqual(self.client.post("/v1/proposals",headers=self.headers(),content="x"*40000).status_code,415)
        self.assertEqual(self.client.post("/v1/proposals",headers=self.headers()|{"Content-Type":"application/json"},content="x"*40000).status_code,413)

    def test_source_path_boundaries_symlinks_and_customer_binding(self):
        root=self.root/"sources"
        (root/"linked.json").symlink_to(self.root/"jwks.json")
        with self.assertRaises(SourceUnavailable): self.service.source.get("linked")
        self.write_record(self.record.model_copy(update={"tenant_id":"other"}))
        with self.assertRaises(SourceUnavailable): self.service.source.get(self.record.source_id)
        for source_id in ("../jwks","/etc/passwd","."):
            with self.assertRaises(ValidationError): self.service.source.get(source_id)

    def test_fifo_source_is_rejected_without_blocking(self):
        os.mkfifo(self.root/"sources/fifo-message.json")
        ctx=multiprocessing.get_context("spawn"); queue=ctx.Queue()
        child=ctx.Process(target=process_fifo,args=(self.settings.model_dump(mode="json"),queue))
        child.start(); child.join(timeout=5)
        if child.is_alive(): child.terminate(); child.join(timeout=5); self.fail("FIFO source blocked")
        self.assertEqual(child.exitcode,0)
        self.assertEqual(queue.get(timeout=2),"rejected")

    def test_stale_source_proposal_rejected(self):
        with self.assertRaises(StoreError): self.propose(source_fingerprint="0"*64)
        with self.assertRaises(StoreError): self.propose(source_version="different")

    def test_idempotency_same_key_returns_stored_and_changed_body_conflicts(self):
        first=self.propose()
        self.clock.now+=5
        again=self.propose()
        self.assertEqual(first["id"],again["id"])
        with self.assertRaises(StoreError) as context: self.propose(body="different body")
        self.assertEqual(context.exception.code,"idempotency_conflict")

    def test_missing_or_wrong_digest_never_approves(self):
        proposal=self.propose()
        with self.assertRaises(StoreError): self.service.store.decide(proposal["id"],"0"*64,"approve",REVIEWER,str(uuid.uuid4()))
        response=self.client.post(f'/v1/reviews/{proposal["id"]}/decision',headers=self.headers(True),json={"decision":"approve","approved":True})
        self.assertEqual(response.status_code,422)
        self.run_worker()
        self.assertEqual(self.transport.calls,[])

    def test_approve_and_reservation_atomic_and_replay_one_submission(self):
        proposal=self.propose()
        approved=self.approve(proposal)
        again=self.approve(proposal)
        self.assertEqual(approved["approval_id"],again["approval_id"])
        self.assertIsNone(approved["consumed_at"])
        self.run_worker(); self.run_worker()
        final=self.service.store.get(proposal["id"])
        self.assertEqual(final["execution"]["state"],"accepted")
        self.assertEqual(final["execution"]["result"]["delivery"],"unverified")
        self.assertEqual(len(self.transport.calls),1)
        self.assertIsNotNone(final["consumed_at"])

    def test_cross_proposal_digest_cannot_approve(self):
        first=self.propose()
        second=self.propose(operation_key="ticket-002-reply-1")
        with self.assertRaises(StoreError): self.service.store.decide(second["id"],first["digest"],"approve",REVIEWER,str(uuid.uuid4()))

    def test_denial_terminal_and_revocation_prevents_claim(self):
        proposal=self.propose()
        self.service.store.decide(proposal["id"],proposal["digest"],"deny",REVIEWER,str(uuid.uuid4()))
        with self.assertRaises(StoreError): self.approve(proposal)
        second=self.propose(operation_key="ticket-002-reply-1")
        self.approve(second)
        self.service.store.decide(second["id"],second["digest"],"revoke",REVIEWER,str(uuid.uuid4()))
        self.run_worker()
        self.assertEqual(self.transport.calls,[])
        self.assertEqual(self.service.store.get(second["id"])["execution"]["state"],"revoked")

    def test_expiry_at_boundary_and_before_issuance(self):
        proposal=self.propose()
        self.clock.now=proposal["expires_at"]
        with self.assertRaises(StoreError): self.approve(proposal)
        self.clock.now=proposal["created_at"]-1
        with self.assertRaises(StoreError): self.approve(proposal)

    def test_queued_approval_expiry_blocks_without_consumption(self):
        proposal=self.propose(); approved=self.approve(proposal)
        self.clock.now=approved["approval_expires_at"]
        self.run_worker()
        final=self.service.store.get(proposal["id"])
        self.assertEqual(final["execution"]["state"],"expired")
        self.assertIsNone(final["consumed_at"])
        self.assertEqual(self.transport.calls,[])

    def test_source_drift_and_policy_change_block_execution(self):
        for policy in (False,True):
            with self.subTest(policy=policy):
                proposal=self.propose(operation_key="ticket-source-drift-"+str(policy))
                self.approve(proposal)
                if policy:
                    self.service.store.settings=self.settings.model_copy(update={"policy_version":"policy-2"})
                else:
                    self.write_record(self.record.model_copy(update={"reply_to":"changed@example.invalid"}))
                self.run_worker()
                self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"],"invalidated")
                self.write_record(self.record)
        self.assertEqual(self.transport.calls,[])

    def test_revoked_configured_identity_blocks_queued_approval(self):
        proposal=self.propose(); self.approve(proposal)
        self.service.store.settings=self.settings.model_copy(update={"principals":(self.settings.principals[0],)})
        self.run_worker()
        self.assertEqual(self.transport.calls,[])
        self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"],"invalidated")

    def test_disabled_runtime_records_block_without_consuming_approval(self):
        runtime=Service(self.settings,clock=self.clock)
        proposal=runtime.propose(self.input(),AGENT,str(uuid.uuid4()))
        runtime.store.decide(proposal["id"],proposal["digest"],"approve",REVIEWER,str(uuid.uuid4()))
        asyncio.run(runtime.worker_once())
        result=runtime.store.get(proposal["id"])
        self.assertEqual(result["execution"]["state"],"blocked")
        self.assertEqual(result["execution"]["result"]["error_class"],"delivery_disabled")
        self.assertIsNone(result["consumed_at"])
        self.assertFalse(runtime.readiness()["live_delivery_ready"])

    def test_source_missing_at_execution_blocks_without_consumption(self):
        proposal=self.propose(); self.approve(proposal)
        (self.root/"sources"/(self.record.source_id+".json")).unlink()
        self.run_worker()
        result=self.service.store.get(proposal["id"])
        self.assertEqual(result["execution"]["result"]["error_class"],"source_unavailable")
        self.assertIsNone(result["consumed_at"])
        self.assertEqual(self.transport.calls,[])

    def test_preclaim_audit_failure_rolls_back_approval_consumption(self):
        proposal=self.propose(); self.approve(proposal)
        original=self.service.store.event
        def fail(conn,request_id,actor,action_id,action_digest,policy,kind,outcome,error="none"):
            if kind=="claim": raise StoreError("audit_failure",503)
            return original(conn,request_id,actor,action_id,action_digest,policy,kind,outcome,error)
        with patch.object(self.service.store,"event",side_effect=fail), self.assertRaises(StoreError): self.run_worker()
        final=self.service.store.get(proposal["id"])
        self.assertIsNone(final["consumed_at"])
        self.assertEqual(final["execution"]["state"],"queued")
        self.assertEqual(self.transport.calls,[])

    def test_decision_audit_failure_rolls_back_decision_and_reservation(self):
        proposal=self.propose()
        with patch.object(self.service.store,"event",side_effect=StoreError("audit_failure",503)), self.assertRaises(StoreError): self.approve(proposal)
        result=self.service.store.get(proposal["id"])
        self.assertEqual(result["state"],"pending")
        self.assertIsNone(result["execution"])
        self.assertIsNone(result["approval_id"])

    def test_database_capacity_before_submit_causes_zero_adapter_calls(self):
        proposal=self.propose(); self.approve(proposal)
        with patch.object(self.service.store,"capacity",side_effect=StoreError("storage_capacity",503)), self.assertRaises(StoreError): self.run_worker()
        self.assertEqual(self.transport.calls,[])

    def test_postsubmit_audit_failure_quarantines_no_retry(self):
        proposal=self.propose(); self.approve(proposal)
        original=self.service.store.event
        def fail(conn,request_id,actor,action_id,action_digest,policy,kind,outcome,error="none"):
            if kind=="result" and outcome=="accepted": raise StoreError("audit_failure",503)
            return original(conn,request_id,actor,action_id,action_digest,policy,kind,outcome,error)
        with patch.object(self.service.store,"event",side_effect=fail), self.assertRaises(StoreError): self.run_worker()
        self.assertEqual(len(self.transport.calls),1)
        self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"],"claimed")
        self.clock.now+=self.settings.claim_timeout_seconds
        self.run_worker(); self.run_worker()
        self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"],"uncertain")
        self.assertEqual(len(self.transport.calls),1)

    def test_accepted_sent_copy_failure_is_never_resent(self):
        self.transport.sent_copy="failed"
        proposal=self.propose(); self.approve(proposal)
        self.run_worker(); self.run_worker()
        result=self.service.store.get(proposal["id"])["execution"]["result"]
        self.assertEqual((result["submission"],result["sent_copy"],result["delivery"]),("accepted","failed","unverified"))
        self.assertEqual(len(self.transport.calls),1)

    def test_uncertain_rejected_and_adapter_exception_are_not_retried(self):
        for mode in ("uncertain","rejected","exception"):
            self.transport.submission=mode if mode!="exception" else "accepted"
            self.transport.raises=mode=="exception"
            proposal=self.propose(operation_key="ticket-result-"+mode)
            self.approve(proposal); self.run_worker(); self.run_worker()
            result=self.service.store.get(proposal["id"])["execution"]["result"]
            self.assertEqual(result["submission"],mode if mode!="exception" else "uncertain")
            self.assertNotIn("PASSWORD_CANARY",canonical(result))
        self.assertEqual(len(self.transport.calls),3)

    def test_intentional_second_operation_requires_new_approval(self):
        first=self.propose(); self.approve(first); self.run_worker()
        second=self.propose(operation_key="ticket-001-explicit-repeat")
        self.run_worker(); self.assertEqual(len(self.transport.calls),1)
        self.approve(second); self.run_worker()
        self.assertEqual(len(self.transport.calls),2)

    def test_restart_preserves_result_and_no_duplicate_submission(self):
        proposal=self.propose(); self.approve(proposal); self.run_worker()
        restarted=Service(self.settings,transport=self.transport,clock=self.clock)
        asyncio.run(restarted.worker_once())
        self.assertEqual(restarted.store.get(proposal["id"])["execution"]["state"],"accepted")
        self.assertEqual(len(self.transport.calls),1)
        self.assertEqual(restarted.propose(self.input(),AGENT,str(uuid.uuid4()))["id"],proposal["id"])

    def test_independent_processes_only_one_claim_commits(self):
        proposal=self.propose(); self.approve(proposal)
        ctx=multiprocessing.get_context("spawn")
        barrier=ctx.Barrier(4); queue=ctx.Queue()
        processes=[ctx.Process(target=process_claim,args=(self.settings.model_dump(mode="json"),proposal["id"],digest(self.record),barrier,queue)) for _ in range(4)]
        for process in processes: process.start()
        for process in processes: process.join(timeout=15); self.assertEqual(process.exitcode,0)
        results=[queue.get(timeout=2) for _ in processes]
        self.assertEqual(results.count(True),1)
        self.assertEqual(results.count(False),3)
        self.assertIsNotNone(self.service.store.get(proposal["id"])["consumed_at"])

    def test_process_crash_after_claim_or_possible_submission_never_reclaims(self):
        for possible in (False,True):
            proposal=self.propose(operation_key="ticket-crash-case-"+str(possible)); self.approve(proposal)
            marker=self.root/("crash-"+str(possible))
            process=multiprocessing.get_context("spawn").Process(target=process_crash,args=(self.settings.model_dump(mode="json"),proposal["id"],digest(self.record),str(marker),possible,self.clock.now))
            process.start(); process.join(timeout=15)
            self.assertEqual(process.exitcode,0); self.assertTrue(marker.exists())
            restarted=Service(self.settings,transport=self.transport,clock=self.clock)
            self.clock.now+=self.settings.claim_timeout_seconds
            asyncio.run(restarted.worker_once()); asyncio.run(restarted.worker_once())
            self.assertEqual(restarted.store.get(proposal["id"])["execution"]["state"],"uncertain")
        self.assertEqual(self.transport.calls,[])

    def test_transaction_contention_uses_fresh_expiry_and_deadline_clock(self):
        def crossing(operation, boundary):
            errors=[]; results=[]; waiting=threading.Event()
            original_connection=self.service.store.connection
            from contextlib import contextmanager
            @contextmanager
            def observed_connection():
                with original_connection() as conn:
                    conn.set_trace_callback(lambda statement: waiting.set() if statement=="BEGIN IMMEDIATE" else None)
                    yield conn
            def invoke():
                try: results.append(operation())
                except Exception as exc: errors.append(exc)
            self.clock.now=boundary-1
            with original_connection() as locked:
                locked.execute("BEGIN IMMEDIATE")
                with patch.object(self.service.store,"connection",observed_connection):
                    thread=threading.Thread(target=invoke); thread.start()
                    self.assertTrue(waiting.wait(timeout=2))
                    self.clock.now=boundary+1
                    locked.commit(); thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
            return results,errors
        pending=self.propose()
        results,errors=crossing(lambda:self.approve(pending),pending["expires_at"])
        self.assertEqual(results,[]); self.assertEqual(errors[0].code,"proposal_expired")
        proposal=self.propose(operation_key="ticket-contended-claim"); approved=self.approve(proposal)
        results,errors=crossing(lambda:self.service.store.claim(proposal["id"],digest(self.record),True,str(uuid.uuid4())),approved["approval_expires_at"])
        self.assertEqual(results,[None]); self.assertEqual(errors,[])
        self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"],"expired")
        claimed=self.propose(operation_key="ticket-contended-finish"); self.approve(claimed)
        action,token=self.service.store.claim(claimed["id"],digest(self.record),True,str(uuid.uuid4()))
        deadline=self.service.store.get(claimed["id"])["execution"]["claim_deadline"]
        results,errors=crossing(lambda:self.service.store.finish(claimed["id"],token,TransportResult(submission="accepted",sent_copy="stored").model_dump(mode="json"),str(uuid.uuid4())),deadline)
        self.assertEqual(results,[]); self.assertEqual(errors[0].code,"claim_fenced")
        interrupted=self.propose(operation_key="ticket-contended-expire"); self.approve(interrupted)
        self.service.store.claim(interrupted["id"],digest(self.record),True,str(uuid.uuid4()))
        deadline=self.service.store.get(interrupted["id"])["execution"]["claim_deadline"]
        results,errors=crossing(lambda:self.service.store.expire_claims(str(uuid.uuid4())),deadline)
        self.assertEqual(errors,[])
        self.assertEqual(self.service.store.get(interrupted["id"])["execution"]["state"],"uncertain")

    def test_late_completion_fenced_and_revoke_after_claim_refused(self):
        proposal=self.propose(); self.approve(proposal)
        action,token=self.service.store.claim(proposal["id"],digest(self.record),True,str(uuid.uuid4()))
        with self.assertRaises(StoreError): self.service.store.decide(proposal["id"],proposal["digest"],"revoke",REVIEWER,str(uuid.uuid4()))
        self.clock.now+=self.settings.claim_timeout_seconds
        self.service.store.expire_claims(str(uuid.uuid4()))
        with self.assertRaises(StoreError): self.service.store.finish(proposal["id"],token,{"submission":"accepted"},str(uuid.uuid4()))

    def test_reconciliation_is_observation_never_reopens_uncertainty(self):
        self.transport.submission="uncertain"
        proposal=self.propose(); self.approve(proposal); self.run_worker()
        self.service.store.reconcile(proposal["id"],proposal["digest"],"accepted","provider-observation-1",REVIEWER,str(uuid.uuid4()))
        self.run_worker()
        self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"],"uncertain")
        self.assertEqual(len(self.transport.calls),1)

    def test_reconciliation_preserves_append_only_evidence_references(self):
        self.transport.submission="uncertain"
        proposal=self.propose(); self.approve(proposal); self.run_worker()
        self.service.store.reconcile(proposal["id"],proposal["digest"],"uncertain","observation-one",REVIEWER,str(uuid.uuid4()))
        self.service.store.reconcile(proposal["id"],proposal["digest"],"accepted","observation-two",REVIEWER,str(uuid.uuid4()))
        observations=self.service.store.reconciliations(proposal["id"])
        self.assertEqual([o["evidence_reference"] for o in observations],["observation-one","observation-two"])
        self.assertEqual(len({o["event_id"] for o in observations}),2)
        with self.service.store.connection() as conn:
            for sql in ("UPDATE reconciliations SET evidence_reference='forged'","DELETE FROM reconciliations"):
                with self.assertRaises(sqlite3.IntegrityError): conn.execute(sql)
        url=f'/v1/reviews/{proposal["id"]}/reconciliations'
        self.assertEqual(self.client.get(url,headers=self.headers()).status_code,403)
        self.assertEqual(len(self.client.get(url,headers=self.headers(True)).json()["observations"]),2)

    def test_restore_quarantine_blocks_without_claim(self):
        proposal=self.propose(); self.approve(proposal)
        with self.service.store.transaction() as conn:
            conn.execute("UPDATE metadata SET value='true' WHERE key='execution_quarantine'")
        self.run_worker()
        result=self.service.store.get(proposal["id"])
        self.assertEqual(result["execution"]["result"]["error_class"],"restore_quarantine")
        self.assertIsNone(result["consumed_at"])
        self.assertEqual(self.transport.calls,[])

    def test_rate_limit_persists_across_service_restart(self):
        limited=Service(self.settings.model_copy(update={"requests_per_minute":1}),clock=self.clock)
        limited.store.rate_limit(AGENT)
        restarted=Service(limited.settings,clock=self.clock)
        with self.assertRaises(StoreError) as context: restarted.store.rate_limit(AGENT)
        self.assertEqual(context.exception.status,429)

    def test_audit_minimization_and_reviewer_only_access(self):
        proposal=self.propose(); self.approve(proposal); self.run_worker()
        events=self.service.store.audit()
        text=canonical({"events":events})
        for forbidden in ("BODY_CANARY","Support question","PASSWORD_CANARY","support@example.invalid","replyto@example.invalid","Bearer"):
            self.assertNotIn(forbidden,text)
        for event in events:
            for required in ("event_id","request_id","occurred_at","tenant_id","actor","client_id","action_id","action_digest","policy_version","event_type","outcome","error_class"):
                self.assertIn(required,event)
        self.assertEqual(self.client.get("/v1/audit",headers=self.headers()).status_code,403)
        self.assertEqual(self.client.get("/v1/audit",headers=self.headers(True)).status_code,200)

    def test_graceful_worker_stop_finishes_inflight_but_does_not_claim_next(self):
        first=self.propose(); self.approve(first)
        second=self.propose(operation_key="ticket-graceful-next"); self.approve(second)
        async def scenario():
            started=asyncio.Event(); release=asyncio.Event(); stop=asyncio.Event()
            class WaitingTransport:
                enabled=True
                async def submit(transport,action):
                    started.set()
                    await release.wait()
                    return TransportResult(submission="accepted",sent_copy="stored")
            self.service.transport=WaitingTransport()
            task=asyncio.create_task(self.service.worker_loop(stop))
            await started.wait(); stop.set(); release.set(); await task
        asyncio.run(scenario())
        self.assertEqual(self.service.store.get(first["id"])["execution"]["state"],"accepted")
        self.assertEqual(self.service.store.get(second["id"])["execution"]["state"],"queued")

    def test_timeout_and_cancellation_after_invocation_remain_uncertain(self):
        for cancel in (False,True):
            proposal=self.propose(operation_key="ticket-timeout-"+str(cancel)); self.approve(proposal)
            self.service.settings=self.settings.model_copy(update={"adapter_timeout_seconds":1})
            calls=[]
            async def scenario():
                started=asyncio.Event()
                class WaitingTransport:
                    enabled=True
                    async def submit(transport,action):
                        calls.append(action)
                        started.set(); await asyncio.sleep(5)
                self.service.transport=WaitingTransport()
                task=asyncio.create_task(self.service.worker_once())
                if cancel:
                    await started.wait(); task.cancel()
                    with self.assertRaises(asyncio.CancelledError): await task
                else: await task
            asyncio.run(scenario())
            self.assertEqual(len(calls),1)
            self.assertEqual(self.service.store.get(proposal["id"])["execution"]["state"],"uncertain")
            self.run_worker(); self.assertEqual(len(calls),1)

    def test_json_duplicate_keys_and_nonfinite_values_rejected(self):
        headers=self.headers()|{"Content-Type":"application/json"}
        for raw in ('{"body":"x","body":"y"}', '{"body":NaN}'):
            self.assertEqual(self.client.post("/v1/proposals",headers=headers,content=raw).status_code,400)

    def test_backup_restore_preserves_ledger_and_quarantines(self):
        from zetbros_service.maintenance import backup, restore
        first=self.propose(); self.approve(first); self.run_worker()
        second=self.propose(operation_key="ticket-backup-queued"); self.approve(second)
        third=self.propose(operation_key="ticket-backup-claimed"); self.approve(third)
        self.service.store.claim(third["id"],digest(self.record),True,str(uuid.uuid4()))
        destination=self.root/"backup.sqlite3"
        backup(self.service.store,destination)
        restored_settings=self.settings.model_copy(update={"database_path":str(self.root/"data/restored.sqlite3")})
        result=restore(restored_settings,destination)
        self.assertTrue(result["execution_quarantine"])
        restored=Service(restored_settings,transport=self.transport,clock=self.clock)
        self.assertEqual(restored.store.get(first["id"])["execution"]["state"],"accepted")
        self.assertEqual(restored.store.get(second["id"])["execution"]["state"],"blocked")
        self.assertEqual(restored.store.get(third["id"])["execution"]["state"],"uncertain")
        asyncio.run(restored.worker_once())
        self.assertEqual(len(self.transport.calls),1)
        with self.assertRaises(StoreError): backup(self.service.store,destination)
        with self.assertRaises(StoreError): restore(restored_settings,destination)

    def test_interrupted_restore_does_not_publish_unquarantined_final_database(self):
        from zetbros_service.maintenance import backup
        proposal=self.propose(); self.approve(proposal)
        backup_file=self.root/"interrupt-backup.sqlite3"; backup(self.service.store,backup_file)
        target=self.root/"data/interrupted-restore.sqlite3"
        restored=self.settings.model_copy(update={"database_path":str(target)})
        child=multiprocessing.get_context("spawn").Process(target=process_interrupted_restore,args=(restored.model_dump(mode="json"),str(backup_file)))
        child.start(); child.join(timeout=10)
        if child.is_alive(): child.terminate(); child.join(timeout=5); self.fail("restore child blocked")
        self.assertEqual(child.exitcode,23)
        self.assertFalse(target.exists())
        self.assertTrue(list(target.parent.glob(".zetbros-restore-*.sqlite3")))
        with self.assertRaises(RuntimeError): Service(restored)
        with self.assertRaises(RuntimeError): Store.initialize(restored)
        self.assertFalse(target.exists())

    def test_restore_final_publication_never_overwrites_racing_destination(self):
        from zetbros_service.maintenance import backup, restore
        backup_file=self.root/"race-backup.sqlite3"; backup(self.service.store,backup_file)
        target=self.root/"data/race-restore.sqlite3"
        settings=self.settings.model_copy(update={"database_path":str(target)})
        original_link=os.link
        def racing_link(source,destination):
            Path(destination).write_text("existing database must be preserved")
            original_link(source,destination)
        with patch("zetbros_service.maintenance.os.link",side_effect=racing_link), self.assertRaises(FileExistsError):
            restore(settings,backup_file)
        self.assertEqual(target.read_text(),"existing database must be preserved")

    def test_health_readiness_truthfully_discloses_disabled_delivery(self):
        self.assertEqual(self.client.get("/healthz").json(),{"status":"up"})
        result=self.client.get("/readyz")
        self.assertEqual(result.status_code,200)
        self.assertTrue(result.json()["control_ready"])
        self.assertFalse(result.json()["live_delivery_ready"])
        self.assertEqual(result.json()["delivery"],"disabled")


if __name__ == "__main__":
    unittest.main()
