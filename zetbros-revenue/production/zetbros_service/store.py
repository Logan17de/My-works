"""Durable single-host ledger. No transaction spans an adapter invocation."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .auth import Principal
from .config import Settings
from .models import ReplyAction, canonical, digest


class StoreError(Exception):
    def __init__(self, code: str, status: int = 409):
        self.code, self.status = code, status
        super().__init__(code)


class Store:
    workflow_contract = "action_v1"

    def __init__(self, settings: Settings, clock=time.time, *, initialize: bool = False, maintenance_only: bool = False):
        self.settings, self.clock = settings, clock
        self.maintenance_only = maintenance_only
        self.path = Path(settings.database_path)
        if not self.path.parent.is_dir() or self.path.is_symlink() or not self.path.is_file():
            raise RuntimeError("initialized local database required; use explicit init only for a new deployment")
        with self.connection() as conn:
            if conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise RuntimeError("WAL required")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1) or (version == 0 and not initialize):
                raise RuntimeError("initialized supported database schema required")
            if version == 0:
                conn.executescript("""
BEGIN IMMEDIATE;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE proposals (
 id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, connector_id TEXT NOT NULL,
 operation_key TEXT NOT NULL, digest TEXT NOT NULL, request_digest TEXT NOT NULL,
 payload TEXT NOT NULL, proposer_subject TEXT NOT NULL, proposer_client TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('pending','approved','denied','revoked')),
 created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, approval_id TEXT,
 reviewer_subject TEXT, reviewer_client TEXT, approval_at INTEGER,
 approval_expires_at INTEGER, consumed_at INTEGER,
 UNIQUE(tenant_id, connector_id, operation_key));
CREATE TABLE decisions (
 id TEXT PRIMARY KEY, proposal_id TEXT NOT NULL REFERENCES proposals(id),
 actor TEXT NOT NULL, client_id TEXT NOT NULL, digest TEXT NOT NULL,
 decision TEXT NOT NULL CHECK(decision IN ('approve','deny','revoke')),
 created_at INTEGER NOT NULL, expires_at INTEGER);
CREATE TABLE executions (
 id TEXT PRIMARY KEY, proposal_id TEXT NOT NULL UNIQUE REFERENCES proposals(id),
 tenant_id TEXT NOT NULL, connector_id TEXT NOT NULL, operation_key TEXT NOT NULL,
 digest TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN
 ('queued','blocked','claimed','accepted','rejected','uncertain','revoked','expired','invalidated')),
 claim_token TEXT, claimed_at INTEGER, claim_deadline INTEGER,
 completed_at INTEGER, result TEXT,
 UNIQUE(tenant_id,connector_id,operation_key));
CREATE TABLE audit (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
 request_id TEXT NOT NULL, occurred_at INTEGER NOT NULL, tenant_id TEXT NOT NULL,
 connector_id TEXT NOT NULL, actor TEXT NOT NULL, client_id TEXT NOT NULL,
 action_id TEXT NOT NULL, action_digest TEXT NOT NULL, policy_version TEXT NOT NULL,
 event_type TEXT NOT NULL, outcome TEXT NOT NULL, error_class TEXT NOT NULL);
CREATE TABLE reconciliations (
 id TEXT PRIMARY KEY, proposal_id TEXT NOT NULL REFERENCES proposals(id),
 digest TEXT NOT NULL, actor TEXT NOT NULL, client_id TEXT NOT NULL,
 observed_submission TEXT NOT NULL CHECK(observed_submission IN ('accepted','rejected','uncertain')),
 evidence_reference TEXT NOT NULL, occurred_at INTEGER NOT NULL,
 event_id TEXT NOT NULL UNIQUE REFERENCES audit(event_id));
CREATE TRIGGER reconciliation_no_update BEFORE UPDATE ON reconciliations BEGIN SELECT RAISE(ABORT,'append_only'); END;
CREATE TRIGGER reconciliation_no_delete BEFORE DELETE ON reconciliations BEGIN SELECT RAISE(ABORT,'append_only'); END;
CREATE TABLE rate_limits (
 principal TEXT NOT NULL, bucket INTEGER NOT NULL, count INTEGER NOT NULL,
 PRIMARY KEY(principal,bucket));
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit BEGIN SELECT RAISE(ABORT,'append_only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit BEGIN SELECT RAISE(ABORT,'append_only'); END;
CREATE TRIGGER decision_no_update BEFORE UPDATE ON decisions BEGIN SELECT RAISE(ABORT,'append_only'); END;
CREATE TRIGGER decision_no_delete BEFORE DELETE ON decisions BEGIN SELECT RAISE(ABORT,'append_only'); END;
CREATE TRIGGER immutable_proposal BEFORE UPDATE OF payload,digest,request_digest,tenant_id,connector_id,operation_key ON proposals
 BEGIN SELECT RAISE(ABORT,'immutable_proposal'); END;
PRAGMA user_version=1;
COMMIT;
""")
        with self.transaction() as conn:
            mail_schema = conn.execute("SELECT value FROM metadata WHERE key='mail_review_schema'").fetchone()
            mail_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='mail_previews'").fetchone()
            if bool(mail_schema) != bool(mail_table):
                raise RuntimeError("mail-review ledger extension metadata/table is incomplete")
            if mail_schema and (mail_schema[0] != "1" or (self.workflow_contract != "mail_exact_wire_v1" and not maintenance_only)):
                raise RuntimeError("mail-review ledger cannot use the action-only workflow")
            existing = conn.execute("SELECT value FROM metadata WHERE key='deployment_identity'").fetchone()
            if not existing and not initialize:
                raise RuntimeError("deployment identity metadata is missing")
            if existing and existing[0] != settings.identity_digest:
                raise RuntimeError("existing database cannot be rebound to another deployment identity")
            conn.execute("INSERT OR IGNORE INTO metadata VALUES ('deployment_identity',?)", (settings.identity_digest,))
            quarantine = conn.execute("SELECT value FROM metadata WHERE key='execution_quarantine'").fetchone()
            if not initialize and (not quarantine or quarantine[0] not in ("true","false")):
                raise RuntimeError("execution quarantine metadata is missing or invalid")
            if initialize:
                conn.execute("INSERT OR IGNORE INTO metadata VALUES ('execution_quarantine','false')")

    @classmethod
    def initialize(cls, settings: Settings, clock=time.time):
        path = Path(settings.database_path)
        if not path.parent.is_dir() or list(path.parent.glob(".zetbros-restore-*.sqlite3")):
            raise RuntimeError("fresh deployment directory required; pending restore candidates must be reconciled")
        fd = os.open(path,os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,0o600)
        os.close(fd)
        return cls(settings,clock,initialize=True)

    @contextmanager
    def connection(self):
        # mode=rw is essential: a lost/unmounted/deleted ledger must never be
        # silently recreated by startup, rate limiting, reads or worker reconnects.
        conn = sqlite3.connect(self.path.as_uri()+"?mode=rw", uri=True, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA busy_timeout=5000")
            if conn.execute("PRAGMA synchronous").fetchone()[0] != 2:
                raise RuntimeError("FULL database synchronization required")
            yield conn
        finally:
            conn.close()

    def capacity(self):
        size = sum(p.stat().st_size for p in (self.path, Path(str(self.path)+"-wal"), Path(str(self.path)+"-shm")) if p.exists())
        if size >= self.settings.max_database_bytes or shutil.disk_usage(self.path.parent).free < 1048576:
            raise StoreError("storage_capacity", 503)

    @contextmanager
    def transaction(self):
        self.capacity()
        try:
            with self.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
        except sqlite3.Error as exc:
            raise StoreError("storage_unavailable", 503) from exc

    def event(self, conn, request_id: str, actor: Principal, action_id: str, action_digest: str,
              policy: str, kind: str, outcome: str, error: str = "none"):
        event_id = str(uuid.uuid4())
        conn.execute("INSERT INTO audit(event_id,request_id,occurred_at,tenant_id,connector_id,actor,client_id,action_id,action_digest,policy_version,event_type,outcome,error_class) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (event_id, request_id, int(self.clock()), self.settings.tenant_id,
                      self.settings.connector_id, actor.subject, actor.client_id, action_id,
                      action_digest, policy, kind, outcome, error))
        return event_id

    def rate_limit(self, principal: Principal):
        with self.transaction() as conn:
            bucket = int(self.clock()) // 60
            key = canonical({"sub": principal.subject, "client": principal.client_id})
            conn.execute("DELETE FROM rate_limits WHERE bucket < ?", (bucket-1,))
            row = conn.execute("SELECT count FROM rate_limits WHERE principal=? AND bucket=?", (key,bucket)).fetchone()
            if row and row[0] >= self.settings.requests_per_minute:
                raise StoreError("rate_limit", 429)
            conn.execute("INSERT INTO rate_limits VALUES(?,?,1) ON CONFLICT(principal,bucket) DO UPDATE SET count=count+1", (key,bucket))

    def create(self, action: ReplyAction, request_digest: str, actor: Principal, request_id: str):
        if self.maintenance_only:
            raise StoreError("maintenance_only")
        with self.transaction() as conn:
            old = conn.execute("SELECT * FROM proposals WHERE tenant_id=? AND connector_id=? AND operation_key=?",
                               (action.tenant_id,action.connector_id,action.operation_key)).fetchone()
            if old:
                if old["request_digest"] != request_digest or old["proposer_subject"] != actor.subject or old["proposer_client"] != actor.client_id:
                    raise StoreError("idempotency_conflict")
                return self.view(conn, old["id"])
            proposal_id = str(uuid.uuid4())
            conn.execute("INSERT INTO proposals(id,tenant_id,connector_id,operation_key,digest,request_digest,payload,proposer_subject,proposer_client,state,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,'pending',?,?)",
                         (proposal_id, action.tenant_id, action.connector_id, action.operation_key,
                          digest(action),request_digest,canonical(action),actor.subject,actor.client_id,
                          action.created_at,action.expires_at))
            self.event(conn,request_id,actor,proposal_id,digest(action),action.policy_version,"proposal","pending")
            return self.view(conn,proposal_id)

    def view(self, conn, proposal_id: str):
        row = conn.execute("SELECT * FROM proposals WHERE id=?", (proposal_id,)).fetchone()
        if row is None:
            raise StoreError("not_found",404)
        proposal = dict(row)
        proposal["payload"] = json.loads(proposal["payload"])
        proposal.pop("request_digest")
        execution = conn.execute("SELECT * FROM executions WHERE proposal_id=?", (proposal_id,)).fetchone()
        proposal["execution"] = None
        if execution:
            execution = dict(execution)
            execution.pop("claim_token")
            execution["result"] = json.loads(execution["result"]) if execution["result"] else None
            proposal["execution"] = execution
        return proposal

    def get(self, proposal_id: str):
        try:
            with self.connection() as conn:
                return self.view(conn,proposal_id)
        except sqlite3.Error as exc:
            raise StoreError("storage_unavailable",503) from exc

    def bound_digest(self, conn, proposal_id: str, action: ReplyAction) -> str:
        """Default action-only contract; specialized ledgers bind exact previews."""
        return digest(action)

    def decide(self, proposal_id: str, shown_digest: str, decision: str, actor: Principal, request_id: str):
        if self.maintenance_only:
            raise StoreError("maintenance_only")
        if actor.role != "reviewer" or self.settings.role(actor.subject,actor.client_id) != "reviewer":
            raise StoreError("reviewer_required",403)
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM proposals WHERE id=?",(proposal_id,)).fetchone()
            if not row:
                raise StoreError("not_found",404)
            if row["digest"] != shown_digest:
                raise StoreError("digest_mismatch")
            action = ReplyAction.model_validate_json(row["payload"])
            if self.bound_digest(conn,proposal_id,action) != shown_digest or action.policy_version != self.settings.policy_version:
                raise StoreError("policy_or_payload_changed")
            now = int(self.clock())
            if now < row["created_at"] or now >= row["expires_at"]:
                raise StoreError("proposal_expired")
            if row["state"] == "approved" and decision == "approve":
                # Returning a stored decision is not issuing another approval.
                return self.view(conn,proposal_id)
            if row["state"] in ("denied","revoked"):
                raise StoreError("decision_terminal")
            execution = conn.execute("SELECT state FROM executions WHERE proposal_id=?",(proposal_id,)).fetchone()
            if execution and execution[0] not in ("queued","blocked"):
                raise StoreError("already_claimed_cannot_revoke")
            if row["state"] == "approved" and decision != "revoke":
                raise StoreError("use_revoke_for_approved_action")
            decision_id = str(uuid.uuid4())
            expiry = min(row["expires_at"],now+self.settings.approval_ttl_seconds) if decision == "approve" else None
            conn.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?)",(decision_id,proposal_id,actor.subject,actor.client_id,shown_digest,decision,now,expiry))
            state = {"approve":"approved","deny":"denied","revoke":"revoked"}[decision]
            conn.execute("UPDATE proposals SET state=?,approval_id=?,reviewer_subject=?,reviewer_client=?,approval_at=?,approval_expires_at=? WHERE id=?",
                         (state,decision_id,actor.subject,actor.client_id,now,expiry,proposal_id))
            if decision == "approve":
                conn.execute("INSERT INTO executions(id,proposal_id,tenant_id,connector_id,operation_key,digest,state) VALUES(?,?,?,?,?,?,'queued')",
                             (str(uuid.uuid4()),proposal_id,action.tenant_id,action.connector_id,action.operation_key,shown_digest))
            elif execution:
                conn.execute("UPDATE executions SET state='revoked',completed_at=?,result=? WHERE proposal_id=?",
                             (now,canonical({"submission":"not_attempted","error_class":"approval_revoked","delivery":"unverified"}),proposal_id))
            self.event(conn,request_id,actor,proposal_id,shown_digest,action.policy_version,"decision",state)
            return self.view(conn,proposal_id)

    def queued(self):
        with self.connection() as conn:
            return [r[0] for r in conn.execute("SELECT proposal_id FROM executions WHERE state='queued' ORDER BY rowid LIMIT 25")]

    def expire_claims(self, request_id: str):
        with self.transaction() as conn:
            now = int(self.clock())
            rows = conn.execute("SELECT e.proposal_id,e.digest,p.payload FROM executions e JOIN proposals p ON p.id=e.proposal_id WHERE e.state='claimed' AND e.claim_deadline<=?",(now,)).fetchall()
            for row in rows:
                action = ReplyAction.model_validate_json(row["payload"])
                conn.execute("UPDATE executions SET state='uncertain',completed_at=?,result=? WHERE proposal_id=? AND state='claimed'",
                             (now,canonical({"submission":"uncertain","sent_copy":"unknown","delivery":"unverified","error_class":"claim_interrupted"}),row["proposal_id"]))
                self.event(conn,request_id,Principal("service-worker","service-worker","worker"),row["proposal_id"],row["digest"],action.policy_version,"uncertainty","uncertain","claim_interrupted")

    def claim(self, proposal_id: str, source_fingerprint: str | None, transport_enabled: bool, request_id: str):
        if self.maintenance_only:
            raise StoreError("maintenance_only")
        with self.transaction() as conn:
            row = conn.execute("SELECT p.*,e.state AS execution_state FROM proposals p JOIN executions e ON p.id=e.proposal_id WHERE p.id=?",(proposal_id,)).fetchone()
            if not row or row["execution_state"] != "queued":
                return None
            action = ReplyAction.model_validate_json(row["payload"])
            now = int(self.clock())
            error = None
            if not transport_enabled:
                error = "delivery_disabled"
            elif source_fingerprint is None:
                error = "source_unavailable"
            elif conn.execute("SELECT value FROM metadata WHERE key='execution_quarantine'").fetchone()[0] != "false":
                error = "restore_quarantine"
            elif row["state"] != "approved" or self.settings.role(row["reviewer_subject"],row["reviewer_client"]) != "reviewer":
                error = "approval_revoked"
            elif self.settings.role(action.proposer_subject,action.proposer_client) != "agent":
                error = "proposer_revoked"
            elif not row["approval_at"] <= now < row["approval_expires_at"] or now >= action.expires_at:
                error = "approval_expired"
            elif action.policy_version != self.settings.policy_version or self.bound_digest(conn,proposal_id,action) != row["digest"] or action.source_fingerprint != source_fingerprint:
                error = "source_policy_or_payload_changed"
            if error:
                state = "expired" if error == "approval_expired" else "blocked" if error in ("delivery_disabled","source_unavailable","restore_quarantine") else "invalidated"
                conn.execute("UPDATE executions SET state=?,completed_at=?,result=? WHERE proposal_id=?",(state,now,canonical({"submission":"not_attempted","sent_copy":"not_attempted","delivery":"unverified","error_class":error}),proposal_id))
                self.event(conn,request_id,Principal("service-worker","service-worker","worker"),proposal_id,row["digest"],action.policy_version,"result",state,error)
                # A preflight block does not consume the approval or claim send permission.
                return None
            token = str(uuid.uuid4())
            conn.execute("UPDATE executions SET state='claimed',claim_token=?,claimed_at=?,claim_deadline=? WHERE proposal_id=? AND state='queued'",(token,now,now+self.settings.claim_timeout_seconds,proposal_id))
            conn.execute("UPDATE proposals SET consumed_at=? WHERE id=? AND consumed_at IS NULL",(now,proposal_id))
            self.event(conn,request_id,Principal("service-worker","service-worker","worker"),proposal_id,row["digest"],action.policy_version,"claim","claimed")
            return action, token

    def finish(self, proposal_id: str, token: str, result: dict, request_id: str):
        with self.transaction() as conn:
            row = conn.execute("SELECT p.digest,p.payload,e.state,e.claim_token,e.claim_deadline FROM proposals p JOIN executions e ON p.id=e.proposal_id WHERE p.id=?",(proposal_id,)).fetchone()
            now = int(self.clock())
            if not row or row["state"] != "claimed" or row["claim_token"] != token or now >= row["claim_deadline"]:
                raise StoreError("claim_fenced")
            action = ReplyAction.model_validate_json(row["payload"])
            conn.execute("UPDATE executions SET state=?,completed_at=?,result=? WHERE proposal_id=? AND state='claimed' AND claim_token=?",(result["submission"],now,canonical(result),proposal_id,token))
            self.event(conn,request_id,Principal("service-worker","service-worker","worker"),proposal_id,row["digest"],action.policy_version,"result",result["submission"],result["error_class"])

    def reconcile(self, proposal_id: str, shown_digest: str, observed: str, evidence: str, actor: Principal, request_id: str):
        if actor.role != "reviewer":
            raise StoreError("reviewer_required",403)
        with self.transaction() as conn:
            view = self.view(conn,proposal_id)
            if view["digest"] != shown_digest or not view["execution"] or view["execution"]["state"] != "uncertain":
                raise StoreError("reconciliation_requires_uncertain")
            # Supplemental observation only; never reopen, retry or erase uncertainty.
            event_id = self.event(conn,request_id,actor,proposal_id,shown_digest,view["payload"]["policy_version"],"reconciliation",observed)
            conn.execute("INSERT INTO reconciliations VALUES(?,?,?,?,?,?,?,?,?)",
                         (str(uuid.uuid4()),proposal_id,shown_digest,actor.subject,actor.client_id,observed,evidence,int(self.clock()),event_id))
            return self.view(conn,proposal_id)

    def reconciliations(self, proposal_id: str):
        with self.connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM reconciliations WHERE proposal_id=? ORDER BY rowid LIMIT 100",(proposal_id,))]

    def audit(self, after: int = 0):
        with self.connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM audit WHERE sequence>? ORDER BY sequence LIMIT 100",(after,))]

    def healthy(self):
        try:
            self.capacity()
            with self.connection() as conn:
                return conn.execute("SELECT value FROM metadata WHERE key='deployment_identity'").fetchone()[0] == self.settings.identity_digest
        except (OSError,sqlite3.Error,StoreError):
            return False
