"""Offline exact-preview integration candidate, never selected by configured_app.

A prepared-wire transport is an explicitly injected trusted test port. The
inspected owner's reply_email/_send are incompatible and are never invoked.
There is no live port, credential or activation configuration in this release.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
import uuid
from typing import Callable, Protocol

from .adapters import DisabledTransport, SourceUnavailable
from .auth import Principal, Verifier
from .mail_owner import OwnerMailSource
from .models import ReplyAction, TransportResult, canonical, digest
from .service import Service
from .spacemail_contract import ContractError, PreparedReply, WirePreview, prepare_reply
from .store import Store, StoreError


def review_digest(action: ReplyAction, preview: WirePreview) -> str:
    return digest({"action_digest": digest(action), "wire_preview_digest": digest(preview)})


class PreparedWirePort(Protocol):
    enabled: bool
    async def submit_prepared(self, prepared: PreparedReply, before_body: Callable[[], None]) -> TransportResult: ...


class MailStore(Store):
    """Separate explicit offline ledger extension; base startup never migrates it."""
    workflow_contract = "mail_exact_wire_v1"
    def __init__(self, settings, clock=time.time, *, source: OwnerMailSource):
        super().__init__(settings, clock)
        self.source = source
        with self.connection() as conn:
            marker = conn.execute("SELECT value FROM metadata WHERE key='mail_review_schema'").fetchone()
            if marker is None or marker[0] != "1":
                raise RuntimeError("explicit new offline mail-review ledger required")
            # Missing table is a fail-closed startup error, never a fallback.
            conn.execute("SELECT proposal_id,preview_digest,preview,wire_base64 FROM mail_previews LIMIT 0")

    @classmethod
    def initialize_offline(cls, settings, clock=time.time, *, source: OwnerMailSource):
        base = Store.initialize(settings, clock)
        with base.transaction() as conn:
            conn.execute("""CREATE TABLE mail_previews (
                proposal_id TEXT PRIMARY KEY REFERENCES proposals(id),
                preview_digest TEXT NOT NULL, preview TEXT NOT NULL, wire_base64 TEXT NOT NULL)""")
            conn.execute("CREATE TRIGGER mail_preview_no_update BEFORE UPDATE ON mail_previews BEGIN SELECT RAISE(ABORT,'immutable_preview'); END")
            conn.execute("CREATE TRIGGER mail_preview_no_delete BEFORE DELETE ON mail_previews BEGIN SELECT RAISE(ABORT,'immutable_preview'); END")
            conn.execute("INSERT INTO metadata VALUES ('mail_review_schema','1')")
        return cls(settings, clock, source=source)

    def prepared(self, conn, proposal_id: str, action: ReplyAction) -> PreparedReply:
        try:
            row = conn.execute("SELECT * FROM mail_previews WHERE proposal_id=?", (proposal_id,)).fetchone()
            if row is None:
                raise ValueError
            preview = WirePreview.model_validate_json(row["preview"])
            wire = base64.b64decode(row["wire_base64"], validate=True)
            if (preview.action_digest != digest(action) or digest(preview) != row["preview_digest"]
                    or hashlib.sha256(wire).hexdigest() != preview.wire_sha256
                    or len(wire) > 131072
                    or any(getattr(preview, key) != getattr(action, key) for key in
                           ("source_id", "source_version", "source_fingerprint", "sender", "to", "subject", "body"))):
                raise ValueError
            return PreparedReply(preview, wire)
        except Exception:
            raise StoreError("wire_preview_invalid") from None

    def bound_digest(self, conn, proposal_id: str, action: ReplyAction) -> str:
        return review_digest(action, self.prepared(conn, proposal_id, action).preview)

    def create(self, action, request_digest, actor, request_id):
        # Provider work is outside all ledger transactions. Re-fetch immediately
        # before preparing: stale source/routing/thread changes create no proposal.
        try:
            fresh = self.source.get(action.source_id)
            prepared = prepare_reply(self.source.binding, action, fresh, now=int(self.clock()))
        except SourceUnavailable:
            raise StoreError("source_unavailable", 503) from None
        except ContractError:
            raise StoreError("source_or_preview_changed") from None
        bound = review_digest(action, prepared.preview)
        with self.transaction() as conn:
            old = conn.execute("SELECT * FROM proposals WHERE tenant_id=? AND connector_id=? AND operation_key=?",
                               (action.tenant_id, action.connector_id, action.operation_key)).fetchone()
            if old:
                if old["request_digest"] != request_digest or (old["proposer_subject"], old["proposer_client"]) != (actor.subject, actor.client_id):
                    raise StoreError("idempotency_conflict")
                return self.view(conn, old["id"])
            proposal_id = str(uuid.uuid4())
            conn.execute("INSERT INTO proposals(id,tenant_id,connector_id,operation_key,digest,request_digest,payload,proposer_subject,proposer_client,state,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,'pending',?,?)",
                         (proposal_id, action.tenant_id, action.connector_id, action.operation_key,
                          bound, request_digest, canonical(action), actor.subject, actor.client_id,
                          action.created_at, action.expires_at))
            conn.execute("INSERT INTO mail_previews VALUES(?,?,?,?)", (proposal_id,
                         prepared.preview_digest, canonical(prepared.preview), base64.b64encode(prepared.wire).decode("ascii")))
            self.event(conn, request_id, actor, proposal_id, bound, action.policy_version, "proposal", "pending")
            return self.view(conn, proposal_id)

    def view(self, conn, proposal_id):
        view = super().view(conn, proposal_id)
        action = ReplyAction.model_validate_json(canonical(view["payload"]))
        prepared = self.prepared(conn, proposal_id, action)
        if view["digest"] != review_digest(action, prepared.preview):
            raise StoreError("wire_preview_invalid")
        view.update(action_digest=digest(action), wire_preview=prepared.preview.model_dump(mode="json"),
                    wire_preview_digest=prepared.preview_digest, review_contract="action_and_exact_wire_v1")
        return view

    def submission_gate(self, proposal_id, token, prepared, request_id, stage):
        if stage not in ("pre_submit", "before_body"):
            raise StoreError("submission_gate_refused")
        with self.transaction() as conn:
            row = conn.execute("SELECT p.*,e.state AS execution_state,e.claim_token,e.claim_deadline,e.digest AS execution_digest FROM proposals p JOIN executions e ON p.id=e.proposal_id WHERE p.id=?", (proposal_id,)).fetchone()
            if row is None:
                raise StoreError("submission_gate_refused")
            action = ReplyAction.model_validate_json(row["payload"])
            stored = self.prepared(conn, proposal_id, action)
            now = int(self.clock())  # After obtaining the transaction lock.
            if (row["execution_state"] != "claimed" or row["claim_token"] != token
                    or now >= row["claim_deadline"] or row["consumed_at"] is None
                    or row["state"] != "approved"
                    or self.settings.role(row["reviewer_subject"], row["reviewer_client"]) != "reviewer"
                    or self.settings.role(action.proposer_subject, action.proposer_client) != "agent"
                    or not row["approval_at"] <= now < row["approval_expires_at"]
                    or not action.created_at <= now < action.expires_at
                    or action.policy_version != self.settings.policy_version
                    or conn.execute("SELECT value FROM metadata WHERE key='execution_quarantine'").fetchone()[0] != "false"
                    or row["digest"] != review_digest(action, stored.preview)
                    or row["execution_digest"] != row["digest"]
                    or prepared.preview != stored.preview or prepared.wire != stored.wire):
                raise StoreError("submission_gate_refused")
            self.event(conn, request_id, Principal("service-worker", "service-worker", "worker"),
                       proposal_id, row["digest"], action.policy_version, "submission_gate", stage)


class MailService(Service):
    """Injected offline service using existing authenticated API/reviewer endpoints."""
    def __init__(self, settings, *, source: OwnerMailSource, prepared_transport=None, clock=time.time):
        for key in ("tenant_id", "connector_id", "account_id", "sender_address", "policy_version"):
            if getattr(source.binding, key) != getattr(settings, key):
                raise RuntimeError("owner interface deployment binding mismatch")
        self.settings, self.clock = settings, clock
        self.verifier = Verifier(settings)
        self.source, self.transport = source, DisabledTransport()
        self.worker_error = None
        self.store = MailStore(settings, clock, source=source)
        self.prepared_transport = prepared_transport

    def read_source(self, source_id):
        result = super().read_source(source_id)
        result["source_kind"] = "owner_mail_interface_offline_candidate"
        return result

    def readiness(self):
        result = super().readiness()
        result["source"] = "owner_mail_interface_offline_candidate"
        result["owner_account_attestation"] = "unverified"
        return result

    async def worker_once(self, stop=None):
        request_id = str(uuid.uuid4())
        self.store.expire_claims(request_id)
        port = self.prepared_transport
        for proposal_id in self.store.queued():
            if stop is not None and stop.is_set():
                break
            current = self.store.get(proposal_id)
            source_fingerprint = None
            try:
                source_fingerprint = digest(self.source.get(current["payload"]["source_id"]))
            except SourceUnavailable:
                pass
            claimed = self.store.claim(proposal_id, source_fingerprint, bool(port is not None and port.enabled), request_id)
            if not claimed:
                continue
            action, token = claimed
            preflight_error = "source_revalidation_failed"
            try:
                fresh = self.source.get(action.source_id)
                prepared = prepare_reply(self.source.binding, action, fresh, now=int(self.clock()))
                preflight_error = "submission_gate_refused"
                self.store.submission_gate(proposal_id, token, prepared, request_id, "pre_submit")
            except Exception:
                # No port invocation occurred. Approval remains consumed and
                # execution terminal: a later source repair cannot permit retry.
                result = TransportResult(submission="rejected", sent_copy="not_attempted", error_class=preflight_error)
                self.store.finish(proposal_id, token, result.model_dump(mode="json"), request_id)
                continue
            gate_state = {"attempts": 0, "successes": 0, "failed": False}
            def before_body(proposal_id=proposal_id, token=token, prepared=prepared, request_id=request_id, gate_state=gate_state):
                gate_state["attempts"] += 1
                if gate_state["attempts"] != 1:
                    gate_state["failed"] = True
                    raise StoreError("submission_gate_refused")
                try:
                    self.store.submission_gate(proposal_id, token, prepared, request_id, "before_body")
                except Exception:
                    gate_state["failed"] = True
                    raise
                gate_state["successes"] += 1
            cancellation = False
            try:
                raw = await asyncio.wait_for(port.submit_prepared(prepared, before_body), timeout=self.settings.adapter_timeout_seconds)
                # Pydantic model instances can be forged via model_construct or
                # model_copy. Revalidate JSON fields without raw warning output.
                if isinstance(raw, TransportResult):
                    raw = raw.model_dump(mode="json", warnings=False)
                result = TransportResult.model_validate_json(canonical(raw))
                # Accepted submission must be preceded by the test port's gate.
                if result.submission == "accepted" and (gate_state["attempts"] != 1 or gate_state["successes"] != 1 or gate_state["failed"]):
                    result = TransportResult(submission="uncertain", sent_copy="unknown", error_class="invalid_adapter_result")
            except asyncio.TimeoutError:
                result = TransportResult(submission="uncertain", sent_copy="unknown", error_class="transport_timeout")
            except asyncio.CancelledError:
                result = TransportResult(submission="uncertain", sent_copy="unknown", error_class="adapter_exception")
                cancellation = True
            except Exception:
                result = TransportResult(submission="uncertain", sent_copy="unknown", error_class="adapter_exception")
            self.store.finish(proposal_id, token, result.model_dump(mode="json"), request_id)
            if cancellation:
                raise asyncio.CancelledError()
