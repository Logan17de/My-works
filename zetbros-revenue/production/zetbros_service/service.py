"""One bounded workflow: read source, propose reply, human decision, durable worker."""
from __future__ import annotations

import asyncio
import time
import uuid

from .adapters import DisabledTransport, SnapshotSource, SourceNotFound, SourceUnavailable
from .auth import Principal, Verifier
from .config import Settings
from .models import ProposalInput, ReplyAction, TransportResult, canonical, digest
from .store import Store, StoreError


class Service:
    def __init__(self, settings: Settings, *, source=None, transport=None, clock=time.time):
        self.settings, self.clock = settings, clock
        self.verifier = Verifier(settings)
        self.store = Store(settings,clock)
        self.source = source if source is not None else SnapshotSource(settings)
        self.transport = transport if transport is not None else DisabledTransport()
        self.worker_error = None

    def read_source(self, source_id: str):
        try:
            record = self.source.get(source_id)
            return {"source": record.model_dump(mode="json"), "source_fingerprint":digest(record),
                    "source_kind":"customer_staged_snapshot", "live_source_verified":False}
        except SourceNotFound as exc:
            raise StoreError("source_not_found",404) from exc
        except SourceUnavailable as exc:
            raise StoreError("source_unavailable",503) from exc

    def propose(self, request: ProposalInput, actor: Principal, request_id: str):
        if actor.role != "agent" or self.settings.role(actor.subject,actor.client_id) != "agent":
            raise StoreError("agent_required",403)
        source = self.read_source(request.source_id)
        record = source["source"]
        if request.source_version != record["version"] or request.source_fingerprint != source["source_fingerprint"]:
            raise StoreError("stale_source")
        recipient = record["reply_to"] or record["sender"]
        if recipient == self.settings.sender_address:
            raise StoreError("self_reply_outside_workflow")
        now = int(self.clock())
        action = ReplyAction(tenant_id=self.settings.tenant_id,connector_id=self.settings.connector_id,
                             account_id=self.settings.account_id,operation_key=request.operation_key,
                             source_id=request.source_id,source_version=request.source_version,
                             source_fingerprint=request.source_fingerprint,sender=self.settings.sender_address,
                             to=(recipient,),subject=record["subject"] if record["subject"].lower().startswith("re:") else "Re: "+record["subject"],
                             body=request.body,policy_version=self.settings.policy_version,
                             proposer_subject=actor.subject,proposer_client=actor.client_id,
                             created_at=now,expires_at=now+self.settings.proposal_ttl_seconds)
        return self.store.create(action,digest(request),actor,request_id)

    def view(self, proposal_id: str, actor: Principal):
        result = self.store.get(proposal_id)
        if actor.role == "agent" and (result["proposer_subject"],result["proposer_client"]) != (actor.subject,actor.client_id):
            raise StoreError("not_found",404)
        return result

    async def worker_once(self, stop: asyncio.Event | None = None):
        request_id = str(uuid.uuid4())
        self.store.expire_claims(request_id)
        for proposal_id in self.store.queued():
            if stop is not None and stop.is_set():
                break
            current = self.store.get(proposal_id)
            source_fingerprint = None
            try:
                source_fingerprint = digest(self.source.get(current["payload"]["source_id"]))
            except SourceUnavailable:
                pass
            claimed = self.store.claim(proposal_id,source_fingerprint,self.transport.enabled,request_id)
            if not claimed:
                continue
            action, token = claimed
            cancellation = False
            try:
                raw = await asyncio.wait_for(self.transport.submit(action),timeout=self.settings.adapter_timeout_seconds)
                if isinstance(raw, TransportResult):
                    raw = raw.model_dump(mode="json", warnings=False)
                result = TransportResult.model_validate_json(canonical(raw)).model_dump(mode="json")
            except asyncio.TimeoutError:
                result = TransportResult(submission="uncertain",sent_copy="unknown",error_class="transport_timeout").model_dump(mode="json")
            except asyncio.CancelledError:
                result = TransportResult(submission="uncertain",sent_copy="unknown",error_class="adapter_exception").model_dump(mode="json")
                cancellation = True
            except Exception:
                # An invocation may already have submitted. Never infer rejection or retry.
                result = TransportResult(submission="uncertain",sent_copy="unknown",error_class="adapter_exception").model_dump(mode="json")
            self.store.finish(proposal_id,token,result,request_id)
            if cancellation:
                raise asyncio.CancelledError()

    async def worker_loop(self, stop: asyncio.Event):
        while not stop.is_set():
            try:
                await self.worker_once(stop)
                self.worker_error = None
            except StoreError as exc:
                self.worker_error = exc.code
            except Exception:
                self.worker_error = "worker_unavailable"
            try:
                await asyncio.wait_for(stop.wait(),timeout=1)
            except asyncio.TimeoutError:
                pass

    def readiness(self):
        control_ready = self.store.healthy() and self.source.ready() and self.worker_error is None
        return {"control_ready":control_ready,"live_delivery_ready":False,
                "delivery":"disabled","source":"customer_staged_snapshot",
                "live_source_verified":False,"deployment_profile":self.settings.deployment_profile,
                "error_class":self.worker_error or "none"}
