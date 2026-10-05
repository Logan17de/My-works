"""Own offline integration with the inspected Mail 0.1.2 interface.

No supplied implementation is imported or copied. No credential, socket, login,
MCP send tool or live adapter factory exists here. The injected owner is a
trusted private boundary; its account/endpoint attestation remains a live gate.
"""
from __future__ import annotations

import base64
import json
from typing import ContextManager, Protocol

from .adapters import SourceUnavailable
from .models import TransportResult
from .spacemail_contract import (ContractError, ImapReadPlan, PreparedReply,
    ProviderBinding, parse_body_response, parse_size_response, parse_source,
    read_plan, validate_uidvalidity)


class OwnerReadPort(Protocol):
    def mailbox(self, folder: str, readonly: bool = True) -> ContextManager: ...
    def list_senders(self) -> dict: ...


def owner_message_id(plan: ImapReadPlan) -> str:
    """Interoperate with the owner's canonical opaque IDs, with tighter bounds."""
    raw = json.dumps([plan.mailbox, str(plan.uidvalidity), str(plan.uid)],
                     separators=(",", ":")).encode("ascii")
    return "v1_" + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_owner_message_id(message_id: str) -> ImapReadPlan:
    try:
        if type(message_id) is not str or not 4 <= len(message_id) <= 96 or not message_id.startswith("v1_"):
            raise ValueError
        encoded = message_id[3:]
        raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        values = json.loads(raw)
        if type(values) is not list or len(values) != 3 or any(type(v) is not str for v in values) or values[0] != "INBOX":
            raise ValueError
        plan = read_plan(f"imap-v1:{values[1]}:{values[2]}")
        if owner_message_id(plan) != message_id:
            raise ValueError
        return plan
    except Exception:
        raise ContractError("invalid_owner_message_id") from None


class OwnerMailSource:
    """Owner-compatible read-only selected mailbox, bounded raw UID fetches.

    Source IDs at the API are the bounded imap-v1 IDs. Opaque owner IDs are
    decoded locally, never used as paths or an identity/approval authority.
    The injected port's secure account binding is not proven by this class.
    """
    def __init__(self, binding: ProviderBinding, owner: OwnerReadPort):
        self.binding, self.owner = binding, owner

    def ready(self) -> bool:
        return True  # Interface only; never claims provider or live readiness.

    def get_owner(self, message_id: str):
        return self.get(decode_owner_message_id(message_id).source_id)

    def get(self, source_id: str):
        try:
            plan = read_plan(source_id)
            # Alias discovery is advisory, never authorization. The fixed sender
            # still comes exclusively from backend deployment configuration.
            senders = self.owner.list_senders()
            if (type(senders) is not dict or set(senders) != {"default_sender", "allowed_senders"}
                    or type(senders["allowed_senders"]) is not list
                    or not 1 <= len(senders["allowed_senders"]) <= 100
                    or self.binding.sender_address not in senders["allowed_senders"]):
                raise SourceUnavailable
            # mailbox() in the inspected owner has already EXAMINEd this folder.
            # Do not use read_email: it drops raw/version/routing/thread evidence.
            with self.owner.mailbox("INBOX", readonly=True) as conn:
                validate_uidvalidity(plan, ("UIDVALIDITY", conn.untagged_responses.get("UIDVALIDITY")))
                size = parse_size_response(plan, *conn.uid("FETCH", str(plan.uid), plan.size_items))
                raw = parse_body_response(plan, *conn.uid("FETCH", str(plan.uid), plan.body_items), size)
                return parse_source(self.binding, plan, raw)
        except SourceUnavailable:
            raise
        except Exception:
            raise SourceUnavailable() from None


def normalize_owner_result(raw: object, prepared: PreparedReply) -> TransportResult:
    """Conservatively normalize owner-shaped results; never call its send tool.

    Only a matching deterministic wire identity can be credited to this action.
    The current owner's reply_email generates another identity and therefore
    cannot be treated as an approved prepared-wire bridge.
    """
    uncertain = TransportResult(submission="uncertain", sent_copy="unknown", error_class="invalid_adapter_result")
    if type(raw) is not dict or raw.get("from_email") != prepared.preview.sender or raw.get("message_header_id") != prepared.preview.message_id:
        return uncertain
    status, sent = raw.get("status"), raw.get("sent")
    if type(status) is not str:
        return uncertain
    common_keys = {"from_email", "message_header_id", "sent_folder_copy", "sent_folder_copy_status", "sent", "status"}
    optional_keys = {"sent_folder", "sent_folder_copy_detail", "detail"}
    status_keys = {"accepted_by_smtp": {"refused_recipients"}, "partially_accepted": {"refused_recipients"},
                   "rejected_by_smtp": {"smtp_code"}, "all_recipients_refused": set(), "delivery_unknown": set()}
    if not common_keys <= set(raw) or status not in status_keys or set(raw) - (common_keys | optional_keys | status_keys[status]):
        return uncertain
    if sent is True and status == "accepted_by_smtp" and raw.get("refused_recipients") == []:
        copy_status = raw.get("sent_folder_copy_status")
        if type(copy_status) is not str:
            return uncertain
        copy = {"saved": (True, "stored"), "already_present": (True, "stored"),
                "failed": (False, "failed"), "not_attempted": (False, "not_attempted"),
                "unknown": (None, "unknown")}.get(copy_status)
        if copy is None or raw.get("sent_folder_copy") is not copy[0]:
            return uncertain
        return TransportResult(submission="accepted", sent_copy=copy[1])
    if sent is False and raw.get("sent_folder_copy") is False and raw.get("sent_folder_copy_status") == "not_attempted":
        if status == "all_recipients_refused" or (status == "rejected_by_smtp" and type(raw.get("smtp_code")) is int and 400 <= raw["smtp_code"] <= 599):
            return TransportResult(submission="rejected", sent_copy="not_attempted", error_class="provider_rejected")
    if sent is None and status == "delivery_unknown":
        return TransportResult(submission="uncertain", sent_copy="unknown", error_class="provider_uncertain")
    # Partial acceptance is outside the single-destination workflow. Never retry.
    return uncertain
