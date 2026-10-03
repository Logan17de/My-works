"""Strict, bounded wire and storage contracts. Mail contents never grant authority."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Identifier = Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
PrincipalId = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[^\x00-\x20\x7f]+$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Address = Annotated[str, Field(min_length=3, max_length=254)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def canonical(value: BaseModel | dict) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: BaseModel | dict) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def address(value: str) -> str:
    # One plain ASCII addr-spec only; no display names, groups or lists.
    if not value.isascii() or not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", value):
        raise ValueError("one plain ASCII mailbox address required")
    if ".." in value or len(value.split("@")[0]) > 64:
        raise ValueError("invalid mailbox address")
    return value


def plain_header(value: str) -> str:
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("control characters are forbidden in headers")
    return value


class SourceMessage(StrictModel):
    source_id: Identifier
    version: Identifier
    tenant_id: Identifier
    connector_id: Identifier
    account_id: Identifier
    sender: Address
    reply_to: Address | None = None
    subject: Annotated[str, Field(max_length=300)]
    body: Annotated[str, Field(max_length=16000)]

    _address = field_validator("sender", "reply_to")(lambda v: address(v) if v is not None else v)
    _header = field_validator("subject")(plain_header)


class ProposalInput(StrictModel):
    operation_key: Annotated[str, Field(min_length=16, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
    source_id: Identifier
    source_version: Identifier
    source_fingerprint: Digest
    body: Annotated[str, Field(min_length=1, max_length=16000)]

    @field_validator("body")
    @classmethod
    def text_body(cls, value: str) -> str:
        if "\x00" in value or any(ord(c) < 32 and c not in "\r\n\t" for c in value):
            raise ValueError("invalid text body")
        return value


class DecisionInput(StrictModel):
    digest: Digest
    decision: Literal["approve", "deny", "revoke"]


class ReconciliationInput(StrictModel):
    digest: Digest
    observed_submission: Literal["accepted", "rejected", "uncertain"]
    evidence_reference: Identifier


class ReplyAction(StrictModel):
    tenant_id: Identifier
    connector_id: Identifier
    account_id: Identifier
    operation: Literal["reply"] = "reply"
    operation_key: str
    source_id: Identifier
    source_version: Identifier
    source_fingerprint: Digest
    sender: Address
    to: tuple[Address, ...]
    cc: tuple[Address, ...] = ()
    subject: Annotated[str, Field(max_length=304)]
    body: Annotated[str, Field(min_length=1, max_length=16000)]
    policy_version: Identifier
    proposer_subject: PrincipalId
    proposer_client: PrincipalId
    created_at: int
    expires_at: int

    _header = field_validator("subject")(plain_header)
    _sender = field_validator("sender")(address)

    @field_validator("to")
    @classmethod
    def one_destination(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 1:
            raise ValueError("exactly one reply destination required")
        address(value[0])
        return value

    @field_validator("cc")
    @classmethod
    def no_cc(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value:
            raise ValueError("CC is outside this workflow")
        return value


class TransportResult(StrictModel):
    submission: Literal["accepted", "rejected", "uncertain"]
    sent_copy: Literal["stored", "failed", "not_attempted", "unknown"]
    error_class: Literal["none", "provider_rejected", "provider_uncertain", "adapter_exception", "transport_timeout", "invalid_adapter_result"] = "none"
    delivery: Literal["unverified"] = "unverified"
