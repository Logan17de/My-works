"""Dormant SpaceMail provider primitives. No sockets, login, credentials or sends.

The owning connector must supply a reviewed, bounded protocol bridge before these
primitives can be used live. Production configuration does not import this module.
"""
from __future__ import annotations

import base64
import hashlib
import quopri
import re
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import format_datetime
from typing import Annotated, Literal

from pydantic import Field, field_validator

from .adapters import SourceNotFound, SourceUnavailable
from .models import (Address, Digest, Identifier, ReplyAction, SourceMessage,
                     StrictModel, TransportResult, address, canonical, digest, plain_header)

MAX_SOURCE_BYTES = 65536
MAX_HEADER_BYTES = 16384
MAX_WIRE_BYTES = 131072
MAX_UID = 4294967295
MAX_REFERENCES = 20
_MSG_ID = re.compile(r"<[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?>")
_SOURCE_ID = re.compile(r"imap-v1:([1-9][0-9]{0,9}):([1-9][0-9]{0,9})")
_ENCODED_WORD = re.compile(r"=\?([^?\s]+)\?([bBqQ])\?([^?\s]+)\?=")


class ContractError(ValueError):
    """Sanitized classification only; provider responses and mail are never logged."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class SpaceMailEndpoints(StrictModel):
    host: Literal["mail.spacemail.com"] = "mail.spacemail.com"
    imap_port: Literal[993] = 993
    smtp_port: Literal[465] = 465
    security: Literal["implicit_tls_verified"] = "implicit_tls_verified"
    inbox: Literal["INBOX"] = "INBOX"
    operation_timeout_seconds: Annotated[int, Field(ge=1, le=30)] = 10
    source_byte_limit: Literal[65536] = MAX_SOURCE_BYTES


def verified_tls_context() -> ssl.SSLContext:
    """Build a verified context only; this function never opens a connection."""
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise ContractError("tls_verification_required")
    return context


@dataclass(frozen=True)
class ImapReadPlan:
    uidvalidity: int
    uid: int
    mailbox: str = "INBOX"
    readonly: bool = True
    size_items: str = "(UID RFC822.SIZE)"
    body_items: str = "(UID BODY.PEEK[]<0.65537>)"

    def __post_init__(self):
        if any(type(value) is not int or not 1 <= value <= MAX_UID for value in (self.uidvalidity, self.uid)):
            raise ContractError("invalid_uid_locator")
        if type(self.readonly) is not bool or (self.mailbox, self.readonly, self.size_items, self.body_items) != (
                "INBOX", True, "(UID RFC822.SIZE)", "(UID BODY.PEEK[]<0.65537>)"):
            raise ContractError("read_only_inbox_required")

    @property
    def source_id(self) -> str:
        return f"imap-v1:{self.uidvalidity}:{self.uid}"


def read_plan(source_id: str) -> ImapReadPlan:
    if type(source_id) is not str or not (match := _SOURCE_ID.fullmatch(source_id)):
        raise ContractError("invalid_uid_locator")
    return ImapReadPlan(int(match[1]), int(match[2]))


def _response_parts(data: object) -> list:
    if type(data) is not list or not 1 <= len(data) <= 3:
        raise SourceUnavailable() from None
    return data


def validate_uidvalidity(plan: ImapReadPlan, response: object) -> None:
    # imaplib.response('UIDVALIDITY') -> ('UIDVALIDITY', [b'123']).
    if (type(response) is not tuple or len(response) != 2 or response[0] != "UIDVALIDITY"
            or response[1] != [str(plan.uidvalidity).encode("ascii")]):
        raise SourceUnavailable() from None


def parse_size_response(plan: ImapReadPlan, status: str, data: object) -> int:
    if status != "OK":
        raise SourceUnavailable() from None
    parts = _response_parts(data)
    if parts == [None]:
        raise SourceNotFound() from None
    if len(parts) != 1 or type(parts[0]) is not bytes or len(parts[0]) > 256:
        raise SourceUnavailable() from None
    match = re.fullmatch(rb"[1-9][0-9]* \((?:UID ([1-9][0-9]*) RFC822.SIZE ([0-9]+)|RFC822.SIZE ([0-9]+) UID ([1-9][0-9]*))\)", parts[0])
    if not match:
        raise SourceUnavailable() from None
    uid, size = (int(match[1]), int(match[2])) if match[1] else (int(match[4]), int(match[3]))
    if uid != plan.uid or not 1 <= size <= MAX_SOURCE_BYTES:
        raise SourceUnavailable() from None
    return size


def parse_body_response(plan: ImapReadPlan, status: str, data: object, expected_size: int) -> bytes:
    """Refuse wrong UID, multiple literals, partial/malformed/oversized records."""
    if status != "OK" or type(expected_size) is not int or not 1 <= expected_size <= MAX_SOURCE_BYTES:
        raise SourceUnavailable() from None
    parts = _response_parts(data)
    if parts == [None]:
        raise SourceNotFound() from None
    if len(parts) != 2 or type(parts[0]) is not tuple or len(parts[0]) != 2 or type(parts[1]) is not bytes:
        raise SourceUnavailable() from None
    prefix, raw = parts[0]
    suffix = parts[1]
    if type(prefix) is not bytes or type(raw) is not bytes or len(prefix) + len(suffix) > 1024:
        raise SourceUnavailable() from None
    # Exact framing permits one UID either before or after one body literal.
    # Counting then stripping tokens can disagree on tabs and hide a wrong UID.
    before = re.fullmatch(rb"[1-9][0-9]* \(UID ([1-9][0-9]*) BODY\[\](?:<0>)? \{([0-9]+)\}", prefix)
    after = re.fullmatch(rb"[1-9][0-9]* \(BODY\[\](?:<0>)? \{([0-9]+)\}", prefix)
    tail = re.fullmatch(rb" UID ([1-9][0-9]*)\)", suffix)
    if before is not None and suffix == b")":
        uid, literal_size = int(before[1]), int(before[2])
    elif after is not None and tail is not None:
        uid, literal_size = int(tail[1]), int(after[1])
    else:
        raise SourceUnavailable() from None
    if uid != plan.uid or len(raw) != expected_size or literal_size != expected_size:
        raise SourceUnavailable() from None
    return raw


class ProviderBinding(StrictModel):
    tenant_id: Identifier
    connector_id: Identifier
    account_id: Identifier
    sender_address: Address
    policy_version: Identifier

    _sender = field_validator("sender_address")(address)


def _message_id(value: str) -> str:
    if type(value) is not str or len(value) > 254 or not _MSG_ID.fullmatch(value) or ".." in value:
        raise ContractError("unsupported_thread_header")
    local = value[1:].split("@", 1)[0]
    if local.startswith(".") or local.endswith("."):
        raise ContractError("unsupported_thread_header")
    return value


class SpaceMailRecord(SourceMessage):
    provider: Literal["spacemail"] = "spacemail"
    mailbox: Literal["INBOX"] = "INBOX"
    uidvalidity: Annotated[int, Field(ge=1, le=MAX_UID)]
    uid: Annotated[int, Field(ge=1, le=MAX_UID)]
    rfc_message_id: Annotated[str, Field(max_length=254)]
    rfc_in_reply_to: Annotated[str, Field(max_length=254)] | None = None
    rfc_references: Annotated[tuple[str, ...], Field(max_length=MAX_REFERENCES)] = ()

    _thread = field_validator("rfc_message_id")(_message_id)
    _parent = field_validator("rfc_in_reply_to")(lambda value: _message_id(value) if value is not None else value)

    @field_validator("rfc_references")
    @classmethod
    def valid_references(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _message_id(value)
        if len(set(values)) != len(values):
            raise ContractError("ambiguous_thread_header")
        return values


def _one_header(message: EmailMessage, name: str, *, required=False):
    values = message.get_all(name, [])
    if len(values) > 1 or (required and not values):
        raise ContractError("ambiguous_source_header")
    if not values:
        return None
    value = values[0]
    if getattr(value, "defects", ()):
        raise ContractError("malformed_source_header")
    plain_header(str(value))
    return value


def _validate_raw_encoded_words(message: EmailMessage) -> None:
    # Python's UnstructuredHeader can silently repair encoded-word defects.
    # Validate raw words independently; never depend on its private parse tree.
    for name, value in message.raw_items():
        lower = name.lower()
        if lower in ("message-id", "references", "in-reply-to") and "=?" in value:
            raise ContractError("encoded_thread_header_forbidden")
        if lower not in ("subject", "from", "reply-to", "to", "cc"):
            continue
        position = 0
        while (start := value.find("=?", position)) != -1:
            word = _ENCODED_WORD.match(value, start)
            if word is None or len(word[0]) > 75:
                raise ContractError("malformed_source_encoded_word")
            charset, encoding, encoded = word.groups()
            supported = {"utf-8":"utf-8", "utf8":"utf-8", "us-ascii":"ascii", "ascii":"ascii"}
            if charset.lower() not in supported:
                raise ContractError("unsupported_source_header_charset")
            if encoding.lower() == "b":
                decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
                if base64.b64encode(decoded).decode("ascii") != encoded:
                    raise ContractError("noncanonical_source_encoded_word")
            else:
                if re.search(r"=(?![0-9A-Fa-f]{2})", encoded):
                    raise ContractError("malformed_source_encoded_word")
                decoded = quopri.decodestring(encoded.replace("_", " ").encode("ascii"))
            plain_header(decoded.decode(supported[charset.lower()], errors="strict"))
            position = word.end()


def _one_mailbox(header) -> str:
    addresses = getattr(header, "addresses", ())
    groups = getattr(header, "groups", ())
    if len(addresses) != 1 or any(group.display_name is not None for group in groups):
        raise ContractError("one_source_mailbox_required")
    try:
        return address(addresses[0].addr_spec)
    except ValueError:
        raise ContractError("one_source_mailbox_required") from None


def parse_source(binding: ProviderBinding, plan: ImapReadPlan, raw: bytes) -> SpaceMailRecord:
    """Parse one bounded plain-text message; full original bytes bind the version."""
    try:
        if type(raw) is not bytes or not 1 <= len(raw) <= MAX_SOURCE_BYTES or b"\x00" in raw:
            raise ContractError("source_bounds")
        if re.search(rb"(?<!\r)\n|\r(?!\n)", raw):
            raise ContractError("unsupported_source_line_endings")
        header_block, separator, _ = raw.partition(b"\r\n\r\n")
        if not separator or len(header_block) > MAX_HEADER_BYTES or not header_block.isascii():
            raise ContractError("source_header_bounds")
        message = BytesParser(policy=policy.default.clone(raise_on_defect=True)).parsebytes(raw)
        _validate_raw_encoded_words(message)
        if message.defects or message.is_multipart() or message.get_content_type() != "text/plain":
            raise ContractError("unsupported_source_mime")
        for name in ("From", "Reply-To", "Subject", "Message-ID", "References", "In-Reply-To", "To", "Cc",
                     "Content-Type", "Content-Transfer-Encoding", "Content-Disposition", "MIME-Version"):
            _one_header(message, name)
        if message.get("Content-Disposition") is not None or message.get_filename() is not None:
            raise ContractError("unsupported_source_attachment")
        if message.get("MIME-Version") is not None and str(message["MIME-Version"]) != "1.0":
            raise ContractError("unsupported_source_mime_version")
        if any(name.lower().startswith("resent-") for name in message.keys()) or message.get("Sender") is not None:
            raise ContractError("unsupported_source_routing")
        sender = _one_mailbox(_one_header(message, "From", required=True))
        reply_header = _one_header(message, "Reply-To")
        reply_to = _one_mailbox(reply_header) if reply_header is not None else None
        subject = str(_one_header(message, "Subject", required=True))
        msg_id = _message_id(str(_one_header(message, "Message-ID", required=True)))
        refs_header = _one_header(message, "References")
        refs = tuple(str(refs_header).split()) if refs_header is not None else ()
        if len(refs) > MAX_REFERENCES - 1 or (refs_header is not None and not refs):
            raise ContractError("thread_bounds")
        for reference in refs:
            _message_id(reference)
        if len(set(refs)) != len(refs) or msg_id in refs:
            raise ContractError("ambiguous_thread_header")
        in_reply_to = _one_header(message, "In-Reply-To")
        if in_reply_to is not None:
            _message_id(str(in_reply_to))
            if str(in_reply_to) == msg_id:
                raise ContractError("ambiguous_thread_header")
        charset = (message.get_content_charset() or "us-ascii").lower()
        if charset not in ("utf-8", "us-ascii"):
            raise ContractError("unsupported_source_charset")
        transfer = str(message.get("Content-Transfer-Encoding", "7bit")).lower()
        payload = message.get_payload(decode=False)
        if type(payload) is not str:
            raise ContractError("unsupported_source_mime")
        if transfer == "base64":
            body_bytes = base64.b64decode(re.sub(rb"[ \t\r\n]", b"", payload.encode("ascii")), validate=True)
        elif transfer == "quoted-printable":
            encoded = payload.encode("ascii")
            if re.search(rb"=(?![0-9A-Fa-f]{2}|\r\n)", encoded):
                raise ContractError("malformed_source_encoding")
            body_bytes = quopri.decodestring(encoded)
        elif transfer in ("7bit", "8bit"):
            body_bytes = message.get_payload(decode=True)
            if type(body_bytes) is not bytes or (transfer == "7bit" and not body_bytes.isascii()):
                raise ContractError("malformed_source_encoding")
        else:
            raise ContractError("unsupported_source_encoding")
        body = body_bytes.decode(charset, errors="strict")
        if any(ord(char) < 32 and char not in "\r\n\t" for char in body):
            raise ContractError("source_body_controls")
        return SpaceMailRecord(source_id=plan.source_id, version="rfc822:"+hashlib.sha256(raw).hexdigest(),
            tenant_id=binding.tenant_id, connector_id=binding.connector_id, account_id=binding.account_id,
            sender=sender, reply_to=reply_to, subject=subject, body=body, uidvalidity=plan.uidvalidity,
            uid=plan.uid, rfc_message_id=msg_id, rfc_in_reply_to=str(in_reply_to) if in_reply_to is not None else None,
            rfc_references=refs)
    except Exception:
        # Discard raw parser/provider exception text, which may include message content.
        raise SourceUnavailable() from None


class WirePreview(StrictModel):
    action_digest: Digest
    source_fingerprint: Digest
    source_id: Identifier
    source_version: Identifier
    sender: Address
    to: tuple[Address, ...]
    subject: str
    body: str
    message_id: str
    in_reply_to: str
    references: tuple[str, ...]
    date: str
    wire_sha256: Digest


@dataclass(frozen=True)
class PreparedReply:
    preview: WirePreview
    wire: bytes

    @property
    def preview_digest(self) -> str:
        return digest(self.preview)


def prepare_reply(binding: ProviderBinding, action: ReplyAction, fresh: SpaceMailRecord, *, now: int) -> PreparedReply:
    """Revalidate exact action/source before producing any SMTP envelope or bytes.

    A live integration must approve this visible preview, atomically reserve its
    digest, and recheck expiry/policy/revocation before DATA. This does not approve,
    claim, consume approval, send, retry or open a connection.
    """
    try:
        # Revalidation rejects Pydantic model_copy/model_construct bypasses.
        if (type(action.created_at) is not int or type(action.expires_at) is not int
                or type(fresh.uid) is not int or type(fresh.uidvalidity) is not int
                or len(fresh.rfc_references) >= MAX_REFERENCES):
            raise ContractError("invalid_action_or_thread_bounds")
        action = ReplyAction.model_validate_json(canonical(action))
        fresh = SpaceMailRecord.model_validate_json(canonical(fresh))
        plan = read_plan(fresh.source_id)
        if (fresh.uidvalidity, fresh.uid) != (plan.uidvalidity, plan.uid):
            raise ContractError("source_locator_mismatch")
        for key in ("tenant_id", "connector_id", "account_id"):
            if getattr(action, key) != getattr(binding, key) or getattr(fresh, key) != getattr(binding, key):
                raise ContractError("deployment_identity_mismatch")
        if (action.source_id, action.source_version, action.source_fingerprint) != (fresh.source_id, fresh.version, digest(fresh)):
            raise ContractError("stale_source")
        subject = fresh.subject if fresh.subject.lower().startswith("re:") else "Re: "+fresh.subject
        recipient = fresh.reply_to or fresh.sender
        if (action.sender != binding.sender_address or action.policy_version != binding.policy_version
                or action.to != (recipient,) or action.cc or action.subject != subject
                or recipient == binding.sender_address):
            raise ContractError("reply_preview_drift")
        if (type(now) is not int or not action.created_at <= now < action.expires_at
                or len(action.operation_key) < 16 or len(action.operation_key) > 96
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", action.operation_key)
                or any(ord(char) < 32 and char not in "\r\n\t" for char in action.body)):
            raise ContractError("invalid_or_expired_action")
        action_digest = digest(action)
        message_id = f"<zetbros.{action_digest}@{binding.sender_address.split('@')[1]}>"
        # RFC 5322 3.6.4: a single parent In-Reply-To is the fallback when the
        # parent has no References. It is part of the immutable source binding.
        parents = fresh.rfc_references or ((fresh.rfc_in_reply_to,) if fresh.rfc_in_reply_to is not None else ())
        refs = (*parents, fresh.rfc_message_id)
        date = format_datetime(datetime.fromtimestamp(action.created_at, timezone.utc))
        message = EmailMessage(policy=policy.SMTP.clone(max_line_length=78))
        for header, value in (("From", action.sender), ("To", recipient), ("Subject", action.subject),
                              ("Message-ID", message_id), ("In-Reply-To", fresh.rfc_message_id),
                              ("References", " ".join(refs)), ("Date", date), ("MIME-Version", "1.0"),
                              ("Content-Type", 'text/plain; charset="utf-8"'), ("Content-Transfer-Encoding", "base64")):
            message[header] = value
        # Preserve the exact approved text, including newline choices and final newline.
        message.set_payload(base64.encodebytes(action.body.encode("utf-8")).decode("ascii"))
        wire = message.as_bytes()
        if len(wire) > MAX_WIRE_BYTES or any(len(line) > 998 for line in wire.split(b"\r\n")):
            raise ContractError("wire_bounds")
        # Header assignment/folding can interpret an encoded-word-looking
        # literal again. Refuse any wire that changes the exact visible preview.
        rendered = BytesParser(policy=policy.default.clone(raise_on_defect=True)).parsebytes(wire)
        expected = {"From":action.sender, "To":recipient, "Subject":action.subject, "Message-ID":message_id,
                    "In-Reply-To":fresh.rfc_message_id, "Date":date}
        if (any(str(rendered[name]) != value for name, value in expected.items())
                or " ".join(str(rendered["References"]).split()) != " ".join(refs)
                or rendered.is_multipart() or rendered.get_payload(decode=True) != action.body.encode("utf-8")):
            raise ContractError("wire_preview_round_trip_mismatch")
        preview = WirePreview(action_digest=action_digest, source_fingerprint=digest(fresh), source_id=fresh.source_id,
            source_version=fresh.version, sender=action.sender, to=action.to, subject=action.subject, body=action.body,
            message_id=message_id, in_reply_to=fresh.rfc_message_id, references=refs, date=date,
            wire_sha256=hashlib.sha256(wire).hexdigest())
        return PreparedReply(preview, wire)
    except Exception:
        raise ContractError("reply_preparation_refused") from None


class SmtpOutcome:
    """One SMTP transaction's conservative result reducer, with no I/O or retry.

    Events describe MAIL, RCPT, DATA-command, body-start, final DATA acknowledgment.
    Only body-start permits possible submission. Only final 250 confirms accepted.
    Duplicate, malformed and out-of-order events cannot establish acceptance.
    """
    def __init__(self):
        self.phase = "mail"
        self._result: TransportResult | None = None

    def _close(self, submission, error_class="none") -> TransportResult:
        if self._result is None:
            self._result = TransportResult(submission=submission, sent_copy="not_attempted", error_class=error_class)
            self.phase = "closed"
        return self._result

    def response(self, stage: Literal["mail", "rcpt", "data", "final"], code: int) -> TransportResult | None:
        if self._result is not None:
            return self._result
        if type(stage) is not str or stage not in ("mail", "rcpt", "data", "final") or stage != self.phase or type(code) is not int or not 100 <= code <= 599:
            return self._close("uncertain", "provider_uncertain")
        accepted = {"mail": (250,), "rcpt": (250, 251), "data": (354,), "final": (250,)}
        if code in accepted[stage]:
            if stage == "final":
                return self._close("accepted")
            self.phase = {"mail": "rcpt", "rcpt": "data", "data": "body"}[stage]
            return None
        if 400 <= code <= 599:
            return self._close("rejected", "provider_rejected")
        return self._close("uncertain", "provider_uncertain")

    def begin_body(self) -> TransportResult | None:
        if self._result is not None:
            return self._result
        if self.phase != "body":
            return self._close("uncertain", "provider_uncertain")
        self.phase = "final"
        return None

    def interrupted(self, error_class: Literal["adapter_exception", "transport_timeout"] = "adapter_exception") -> TransportResult:
        if error_class not in ("adapter_exception", "transport_timeout"):
            raise ContractError("invalid_interruption_class")
        return self._close("uncertain" if self.phase == "final" else "rejected", error_class)

    def sent_copy(self, outcome: Literal["stored", "failed", "unknown"]) -> TransportResult:
        if outcome not in ("stored", "failed", "unknown") or self._result is None or self._result.submission != "accepted":
            raise ContractError("sent_copy_without_acceptance")
        if self._result.sent_copy != "not_attempted":
            raise ContractError("sent_copy_already_recorded")
        self._result = self._result.model_copy(update={"sent_copy": outcome})
        return self._result

    def result(self) -> TransportResult:
        if self._result is None:
            raise ContractError("smtp_result_not_final")
        return self._result
