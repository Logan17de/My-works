"""Own dormant private bridge with injected stdlib-shaped sessions only.

No network constructor, credential lookup, login, deployed factory or config
activation is supplied. Fake sessions exercise the actual phase orchestration.
A trusted private session factory must establish account/TLS/deadline guarantees
before any live use. Session metadata alone is not evidence of those guarantees.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from typing import Annotated, Callable, ContextManager, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from .adapters import SourceUnavailable
from .models import Address, ReplyAction, StrictModel, TransportResult, address, canonical, digest
from .spacemail_contract import (ContractError, MAX_WIRE_BYTES, PreparedReply,
    ImapReadPlan, ProviderBinding, SmtpOutcome, SpaceMailEndpoints, WirePreview,
    parse_body_response, parse_size_response)


class PrivateBridgeProfile(StrictModel):
    binding: ProviderBinding
    mailbox_address: Address
    endpoints: SpaceMailEndpoints = SpaceMailEndpoints()
    sent_folder: Literal["Sent"] = "Sent"
    approved_aliases: tuple[Address, ...] = ()
    total_timeout_seconds: Annotated[int, Field(ge=1, le=15)] = 10
    phase_timeout_seconds: Annotated[int, Field(ge=1, le=5)] = 3
    deployment_profile: Literal["single_customer_single_host"] = "single_customer_single_host"
    outbound: Literal["disabled"] = "disabled"

    _mailbox = field_validator("mailbox_address")(address)

    @field_validator("approved_aliases")
    @classmethod
    def aliases(cls, values):
        if len(values) > 20 or len(set(values)) != len(values):
            raise ValueError("bounded distinct aliases required")
        for value in values: address(value)
        return values

    @model_validator(mode="after")
    def configured_sender(self):
        if self.binding.sender_address not in (self.mailbox_address, *self.approved_aliases):
            raise ValueError("sender must be the bound mailbox or an approved alias")
        if self.phase_timeout_seconds > self.total_timeout_seconds:
            raise ValueError("phase timeout must not exceed total budget")
        return self


class SessionIdentity(StrictModel):
    binding: ProviderBinding
    mailbox_address: Address
    host: Literal["mail.spacemail.com"] = "mail.spacemail.com"
    port: Literal[993, 465]
    purpose: Literal["imap", "smtp"]
    security: Literal["implicit_tls_verified"] = "implicit_tls_verified"

    _mailbox = field_validator("mailbox_address")(address)

    @model_validator(mode="after")
    def port_matches(self):
        if self.port != {"imap": 993, "smtp": 465}[self.purpose]:
            raise ValueError("session purpose/port mismatch")
        return self


@dataclass(frozen=True)
class BoundSession:
    identity: SessionIdentity
    session: SmtpSessionPort | ImapSessionPort


class SocketPort(Protocol):
    def settimeout(self, value: float) -> None: ...
    def shutdown(self, how: int) -> None: ...
    def close(self) -> None: ...


class SmtpSessionPort(Protocol):
    sock: SocketPort
    def mail(self, sender: str) -> tuple[int, bytes]: ...
    def rcpt(self, recipient: str) -> tuple[int, bytes]: ...
    def docmd(self, command: str) -> tuple[int, bytes]: ...
    def send(self, wire: bytes) -> None: ...
    def getreply(self) -> tuple[int, bytes]: ...


class ImapSessionPort(Protocol):
    sock: SocketPort
    untagged_responses: dict
    def uid(self, command: str, *args) -> tuple[str, list]: ...
    def append(self, folder: str, flags: str, date: None, wire: bytes) -> tuple[str, list]: ...


class PrivateSessionFactory(Protocol):
    # True is reserved for fake offline factories. No production enable switch.
    offline_only: bool
    def smtp(self, profile: PrivateBridgeProfile, *, deadline: float, timeout: float) -> ContextManager[BoundSession]: ...
    def imap(self, profile: PrivateBridgeProfile, *, folder: str, readonly: bool,
             deadline: float, timeout: float) -> ContextManager[BoundSession]: ...


class Budget:
    def __init__(self, profile, *, monotonic=time.monotonic, cancelled=None):
        self.profile, self.monotonic = profile, monotonic
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.deadline = monotonic() + profile.total_timeout_seconds
        self._lock = threading.Lock()
        self._session = None
        self.accepted = None
        self.copy_started = False
        self.timer = threading.Timer(profile.total_timeout_seconds, self.abort)
        self.timer.daemon = True
        self.timer.start()

    def attach(self, session):
        with self._lock:
            self._session = session
        try:
            self.remaining()
        except Exception:
            self.abort()
            raise

    def abort(self):
        self.cancelled.set()
        with self._lock:
            session = self._session
        if session is not None:
            # The private factory owns one isolated connection. A watchdog
            # actively interrupts in-flight reads/writes, including multiline
            # trickling responses. The fake socket implements these methods.
            for method, args in (("shutdown", (2,)), ("close", ())):
                try: getattr(session.sock, method)(*args)
                except Exception: pass

    def close(self):
        self.timer.cancel()

    def remember_acceptance(self, result):
        with self._lock:
            self.accepted = result

    def accepted_on_interruption(self):
        with self._lock:
            result = self.accepted
            copy_started = self.copy_started
        if result is not None and copy_started:
            return result.model_copy(update={"sent_copy": "unknown"})
        return result

    def remaining(self):
        if self.cancelled.is_set():
            raise ContractError("bridge_cancelled")
        remaining = self.deadline - self.monotonic()
        if remaining <= 0:
            raise ContractError("bridge_deadline")
        return min(float(self.profile.phase_timeout_seconds), remaining)

    def phase(self, session, call, *args):
        # This setter bounds a synchronous operation's inactivity timeout. A
        # private factory additionally needs absolute-deadline-aware I/O; this
        # orchestration does not claim a timeout setter interrupts blocked code.
        session.sock.settimeout(self.remaining())
        result = call(*args)
        return result


def strict_revalidate(value, model):
    if isinstance(value, StrictModel):
        value = value.model_dump(mode="json", warnings=False)
    return model.model_validate_json(canonical(value))


def validate_session(profile, bound, purpose):
    try:
        identity = strict_revalidate(bound.identity, SessionIdentity)
        expected = SessionIdentity(binding=profile.binding, mailbox_address=profile.mailbox_address,
                                   purpose=purpose, port=465 if purpose == "smtp" else 993)
        if identity != expected:
            raise ValueError
        return bound.session
    except Exception:
        raise ContractError("session_binding_mismatch") from None


def validate_prepared(profile, prepared, now):
    """Defense in depth for tenant/action/preview/wire, before any session call."""
    try:
        if not isinstance(prepared, PreparedReply) or prepared.action is None or type(prepared.wire) is not bytes:
            raise ValueError
        action = strict_revalidate(prepared.action, ReplyAction)
        preview = strict_revalidate(prepared.preview, WirePreview)
        if (any(getattr(action, key) != getattr(profile.binding, key) for key in ("tenant_id", "connector_id", "account_id", "policy_version"))
                or action.sender != profile.binding.sender_address or action.to == (action.sender,)
                or not 16 <= len(action.operation_key) <= 96 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", action.operation_key)
                or any(ord(c) < 32 and c not in "\r\n\t" for c in action.body)
                or preview.action_digest != digest(action)
                or preview.message_id != f"<zetbros.{digest(action)}@{action.sender.split('@')[1]}>"
                or any(getattr(preview, key) != getattr(action, key) for key in
                       ("source_id", "source_version", "source_fingerprint", "sender", "to", "subject", "body"))
                or type(now) is not int or not action.created_at <= now < action.expires_at
                or not 1 <= len(prepared.wire) <= MAX_WIRE_BYTES
                or hashlib.sha256(prepared.wire).hexdigest() != preview.wire_sha256
                or not prepared.wire.endswith(b"\r\n")
                or re.search(rb"(?<!\r)\n|\r(?!\n)", prepared.wire)
                or any(len(line) > 998 for line in prepared.wire.split(b"\r\n"))):
            raise ValueError
        message = BytesParser(policy=policy.default.clone(raise_on_defect=True)).parsebytes(prepared.wire)
        expected = {"from": preview.sender, "to": preview.to[0], "subject": preview.subject,
                    "message-id": preview.message_id, "in-reply-to": preview.in_reply_to,
                    "date": preview.date, "mime-version": "1.0",
                    "content-type": 'text/plain; charset="utf-8"', "content-transfer-encoding": "base64"}
        names = [name.lower() for name in message.keys()]
        if (len(names) != len(set(names)) or set(names) != set(expected) | {"references"}
                or any(str(message[name]) != value for name, value in expected.items())
                or " ".join(str(message["references"]).split()) != " ".join(preview.references)
                or message.is_multipart() or message.get_payload(decode=True) != action.body.encode("utf-8")):
            raise ValueError
        return PreparedReply(preview, prepared.wire, action)
    except Exception:
        raise ContractError("prepared_wire_invalid") from None


def smtp_data_frame(wire: bytes) -> bytes:
    """SMTP transparency framing only; receiver reconstructs the exact MIME wire."""
    if type(wire) is not bytes or not wire.endswith(b"\r\n") or len(wire) > MAX_WIRE_BYTES:
        raise ContractError("prepared_wire_invalid")
    # No newline normalization, text encoding or MIME/header regeneration.
    escaped = b"\r\n".join(b"." + line if line.startswith(b".") else line for line in wire.split(b"\r\n"))
    return escaped + b".\r\n"


def response_code(value):
    if (type(value) is not tuple or len(value) != 2 or type(value[0]) is not int
            or type(value[1]) is not bytes or len(value[1]) > 8192):
        raise ContractError("malformed_protocol_response")
    return value[0]  # Raw provider response text is never retained or logged.


class ReadOnlySession:
    """Narrow proxy prevents a source reader from adding mailbox mutations."""
    def __init__(self, session, budget):
        self.session, self.budget = session, budget
        self.untagged_responses = session.untagged_responses

    def uid(self, command, uid, items):
        if (command != "FETCH" or type(uid) is not str or not re.fullmatch(r"[1-9][0-9]{0,9}", uid)
                or int(uid) > 4294967295 or items not in ("(UID RFC822.SIZE)", "(UID BODY.PEEK[]<0.65537>)")):
            raise SourceUnavailable()
        return self.budget.phase(self.session, self.session.uid, command, uid, items)


class PrivateReadPort:
    """Per-instance single-account port compatible with OwnerMailSource."""
    def __init__(self, profile, factory, *, monotonic=time.monotonic):
        self.profile, self.factory, self.monotonic = strict_revalidate(profile, PrivateBridgeProfile), factory, monotonic

    def list_senders(self):
        return {"default_sender": self.profile.binding.sender_address,
                "allowed_senders": list(dict.fromkeys((self.profile.mailbox_address, *self.profile.approved_aliases)))}

    @contextmanager
    def mailbox(self, folder, readonly=True):
        if folder != "INBOX" or readonly is not True:
            raise SourceUnavailable()
        budget = Budget(self.profile, monotonic=self.monotonic)
        try:
            with self.factory.imap(self.profile, folder="INBOX", readonly=True,
                                   deadline=budget.deadline, timeout=budget.remaining()) as bound:
                session = validate_session(self.profile, bound, "imap")
                budget.attach(session)
                yield ReadOnlySession(session, budget)
        except Exception:
            raise SourceUnavailable() from None
        finally:
            budget.close()


class PrivatePreparedWireTransport:
    """Dormant transport; only an explicitly fake factory can exercise offline."""
    def __init__(self, profile, factory, *, offline_test_mode=False, clock=time.time, monotonic=time.monotonic,
                 sent_copy=False):
        self.profile, self.factory = strict_revalidate(profile, PrivateBridgeProfile), factory
        self.clock, self.monotonic, self.copy_enabled = clock, monotonic, sent_copy
        if type(offline_test_mode) is not bool or type(sent_copy) is not bool:
            raise ValueError("strict offline options required")
        if offline_test_mode and getattr(factory, "offline_only", None) is not True:
            raise ValueError("only an explicitly offline fake factory may exercise this release")
        self.enabled = offline_test_mode

    async def submit_prepared(self, prepared, before_body):
        if not self.enabled:
            raise RuntimeError("live private bridge is disabled in this release")
        budget = Budget(self.profile, monotonic=self.monotonic)
        task = asyncio.create_task(asyncio.to_thread(self._transaction, prepared, before_body, budget))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            budget.abort()
            try:
                # A conforming factory/connection exits under its phase bound.
                # Shield preserves a known final 250 through Sent/QUIT cleanup.
                return await asyncio.wait_for(asyncio.shield(task), timeout=self.profile.phase_timeout_seconds + 1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                known = budget.accepted_on_interruption()
                return known if known is not None else TransportResult(submission="uncertain", sent_copy="unknown", error_class="transport_timeout")
        finally:
            budget.close()

    def _transaction(self, prepared, before_body, budget):
        state = SmtpOutcome()
        try:
            prepared = validate_prepared(self.profile, prepared, int(self.clock()))
            frame = smtp_data_frame(prepared.wire)
            with self.factory.smtp(self.profile, deadline=budget.deadline, timeout=budget.remaining()) as bound:
                session = validate_session(self.profile, bound, "smtp")
                budget.attach(session)
                for stage, call, args in (("mail", session.mail, (prepared.preview.sender,)),
                                          ("rcpt", session.rcpt, (prepared.preview.to[0],)),
                                          ("data", session.docmd, ("DATA",))):
                    code = response_code(budget.phase(session, call, *args))
                    outcome = state.response(stage, code)
                    if outcome is not None:
                        return outcome
                # DATA 354 was received. No body bytes have been written yet.
                session.sock.settimeout(budget.remaining())
                before_body()  # Durable exact-preview gate, no provider I/O inside it.
                budget.remaining()
                if int(self.clock()) >= prepared.action.expires_at:
                    raise ContractError("action_expired_before_body")
                state.begin_body()  # Any following write exception is uncertain.
                session.send(frame)
                final = response_code(budget.phase(session, session.getreply))
                final_result = state.response("final", final)
                if final_result.submission == "accepted":
                    budget.remember_acceptance(final_result)
        except Exception:
            # Preserve a known final outcome even if context cleanup/QUIT fails.
            state.interrupted()
        result = state.result()
        if result.submission == "accepted" and self.copy_enabled:
            state.sent_copy(self._sent_copy(prepared, budget))
        return state.result()

    def _sent_copy(self, prepared, budget):
        attempted = False
        outcome = None
        try:
            with self.factory.imap(self.profile, folder=self.profile.sent_folder, readonly=True,
                                   deadline=budget.deadline, timeout=budget.remaining()) as bound:
                session = validate_session(self.profile, bound, "imap")
                budget.attach(session)
                status, data = budget.phase(session, session.uid, "SEARCH", None, "HEADER", "Message-ID", json.dumps(prepared.preview.message_id))
                if (status != "OK" or type(data) is not list or len(data) != 1 or type(data[0]) is not bytes
                        or len(data[0]) > 4096):
                    outcome = "failed"
                    return outcome
                uids = data[0].split()
                if len(uids) > 100 or any(not re.fullmatch(rb"[1-9][0-9]{0,9}", uid) or int(uid) > 4294967295 for uid in uids):
                    outcome = "failed"
                    return outcome
                if uids:
                    if len(uids) != 1:
                        outcome = "failed"
                        return outcome
                    validity = session.untagged_responses.get("UIDVALIDITY")
                    if type(validity) is not list or len(validity) != 1 or type(validity[0]) is not bytes or not re.fullmatch(rb"[1-9][0-9]{0,9}", validity[0]):
                        outcome = "failed"
                        return outcome
                    plan = ImapReadPlan(int(validity[0]), int(uids[0]))
                    # Existing-copy verification uses the same bounded 64-KiB
                    # read contract. A larger/ambiguous match fails without APPEND.
                    size = parse_size_response(plan, *budget.phase(session, session.uid, "FETCH", str(plan.uid), plan.size_items))
                    raw = parse_body_response(plan, *budget.phase(session, session.uid, "FETCH", str(plan.uid), plan.body_items), size)
                    outcome = "stored" if raw == prepared.wire else "failed"
                    return outcome
                budget.remaining()
                attempted = True
                budget.copy_started = True
                status, _ = budget.phase(session, session.append, json.dumps(self.profile.sent_folder), "\\Seen", None, prepared.wire)
                outcome = "stored" if status == "OK" else "failed"
                return outcome
        except Exception:
            return outcome if outcome is not None else "unknown" if attempted else "failed"
