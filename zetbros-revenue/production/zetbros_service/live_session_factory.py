"""Own live-capable pinned session factory; never invoked by dormant defaults.

Provider credentials are read only from an operator-approved systemd credential
boundary when the explicit live runtime invokes a session. No credential is
included in configuration, API inputs, exception text, logs or public evidence.
Offline tests inject fake connectors/clients/credential sources, never sockets.
"""
from __future__ import annotations

import imaplib
import json
import os
import re
import socket
import ssl
import stat
import smtplib
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .private_wire_bridge import BoundSession, PrivateBridgeProfile, SessionIdentity, strict_revalidate
from .spacemail_contract import ContractError, verified_tls_context


class SystemdCredentialSource:
    name = "spacemail-password"

    def read(self) -> str:
        directory = os.environ.get("CREDENTIALS_DIRECTORY")
        try:
            if not directory or not Path(directory).is_absolute():
                raise ValueError
            root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                meta = os.fstat(root)
                if meta.st_uid != os.geteuid() or stat.S_IMODE(meta.st_mode) & 0o077:
                    raise ValueError
                fd = os.open(self.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root)
                with os.fdopen(fd, "rb") as handle:
                    meta = os.fstat(handle.fileno())
                    if (not stat.S_ISREG(meta.st_mode) or meta.st_uid != os.geteuid()
                            or stat.S_IMODE(meta.st_mode) & 0o077 or not 1 <= meta.st_size <= 512):
                        raise ValueError
                    raw = handle.read(513)
            finally:
                os.close(root)
            value = raw.decode("utf-8", errors="strict")
            if not value or len(raw) > 512 or any(c in value for c in "\r\n\x00"):
                raise ValueError
            return value
        except Exception:
            raise ContractError("protected_credential_unavailable") from None


@dataclass
class _Resolution:
    ready: threading.Event
    addresses: list | None = None


class BoundedResolver:
    """One bounded-wait resolver job per factory, never an unbounded thread pool.

    getaddrinfo itself cannot be killed portably. A stuck DNS job remains one
    daemon; callers time out without connecting, and no replacement is spawned
    while that job is outstanding. Only the fixed provider hostname is resolved.
    """
    def __init__(self, *, lookup=socket.getaddrinfo, monotonic=time.monotonic):
        self.lookup, self.monotonic = lookup, monotonic
        self.lock = threading.Lock()
        self.job = None
        self.cached_at = 0.0

    def resolve(self, host, port, timeout):
        if host != "mail.spacemail.com" or port not in (993, 465) or not 0 < timeout <= 5:
            raise ContractError("fixed_endpoint_required")
        with self.lock:
            if self.job is None or (self.job.ready.is_set() and self.monotonic() - self.cached_at > 60):
                job = _Resolution(threading.Event())
                self.job = job
                def run():
                    try:
                        rows = self.lookup(host, None, type=socket.SOCK_STREAM)
                        valid = []
                        for family, kind, proto, _, location in rows:
                            if family not in (socket.AF_INET, socket.AF_INET6) or kind != socket.SOCK_STREAM:
                                continue
                            if family == socket.AF_INET:
                                valid.append((family, kind, proto, (location[0], port)))
                            else:
                                valid.append((family, kind, proto, (location[0], port, *location[2:])))
                        job.addresses = valid[:8] or None
                    except Exception:
                        job.addresses = None
                    finally:
                        self.cached_at = self.monotonic()
                        job.ready.set()
                threading.Thread(target=run, name="zetbros-dns", daemon=True).start()
            job = self.job
        if not job.ready.wait(timeout=timeout) or not job.addresses:
            raise ContractError("provider_resolution_unavailable")
        # Cache IP/family, not another session's port.
        return [(f, k, p, (loc[0], port, *loc[2:])) for f, k, p, loc in job.addresses]


class ConnectionGuard:
    def __init__(self, deadline, phase_timeout, *, monotonic=time.monotonic):
        self.deadline, self.phase_timeout, self.monotonic = deadline, phase_timeout, monotonic
        self.lock = threading.Lock()
        self.sockets = []
        self.stopped = threading.Event()
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ContractError("private_session_deadline")
        self.timer = threading.Timer(remaining, self.abort)
        self.timer.daemon = True
        self.timer.start()

    def remaining(self):
        remaining = self.deadline - self.monotonic()
        if self.stopped.is_set() or remaining <= 0:
            raise ContractError("private_session_deadline")
        return min(float(self.phase_timeout), remaining)

    def register(self, sock):
        with self.lock:
            self.sockets.append(sock)
        try:
            sock.settimeout(self.remaining())
        except Exception:
            self.abort()
            raise
        return sock

    def abort(self):
        self.stopped.set()
        with self.lock: sockets = list(self.sockets)
        for sock in sockets:
            for method, args in (("shutdown", (socket.SHUT_RDWR,)), ("close", ())):
                try: getattr(sock, method)(*args)
                except Exception: pass

    def close(self):
        self.timer.cancel()
        self.abort()


class PinnedConnector:
    def __init__(self, *, resolver=None, socket_factory=socket.socket, tls_context=verified_tls_context):
        self.resolver = resolver if resolver is not None else BoundedResolver()
        self.socket_factory, self.tls_context = socket_factory, tls_context

    def connect(self, host, port, guard):
        if host != "mail.spacemail.com" or port not in (993, 465):
            raise ContractError("fixed_endpoint_required")
        context = self.tls_context()
        if (context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname
                or context.minimum_version < ssl.TLSVersion.TLSv1_2):
            raise ContractError("tls_verification_required")
        addresses = self.resolver.resolve(host, port, guard.remaining())
        for family, kind, proto, location in addresses:
            sock = None
            try:
                sock = guard.register(self.socket_factory(family, kind, proto))
                sock.connect(location)
                sock.settimeout(guard.remaining())
                secured = guard.register(context.wrap_socket(sock, server_hostname=host, do_handshake_on_connect=False))
                secured.do_handshake()
                guard.remaining()
                return secured
            except Exception:
                if sock is not None:
                    try: sock.close()
                    except Exception: pass
                guard.remaining()
        raise ContractError("private_connection_unavailable")


class _PinnedSmtp(smtplib.SMTP_SSL):
    def __init__(self, host, port, guard, connector):
        self._guard, self._connector = guard, connector
        super().__init__(host=host, port=port, timeout=guard.remaining(), context=verified_tls_context(), local_hostname="[127.0.0.1]")

    def _get_socket(self, host, port, timeout):
        return self._connector.connect(host, port, self._guard)

    def getreply(self):
        if self.file is None:
            self.file = self.sock.makefile("rb")
        lines, total, expected = [], 0, None
        for _ in range(32):
            self.sock.settimeout(self._guard.remaining())
            line = self.file.readline(513)
            total += len(line)
            match = re.fullmatch(rb"([1-5][0-9]{2})([- ])([^\r\n]*)\r\n", line)
            if not match or len(line) > 512 or total > 8192:
                raise ContractError("bounded_smtp_response_required")
            code = int(match[1])
            if expected is not None and code != expected:
                raise ContractError("bounded_smtp_response_required")
            expected = code
            lines.append(match[3].strip(b" \t"))
            if match[2] == b" ":
                return code, b"\n".join(lines)
        raise ContractError("bounded_smtp_response_required")


class _PinnedImap(imaplib.IMAP4_SSL):
    def __init__(self, host, port, guard, connector):
        self._guard, self._connector = guard, connector
        self._literal_count, self._response_count = 0, 0
        super().__init__(host=host, port=port, timeout=guard.remaining(), ssl_context=verified_tls_context())

    def _create_socket(self, timeout):
        return self._connector.connect(self.host, self.port, self._guard)

    def _log(self, line):
        # imaplib caches complete outgoing LOGIN commands even at debug=0.
        # A credential must never enter that diagnostic cache.
        return None

    def _mesg(self, message, secs=None):
        return None

    def _simple_command(self, name, *args):
        self._literal_count, self._response_count = 0, 0
        return super()._simple_command(name, *args)

    def read(self, size):
        self._literal_count += 1
        if type(size) is not int or not 0 <= size <= 65536 or self._literal_count > 1:
            raise ContractError("bounded_imap_literal_required")
        self.sock.settimeout(self._guard.remaining())
        return super().read(size)

    def readline(self):
        self.sock.settimeout(self._guard.remaining())
        line = self.file.readline(8193)
        if len(line) > 8192:
            raise ContractError("bounded_imap_response_required")
        return line

    def _get_response(self):
        self._response_count += 1
        if self._response_count > 64:
            raise ContractError("bounded_imap_response_required")
        return super()._get_response()

    def _append_untagged(self, typ, dat):
        def size(value):
            if type(value) is bytes: return len(value)
            if type(value) is tuple and len(value) == 2 and all(type(v) is bytes for v in value):
                return sum(len(v) for v in value)
            raise ContractError("bounded_imap_response_required")
        values = [v for rows in self.untagged_responses.values() for v in rows]
        if len(values) >= 64 or sum(size(v) for v in values) + size(dat or b"") > 98304:
            raise ContractError("bounded_imap_response_required")
        return super()._append_untagged(typ, dat)


class LiveSessionFactory:
    offline_only = False

    def __init__(self, profile, *, credentials=None, connector=None,
                 smtp_client=_PinnedSmtp, imap_client=_PinnedImap, monotonic=time.monotonic):
        self.profile = strict_revalidate(profile, PrivateBridgeProfile)
        if self.profile.outbound != "owner_approved_live":
            raise ContractError("live_factory_requires_operator_profile")
        self.credentials = credentials if credentials is not None else SystemdCredentialSource()
        self.connector = connector if connector is not None else PinnedConnector()
        self.smtp_client, self.imap_client, self.monotonic = smtp_client, imap_client, monotonic

    def check(self, profile):
        if strict_revalidate(profile, PrivateBridgeProfile) != self.profile:
            raise ContractError("private_factory_binding_mismatch")

    def identity(self, purpose):
        return SessionIdentity(binding=self.profile.binding, mailbox_address=self.profile.mailbox_address,
                               purpose=purpose, port=465 if purpose == "smtp" else 993)

    @contextmanager
    def smtp(self, profile, *, deadline, timeout):
        self.check(profile)
        guard = ConnectionGuard(deadline, min(timeout, profile.phase_timeout_seconds), monotonic=self.monotonic)
        client = None
        try:
            client = self.smtp_client(profile.endpoints.host, 465, guard, self.connector)
            client.sock.settimeout(guard.remaining())
            result = client.ehlo()
            if type(result) is not tuple or result[0] != 250:
                raise ValueError
            client.sock.settimeout(guard.remaining())
            password = self.credentials.read()
            try: result = client.login(profile.mailbox_address, password)
            finally:
                password = None
                client.user = client.password = None
            if type(result) is not tuple or len(result) != 2 or type(result[0]) is not int or result[0] != 235:
                raise ValueError
            guard.remaining()
            yield BoundSession(self.identity("smtp"), client)
        except Exception:
            raise ContractError("private_session_unavailable") from None
        finally:
            if client is not None:
                try: client.close()
                except Exception: pass
            guard.close()

    @contextmanager
    def imap(self, profile, *, folder, readonly, deadline, timeout):
        self.check(profile)
        if folder not in ("INBOX", profile.sent_folder) or readonly is not True:
            raise ContractError("private_mailbox_scope")
        guard = ConnectionGuard(deadline, min(timeout, profile.phase_timeout_seconds), monotonic=self.monotonic)
        client = None
        try:
            client = self.imap_client(profile.endpoints.host, 993, guard, self.connector)
            client.sock.settimeout(guard.remaining())
            password = self.credentials.read()
            try: result = client.login(profile.mailbox_address, password)
            finally: password = None
            if type(result) is not tuple or result[0] != "OK": raise ValueError
            client.sock.settimeout(guard.remaining())
            result = client.select(json.dumps(folder), readonly=True)
            if type(result) is not tuple or result[0] != "OK": raise ValueError
            guard.remaining()
            yield BoundSession(self.identity("imap"), client)
        except Exception:
            raise ContractError("private_session_unavailable") from None
        finally:
            if client is not None:
                try: client.shutdown()
                except Exception: pass
            guard.close()
