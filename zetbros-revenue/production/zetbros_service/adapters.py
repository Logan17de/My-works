"""One source contract; runtime has no SMTP, IMAP, provider credentials or send path."""
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from .config import Settings
from .models import Identifier, ReplyAction, SourceMessage, TransportResult


class SourceUnavailable(Exception):
    pass


class SourceNotFound(SourceUnavailable):
    pass


class SnapshotSource:
    """Customer-staged read-only records, explicitly not authoritative live Mail."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.source_directory)
        if not self.root.is_dir() or self.root.is_symlink():
            raise RuntimeError("customer-staged source directory missing or invalid")

    def get(self, source_id: str) -> SourceMessage:
        TypeAdapter(Identifier).validate_python(source_id, strict=True)
        # The ID never becomes a caller-controlled path or provider URL.
        path = self.root / (source_id + ".json")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as handle:
                metadata = os.fstat(handle.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 65536:
                    raise ValueError("invalid source file")
                record = SourceMessage.model_validate_json(handle.read(65537))
            if record.source_id != source_id or any(getattr(record, k) != getattr(self.settings, k) for k in ("tenant_id", "connector_id", "account_id")):
                raise ValueError("source identity mismatch")
            return record
        except FileNotFoundError as exc:
            raise SourceNotFound() from exc
        except (OSError, ValueError) as exc:
            raise SourceUnavailable() from exc

    def ready(self) -> bool:
        return self.root.is_dir() and not self.root.is_symlink()


class ReplyTransport(Protocol):
    enabled: bool

    async def submit(self, action: ReplyAction) -> TransportResult: ...


class DisabledTransport:
    enabled = False

    async def submit(self, action: ReplyAction) -> TransportResult:
        # Never called by the worker; defense in depth without network clients.
        raise RuntimeError("outbound delivery is disabled in this release")
