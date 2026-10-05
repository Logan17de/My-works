"""Explicit operator-configured pilot wiring; shipped defaults remain disabled."""
from __future__ import annotations

import os
import stat
from pathlib import Path

from .config import Settings, load_settings
from .live_session_factory import LiveSessionFactory
from .mail_integration import MailService, MailStore
from .mail_owner import OwnerMailSource
from .private_wire_bridge import PrivateBridgeProfile, PrivatePreparedWireTransport, PrivateReadPort
from .service import Service
from .spacemail_contract import ProviderBinding
from .models import canonical


def load_profile(settings):
    try:
        path = settings.private_bridge_profile_file
        if not path: raise ValueError
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            meta = os.fstat(handle.fileno())
            if not stat.S_ISREG(meta.st_mode) or not 1 <= meta.st_size <= 16384: raise ValueError
            profile = PrivateBridgeProfile.model_validate_json(handle.read(16385))
        expected = ProviderBinding(**{key: getattr(settings, key) for key in
            ("tenant_id", "connector_id", "account_id", "sender_address", "policy_version")})
        if profile.binding != expected or profile.outbound != "owner_approved_live": raise ValueError
        if profile.total_timeout_seconds >= settings.adapter_timeout_seconds: raise ValueError
        return profile
    except Exception:
        raise RuntimeError("operator private profile is missing, mismatched or invalid") from None


def assemble_source(settings, *, factory_builder=LiveSessionFactory):
    profile = load_profile(settings)
    factory = factory_builder(profile)
    return profile, factory, OwnerMailSource(profile.binding, PrivateReadPort(profile, factory))


def build_service(settings, *, factory_builder=LiveSessionFactory, clock=None):
    settings = Settings.model_validate_json(canonical(settings))
    options = {} if clock is None else {"clock": clock}
    if settings.source_adapter == "customer_staged_snapshot" and settings.outbound_adapter == "disabled":
        return Service(settings, **options)
    if settings.source_adapter != "private_spacemail" or settings.outbound_adapter != "private_spacemail":
        raise RuntimeError("explicit paired private pilot adapters required")
    profile, factory, source = assemble_source(settings, factory_builder=factory_builder)
    transport = PrivatePreparedWireTransport(profile, factory, sent_copy=True, **options)
    return MailService(settings, source=source, prepared_transport=transport, **options)


def initialize_new_pilot(settings, *, factory_builder=LiveSessionFactory, clock=None):
    settings = Settings.model_validate_json(canonical(settings))
    if settings.source_adapter != "private_spacemail" or settings.outbound_adapter != "private_spacemail":
        raise RuntimeError("explicit private pilot configuration required")
    _, _, source = assemble_source(settings, factory_builder=factory_builder)
    options = {} if clock is None else {"clock": clock}
    # Store.initialize uses O_EXCL and refuses any existing final path. This
    # creates a new ledger only; it is never a migration or restore substitute.
    return MailStore.initialize_offline(settings, source=source, **options)


def configured_pilot_app():
    from .api import create_app
    settings = load_settings()
    return create_app(build_service(settings))
