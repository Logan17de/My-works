"""Disabled staging first; initialize a pinned ledger only after identity setup."""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from urllib.parse import urlsplit
from contextlib import contextmanager, asynccontextmanager

from .adapters import SourceUnavailable
from .auth import Verifier
from .config import PendingPilotConfig, ReviewerPendingConfig, Settings, load_settings
from .live_session_factory import LiveSessionFactory
from .mail_integration import MailService, MailStore
from .mail_owner import OwnerMailSource
from .private_wire_bridge import PrivateBridgeProfile, PrivatePreparedWireTransport, PrivateReadPort
from .service import Service
from .spacemail_contract import ProviderBinding
from .models import canonical


def pending_status():
    return {"runtime_state": "disabled_pending_identity", "identity_configuration": "pending",
            "control_ready": False, "live_delivery_ready": False, "delivery": "disabled",
            "ledger_initialized": False, "ledger_access": "not_opened", "provider_connections": "not_performed",
            "error_class": "identity_configuration_pending"}


def stage_without_ledger(settings: PendingPilotConfig):
    settings = PendingPilotConfig.model_validate_json(canonical(settings))
    if os.path.lexists(settings.database_path):
        raise FileExistsError("existing ledger is never initialized or rebound by staging")
    return pending_status() | {"ledger_created": False, "stage": "configuration_validated_no_ledger"}


def validate_identity(settings):
    values = [getattr(settings, k) for k in ("tenant_id", "connector_id", "account_id", "sender_address",
                                           "policy_version", "issuer", "audience")]
    values += [v for grant in settings.principals for v in (grant.subject, grant.client_id)]
    domains = [settings.sender_address.rsplit("@", 1)[-1].lower(), (urlsplit(settings.issuer).hostname or "").lower()]
    if (any("REPLACE" in value.upper() for value in values)
            or any(domain.endswith((".invalid", ".test", ".example"))
                   or domain in ("example.com", "example.net", "example.org")
                   or any(domain.endswith("." + example) for example in ("example.com", "example.net", "example.org"))
                   for domain in domains)):
        raise RuntimeError("actual owner identity configuration required; placeholders cannot pin a pilot ledger")
    # Public-key and configured-principal verification precedes any ledger write.
    Verifier(settings)


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
        required = "owner_approved_live" if settings.outbound_adapter == "private_spacemail" else "disabled"
        if profile.binding != expected or profile.outbound != required: raise ValueError
        if profile.total_timeout_seconds >= settings.adapter_timeout_seconds: raise ValueError
        return profile
    except Exception:
        raise RuntimeError("operator private profile is missing, mismatched or invalid") from None


class DisabledPrivateReadPort:
    """Same mail-review contract while disabled, without a factory or credentials."""
    def __init__(self, profile): self.profile = profile
    def list_senders(self):
        return {"default_sender": self.profile.binding.sender_address,
                "allowed_senders": list(dict.fromkeys((self.profile.mailbox_address, *self.profile.approved_aliases)))}
    @contextmanager
    def mailbox(self, folder, readonly=True):
        raise SourceUnavailable()
        yield  # Structural context-manager interface; no operation is performed.


def disabled_source(profile):
    return OwnerMailSource(profile.binding, DisabledPrivateReadPort(profile))


def assemble_source(settings, *, factory_builder=LiveSessionFactory):
    profile = load_profile(settings)
    if settings.outbound_adapter == "disabled":
        return profile, None, disabled_source(profile)
    factory = factory_builder(profile)
    return profile, factory, OwnerMailSource(profile.binding, PrivateReadPort(profile, factory))


class ReviewerPendingRuntime:
    """Authentication only; deliberately has no store, verifier, provider or worker."""
    def __init__(self, settings, clock=None):
        self.settings = settings
        self.clock = time.time if clock is None else clock

    def readiness(self):
        return {"runtime_state": "reviewer_enabled_workflow_pending",
                "reviewer_login_enabled": True, "workflow_ready": False,
                "agent_api_enabled": False, "delivery": "disabled",
                "ledger_initialized": False, "provider_connections": "not_performed"}


def build_service(settings, *, factory_builder=LiveSessionFactory, clock=None):
    if isinstance(settings, PendingPilotConfig):
        raise RuntimeError("pending identity uses the staging app without a service ledger")
    if isinstance(settings, ReviewerPendingConfig):
        settings = ReviewerPendingConfig.model_validate_json(canonical(settings))
        from .google_auth_runtime import configure_google_review
        return configure_google_review(ReviewerPendingRuntime(settings, clock))
    settings = Settings.model_validate_json(canonical(settings))
    options = {} if clock is None else {"clock": clock}
    if settings.source_adapter == "customer_staged_snapshot" and settings.outbound_adapter == "disabled":
        if settings.reviewer_auth != "existing_access_token":
            raise RuntimeError("Google reviewer mode requires the exact-wire private mail contract")
        return Service(settings, **options)
    if settings.source_adapter != "private_spacemail":
        raise RuntimeError("private pilot source contract required")
    validate_identity(settings)
    profile, factory, source = assemble_source(settings, factory_builder=factory_builder)
    transport = None if factory is None else PrivatePreparedWireTransport(profile, factory, sent_copy=True, **options)
    service = MailService(settings, source=source, prepared_transport=transport, **options)
    from .google_auth_runtime import configure_google_review
    return configure_google_review(service)


def initialize_new_pilot(settings, *, factory_builder=LiveSessionFactory, clock=None):
    if isinstance(settings, PendingPilotConfig):
        return stage_without_ledger(settings)
    if isinstance(settings, ReviewerPendingConfig):
        return build_service(settings, clock=clock).readiness() | {"ledger_created": False}
    settings = Settings.model_validate_json(canonical(settings))
    if os.path.lexists(settings.database_path):
        raise FileExistsError("new pilot initialization refuses existing ledger paths")
    if settings.source_adapter != "private_spacemail":
        raise RuntimeError("identity-configured private pilot source contract required")
    validate_identity(settings)
    if settings.reviewer_auth == "google_oidc":
        from .google_auth_runtime import load_google_review_profile
        load_google_review_profile(settings)
    profile = load_profile(settings)
    options = {} if clock is None else {"clock": clock}
    # Initialization never constructs a live factory, even with live config.
    # Store.initialize still uses O_EXCL to reject a racing existing final path.
    return MailStore.initialize_offline(settings, source=disabled_source(profile), **options)


def create_staging_app(settings: PendingPilotConfig):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    PendingPilotConfig.model_validate_json(canonical(settings))
    @asynccontextmanager
    async def lifespan(app):
        import anyio
        anyio.to_thread.current_default_thread_limiter().total_tokens = 4
        yield
    app = FastAPI(title="Zetbros disabled staging: identity pending", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    @app.middleware("http")
    async def boundary(request: Request, call_next):
        if request.query_params:
            return JSONResponse({"error": "staging_queries_forbidden"}, status_code=400)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    @app.get("/healthz")
    def health(): return {"status": "up", "runtime_state": "disabled_pending_identity"}
    @app.get("/readyz")
    def ready(): return JSONResponse(pending_status(), status_code=503)
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    def unavailable(path: str):
        return JSONResponse({"error": "identity_configuration_pending", "delivery": "disabled"}, status_code=503)
    return app


def create_reviewer_pending_app(service):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from .review_ui import install_review_ui
    from .store import StoreError
    if not isinstance(service.settings, ReviewerPendingConfig):
        raise ValueError("reviewer pending configuration required")
    @asynccontextmanager
    async def lifespan(app):
        import anyio
        anyio.to_thread.current_default_thread_limiter().total_tokens = 4
        yield
    app = FastAPI(title="Zetbros reviewer login", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    @app.exception_handler(StoreError)
    async def failure(request, exc):
        return JSONResponse({"error": "review_unavailable"}, status_code=404)
    @app.get('/healthz')
    def health():
        return {"status": "up"} | service.readiness()
    @app.get('/readyz')
    def ready():
        return JSONResponse(service.readiness(), status_code=503)
    install_review_ui(app, service, service.settings.review_origin)
    @app.api_route('/{path:path}', methods=['GET','POST','PUT','PATCH','DELETE','OPTIONS','HEAD'])
    def unavailable(path):
        return JSONResponse({"error": "workflow_configuration_pending",
                             "agent_api_enabled": False, "delivery": "disabled"}, status_code=503)
    return app


def configured_pilot_app():
    from .api import configured_app
    return configured_app()
