"""Load deployment-owned configuration only; no secret discovery or defaults."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from .models import Address, Identifier, PrincipalId, StrictModel, address, digest


class Grant(StrictModel):
    subject: PrincipalId
    client_id: PrincipalId
    role: Literal["agent", "reviewer"]


class PendingPilotConfig(StrictModel):
    """Disabled install configuration, deliberately containing no identity values."""
    runtime_state: Literal["disabled_pending_identity"]
    deployment_profile: Literal["single_host_local_volume"] = "single_host_local_volume"
    source_adapter: Literal["disabled"] = "disabled"
    outbound_adapter: Literal["disabled"] = "disabled"
    database_path: str

    @field_validator("database_path")
    @classmethod
    def proposed_database_path(cls, value):
        if not Path(value).is_absolute() or "\x00" in value:
            raise ValueError("proposed database path must be absolute")
        return value


class ReviewerPendingConfig(StrictModel):
    """Approved reviewer login, with no agent, workflow, ledger or Mail settings."""
    runtime_state: Literal["reviewer_enabled_workflow_pending"]
    reviewer_auth: Literal["google_oidc"] = "google_oidc"
    review_origin: str
    google_reviewer_profile_file: str
    principals: tuple[Grant, ...]
    outbound_adapter: Literal["disabled"] = "disabled"

    @field_validator("review_origin")
    @classmethod
    def origin(cls, value):
        from .review_ui import validate_review_origin
        if not value.startswith("https://"):
            raise ValueError("reviewer login requires HTTPS")
        return validate_review_origin(value)

    @field_validator("google_reviewer_profile_file")
    @classmethod
    def profile_path(cls, value):
        if not Path(value).is_absolute() or "\x00" in value:
            raise ValueError("absolute reviewer profile path required")
        return value

    @model_validator(mode="after")
    def reviewers_only(self):
        pairs = [(g.subject, g.client_id) for g in self.principals]
        if (not 1 <= len(pairs) <= 32 or len(pairs) != len(set(pairs))
                or any(g.role != "reviewer" for g in self.principals)):
            raise ValueError("bounded distinct reviewer mappings only")
        return self

    def role(self, subject, client_id):
        return next((g.role for g in self.principals
                     if (g.subject, g.client_id) == (subject, client_id)), None)


class Settings(StrictModel):
    tenant_id: Identifier
    connector_id: Identifier
    account_id: Identifier
    sender_address: Address
    policy_version: Identifier
    issuer: Annotated[str, Field(min_length=8, max_length=512)]
    audience: Annotated[str, Field(min_length=1, max_length=160)]
    jwks_file: str
    database_path: str
    source_directory: str
    principals: tuple[Grant, ...]
    proposal_ttl_seconds: Annotated[int, Field(ge=30, le=3600)] = 600
    approval_ttl_seconds: Annotated[int, Field(ge=30, le=900)] = 300
    claim_timeout_seconds: Annotated[int, Field(ge=30, le=300)] = 60
    adapter_timeout_seconds: Annotated[int, Field(ge=1, le=60)] = 20
    token_max_lifetime_seconds: Annotated[int, Field(ge=60, le=3600)] = 900
    requests_per_minute: Annotated[int, Field(ge=1, le=600)] = 60
    max_database_bytes: Annotated[int, Field(ge=1048576, le=107374182400)] = 1073741824
    token_client_claim: Literal["azp", "client_id"] = "azp"
    token_header_type: Literal["at+jwt", "JWT"] = "at+jwt"
    deployment_profile: Literal["single_host_local_volume"] = "single_host_local_volume"
    outbound_adapter: Literal["disabled", "private_spacemail"] = "disabled"
    source_adapter: Literal["customer_staged_snapshot", "private_spacemail"] = "customer_staged_snapshot"
    private_bridge_profile_file: str | None = None
    review_origin: str | None = None
    reviewer_auth: Literal["existing_access_token", "google_oidc"] = "existing_access_token"
    google_reviewer_profile_file: str | None = None

    _sender = field_validator("sender_address")(address)

    @field_validator("private_bridge_profile_file", "google_reviewer_profile_file")
    @classmethod
    def optional_profile_path(cls, value):
        if value is not None and (not Path(value).is_absolute() or "\x00" in value):
            raise ValueError("private profile path must be absolute")
        return value

    @field_validator("review_origin")
    @classmethod
    def fixed_review_origin(cls, value):
        if value is None: return value
        parsed = urlsplit(value)
        if (parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment or value.endswith("/")):
            raise ValueError("exact reviewer origin required")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("reviewer HTTP is permitted only on loopback")
        return value

    @field_validator("issuer")
    @classmethod
    def fixed_https_issuer(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("issuer must be an HTTPS identity-provider URL")
        return value

    @field_validator("jwks_file", "database_path", "source_directory")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute() or "\x00" in value:
            raise ValueError("deployment paths must be absolute")
        return value

    @model_validator(mode="after")
    def separate_principals(self) -> "Settings":
        if self.reviewer_auth == "google_oidc" and (self.google_reviewer_profile_file is None or self.review_origin is None):
            raise ValueError("Google reviewer mode requires its explicit profile and origin")
        live_source = self.source_adapter == "private_spacemail"
        live_transport = self.outbound_adapter == "private_spacemail"
        if live_transport and not live_source:
            raise ValueError("private delivery requires its private source contract")
        if live_source and (self.private_bridge_profile_file is None or self.review_origin is None):
            raise ValueError("private source contract requires profile and reviewer origin")
        keys = [(g.subject, g.client_id) for g in self.principals]
        if not keys or len(keys) != len(set(keys)) or {g.role for g in self.principals} != {"agent", "reviewer"}:
            raise ValueError("distinct configured agent and reviewer principal/client pairs required")
        agent_clients = {g.client_id for g in self.principals if g.role == "agent"}
        reviewer_clients = {g.client_id for g in self.principals if g.role == "reviewer"}
        if agent_clients & reviewer_clients:
            raise ValueError("agent and reviewer client ID sets must be disjoint")
        if self.adapter_timeout_seconds >= self.claim_timeout_seconds:
            raise ValueError("adapter timeout must be shorter than claim timeout")
        db, source = Path(self.database_path), Path(self.source_directory)
        if source == db.parent or source in db.parents or db == Path(self.jwks_file):
            raise ValueError("source, database and identity-key storage must be separate")
        return self

    @property
    def identity_digest(self) -> str:
        return digest({k: getattr(self, k) for k in ("tenant_id", "connector_id", "account_id", "sender_address", "issuer", "audience")})

    def role(self, subject: str, client_id: str) -> str | None:
        return next((g.role for g in self.principals if g.subject == subject and g.client_id == client_id), None)


def load_settings() -> Settings | PendingPilotConfig | ReviewerPendingConfig:
    path = os.environ.get("ZETBROS_CONFIG_FILE")
    if not path:
        raise RuntimeError("ZETBROS_CONFIG_FILE is required; service cannot start unconfigured")
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 65536:
            raise ValueError("configuration too large")
        value = json.loads(raw)
        if isinstance(value, dict) and value.get("runtime_state") == "disabled_pending_identity":
            return PendingPilotConfig.model_validate_json(raw)
        if isinstance(value, dict) and value.get("runtime_state") == "reviewer_enabled_workflow_pending":
            return ReviewerPendingConfig.model_validate_json(raw)
        return Settings.model_validate_json(raw)
    except (OSError, ValueError) as exc:
        raise RuntimeError("deployment configuration is missing or invalid") from exc
