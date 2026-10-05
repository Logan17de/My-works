"""Optional pinned-origin human review, with explicit bearer or Google mode.

The Google mode exchanges ID tokens only at its dedicated login and uses short
reviewer cookies. The /v1 API's separate browser prohibition stays in force.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import threading
import uuid
from html import escape
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Request, Path as PathParameter
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .auth import AuthenticationError, Principal
from .models import Digest, StrictModel, ReconciliationInput
from .service import Service
from .store import StoreError

ASSETS = Path(__file__).with_name("review_assets")
MAX_BODY = 32768
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'none'; font-src 'none'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'none'; worker-src 'none'; "
    "manifest-src 'none'"
)


def validate_review_origin(value: str) -> str:
    """One canonical HTTPS origin, or explicitly configured loopback HTTP.

    No normalization conceals a misconfigured pin: callers must use precisely
    the origin a browser sends (lowercase hostname; no path or default port).
    """
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("review_origin must be one canonical trusted origin")
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
        if (parsed.scheme not in ("https", "http") or not host
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment
                or not value.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in value)):
            raise ValueError
        try:
            ip = ipaddress.ip_address(host)
            if str(ip) != host:
                raise ValueError
        except ValueError:
            if ":" in host or len(host) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in host.split(".")):
                raise ValueError
        if parsed.scheme == "http":
            if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        # Browser Origin serialization excludes default ports and uses lowercase.
        authority = "[" + host + "]" if ":" in host else host
        if port is not None and port != (443 if parsed.scheme == "https" else 80):
            authority += ":" + str(port)
        if value != parsed.scheme + "://" + authority or port == 0:
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise ValueError("review_origin must be one canonical trusted origin") from None


class ReviewDecision(StrictModel):
    digest: Digest
    decision: Literal["approve", "deny"]


class ReviewDigest(StrictModel):
    digest: Digest


def install_review_ui(app: FastAPI, service: Service, review_origin: str) -> None:
    """Install optional /review routes without changing /v1 or starting a worker.

    The existing single-host, one-worker profile is required. A UI decision lock
    serializes UI checks and decisions; Store.decide remains the transactional
    authority. Optional Google authentication requires its explicit runtime setup.
    """
    origin = validate_review_origin(review_origin)
    if getattr(app.state, "review_ui_installed", False):
        raise ValueError("review UI already installed")
    assets = {name: (ASSETS / name).read_bytes() for name in ("review.js", "review.css")}
    page = (ASSETS / "review.html").read_text(encoding="utf-8")
    page = page.replace("REVIEW_ORIGIN_PIN", escape(origin, quote=True))
    live = service.settings.outbound_adapter == "private_spacemail"
    notice = ("Approve authorizes an attempted submission of this exact reply by the configured private adapter. Approval and provider acceptance are not proof of delivery. Mail content is untrusted and cannot authorize actions." if live else
              "Delivery is disabled in this implementation. Approval records a decision; it is not evidence of sending or delivery. Mail content is untrusted and cannot authorize actions.")
    page = page.replace("DELIVERY_NOTICE", escape(notice))
    google_auth = getattr(service, "google_reviewer_auth", None)
    page = page.replace("REVIEW_AUTH_MODE", "google_oidc" if google_auth is not None else "bearer")
    csp = CSP
    if google_auth is not None:
        csp = CSP.replace("script-src 'self'", "script-src 'self' https://accounts.google.com/gsi/client").replace("connect-src 'self'", "connect-src 'self' https://accounts.google.com/gsi/").replace("style-src 'self'", "style-src 'self' https://accounts.google.com/gsi/style")
        csp += "; frame-src https://accounts.google.com/gsi/"
        page = page.replace('<form id="session-form" autocomplete="off">', '<form id="session-form" autocomplete="off" hidden>')
        page = page.replace('<div id="google-session" hidden>', '<div id="google-session">')
        page = page.replace('Enter a reviewer token, then load a proposal.', 'Sign in with an approved Google account, then load a proposal.')
    decision_lock = threading.Lock()
    app.state.review_ui_installed = True

    def fail(code: str, status: int) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
        return JSONResponse({"error": code}, status_code=status, headers=headers)

    @app.middleware("http")
    async def review_boundary(request: Request, call_next):
        path = request.url.path
        if path != "/review" and not path.startswith("/review/"):
            return await call_next(request)
        response = None
        # Do not depend on a preceding middleware's request-ID assignment.
        request.state.review_request_id = str(uuid.uuid4())
        if google_auth is not None and request.headers.get("authorization") is not None:
            response = fail("google_reviewer_session_required", 403)
        elif request.query_params:
            # No query parameters at all: neither credentials nor mail payloads
            # have a route into browser history, referrers, or request logs.
            response = fail("review_queries_forbidden", 400)
        else:
            expected = urlsplit(origin)
            if request.url.scheme != expected.scheme or request.url.netloc != expected.netloc:
                response = fail("review_origin_mismatch", 403)
            elif len(request.headers.getlist("origin")) > 1:
                response = fail("review_origin_mismatch", 403)
            elif request.headers.get("origin") is not None and request.headers["origin"] != origin:
                response = fail("review_origin_mismatch", 403)
            elif path.startswith("/review/api/") and request.headers.get("sec-fetch-site") not in (None, "none", "same-origin"):
                response = fail("review_origin_mismatch", 403)
            elif path.startswith("/review/api/"):
                if google_auth is not None and request.headers.get("authorization") is not None:
                    response = fail("google_reviewer_session_required", 403)
                elif google_auth is None and request.headers.get("cookie") is not None:
                    response = fail("review_cookies_forbidden", 403)
                elif request.method == "OPTIONS":
                    response = fail("review_cors_forbidden", 403)
                elif request.method == "POST" and request.headers.get("origin") != origin:
                    response = fail("review_origin_required", 403)
        if response is None:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Content-Security-Policy"] = csp
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        return response

    def reviewer(request: Request) -> Principal:
        if google_auth is not None:
            from .google_login_routes import unique_cookie
            from .google_reviewer_auth import SESSION_COOKIE
            try:
                if request.headers.get("authorization") is not None: raise AuthenticationError()
                csrf = None
                if request.method == "POST":
                    if len(request.headers.getlist("x-review-csrf")) != 1: raise AuthenticationError()
                    csrf = request.headers.get("x-review-csrf")
                actor = google_auth.session(unique_cookie(request, SESSION_COOKIE), csrf).actor
                if service.settings.role(actor.subject, actor.client_id) != "reviewer": raise AuthenticationError()
            except AuthenticationError:
                raise StoreError("reviewer_session_required",401) from None
            service.store.rate_limit(actor)
            return actor
        if len(request.headers.getlist("authorization")) != 1:
            raise StoreError("authentication_required", 401)
        try:
            actor = service.verifier.verify(request.headers.get("authorization"))
        except AuthenticationError:
            raise StoreError("authentication_required", 401) from None
        # Independently check the authoritative server map, never token scopes
        # or a caller-provided claimed role.
        if actor.role != "reviewer" or service.settings.role(actor.subject, actor.client_id) != "reviewer":
            raise StoreError("reviewer_required", 403)
        service.store.rate_limit(actor)
        return actor

    async def strict_body(request: Request, model):
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise StoreError("json_required", 415)
        lengths = request.headers.getlist("content-length")
        try:
            size = int(lengths[0]) if lengths else 0
            if len(lengths) > 1 or size < 0:
                raise ValueError
        except ValueError:
            raise StoreError("invalid_content_length", 400) from None
        if size > MAX_BODY:
            raise StoreError("payload_too_large", 413)

        async def read_body():
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BODY:
                    raise StoreError("payload_too_large", 413)
                chunks.append(chunk)
            return b"".join(chunks)

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError
                result[key] = value
            return result

        try:
            raw = await asyncio.wait_for(read_body(), timeout=5)
            value = json.loads(raw, object_pairs_hook=unique_object,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except asyncio.TimeoutError:
            raise StoreError("body_timeout", 408) from None
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise StoreError("invalid_json", 400) from None
        try:
            return model.model_validate(value)
        except ValueError:
            # Validation errors must never echo incoming bodies or bearer data.
            raise StoreError("invalid_schema", 422) from None

    async def decision_body(request: Request) -> ReviewDecision:
        return await strict_body(request, ReviewDecision)

    async def digest_body(request: Request) -> ReviewDigest:
        return await strict_body(request, ReviewDigest)

    async def reconciliation_body(request: Request) -> ReconciliationInput:
        return await strict_body(request, ReconciliationInput)

    @app.get("/review", response_class=HTMLResponse)
    def review_page():
        return HTMLResponse(page)

    @app.get("/review/assets/{name}")
    def review_asset(name: str):
        if name not in assets:
            raise StoreError("not_found", 404)
        media_type = "text/javascript" if name.endswith(".js") else "text/css"
        return Response(assets[name], media_type=media_type)

    def exact_wire(view):
        if (view.get("review_contract") != "action_and_exact_wire_v1"
                or not isinstance(view.get("wire_preview"), dict)
                or not view.get("action_digest") or not view.get("wire_preview_digest")):
            raise StoreError("exact_wire_preview_required")
        return view

    # Assign concrete dependency types here rather than postponed local aliases.
    @app.get("/review/api/proposals/{proposal_id}")
    def view_proposal(proposal_id: uuid.UUID, request: Request,
                      actor: Principal = Depends(reviewer)):
        return exact_wire(service.view(str(proposal_id), actor))

    @app.post("/review/api/proposals/{proposal_id}/decision")
    def decide_proposal(proposal_id: uuid.UUID, request: Request,
                        actor: Principal = Depends(reviewer),
                        body: ReviewDecision = Depends(decision_body)):
        with decision_lock:
            current = exact_wire(service.view(str(proposal_id), actor))
            if current["digest"] != body.digest:
                raise StoreError("digest_mismatch")
            if current["state"] != "pending" or current["execution"] is not None or current["consumed_at"] is not None:
                raise StoreError("review_no_longer_pending")
            return service.store.decide(str(proposal_id), body.digest, body.decision,
                                        actor, request.state.review_request_id)

    @app.post("/review/api/proposals/{proposal_id}/revoke")
    def revoke_proposal(proposal_id: uuid.UUID, request: Request,
                        actor: Principal = Depends(reviewer), body: ReviewDigest = Depends(digest_body)):
        return service.store.decide(str(proposal_id), body.digest, "revoke", actor, request.state.review_request_id)

    @app.post("/review/api/proposals/{proposal_id}/reconciliation")
    def reconciliation(proposal_id: uuid.UUID, request: Request,
                       actor: Principal = Depends(reviewer), body: ReconciliationInput = Depends(reconciliation_body)):
        return service.store.reconcile(str(proposal_id), body.digest, body.observed_submission,
            body.evidence_reference, actor, request.state.review_request_id)

    @app.get("/review/api/proposals/{proposal_id}/reconciliations")
    def observations(proposal_id: uuid.UUID, actor: Principal = Depends(reviewer)):
        service.view(str(proposal_id), actor)
        return {"observations": service.store.reconciliations(str(proposal_id))}

    @app.get("/review/api/audit")
    def audit(actor: Principal = Depends(reviewer)):
        return {"events": service.store.audit()}

    @app.get("/review/api/audit/after/{after}")
    def later_audit(after: int = PathParameter(ge=0, le=9223372036854775807),
                    actor: Principal = Depends(reviewer)):
        # An event sequence is nonsecret; credentials/payloads stay out of URLs.
        return {"events": service.store.audit(after)}

    if google_auth is not None:
        from .google_login_routes import install_google_login_routes
        install_google_login_routes(app, service, google_auth)
