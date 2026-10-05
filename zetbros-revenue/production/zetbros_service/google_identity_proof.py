"""Standalone Google identity proof: no enrollment, service, ledger or authority.

The caller sees only its cryptographically verified immutable subject. Proof
never creates a reviewer/agent session or modifies configuration. Real Google
I/O is dormant until this separately configured app receives an approved login.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import Field, model_validator

from .auth import AuthenticationError
from .google_login_routes import unique_cookie
from .google_public_keys import GooglePublicKeys
from .google_reviewer_auth import GoogleIdTokenVerifier, GOOGLE_JWKS_URL
from .models import StrictModel
from .review_ui import validate_review_origin, CSP

COOKIE = '__Secure-zetbros-identity-proof'
ASSETS = Path(__file__).with_name('identity_assets')


class IdentityProofProfile(StrictModel):
    runtime_state: Literal['identity_proof_only'] = 'identity_proof_only'
    enabled: bool = False
    client_id: str | None = None
    origin: str | None = None
    challenge_seconds: Annotated[int, Field(ge=60, le=180)] = 120
    max_challenges: Annotated[int, Field(ge=1, le=32)] = 16
    bootstrap_per_peer_per_minute: Annotated[int, Field(ge=1, le=8)] = 4
    # None explicitly opts into continuous page availability, not authority.
    capture_window_seconds: Annotated[int, Field(ge=60, le=600)] | None = 600
    jwks_url: Literal['https://www.googleapis.com/oauth2/v3/certs'] = GOOGLE_JWKS_URL

    @model_validator(mode='after')
    def explicit_configuration(self):
        if not self.enabled:
            if self.client_id is not None or self.origin is not None:
                raise ValueError('disabled proof contains no placeholder identity')
        elif (not self.client_id or not re.fullmatch(r'[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com', self.client_id)
              or not self.origin or not self.origin.startswith('https://')
              or validate_review_origin(self.origin) != self.origin):
            raise ValueError('actual dedicated web client and canonical HTTPS origin required')
        return self


@dataclass(frozen=True)
class ProofChallenge:
    nonce: str
    csrf: str
    issued: int
    expires: int
    issued_monotonic: float
    expires_monotonic: float


class IdentityProof:
    """Bounded short-lived challenge state only, never authentication sessions."""
    def __init__(self, profile, *, key_source=None, clock=time.time, monotonic=time.monotonic):
        self.profile = IdentityProofProfile.model_validate_json(profile.model_dump_json())
        if not self.profile.enabled:
            raise ValueError('identity proof is disabled')
        self.clock = clock
        self.started = int(clock())
        self.monotonic, self.started_monotonic = monotonic, monotonic()
        self.window_closed = False
        self.verifier = GoogleIdTokenVerifier(self.profile,
            key_source if key_source is not None else GooglePublicKeys(), clock=clock)
        self.lock = threading.Lock()
        self.challenges = {}
        self.rates = {}

    @staticmethod
    def key(value):
        if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{43}', value):
            raise AuthenticationError()
        return hashlib.sha256(value.encode('ascii')).hexdigest()

    def challenge_current(self, challenge):
        return (challenge.issued <= int(self.clock()) < challenge.expires
                and challenge.issued_monotonic <= self.monotonic() < challenge.expires_monotonic)

    def prune(self):
        self.challenges = {key: value for key, value in self.challenges.items()
                           if self.challenge_current(value)}

    def require_window(self):
        if self.window_closed:
            raise AuthenticationError()
        seconds = self.profile.capture_window_seconds
        if seconds is None:
            return
        if (not self.started <= int(self.clock()) < self.started + seconds
                or not 0 <= self.monotonic() - self.started_monotonic < seconds):
            self.window_closed = True  # Clock correction cannot reopen a closed capture.
            raise AuthenticationError()

    def bootstrap(self, peer, previous=None):
        self.require_window()
        if type(peer) is not str or not 1 <= len(peer) <= 128:
            raise AuthenticationError()
        with self.lock:
            self.prune()
            bucket = int(self.clock()) // 60
            self.rates = {key: value for key, value in self.rates.items() if value[0] == bucket}
            key = hashlib.sha256(peer.encode('utf-8')).hexdigest()
            count = self.rates.get(key, (bucket, 0))[1]
            if count >= self.profile.bootstrap_per_peer_per_minute or (key not in self.rates and len(self.rates) >= 64):
                raise AuthenticationError()
            self.rates[key] = (bucket, count + 1)
            if previous is not None:
                self.challenges.pop(self.key(previous), None)
            if len(self.challenges) >= self.profile.max_challenges:
                raise AuthenticationError()
            cookie, nonce, csrf = (secrets.token_urlsafe(32) for _ in range(3))
            now = int(self.clock())
            elapsed = self.monotonic()
            self.challenges[self.key(cookie)] = ProofChallenge(nonce, csrf, now,
                now + self.profile.challenge_seconds, elapsed, elapsed + self.profile.challenge_seconds)
            return cookie, nonce, csrf

    def consume(self, cookie, csrf):
        self.require_window()
        key = self.key(cookie)
        with self.lock:
            self.prune()
            challenge = self.challenges.pop(key, None)
        if (challenge is None or type(csrf) is not str
                or not re.fullmatch(r'[A-Za-z0-9_-]{43}', csrf)
                or not hmac.compare_digest(challenge.csrf, csrf)):
            raise AuthenticationError()
        return challenge

    def prove(self, cookie, csrf, credential):
        challenge = self.consume(cookie, csrf)
        # No claims or token retained, and no role/allowlist/session side effect.
        subject = self.verifier.verify_identity(credential, challenge.nonce)
        self.require_window()
        if not self.challenge_current(challenge):
            raise AuthenticationError()
        return subject


def create_identity_proof_app(profile, *, key_source=None, clock=time.time, monotonic=time.monotonic):
    proof = IdentityProof(profile, key_source=key_source, clock=clock, monotonic=monotonic)
    origin = proof.profile.origin
    page = (ASSETS / 'identity.html').read_text().replace('IDENTITY_ORIGIN_PIN', escape(origin, quote=True))
    assets = {name: (ASSETS / name).read_bytes() for name in ('identity.js', 'identity.css')}
    csp = CSP.replace("script-src 'self'", "script-src 'self' https://accounts.google.com/gsi/client")
    csp = csp.replace("connect-src 'self'", "connect-src 'self' https://accounts.google.com/gsi/")
    csp = csp.replace("style-src 'self'", "style-src 'self' https://accounts.google.com/gsi/style")
    csp += '; frame-src https://accounts.google.com/gsi/'

    @asynccontextmanager
    async def lifespan(app):
        import anyio
        anyio.to_thread.current_default_thread_limiter().total_tokens = 2
        yield
        with proof.lock:
            proof.challenges.clear()
            proof.rates.clear()

    app = FastAPI(title='Zetbros identity proof only', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.identity_proof = proof

    def failure(status=401, code='identity_proof_failed'):
        response = JSONResponse({'error': code, 'authority': 'none'}, status_code=status)
        response.delete_cookie(COOKIE, path='/identity', secure=True, httponly=True, samesite='lax')
        return response

    @app.middleware('http')
    async def boundary(request: Request, call_next):
        response = None
        try:
            proof.require_window()
        except AuthenticationError:
            with proof.lock:
                proof.challenges.clear()
                proof.rates.clear()
            response = failure(503, 'identity_capture_window_closed')
        if response is not None:
            pass
        elif request.query_params:
            response = failure(400, 'identity_queries_forbidden')
        elif request.headers.get('authorization') is not None:
            response = failure(403)
        elif request.url.path == '/identity' or request.url.path.startswith('/identity/'):
            if (str(request.base_url).rstrip('/') != origin
                    or len(request.headers.getlist('host')) != 1
                    or len(request.headers.getlist('origin')) > 1
                    or request.headers.get('origin') not in (None, origin)
                    or (request.url.path != '/identity'
                        and request.headers.get('sec-fetch-site') not in (None, 'none', 'same-origin'))
                    or request.method == 'OPTIONS'
                    or (request.method == 'POST' and request.headers.get('origin') != origin)):
                response = failure(403)
        if response is None:
            response = await call_next(request)
        response.headers.update({'Cache-Control': 'no-store', 'Pragma': 'no-cache',
            'Content-Security-Policy': csp, 'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY', 'Referrer-Policy': 'no-referrer',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
            'Cross-Origin-Resource-Policy': 'same-origin'})
        return response

    @app.get('/healthz')
    def health():
        return {'status': 'up', 'runtime_state': 'identity_proof_only', 'authority': 'none'}

    @app.get('/readyz')
    def ready():
        return JSONResponse({'control_ready': False, 'live_delivery_ready': False,
            'runtime_state': 'identity_proof_only', 'authority': 'none'}, status_code=503)

    @app.get('/identity', response_class=HTMLResponse)
    def shell():
        return HTMLResponse(page)

    @app.get('/identity/assets/{name}')
    def asset(name: str):
        if name not in assets:
            return failure(404)
        return Response(assets[name], media_type='text/javascript' if name.endswith('.js') else 'text/css')

    @app.get('/identity/bootstrap')
    def bootstrap(request: Request):
        try:
            if request.client is None:
                raise AuthenticationError()
            previous = unique_cookie(request, COOKIE) if request.cookies.get(COOKIE) is not None else None
            cookie, nonce, csrf = proof.bootstrap(request.client.host, previous)
            response = JSONResponse({'client_id': proof.profile.client_id, 'nonce': nonce, 'csrf': csrf})
            response.set_cookie(COOKIE, cookie, max_age=proof.profile.challenge_seconds,
                path='/identity', secure=True, httponly=True, samesite='lax')
            return response
        except AuthenticationError:
            return failure()

    def csrf(request):
        if len(request.headers.getlist('x-proof-csrf')) != 1:
            raise AuthenticationError()
        return request.headers['x-proof-csrf']

    @app.post('/identity/proof')
    async def verify(request: Request):
        try:
            if request.headers.get('content-type', '').split(';')[0].strip() != 'application/json':
                raise AuthenticationError()
            cookie, submitted_csrf = unique_cookie(request, COOKIE), csrf(request)
            chunks = []
            size = 0
            async def read_body():
                nonlocal size
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 12288:
                        raise AuthenticationError()
                    chunks.append(chunk)
                return b''.join(chunks)
            raw = await asyncio.wait_for(read_body(), timeout=5)
            def unique(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise AuthenticationError()
                    value[key] = item
                return value
            value = json.loads(raw, object_pairs_hook=unique,
                parse_constant=lambda _: (_ for _ in ()).throw(AuthenticationError()))
            if type(value) is not dict or set(value) != {'credential'}:
                raise AuthenticationError()
            subject = await asyncio.to_thread(proof.prove, cookie, submitted_csrf, value['credential'])
            response = JSONResponse({'verified': True, 'subject': subject, 'authority': 'none'})
            response.delete_cookie(COOKIE, path='/identity', secure=True, httponly=True, samesite='lax')
            return response
        except Exception:
            return failure()

    @app.post('/identity/cancel')
    def cancel(request: Request):
        try:
            proof.consume(unique_cookie(request, COOKIE), csrf(request))
            response = JSONResponse({'cancelled': True, 'authority': 'none'})
            response.delete_cookie(COOKIE, path='/identity', secure=True, httponly=True, samesite='lax')
            return response
        except AuthenticationError:
            return failure()

    @app.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
    def unavailable(path: str):
        return failure(503, 'identity_proof_only_no_application_authority')

    return app


def load_identity_proof_profile():
    path = os.environ.get('ZETBROS_IDENTITY_PROOF_CONFIG_FILE')
    try:
        if not path or not Path(path).is_absolute():
            raise ValueError
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= 16384:
                raise ValueError
            profile = IdentityProofProfile.model_validate_json(handle.read(16385))
        if not profile.enabled:
            raise ValueError
        return profile
    except Exception:
        raise RuntimeError('explicit enabled identity-proof-only configuration required') from None


def configured_identity_proof_app():
    return create_identity_proof_app(load_identity_proof_profile())
