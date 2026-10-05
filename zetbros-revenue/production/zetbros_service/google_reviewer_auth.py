"""Dormant Google GIS login and isolated short reviewer sessions.

ID tokens are accepted only by this dedicated login exchange. Google API access
and refresh tokens are never used. Disabled profiles perform no network, login,
credential generation or key reads. Sessions/challenges are bounded process-local
state and disappear on restart; they never grant agent authority.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Literal
from urllib.parse import urlsplit

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from pydantic import Field, field_validator, model_validator

from .auth import AuthenticationError, Principal, Verifier
from .models import PrincipalId, StrictModel

CHALLENGE_COOKIE = "__Secure-zetbros-login"
SESSION_COOKIE = "__Secure-zetbros-review"
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"


class GoogleReviewerProfile(StrictModel):
    enabled: bool = False
    client_id: str | None = None
    origin: str | None = None
    reviewer_subjects: tuple[PrincipalId, ...] = ()
    session_seconds: Annotated[int, Field(ge=60, le=300)] = 300
    challenge_seconds: Annotated[int, Field(ge=60, le=300)] = 180
    max_sessions: Annotated[int, Field(ge=1, le=128)] = 32
    bootstrap_per_peer_per_minute: Annotated[int, Field(ge=1, le=12)] = 8
    jwks_url: Literal["https://www.googleapis.com/oauth2/v3/certs"] = GOOGLE_JWKS_URL

    @model_validator(mode="after")
    def configuration(self):
        if not self.enabled:
            if self.client_id is not None or self.origin is not None or self.reviewer_subjects:
                raise ValueError("disabled Google profile contains no placeholder identity values")
            return self
        if not self.client_id or not re.fullmatch(r"[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com", self.client_id):
            raise ValueError("actual dedicated Google web client ID required")
        from .review_ui import validate_review_origin
        if not self.origin or validate_review_origin(self.origin) != self.origin or urlsplit(self.origin).scheme != "https":
            raise ValueError("exact HTTPS reviewer origin required")
        if not self.reviewer_subjects or len(set(self.reviewer_subjects)) != len(self.reviewer_subjects) or len(self.reviewer_subjects) > 32:
            raise ValueError("bounded explicit Google subject allowlist required")
        if any(not sub.isascii() or "@" in sub or "REPLACE" in sub.upper() for sub in self.reviewer_subjects):
            raise ValueError("Google sub identifiers, never email addresses, are required")
        return self


class AgentOnlyVerifier:
    """Existing own-audience access JWT interface; Google ID tokens never qualify."""
    def __init__(self, verifier): self.verifier = verifier
    def verify(self, authorization):
        actor = self.verifier.verify(authorization)
        if actor.role != "agent": raise AuthenticationError()
        return actor


class GoogleIdTokenVerifier:
    def __init__(self, profile, key_source, *, clock=time.time):
        self.profile, self.key_source, self.clock = profile, key_source, clock
        self.lock = threading.Lock(); self.keys = {}; self.loaded_at = 0

    def public_keys(self):
        with self.lock:
            now = int(self.clock())
            if self.keys and 0 <= now - self.loaded_at < 300: return self.keys
            try:
                data = self.key_source.get_jwks(GOOGLE_JWKS_URL)
                if type(data) is not dict or set(data) != {"keys"} or type(data['keys']) is not list or not 1 <= len(data['keys']) <= 10:
                    raise ValueError
                keys = {}
                for item in data['keys']:
                    if (type(item) is not dict or item.get('kty') != 'RSA' or item.get('alg') != 'RS256'
                            or item.get('use') != 'sig' or item.get('key_ops',['verify']) != ['verify']
                            or any(k in item for k in ('d','p','q','dp','dq','qi','oth','k'))): raise ValueError
                    kid = item.get('kid')
                    if type(kid) is not str or not 1 <= len(kid) <= 96 or kid in keys: raise ValueError
                    key = jwt.PyJWK.from_dict(item, algorithm='RS256').key
                    if not isinstance(key,RSAPublicKey) or key.key_size < 2048: raise ValueError
                    keys[kid] = key
                self.keys, self.loaded_at = keys, now
                return keys
            except Exception:
                raise AuthenticationError() from None

    def verify_login(self, credential, nonce):
        try:
            if type(credential) is not str or not 1 <= len(credential) <= 8192: raise ValueError
            header = jwt.get_unverified_header(credential)
            if (header.get('alg') != 'RS256' or header.get('typ') not in (None,'JWT')
                    or set(header) - {'alg','kid','typ'} or type(header.get('kid')) is not str): raise ValueError
            key = self.public_keys().get(header['kid'])
            if key is None: raise ValueError  # No token-controlled key URLs or unbounded refresh loops.
            claims = jwt.decode(credential,key,algorithms=['RS256'],audience=self.profile.client_id,
                issuer=('https://accounts.google.com','accounts.google.com'),
                options={'require':['iss','aud','sub','exp','iat','nonce']})
            now = int(self.clock())
            if (claims['aud'] != self.profile.client_id or type(claims['aud']) is not str
                    or type(claims['sub']) is not str or claims['sub'] not in self.profile.reviewer_subjects
                    or type(claims['exp']) is not int or type(claims['iat']) is not int
                    or not claims['iat'] <= now < claims['exp'] or not 0 < claims['exp']-claims['iat'] <= 3600
                    or claims.get('azp',self.profile.client_id) != self.profile.client_id
                    or type(claims['nonce']) is not str or not hmac.compare_digest(claims['nonce'],nonce)):
                raise ValueError
            return claims['sub']
        except Exception:
            raise AuthenticationError() from None


@dataclass(frozen=True)
class Challenge:
    nonce: str
    csrf: str
    issued: int
    expires: int


@dataclass(frozen=True)
class Session:
    actor: Principal
    csrf: str
    issued: int
    expires: int


class GoogleReviewerAuth:
    def __init__(self, profile, settings, key_source, *, clock=time.time):
        self.profile = GoogleReviewerProfile.model_validate_json(profile.model_dump_json())
        if not self.profile.enabled: raise ValueError("Google reviewer login is disabled")
        if settings.review_origin != profile.origin: raise ValueError("review origin mismatch")
        if settings.issuer in ('https://accounts.google.com','accounts.google.com') or settings.audience==profile.client_id:
            raise ValueError("agent access credentials need a separate issuer/resource audience")
        reviewer_pairs={(g.subject,g.client_id) for g in settings.principals if g.role=='reviewer'}
        expected={(sub,profile.client_id) for sub in profile.reviewer_subjects}
        if reviewer_pairs != expected: raise ValueError("exact dedicated Google reviewer mapping required")
        if any(g.client_id==profile.client_id for g in settings.principals if g.role=='agent'):
            raise ValueError("Google reviewer client cannot be an agent client")
        self.settings, self.clock = settings, clock
        self.id_tokens = GoogleIdTokenVerifier(self.profile,key_source,clock=clock)
        self.lock = threading.Lock(); self.challenges = {}; self.sessions = {}; self.bootstrap_rates = {}

    @staticmethod
    def key(value):
        if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{43}',value): raise AuthenticationError()
        return hashlib.sha256(value.encode('ascii')).hexdigest()

    def prune(self):
        now=int(self.clock())
        self.challenges={key:value for key,value in self.challenges.items() if value.issued <= now < value.expires}
        self.sessions={key:value for key,value in self.sessions.items() if value.issued <= now < value.expires}

    def challenge(self, previous_cookie=None, peer=None):
        with self.lock:
            self.prune()
            if peer is not None:
                if type(peer) is not str or not 1 <= len(peer) <= 128: raise AuthenticationError()
                bucket=int(self.clock())//60
                self.bootstrap_rates={key:value for key,value in self.bootstrap_rates.items() if value[0]==bucket}
                key=hashlib.sha256(peer.encode('utf-8')).hexdigest()
                old=self.bootstrap_rates.get(key,(bucket,0))
                if old[1]>=self.profile.bootstrap_per_peer_per_minute or (key not in self.bootstrap_rates and len(self.bootstrap_rates)>=128):
                    raise AuthenticationError()
                self.bootstrap_rates[key]=(bucket,old[1]+1)
            if previous_cookie is not None:
                try:self.challenges.pop(self.key(previous_cookie),None)
                except AuthenticationError:pass
            if len(self.challenges)>=self.profile.max_sessions: raise AuthenticationError()
            token=secrets.token_urlsafe(32);nonce=secrets.token_urlsafe(32);csrf=secrets.token_urlsafe(32)
            now=int(self.clock())
            self.challenges[self.key(token)]=Challenge(nonce,csrf,now,now+self.profile.challenge_seconds)
            return token,nonce,csrf

    def login(self, challenge_cookie, csrf, credential):
        key=self.key(challenge_cookie)
        with self.lock:
            self.prune();challenge=self.challenges.pop(key,None)
        if challenge is None or type(csrf) is not str or not hmac.compare_digest(challenge.csrf,csrf):
            raise AuthenticationError()
        sub=self.id_tokens.verify_login(credential,challenge.nonce)
        actor=Principal(sub,self.profile.client_id,'reviewer')
        if self.settings.role(actor.subject,actor.client_id)!='reviewer': raise AuthenticationError()
        with self.lock:
            self.prune()
            self.sessions={key:value for key,value in self.sessions.items() if value.actor!=actor}
            if len(self.sessions)>=self.profile.max_sessions: raise AuthenticationError()
            token=secrets.token_urlsafe(32);csrf=secrets.token_urlsafe(32)
            now=int(self.clock())
            self.sessions[self.key(token)]=Session(actor,csrf,now,now+self.profile.session_seconds)
            return token,csrf

    def session(self, cookie, csrf=None):
        key=self.key(cookie)
        with self.lock:
            self.prune();session=self.sessions.get(key)
        if session is None or self.settings.role(session.actor.subject,session.actor.client_id)!='reviewer':
            raise AuthenticationError()
        if csrf is not None and (type(csrf) is not str or not hmac.compare_digest(session.csrf,csrf)):
            raise AuthenticationError()
        return session

    def logout(self,cookie,csrf):
        if csrf is None: raise AuthenticationError()
        self.session(cookie,csrf)
        with self.lock:self.sessions.pop(self.key(cookie),None)
