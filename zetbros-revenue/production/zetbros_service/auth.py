"""Verify existing external access tokens. This service never issues credentials."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from .config import Settings


class AuthenticationError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    subject: str
    client_id: str
    role: str


class Verifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.keys = {}
        try:
            raw = Path(settings.jwks_file).read_bytes()
            if len(raw) > 65536:
                raise ValueError("key set too large")
            data = json.loads(raw)
            if set(data) != {"keys"} or not isinstance(data["keys"], list) or not 1 <= len(data["keys"]) <= 10:
                raise ValueError("invalid key set")
            for key in data["keys"]:
                if key.get("kty") != "RSA" or key.get("alg") != "RS256" or key.get("use") != "sig":
                    raise ValueError("RS256 signature public keys required")
                if any(k in key for k in ("d", "p", "q", "dp", "dq", "qi", "oth", "k")) or key.get("key_ops", ["verify"]) != ["verify"]:
                    raise ValueError("private or nonverification key forbidden")
                kid = key.get("kid")
                if not isinstance(kid, str) or not 1 <= len(kid) <= 96 or kid in self.keys:
                    raise ValueError("unique bounded key ID required")
                public = jwt.PyJWK.from_dict(key, algorithm="RS256").key
                if not isinstance(public, RSAPublicKey) or public.key_size < 2048:
                    raise ValueError("RSA key too small")
                self.keys[kid] = public
        except (OSError, ValueError, jwt.PyJWTError, TypeError, AttributeError) as exc:
            raise RuntimeError("trusted pinned JWKS missing or invalid") from exc

    def verify(self, authorization: str | None) -> Principal:
        try:
            if not authorization or not authorization.startswith("Bearer ") or len(authorization) > 16384:
                raise AuthenticationError()
            token = authorization[7:]
            header = jwt.get_unverified_header(token)
            if set(header) - {"alg", "kid", "typ"} or header.get("alg") != "RS256" or header.get("typ") != self.settings.token_header_type:
                raise AuthenticationError()
            key = self.keys.get(header.get("kid"))
            if key is None:
                raise AuthenticationError()
            client_claim = self.settings.token_client_claim
            claims = jwt.decode(token, key, algorithms=["RS256"], issuer=self.settings.issuer, audience=self.settings.audience,
                                options={"require": ["iss", "aud", "exp", "nbf", "iat", "sub", client_claim], "strict_aud": True}, leeway=0)
            now = int(time.time())
            if any(type(claims[c]) is not int for c in ("iat", "nbf", "exp")):
                raise AuthenticationError()
            if claims["exp"] <= now or claims["iat"] > now or claims["nbf"] > now or claims["exp"] <= claims["iat"] or claims["exp"] - claims["iat"] > self.settings.token_max_lifetime_seconds:
                raise AuthenticationError()
            subject, client_id = claims["sub"], claims[client_claim]
            if not isinstance(subject, str) or not isinstance(client_id, str):
                raise AuthenticationError()
            role = self.settings.role(subject, client_id)
            if role is None:
                raise AuthenticationError()
            # Scope/role/tenant/body/header claims cannot upgrade this deployment map.
            return Principal(subject, client_id, role)
        except (jwt.PyJWTError, ValueError, TypeError, KeyError) as exc:
            raise AuthenticationError() from exc
