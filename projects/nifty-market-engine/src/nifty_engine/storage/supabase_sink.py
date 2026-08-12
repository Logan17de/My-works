from __future__ import annotations

import os
from ..models import Signal
from ..serialization import to_primitive


class SupabaseSignalSink:
    def __init__(self, url: str, service_role_key: str) -> None:
        from supabase import create_client
        self.client = create_client(url, service_role_key)

    @classmethod
    def from_env(cls) -> "SupabaseSignalSink":
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        return cls(url, key)

    def write_signal(self, signal: Signal) -> None:
        payload = to_primitive(signal)
        self.client.table("signals").insert({
            "observed_at": payload["timestamp"],
            "event": payload["event"],
            "direction": payload["direction"],
            "confidence": payload["confidence"],
            "combined_direction_score": payload["combined_direction_score"],
            "payload": payload,
        }).execute()
