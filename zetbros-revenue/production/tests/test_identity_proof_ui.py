"""Standalone identity-proof UI tests using only offline DOM/GIS/ASGI doubles.

Fictional public identifiers and credentials are never provider account data.
No listener, Google endpoint, OAuth grant, review session, or ledger is used.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from zetbros_service.google_identity_proof import ASSETS, IdentityProofProfile, create_identity_proof_app

ORIGIN = "https://identity.example.invalid"


class NoProviderKeys:
    def get_jwks(self, url):
        raise AssertionError("Provider key fetch forbidden in static UI checks")


class IdentityProofUiTests(unittest.TestCase):
    def test_shell_is_separate_and_never_contains_google_credentials(self):
        html = (ASSETS / "identity.html").read_text()
        self.assertIn('data-origin="IDENTITY_ORIGIN_PIN"', html)
        self.assertIn('src="/identity/assets/identity.js"', html)
        self.assertIn('id="proof-subject"', html)
        self.assertIn("Authority: none", html)
        self.assertNotIn("accounts.google.com", html)
        for forbidden in ("reviewer-token", "/review/", "Authorization", "onload=", "onclick="):
            self.assertNotIn(forbidden, html)
        self.assertIn("[hidden]", (ASSETS / "identity.css").read_text())

    def test_script_has_no_storage_token_output_or_application_routes(self):
        script = (ASSETS / "identity.js").read_text()
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "localStorage", "sessionStorage",
                          "document.cookie", "console.", "eval(", "new Function", "Authorization", "/review/",
                          "/v1/", "client_secret", "access_token", "refresh_token"):
            self.assertNotIn(forbidden, script)
        self.assertIn('script.src = "https://accounts.google.com/gsi/client"', script)
        self.assertIn('"X-Proof-CSRF": csrf', script)
        self.assertIn('JSON.stringify({credential: response.credential})', script)
        self.assertIn('subject.textContent = value.subject', script)
        self.assertIn('value.authority !== "none"', script)
        self.assertNotIn('textContent = response.credential', script)

    def test_new_shell_assets_are_served_without_starting_a_proof(self):
        profile = IdentityProofProfile(enabled=True, client_id="12345-fictional.apps.googleusercontent.com", origin=ORIGIN)
        with patch("socket.create_connection", side_effect=AssertionError("Network forbidden")):
            app = create_identity_proof_app(profile, key_source=NoProviderKeys())
            with TestClient(app, base_url=ORIGIN) as client:
                for path in ("/identity", "/identity/assets/identity.js", "/identity/assets/identity.css"):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["cache-control"], "no-store")
                    self.assertEqual(response.headers["referrer-policy"], "no-referrer")
                    self.assertNotIn("set-cookie", response.headers)
                html = client.get("/identity").text
                self.assertIn('data-origin="' + ORIGIN + '"', html)
                self.assertNotIn("IDENTITY_ORIGIN_PIN", html)
                self.assertEqual(app.state.identity_proof.challenges, {})
                self.assertFalse(hasattr(app.state.identity_proof, "sessions"))

    def test_socket_free_identity_proof_and_interrupted_flows(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node unavailable; no installation attempted")
        completed = subprocess.run([node, str(Path(__file__).with_name("identity_proof_state_check.cjs"))],
            input=json.dumps({"assets": str(ASSETS), "origin": ORIGIN}), capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["result"], "identity socket-free state checks passed")
        self.assertGreaterEqual(len(report["scenarios"]), 19)


if __name__ == "__main__":
    unittest.main()
