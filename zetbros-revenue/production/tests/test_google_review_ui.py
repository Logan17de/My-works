"""Offline Google-mode DOM state checks; no Google/provider/token network flow.

All identity strings, credentials, and proposal content below are fictional.
Google Identity Services and cookie sessions are mocked in a Node VM; these
checks do not claim browser pixels or actual OAuth provider compatibility.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from zetbros_service.review_ui import ASSETS

ORIGIN = "https://review.example.invalid"
PROPOSAL = {
    "id": "00000000-0000-0000-0000-000000000001",
    "digest": "a" * 64,
    "action_digest": "b" * 64,
    "wire_preview_digest": "c" * 64,
    "review_contract": "action_and_exact_wire_v1",
    "state": "pending",
    "execution": None,
    "consumed_at": None,
    "expires_at": 4102444800,
    "payload": {
        "body": '  Fictional exact body\r\n\t<script>FICTIONAL_CANARY</script>  \n',
        "source_id": "fictional-source", "sender": "sender@example.invalid",
        "to": ["recipient@example.invalid"], "subject": "Fictional reply",
    },
    "wire_preview": {"body": '  Fictional exact body\r\n\t<script>FICTIONAL_CANARY</script>  \n'},
}


class GoogleReviewUiTests(unittest.TestCase):
    def test_google_template_keeps_script_loading_out_of_the_public_shell(self):
        html = (ASSETS / "review.html").read_text()
        self.assertIn('data-auth-mode="REVIEW_AUTH_MODE"', html)
        self.assertIn('id="google-session" hidden', html)
        self.assertIn('id="google-button"', html)
        self.assertIn('id="google-retry"', html)
        self.assertNotIn("accounts.google.com", html)
        self.assertIn('id="session-form"', html)
        self.assertIn('id="reviewer-token" type="password"', html)

    def test_google_script_never_persists_or_renders_identity_credentials(self):
        script = (ASSETS / "review.js").read_text()
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "localStorage",
                          "sessionStorage", "document.cookie", "console.", "eval(", "new Function"):
            self.assertNotIn(forbidden, script)
        self.assertIn('script.src = "https://accounts.google.com/gsi/client"', script)
        self.assertIn('"X-Login-CSRF": loginCsrf', script)
        self.assertIn('headers["X-Review-CSRF"] = reviewCsrf', script)
        self.assertIn('JSON.stringify({credential: result.credential})', script)
        self.assertNotIn("client_secret", script)
        self.assertNotIn("access_token", script)
        self.assertNotIn("refresh_token", script)

    def test_socket_free_google_login_session_and_interrupted_review_flows(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node unavailable; no installation attempted")
        completed = subprocess.run([node, str(Path(__file__).with_name("google_review_state_check.cjs"))],
            input=json.dumps({"assets": str(ASSETS), "origin": ORIGIN, "proposal": PROPOSAL}),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["result"], "google socket-free state checks passed")
        self.assertGreaterEqual(len(result["scenarios"]), 21)


if __name__ == "__main__":
    unittest.main()
