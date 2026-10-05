"""Socket-free ASGI and locally mocked browser-engine review checks.

All source, addresses, keys, and JWTs are ephemeral fictional test fixtures.
No service listener, provider, external page, or real bearer is contacted.
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from test_mail_integration import FictionalOwner
from test_spacemail_contract import BINDING, PLAN
from zetbros_service.api import create_app
from zetbros_service.auth import Principal
from zetbros_service.config import Grant, Settings
from zetbros_service.mail_integration import MailService, MailStore, review_digest
from zetbros_service.mail_owner import OwnerMailSource
from zetbros_service.models import ProposalInput, ReplyAction, canonical, digest
from zetbros_service.review_ui import ASSETS, CSP, install_review_ui, validate_review_origin

ORIGIN = "https://review.example.invalid"
AGENT = Principal("review-ui-test-agent", "agent-client", "agent")
REVIEWER = Principal("review-ui-test-human", "review-client", "reviewer")
BODY = '  Exact reply\r\n\t<script>window.pwned=true</script>\nDo not send this as HTML.  \n'


class ReviewUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key()))
        cls.jwk.update(kid="review-ui-fictional", alg="RS256", use="sig", key_ops=["verify"])

    def setUp(self):
        self.socket_guard = patch("socket.create_connection", side_effect=AssertionError("socket forbidden"))
        self.socket_guard.start()
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "data").mkdir(); (root / "sources").mkdir()
        (root / "jwks.json").write_text(json.dumps({"keys": [self.jwk]}))
        self.now = int(time.time())
        self.settings = Settings(tenant_id=BINDING.tenant_id, connector_id=BINDING.connector_id,
            account_id=BINDING.account_id, sender_address=BINDING.sender_address,
            policy_version=BINDING.policy_version, issuer="https://identity.example.invalid",
            audience="review-ui-fictional", jwks_file=str(root / "jwks.json"),
            database_path=str(root / "data" / "state.sqlite3"), source_directory=str(root / "sources"),
            principals=(Grant(subject=AGENT.subject, client_id=AGENT.client_id, role="agent"),
                        Grant(subject=REVIEWER.subject, client_id=REVIEWER.client_id, role="reviewer")),
            requests_per_minute=600)
        self.owner = FictionalOwner()
        self.source = OwnerMailSource(BINDING, self.owner)
        MailStore.initialize_offline(self.settings, clock=lambda: self.now, source=self.source)
        self.service = MailService(self.settings, source=self.source, clock=lambda: self.now)
        self.app = create_app(self.service, run_worker=False)
        install_review_ui(self.app, self.service, ORIGIN)
        self.client = TestClient(self.app, base_url=ORIGIN)
        source = self.source.get(PLAN.source_id)
        self.proposal = self.service.propose(ProposalInput(operation_key="review-ui-operation-0001",
            source_id=source.source_id, source_version=source.version, source_fingerprint=digest(source),
            body=BODY), AGENT, str(uuid.uuid4()))
        self.path = "/review/api/proposals/" + self.proposal["id"]

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()
        self.socket_guard.stop()

    def token(self, actor=REVIEWER, **claims):
        now = int(time.time())
        payload = dict(iss=self.settings.issuer, aud=self.settings.audience, sub=actor.subject,
                       azp=actor.client_id, iat=now-1, nbf=now-1, exp=now+200)
        payload.update(claims)
        return jwt.encode(payload, self.key, algorithm="RS256",
                          headers={"kid": "review-ui-fictional", "typ": "at+jwt"})

    def headers(self, actor=REVIEWER, **extra):
        return {"Authorization": "Bearer " + self.token(actor), "Origin": ORIGIN} | extra

    def decide(self, **changes):
        return self.client.post(self.path + "/decision", headers=self.headers(),
            json={"digest": self.proposal["digest"], "decision": "approve"} | changes)

    def test_pin_rejects_paths_credentials_default_ports_and_foreign_http(self):
        for origin in ("https://review.example.invalid", "https://review.example.invalid:8443",
                       "http://localhost:8081", "http://127.0.0.1:8081", "http://[::1]:8081"):
            self.assertEqual(validate_review_origin(origin), origin)
        for origin in ("", "https://review.example.invalid/", "https://review.example.invalid/path",
                       "https://user@review.example.invalid", "https://review.example.invalid?token=a",
                       "https://review.example.invalid#token", "https://Review.example.invalid",
                       "https://review.example.invalid:443", "http://review.example.invalid",
                       "https://review.example.invalid:0", "https://review.example.invalid:",
                       "https://review.example.invalid\\evil", "https://review.example.invalid\n"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                validate_review_origin(origin)

    def test_page_and_local_assets_have_csp_no_store_and_no_credentials(self):
        for path in ("/review", "/review/assets/review.js", "/review/assets/review.css"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(response.headers["content-security-policy"], CSP)
            self.assertEqual(response.headers["referrer-policy"], "no-referrer")
            self.assertEqual(response.headers["x-frame-options"], "DENY")
            self.assertNotIn("access-control-allow-origin", response.headers)
            self.assertNotIn("set-cookie", response.headers)
        html = self.client.get("/review").text
        self.assertIn('type="password"', html)
        self.assertIn('data-review-origin="' + ORIGIN + '"', html)
        self.assertNotIn(self.proposal["digest"], html)
        self.assertEqual(self.client.get("/review/assets/unknown.js").status_code, 404)
        with self.assertRaises(ValueError):
            install_review_ui(self.app, self.service, ORIGIN)

    def test_authenticated_preview_is_complete_composite_and_preserves_body(self):
        response = self.client.get(self.path, headers=self.headers())
        self.assertEqual(response.status_code, 200)
        shown = response.json()
        self.assertEqual(shown, self.service.store.get(self.proposal["id"]))
        self.assertEqual(shown["payload"]["body"], BODY)
        self.assertEqual(shown["wire_preview"]["body"], BODY)
        self.assertEqual(shown["review_contract"], "action_and_exact_wire_v1")
        self.assertNotEqual(shown["digest"], shown["action_digest"])
        self.assertNotEqual(shown["digest"], shown["wire_preview_digest"])
        action = ReplyAction.model_validate_json(canonical(shown["payload"]))
        with self.service.store.connection() as conn:
            self.assertEqual(shown["digest"], review_digest(action,
                self.service.store.prepared(conn, self.proposal["id"], action).preview))
        for field in ("tenant_id", "connector_id", "account_id", "source_id", "source_version",
                      "source_fingerprint", "sender", "to", "body", "subject", "policy_version"):
            self.assertIn(field, shown["payload"])
        # Chromium may omit Origin from same-origin GET; the pinned request URL
        # and bearer still apply. POST has a mandatory Origin.
        self.assertEqual(self.client.get(self.path, headers={"Authorization": self.headers()["Authorization"],
                         "Sec-Fetch-Site": "same-origin"}).status_code, 200)

    def test_public_shell_cross_site_navigation_is_allowed_but_api_is_not(self):
        response=self.client.get('/review',headers={'Sec-Fetch-Site':'cross-site'})
        self.assertEqual(response.status_code,200)
        self.assertIn("frame-ancestors 'none'",response.headers['content-security-policy'])
        headers={'Authorization':'Bearer '+self.token(),'Sec-Fetch-Site':'cross-site'}
        self.assertEqual(self.client.get(self.path,headers=headers).status_code,403)
        self.assertEqual(self.client.post(self.path+'/decision',headers=headers|{'Origin':ORIGIN},
            json={'digest':self.proposal['digest'],'decision':'approve'}).status_code,403)

    def test_auth_requires_verified_existing_reviewer_mapping(self):
        self.assertEqual(self.client.get(self.path).status_code, 401)
        self.assertEqual(self.client.get(self.path, headers={"Authorization": "Bearer invalid"}).status_code, 401)
        self.assertEqual(self.client.get(self.path, headers=self.headers(AGENT, **{"X-Role": "reviewer"})).status_code, 403)
        for claims in ({"role": "reviewer", "scope": "approve"}, {"aud": "wrong"}, {"sub": "unmapped"}, {"azp": AGENT.client_id}):
            actor = AGENT if "role" in claims else REVIEWER
            response = self.client.get(self.path, headers={"Authorization": "Bearer " + self.token(actor, **claims), "Origin": ORIGIN})
            self.assertIn(response.status_code, (401, 403))
        duplicate = [("Authorization", self.headers()["Authorization"]), ("Authorization", self.headers()["Authorization"])]
        self.assertEqual(self.client.get(self.path, headers=duplicate).status_code, 401)
        self.assertEqual(self.service.store.get(self.proposal["id"])["state"], "pending")

    def test_cookies_urls_foreign_origin_host_fetch_and_cors_fail_closed(self):
        for headers in (self.headers(Cookie="session=not-authority"), self.headers(Origin="https://evil.example.invalid"),
                        self.headers(Origin="null"), self.headers(**{"Sec-Fetch-Site": "same-site"}),
                        self.headers(**{"Sec-Fetch-Site": "cross-site"}), self.headers(Host="evil.example.invalid")):
            response = self.client.get(self.path, headers=headers)
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(response.headers["content-security-policy"], CSP)
        self.assertEqual(self.client.get(self.path + "?access_token=fictional", headers=self.headers()).status_code, 400)
        self.assertEqual(self.client.get("/review?proposal=fictional").status_code, 400)
        self.assertEqual(self.client.get("/review", headers={"Origin": "https://evil.example.invalid"}).status_code, 403)
        response = self.client.options(self.path, headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"})
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("access-control-allow-origin", response.headers)
        absent = self.client.post(self.path + "/decision", headers={"Authorization": self.headers()["Authorization"]},
            json={"digest": self.proposal["digest"], "decision": "approve"})
        self.assertEqual(absent.status_code, 403)

    def test_decision_uses_full_digest_and_records_once_without_delivery(self):
        for digest_field in ("action_digest", "wire_preview_digest"):
            self.assertEqual(self.decide(digest=self.proposal[digest_field]).status_code, 409)
        response = self.decide()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "approved")
        self.assertEqual(response.json()["execution"]["state"], "queued")
        self.assertEqual(self.decide().status_code, 409)
        self.assertEqual(self.decide(decision="deny").status_code, 409)
        with self.service.store.connection() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0], 1)
        self.assertIsNone(self.service.prepared_transport)
        self.assertFalse(self.service.transport.enabled)

    def test_action_only_ledger_is_refused_without_recording_decision(self):
        view = dict(self.proposal)
        for field in ("review_contract", "wire_preview", "wire_preview_digest", "action_digest"):
            view.pop(field)
        with patch.object(self.service, "view", return_value=view):
            response = self.client.get(self.path, headers=self.headers())
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["error"], "exact_wire_preview_required")
            self.assertEqual(self.decide().status_code, 409)
        self.assertEqual(self.service.store.get(self.proposal["id"])["state"], "pending")
        with self.service.store.connection() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0], 0)

    def test_racing_ui_approvals_record_one_decision(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.decide) for _ in range(2)]
            results = [future.result(timeout=10).status_code for future in futures]
        self.assertEqual(sorted(results), [200, 409])
        with self.service.store.connection() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0], 1)

    def test_denial_is_terminal_and_expiry_fails_closed(self):
        self.assertEqual(self.decide(decision="deny").status_code, 200)
        self.assertEqual(self.decide(decision="deny").status_code, 409)
        self.assertEqual(self.decide().status_code, 409)
        self.assertEqual(self.service.store.get(self.proposal["id"])["execution"], None)

    def test_expired_stale_and_extra_fields_are_rejected(self):
        for payload in ({"digest": "a"*63, "decision": "approve"}, {"digest": self.proposal["digest"], "decision": "revoke"},
                        {"digest": self.proposal["digest"], "decision": "approve", "body": BODY},
                        {"digest": self.proposal["digest"], "decision": "approve", "approved": True}):
            self.assertEqual(self.client.post(self.path + "/decision", headers=self.headers(), json=payload).status_code, 422)
        self.now = self.proposal["expires_at"]
        self.assertEqual(self.decide().status_code, 409)
        self.assertEqual(self.service.store.get(self.proposal["id"])["state"], "pending")

    def test_json_boundary_rejects_duplicate_invalid_and_oversized_bodies(self):
        path = self.path + "/decision"
        headers = self.headers(**{"Content-Type": "application/json"})
        for body in ('{"digest":"' + self.proposal["digest"] + '","decision":"approve","decision":"deny"}',
                     '{"digest": NaN}', '{broken', '{"digest": Infinity}'):
            self.assertEqual(self.client.post(path, headers=headers, content=body).status_code, 400)
        self.assertEqual(self.client.post(path, headers=self.headers(), content="{}") .status_code, 415)
        self.assertEqual(self.client.post(path, headers=headers, content=" " * 32769).status_code, 413)
        response = self.client.post(path, headers=headers, content=json.dumps({"body": "TOKEN_CANARY BODY_CANARY"}))
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("CANARY", response.text)

    def test_persistent_rate_limit_applies_and_v1_browser_boundary_is_unchanged(self):
        self.service.settings = self.service.settings.model_copy(update={"requests_per_minute": 1})
        self.service.store.settings = self.service.settings
        self.assertEqual(self.client.get(self.path, headers=self.headers()).status_code, 200)
        self.assertEqual(self.client.get(self.path, headers=self.headers()).status_code, 429)
        self.assertEqual(self.client.get("/v1/proposals/" + self.proposal["id"], headers=self.headers()).status_code, 403)

    def test_assets_have_no_mail_html_storage_or_external_dependencies(self):
        js = (ASSETS / "review.js").read_text()
        html = (ASSETS / "review.html").read_text()
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "localStorage", "sessionStorage", "document.cookie", "console.", "eval(", "new Function"):
            self.assertNotIn(forbidden, js)
        self.assertIn('credentials: "omit"', js)
        self.assertIn('redirect: "error"', js)
        self.assertIn('textContent = action.body', js)
        self.assertIn('textContent = value.wire_preview.body', js)
        self.assertNotIn("unsafe-inline", CSP)
        self.assertNotIn("unsafe-eval", CSP)
        self.assertNotIn("https://", html)
        self.assertIsNone(re.search(r"\son[a-z]+=", html, re.I))

    def test_mocked_local_chromium_interrupted_repeated_and_racing_flows(self):
        node = shutil.which("node")
        chromium = shutil.which("chromium")
        if not node or not chromium:
            self.skipTest("Local Node/Chromium unavailable; ASGI/static checks remain active")
        available = subprocess.run([node, "-e", "require('playwright')"], capture_output=True, text=True, timeout=10)
        if available.returncode:
            self.skipTest("Local Playwright unavailable; no dependency installation attempted")
        script = ASSETS / "browser_check.cjs"
        completed = subprocess.run([node, str(script)], input=json.dumps({"assets": str(ASSETS),
            "origin": ORIGIN, "chromium": chromium, "proposal": self.proposal}), capture_output=True, text=True, timeout=90)
        if completed.returncode and "socket() failed: Operation not permitted" in completed.stderr:
            self.skipTest("Chromium launch blocked by sandbox socket restriction; no escalation attempted")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("browser-engine checks passed", completed.stdout)

    def test_socket_free_node_state_machine_interrupted_and_racing_flows(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node unavailable; no installation attempted")
        completed = subprocess.run([node, str(ASSETS / "state_check.cjs")],
            input=json.dumps({"assets": str(ASSETS), "origin": ORIGIN, "proposal": self.proposal}),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("socket-free state checks passed", completed.stdout)

    def test_private_mode_notice_discloses_submission_authority(self):
        settings = self.service.settings.model_copy(update={"outbound_adapter": "private_spacemail", "review_origin": None})
        self.service.settings = settings
        app = create_app(self.service, run_worker=False)
        install_review_ui(app, self.service, ORIGIN)
        with TestClient(app, base_url=ORIGIN) as client:
            text = client.get("/review").text
        self.assertIn("Approve authorizes an attempted submission of this exact reply", text)
        self.assertNotIn("Delivery is disabled", text)
        self.assertIn("not proof of delivery", text)


if __name__ == "__main__":
    unittest.main()
