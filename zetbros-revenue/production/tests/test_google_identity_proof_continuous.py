"""Continuous page availability retains expiring single-use proofs and no authority."""
import json
import pathlib
import time
import unittest
from datetime import datetime
from unittest.mock import patch

import jwt
from fastapi.testclient import TestClient
import test_google_reviewer_auth as google_fixtures
import test_google_identity_proof as bounded_fixtures
from zetbros_service.auth import AuthenticationError
from zetbros_service.google_identity_proof import (
    COOKIE, IdentityProof, IdentityProofProfile, create_identity_proof_app)

CID, SUB, ORIGIN = google_fixtures.CID, google_fixtures.SUB, google_fixtures.ORIGIN
ROOT = pathlib.Path(__file__).resolve().parents[1]


class ContinuousProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        google_fixtures.GoogleAuthTests.setUpClass()
        cls.key = google_fixtures.GoogleAuthTests.key
        cls.jwk = google_fixtures.GoogleAuthTests.jwk

    def setUp(self):
        self.now = int(time.time())
        self.elapsed = 100.0
        self.clock = lambda: self.now
        self.monotonic = lambda: self.elapsed
        self.profile = IdentityProofProfile(enabled=True, client_id=CID,
            origin=ORIGIN, capture_window_seconds=None)
        self.keys = google_fixtures.Keys(self.jwk)
        self.app = create_identity_proof_app(self.profile, key_source=self.keys,
            clock=self.clock, monotonic=self.monotonic)
        self.client = TestClient(self.app, base_url=ORIGIN)

    def tearDown(self):
        self.client.close()

    def token(self, expected_nonce, **changes):
        claims = {'iss':'https://accounts.google.com', 'aud':CID, 'sub':SUB,
            'iat':self.now-1, 'exp':self.now+300, 'nonce':expected_nonce,
            'email':'fictional@untrusted.invalid', 'role':'reviewer',
            'user_metadata':{'role':'admin'}}
        claims.update(changes)
        return jwt.encode(claims, self.key, algorithm='RS256',
            headers={'kid':'mock-google', 'typ':'JWT'})

    def bootstrap(self):
        response = self.client.get('/identity/bootstrap')
        self.assertEqual(response.status_code, 200)
        return response.json()

    def prove(self, challenge, token=None, **header_changes):
        headers = {'Origin':ORIGIN, 'X-Proof-CSRF':challenge['csrf']} | header_changes
        return self.client.post('/identity/proof', headers=headers,
            json={'credential':token or self.token(challenge['nonce'])})

    def test_continuous_page_and_new_proof_remain_available_after_thirty_days(self):
        self.bootstrap()
        self.now += 30*24*3600
        self.elapsed += 30*24*3600
        for path in ('/identity', '/identity/assets/identity.js', '/healthz'):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['cache-control'], 'no-store')
        challenge = self.bootstrap()
        future = self.now
        class FutureDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.fromtimestamp(future, tz)
        # PyJWT also validates time against its own UTC clock.
        with patch('jwt.api_jwt.datetime', FutureDateTime):
            self.assertEqual(self.prove(challenge).json(),
                {'verified':True, 'subject':SUB, 'authority':'none'})
        self.assertEqual(self.app.state.identity_proof.challenges, {})

    def test_continuous_is_explicit_and_bounded_default_is_unchanged(self):
        self.assertEqual(IdentityProofProfile().capture_window_seconds, 600)
        self.assertIsNone(self.profile.capture_window_seconds)
        disabled = IdentityProofProfile.model_validate_json(
            (ROOT/'deploy/identity-proof.continuous.disabled.example.json').read_text())
        with self.assertRaises(ValueError):
            IdentityProof(disabled)
        for value in (0, -1, 601, 'unbounded'):
            with self.assertRaises(ValueError):
                IdentityProofProfile(enabled=True, client_id=CID, origin=ORIGIN,
                    capture_window_seconds=value)
        for grant in ({'reviewer_subjects':[SUB]}, {'mail_enabled':True}, {'agent':True}):
            with self.assertRaises(ValueError):
                IdentityProofProfile.model_validate_json(json.dumps(
                    self.profile.model_dump() | grant))

    def test_wall_clock_challenge_expiry_does_not_close_page(self):
        challenge = self.bootstrap()
        self.now += self.profile.challenge_seconds
        self.assertEqual(self.prove(challenge).status_code, 401)
        self.assertEqual(self.keys.calls, [])
        self.assertEqual(self.client.get('/identity').status_code, 200)

    def test_monotonic_challenge_expiry_survives_frozen_wall_clock(self):
        challenge = self.bootstrap()
        self.elapsed += self.profile.challenge_seconds
        self.assertEqual(self.prove(challenge).status_code, 401)
        self.assertEqual(self.keys.calls, [])
        self.assertEqual(self.client.get('/identity').status_code, 200)
        self.assertEqual(self.prove(self.bootstrap()).status_code, 200)

    def test_expiry_rechecked_after_slow_verification_on_monotonic_clock(self):
        challenge = self.bootstrap()
        verifier = self.app.state.identity_proof.verifier
        original = verifier.verify_identity
        def delayed(token, nonce):
            subject = original(token, nonce)
            self.elapsed += self.profile.challenge_seconds
            return subject
        with patch.object(verifier, 'verify_identity', side_effect=delayed):
            self.assertEqual(self.prove(challenge).status_code, 401)
        self.assertEqual(self.client.get('/identity').status_code, 200)

    def test_clock_rollback_discards_challenge_without_expiring_page(self):
        challenge = self.bootstrap()
        self.now -= 1
        self.assertEqual(self.prove(challenge).status_code, 401)
        self.assertEqual(self.client.get('/identity').status_code, 200)
        self.now += 1
        self.assertEqual(self.prove(challenge).status_code, 401)
        challenge = self.bootstrap()
        self.elapsed -= 1
        self.assertEqual(self.prove(challenge).status_code, 401)

    def test_restart_forgets_challenges_and_startup_fetches_no_credentials(self):
        challenge = self.bootstrap()
        restarted = IdentityProof(self.profile, key_source=self.keys,
            clock=self.clock, monotonic=self.monotonic)
        with self.assertRaises(AuthenticationError):
            restarted.prove(self.client.cookies.get(COOKIE), challenge['csrf'],
                self.token(challenge['nonce']))
        with patch('zetbros_service.pilot_runtime.build_service',
                   side_effect=AssertionError('application service forbidden')):
            with TestClient(create_identity_proof_app(self.profile,
                    key_source=self.keys), base_url=ORIGIN) as client:
                self.assertEqual(client.get('/healthz').json()['authority'], 'none')
                self.assertEqual(client.get('/readyz').status_code, 503)
        self.assertEqual(self.keys.calls, [])

    def test_repeated_use_keeps_challenge_and_peer_pools_bounded(self):
        proof = self.app.state.identity_proof
        for minute in range(1440):
            self.now += 60
            self.elapsed += 60
            proof.bootstrap('fictional-peer-' + str(minute))
            self.assertLessEqual(len(proof.challenges), self.profile.max_challenges)
            self.assertLessEqual(len(proof.rates), 64)
        self.assertEqual(self.client.get('/identity').status_code, 200)

    test_no_application_authority = bounded_fixtures.ProofTests.test_review_agent_mail_and_audit_routes_are_absent_with_any_authority
    test_signature_claims_and_nonce_boundaries = bounded_fixtures.ProofTests.test_nonce_signature_audience_issuer_subject_and_times_must_verify
    test_single_use_success_failure_cancel = bounded_fixtures.ProofTests.test_success_failure_cancel_and_expiry_make_challenges_single_use
    test_csrf_cookie_origin_query_boundaries = bounded_fixtures.ProofTests.test_csrf_cookie_origin_cors_queries_and_duplicate_fields_fail_closed
    test_peer_bootstrap_rate_boundaries = bounded_fixtures.ProofTests.test_peer_bounds_ignore_raw_forwarding_and_expire_without_persistence


class ContinuousUnitTests(unittest.TestCase):
    def test_continuous_example_retains_isolation_budgets_and_no_sensitive_logs(self):
        unit = (ROOT/'deploy/zetbros-identity-proof.continuous.service.example').read_text()
        for required in ('RuntimeMaxSec=infinity', 'DynamicUser=yes', 'MemoryMax=80M',
                'MemoryHigh=64M', 'MemorySwapMax=0', 'CPUQuota=25%', 'TasksMax=16',
                'ProtectSystem=strict', 'ProtectHome=yes', 'NoNewPrivileges=yes',
                'StandardOutput=null', 'StandardError=null', '--no-access-log',
                '--host 127.0.0.1', '--port 8082', 'Restart=on-failure',
                'StartLimitBurst=3', 'RestartSec=10', 'WantedBy=multi-user.target'):
            self.assertIn(required, unit)
        for forbidden in ('LoadCredential', 'ReadWritePaths', 'EnvironmentFile', '--port 8081'):
            self.assertNotIn(forbidden, unit)


if __name__ == '__main__':
    unittest.main()
