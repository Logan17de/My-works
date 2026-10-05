"""Socket-free reviewer login with no agent, ledger, provider or action authority."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from zetbros_service.config import Grant, ReviewerPendingConfig, load_settings
from zetbros_service.google_reviewer_auth import GoogleReviewerAuth, GoogleReviewerProfile, SESSION_COOKIE
from zetbros_service.pilot_runtime import ReviewerPendingRuntime, build_service, create_reviewer_pending_app, initialize_new_pilot
import test_google_reviewer_auth as full

CID, SUB, ORIGIN = full.CID, full.SUB, full.ORIGIN


class ReviewerPendingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key()))
        cls.jwk.update(kid='mock-google', alg='RS256', use='sig', key_ops=['verify'])

    def setUp(self):
        self.settings = ReviewerPendingConfig(runtime_state='reviewer_enabled_workflow_pending',
            review_origin=ORIGIN, google_reviewer_profile_file=str(Path(tempfile.gettempdir())/'reviewer-profile.json'),
            principals=(Grant(subject=SUB, client_id=CID, role='reviewer'),))
        self.profile = GoogleReviewerProfile(enabled=True, client_id=CID, origin=ORIGIN, reviewer_subjects=(SUB,))
        self.runtime = ReviewerPendingRuntime(self.settings)
        self.runtime.google_reviewer_auth = GoogleReviewerAuth(self.profile, self.settings, full.Keys(self.jwk))
        self.client = TestClient(create_reviewer_pending_app(self.runtime), base_url=ORIGIN)
        self.addCleanup(self.client.close)

    def login(self, **changes):
        b = self.client.get('/review/auth/bootstrap').json()
        now = int(time.time())
        claims = dict(iss='https://accounts.google.com', aud=CID, sub=SUB, iat=now-1, exp=now+300, nonce=b['nonce'])
        claims.update(changes)
        token = jwt.encode(claims, self.key, algorithm='RS256', headers={'kid':'mock-google','typ':'JWT'})
        return self.client.post('/review/auth/google', headers={'Origin':ORIGIN,'X-Login-CSRF':b['csrf']}, json={'credential':token})

    def test_strict_config_has_no_agent_or_ledger_fields(self):
        for field in ('issuer','audience','jwks_file','database_path','private_bridge_profile_file'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ReviewerPendingConfig.model_validate_json(json.dumps(self.settings.model_dump(mode='json') | {field:'forbidden'}))

    def test_agent_mapping_and_live_adapter_rejected(self):
        for change in ({'principals':[{'subject':SUB,'client_id':CID,'role':'agent'}]}, {'outbound_adapter':'private_spacemail'}):
            with self.assertRaises(ValueError): ReviewerPendingConfig.model_validate_json(json.dumps(self.settings.model_dump(mode='json')|change))

    def test_empty_duplicate_and_non_https_rejected(self):
        for change in ({'principals':[]}, {'principals':[g.model_dump() for g in self.settings.principals]*2}, {'review_origin':'http://127.0.0.1:8081'}):
            with self.assertRaises(ValueError): ReviewerPendingConfig.model_validate_json(json.dumps(self.settings.model_dump(mode='json')|change))

    def test_settings_loader_selects_supported_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'config.json';p.write_text(self.settings.model_dump_json())
            with patch.dict(os.environ, {'ZETBROS_CONFIG_FILE':str(p)}):
                self.assertIsInstance(load_settings(), ReviewerPendingConfig)

    def test_builder_never_constructs_full_runtime(self):
        with patch('zetbros_service.google_auth_runtime.load_google_review_profile', return_value=self.profile), \
             patch('zetbros_service.pilot_runtime.Verifier', side_effect=AssertionError('agent verifier')), \
             patch('zetbros_service.pilot_runtime.MailService', side_effect=AssertionError('mail service')), \
             patch('zetbros_service.pilot_runtime.assemble_source', side_effect=AssertionError('mail source')):
            runtime=build_service(self.settings)
            self.assertIsInstance(runtime, ReviewerPendingRuntime)
            for attribute in ('store','verifier','worker_loop','source','transport'):
                self.assertFalse(hasattr(runtime,attribute))

    def test_initialization_creates_no_ledger(self):
        with patch('zetbros_service.google_auth_runtime.load_google_review_profile', return_value=self.profile):
            result=initialize_new_pilot(self.settings)
            self.assertFalse(result['ledger_created']); self.assertFalse(result['agent_api_enabled'])

    def test_profile_subject_mapping_must_match_exactly(self):
        for profile in (self.profile.model_copy(update={'reviewer_subjects':('another-sub',)}), self.profile.model_copy(update={'client_id':'987654321-another.apps.googleusercontent.com'})):
            with self.assertRaises(ValueError): GoogleReviewerAuth(profile,self.settings,full.Keys(self.jwk))

    def test_status_distinguishes_login_from_workflow_readiness(self):
        health=self.client.get('/healthz').json()
        self.assertTrue(health['reviewer_login_enabled'])
        self.assertFalse(health['workflow_ready']);self.assertFalse(health['ledger_initialized'])
        self.assertEqual(self.client.get('/readyz').status_code,503)

    def test_shell_and_security_headers(self):
        response=self.client.get('/review')
        self.assertEqual(response.status_code,200)
        self.assertIn('data-workflow-state="pending"',response.text)
        self.assertIn('Proposed actions are not available yet',response.text)
        self.assertIn('aria-labelledby="proposal-title" hidden',response.text)
        self.assertEqual(response.headers['cache-control'],'no-store')
        self.assertEqual(response.headers['referrer-policy'],'no-referrer')

    def test_fresh_google_login_creates_only_short_reviewer_session(self):
        response=self.login();self.assertEqual(response.status_code,200)
        self.assertEqual(set(response.json()),{'authenticated','csrf'})
        for value in ('Secure','HttpOnly','SameSite=lax','Path=/review','Max-Age=300'):
            self.assertIn(value,response.headers['set-cookie'])
        self.assertEqual(self.client.get('/review/auth/session').status_code,200)

    def test_unapproved_subject_and_wrong_audience_refused(self):
        for change in ({'sub':'not-approved'},{'aud':'other'},{'iss':'https://attacker.invalid'},{'nonce':'wrong'}):
            with self.subTest(change=change):self.assertEqual(self.login(**change).status_code,401)

    def test_authenticated_agent_api_and_mutations_stay_disabled(self):
        csrf=self.login().json()['csrf']
        for path in ('/v1/proposals','/review/api/proposals/00000000-0000-0000-0000-000000000000/decision','/review/api/proposals/00000000-0000-0000-0000-000000000000/revoke','/review/api/proposals/00000000-0000-0000-0000-000000000000/reconciliation'):
            response=self.client.post(path,headers={'Origin':ORIGIN,'X-Review-CSRF':csrf},json={'decision':'approve'})
            self.assertEqual(response.status_code,503)
        self.assertFalse(hasattr(self.runtime,'store'))

    def test_origin_query_and_bearer_cannot_bypass_boundary(self):
        for path,headers in (('/review',{'Origin':'https://attacker.invalid'}),('/review',{'Authorization':'Bearer fiction'}),('/review?token=fiction',{})):
            self.assertIn(self.client.get(path,headers=headers).status_code,(400,403))

    def test_logout_requires_csrf_and_clears_session(self):
        csrf=self.login().json()['csrf']
        self.assertEqual(self.client.post('/review/auth/logout',headers={'Origin':ORIGIN}).status_code,401)
        self.assertEqual(self.client.post('/review/auth/logout',headers={'Origin':ORIGIN,'X-Review-CSRF':csrf}).status_code,200)
        self.assertEqual(self.client.get('/review/auth/session').status_code,401)

    def test_restart_does_not_restore_cookie_authority(self):
        self.login();cookie=self.client.cookies.get(SESSION_COOKIE)
        fresh=ReviewerPendingRuntime(self.settings)
        fresh.google_reviewer_auth=GoogleReviewerAuth(self.profile,self.settings,full.Keys(self.jwk))
        with TestClient(create_reviewer_pending_app(fresh),base_url=ORIGIN) as client:
            self.assertEqual(client.get('/review/auth/session',headers={'Cookie':SESSION_COOKIE+'='+cookie}).status_code,401)

    def test_transition_to_full_auth_does_not_import_pending_sessions(self):
        self.login();cookie=self.client.cookies.get(SESSION_COOKIE)
        from zetbros_service.config import Settings
        from zetbros_service.auth import AuthenticationError
        directory=Path(tempfile.gettempdir())
        settings=Settings(tenant_id='test-tenant',connector_id='test-connector',account_id='test-account',
            sender_address='owner@owner-runtime.net',policy_version='test-policy',
            issuer='https://auth.owner-runtime.net',audience='test-agent-api',
            jwks_file=str(directory/'test-public-jwks.json'),database_path=str(directory/'db'/'test.sqlite'),
            source_directory=str(directory/'test-source'),review_origin=ORIGIN,
            principals=self.settings.principals+(Grant(subject='test-agent',client_id='test-agent-client',role='agent'),))
        auth=GoogleReviewerAuth(self.profile,settings,full.Keys(self.jwk))
        with self.assertRaises(AuthenticationError):auth.session(cookie)

    def test_configured_entrypoint_selects_login_only_app(self):
        from zetbros_service.api import configured_app
        with patch('zetbros_service.api.load_settings',return_value=self.settings), \
             patch('zetbros_service.google_auth_runtime.load_google_review_profile',return_value=self.profile):
            with TestClient(configured_app(),base_url=ORIGIN) as client:
                self.assertEqual(client.get('/review').status_code,200)
                self.assertEqual(client.post('/v1/proposals',json={}).status_code,503)

    @unittest.skipUnless(hasattr(os,'O_NOFOLLOW'),'Linux profile file protection required')
    def test_protected_profile_loader_and_symlink_rejection(self):
        from zetbros_service.google_auth_runtime import load_google_review_profile
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'profile.json';p.write_text(self.profile.model_dump_json())
            settings=self.settings.model_copy(update={'google_reviewer_profile_file':str(p)})
            self.assertEqual(load_google_review_profile(settings),self.profile)
            p.unlink();p.symlink_to(Path(directory)/'absent')
            with self.assertRaises(RuntimeError):load_google_review_profile(settings)

    def test_full_settings_do_not_accept_reviewer_pending_schema(self):
        from zetbros_service.config import Settings
        with self.assertRaises(ValueError): Settings.model_validate_json(self.settings.model_dump_json())


if __name__=='__main__': unittest.main()
