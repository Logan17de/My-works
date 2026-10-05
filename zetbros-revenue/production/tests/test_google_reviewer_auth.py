"""Offline Google ID-token/session and strict agent separation acceptance."""
import json
import time
import unittest
import uuid
from unittest.mock import patch
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
import test_mail_integration as integration
from zetbros_service.auth import AuthenticationError,Principal
from zetbros_service.config import Grant
from zetbros_service.api import create_app
from zetbros_service.google_reviewer_auth import (AgentOnlyVerifier,GoogleReviewerAuth,GoogleReviewerProfile,
    CHALLENGE_COOKIE,SESSION_COOKIE)
from zetbros_service.review_ui import install_review_ui

CID='123456789-testclient.apps.googleusercontent.com'
SUB='109876543210987654321'
ORIGIN='https://review.owner-runtime.net'


class Keys:
    def __init__(self,key):self.key=key;self.calls=[]
    def get_jwks(self,url):self.calls.append(url);return {'keys':[self.key]}


class GoogleAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        cls.jwk=json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key()))
        cls.jwk.update(kid='mock-google',alg='RS256',use='sig',key_ops=['verify'])
        integration.MailIntegrationTests.setUpClass()
    def setUp(self):
        self.fixture=integration.MailIntegrationTests('test_exact_claim_gate_and_wire_no_retry_survive_restart')
        self.fixture.setUp();self.service=self.fixture.service
        reviewers=(Grant(subject=SUB,client_id=CID,role='reviewer'),)
        agents=tuple(g for g in self.service.settings.principals if g.role=='agent')
        settings=self.service.settings.model_copy(update={'principals':agents+reviewers,'review_origin':ORIGIN})
        self.service.settings=settings;self.service.store.settings=settings
        self.service.verifier=AgentOnlyVerifier(self.service.verifier)
        self.profile=GoogleReviewerProfile(enabled=True,client_id=CID,origin=ORIGIN,reviewer_subjects=(SUB,))
        self.keys=Keys(self.jwk);self.auth=GoogleReviewerAuth(self.profile,settings,self.keys,clock=self.fixture.fixture.clock)
        self.service.google_reviewer_auth=self.auth
        self.client=TestClient(create_app(self.service,run_worker=False),base_url=ORIGIN)
        self.proposal=self.fixture.propose()
    def tearDown(self):self.client.close();self.fixture.tearDown()
    def token(self,expected_nonce,**changes):
        now=int(time.time());claims={'iss':'https://accounts.google.com','aud':CID,'sub':SUB,'iat':now-1,
            'exp':now+300,'nonce':expected_nonce,'email':'ignored@untrusted.invalid','role':'agent','user_metadata':{'role':'admin'}}
        claims.update(changes)
        return jwt.encode(claims,self.key,algorithm='RS256',headers={'kid':'mock-google','typ':'JWT'})
    def bootstrap(self):
        response=self.client.get('/review/auth/bootstrap');self.assertEqual(response.status_code,200);return response.json()
    def login(self,bootstrap=None,token=None):
        bootstrap=bootstrap or self.bootstrap()
        return self.client.post('/review/auth/google',headers={'Origin':ORIGIN,'X-Login-CSRF':bootstrap['csrf']},
            json={'credential':token or self.token(bootstrap['nonce'])})
    def path(self):return '/review/api/proposals/'+self.proposal['id']

    def test_dedicated_login_creates_only_short_secure_opaque_reviewer_cookie(self):
        bootstrap=self.bootstrap();response=self.login(bootstrap)
        self.assertEqual(response.status_code,200)
        self.assertEqual(set(response.json()),{'authenticated','csrf'})
        header=response.headers['set-cookie'];self.assertIn('HttpOnly',header);self.assertIn('Secure',header)
        self.assertIn('Path=/review',header);self.assertIn('SameSite=lax',header)
        cookie=self.client.cookies.get(SESSION_COOKIE)
        self.assertEqual(len(cookie),43);self.assertNotIn(SUB,cookie)
        session=self.client.get('/review/auth/session');self.assertEqual(session.status_code,200)
        preview=self.client.get(self.path());self.assertEqual(preview.status_code,200)
        self.assertEqual(preview.json()['digest'],self.proposal['digest'])
        self.assertNotIn('credential',session.json());self.assertNotIn('access_token',response.json())

    def test_exact_wire_decision_requires_cookie_csrf_origin_and_full_digest(self):
        response=self.login();csrf=response.json()['csrf'];body={'digest':self.proposal['digest'],'decision':'approve'}
        self.assertEqual(self.client.post(self.path()+'/decision',headers={'Origin':ORIGIN},json=body).status_code,401)
        self.assertEqual(self.client.post(self.path()+'/decision',headers={'Origin':ORIGIN,'X-Review-CSRF':'wrong'},json=body).status_code,401)
        self.assertEqual(self.client.post(self.path()+'/decision',headers={'Origin':'https://attacker.invalid','X-Review-CSRF':csrf},json=body).status_code,403)
        result=self.client.post(self.path()+'/decision',headers={'Origin':ORIGIN,'X-Review-CSRF':csrf},json=body)
        self.assertEqual(result.status_code,200);self.assertEqual(result.json()['reviewer_subject'],SUB)
        self.assertEqual(self.client.post(self.path()+'/decision',headers={'Origin':ORIGIN,'X-Review-CSRF':csrf},json=body).status_code,409)

    def test_wrong_google_claims_or_access_token_never_log_in(self):
        for changes in ({'aud':'other.apps.googleusercontent.com'},{'iss':'https://attacker.invalid'},
                        {'sub':'unknown-sub'},{'nonce':'wrong'},{'azp':'other-client'},
                        {'aud':[CID,'other']},{'exp':int(time.time())-1}):
            bootstrap=self.bootstrap();response=self.login(bootstrap,self.token(bootstrap['nonce'],**changes))
            self.assertEqual(response.status_code,401);self.assertNotIn('ignored@',response.text)
        bootstrap=self.bootstrap();self.assertEqual(self.login(bootstrap,'not-a-Google-ID-token').status_code,401)

    def test_state_nonce_and_failed_challenges_are_single_use(self):
        bootstrap=self.bootstrap();bad=self.login(bootstrap,self.token('wrong-nonce'));self.assertEqual(bad.status_code,401)
        again=self.login(bootstrap,self.token(bootstrap['nonce']));self.assertEqual(again.status_code,401)
        bootstrap=self.bootstrap();self.assertEqual(self.login(bootstrap).status_code,200)
        self.assertEqual(self.login(bootstrap).status_code,401)

    def test_no_cookie_wrong_login_csrf_or_bearer_login_fail(self):
        bootstrap=self.bootstrap();self.client.cookies.clear()
        self.assertEqual(self.login(bootstrap).status_code,401)
        bootstrap=self.bootstrap();response=self.client.post('/review/auth/google',
            headers={'Origin':ORIGIN,'X-Login-CSRF':'wrong'},json={'credential':self.token(bootstrap['nonce'])})
        self.assertEqual(response.status_code,401)
        self.assertEqual(self.client.get('/review/auth/bootstrap',headers={'Authorization':'Bearer anything'}).status_code,403)

    def test_agent_bearers_cannot_reach_any_review_reconcile_or_audit_path(self):
        token=self.fixture.fixture.token();headers={'Authorization':'Bearer '+token}
        for path in ('/review','/review/auth/bootstrap',self.path(),'/v1/audit',
                     '/v1/reviews/'+self.proposal['id']+'/reconciliations'):
            response=self.client.get(path,headers=headers);self.assertEqual(response.status_code,403,path)
        for path in ('/v1/reviews/'+self.proposal['id']+'/decision',
                     '/v1/reviews/'+self.proposal['id']+'/reconciliation'):
            self.assertEqual(self.client.post(path,headers=headers,json={}).status_code,403)
        agent=self.client.get('/v1/proposals/'+self.proposal['id'],headers=headers)
        self.assertEqual(agent.status_code,200)

    def test_google_id_token_is_never_a_bearer_or_an_agent_credential(self):
        bootstrap=self.bootstrap();token=self.token(bootstrap['nonce']);headers={'Authorization':'Bearer '+token}
        self.assertEqual(self.client.get('/v1/proposals/'+self.proposal['id'],headers=headers).status_code,401)
        self.assertEqual(self.client.get(self.path(),headers=headers).status_code,403)
        self.login(bootstrap,token)
        cookie=self.client.cookies.get(SESSION_COOKIE)
        self.assertEqual(self.client.get('/v1/proposals/'+self.proposal['id'],headers={'Cookie':SESSION_COOKIE+'='+cookie}).status_code,403)

    def test_logout_expiry_restart_and_role_removal_revoke_session(self):
        response=self.login();csrf=response.json()['csrf'];cookie=self.client.cookies.get(SESSION_COOKIE)
        self.assertEqual(self.client.post('/review/auth/logout',headers={'Origin':ORIGIN}).status_code,401)
        self.assertEqual(self.client.post('/review/auth/logout',headers={'Origin':ORIGIN,'X-Review-CSRF':csrf}).status_code,200)
        self.assertEqual(self.client.get(self.path()).status_code,401)
        with self.assertRaises(AuthenticationError):self.auth.session(cookie)
        self.login();self.fixture.fixture.clock.now+=301
        self.assertEqual(self.client.get(self.path()).status_code,401)
        restarted=GoogleReviewerAuth(self.profile,self.service.settings,self.keys)
        with self.assertRaises(AuthenticationError):restarted.session(cookie)

    def test_forged_role_email_and_user_metadata_do_not_select_authority(self):
        bootstrap=self.bootstrap();token=self.token(bootstrap['nonce'],sub='not-allowed',email='admin@owner-runtime.net',
            email_verified=True,role='reviewer',user_metadata={'role':'reviewer'},service_role=True)
        self.assertEqual(self.login(bootstrap,token).status_code,401)

    def test_disabled_profile_has_no_fake_identity_or_key_fetch(self):
        disabled=GoogleReviewerProfile()
        self.assertFalse(disabled.enabled)
        for changes in ({'client_id':'REPLACE_CLIENT'},{'origin':'https://placeholder.invalid'},
                        {'reviewer_subjects':('REPLACE_SUB',)}):
            with self.assertRaises(ValueError):GoogleReviewerProfile(**changes)
        with self.assertRaises(ValueError):GoogleReviewerAuth(disabled,self.service.settings,self.keys)
        self.assertEqual(self.keys.calls,[])

    def test_duplicate_cookie_and_untrusted_query_or_header_are_rejected(self):
        response=self.login();cookie=self.client.cookies.get(SESSION_COOKIE)
        self.assertEqual(self.client.get(self.path(),headers={'Cookie':SESSION_COOKIE+'='+cookie+'; '+SESSION_COOKIE+'='+cookie}).status_code,401)
        self.assertEqual(self.client.get('/review/auth/session?token=CANARY').status_code,400)
        self.assertEqual(self.client.get('/review/auth/bootstrap',headers={'Origin':'https://attacker.invalid'}).status_code,403)

    def test_scoped_cookie_operator_routes_keep_revoke_and_audit_available(self):
        response=self.login();csrf=response.json()['csrf'];headers={'Origin':ORIGIN,'X-Review-CSRF':csrf}
        approved=self.client.post(self.path()+'/decision',headers=headers,json={'digest':self.proposal['digest'],'decision':'approve'})
        self.assertEqual(approved.status_code,200)
        audit=self.client.get('/review/api/audit');self.assertEqual(audit.status_code,200)
        revoked=self.client.post(self.path()+'/revoke',headers=headers,json={'digest':self.proposal['digest']})
        self.assertEqual(revoked.status_code,200);self.assertEqual(revoked.json()['state'],'revoked')
        token=self.fixture.fixture.token()
        self.assertEqual(self.client.get('/review/api/audit',headers={'Authorization':'Bearer '+token}).status_code,403)

    def test_complete_audit_pages_use_bounded_nonsecret_path_cursor(self):
        self.login()
        with self.service.store.transaction() as connection:
            actor=Principal(SUB,CID,'reviewer')
            for index in range(110):
                self.service.store.event(connection,str(uuid.uuid4()),actor,self.proposal['id'],
                    self.proposal['digest'],'mock-policy','mock-audit','observed')
        first=self.client.get('/review/api/audit').json()['events']
        self.assertEqual(len(first),100)
        later=self.client.get('/review/api/audit/after/'+str(first[-1]['sequence']))
        self.assertEqual(later.status_code,200)
        self.assertTrue(later.json()['events'])
        self.assertTrue(all(event['sequence']>first[-1]['sequence'] for event in later.json()['events']))
        for cursor in ('-1','9223372036854775808','unknown'):
            self.assertEqual(self.client.get('/review/api/audit/after/'+cursor).status_code,422)
        self.assertEqual(self.client.get('/review/api/audit?after=1').status_code,400)
        self.assertEqual(self.client.get('/review/api/audit/after/1',
            headers={'Authorization':'Bearer '+self.fixture.fixture.token()}).status_code,403)

    def test_google_mode_without_login_boundary_never_falls_back_to_bearer(self):
        self.service.settings=self.service.settings.model_copy(update={'reviewer_auth':'google_oidc'})
        saved=self.service.google_reviewer_auth;del self.service.google_reviewer_auth
        with self.assertRaises(RuntimeError):create_app(self.service,run_worker=False)
        self.service.google_reviewer_auth=saved

    def test_new_bootstrap_replaces_old_browser_challenge(self):
        first=self.bootstrap();first_cookie=self.client.cookies.get(CHALLENGE_COOKIE)
        second=self.bootstrap()
        self.assertEqual(len(self.auth.challenges),1)
        with self.assertRaises(AuthenticationError):self.auth.login(first_cookie,first['csrf'],self.token(first['nonce']))
        self.assertEqual(self.login(second).status_code,200)

    def test_bootstrap_peer_throttle_ignores_spoofed_forwarding_headers(self):
        for _ in range(self.profile.bootstrap_per_peer_per_minute):
            self.assertEqual(self.client.get('/review/auth/bootstrap',headers={'X-Forwarded-For':'different-spoofed-ip'}).status_code,200)
        self.assertEqual(self.client.get('/review/auth/bootstrap',headers={'X-Forwarded-For':'another-spoofed-ip'}).status_code,401)
        self.assertEqual(len(self.auth.challenges),1)

    def test_scoped_reconciliation_records_observation_without_resending(self):
        response=self.login();csrf=response.json()['csrf'];headers={'Origin':ORIGIN,'X-Review-CSRF':csrf}
        self.assertEqual(self.client.post(self.path()+'/decision',headers=headers,
            json={'digest':self.proposal['digest'],'decision':'approve'}).status_code,200)
        self.fixture.port.drop_ack=True
        self.fixture.worker()
        result=self.client.post(self.path()+'/reconciliation',headers=headers,json={
            'digest':self.proposal['digest'],'observed_submission':'accepted','evidence_reference':'mock-owner-observation'})
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.json()['execution']['state'],'uncertain')
        self.assertEqual(self.client.get(self.path()+'/reconciliations').status_code,200)
        self.fixture.worker();self.assertEqual(self.fixture.port.calls,1)

    def test_agent_resource_cannot_be_conflated_with_google_login_audience(self):
        for changes in ({'issuer':'https://accounts.google.com'},{'audience':CID}):
            settings=self.service.settings.model_copy(update=changes)
            with self.assertRaises(ValueError):GoogleReviewerAuth(self.profile,settings,self.keys)

    def test_logout_helper_requires_csrf_even_outside_http(self):
        bootstrap=self.bootstrap();self.login(bootstrap);cookie=self.client.cookies.get(SESSION_COOKIE)
        with self.assertRaises(AuthenticationError):self.auth.logout(cookie,None)
