"""Offline identity proof cannot enroll, initialize or authorize applications."""
import json
import os
import pathlib
import tempfile
import time
import unittest
from unittest.mock import patch

import jwt
from fastapi.testclient import TestClient
import test_google_reviewer_auth as google_fixtures
CID, SUB, ORIGIN = google_fixtures.CID, google_fixtures.SUB, google_fixtures.ORIGIN
from zetbros_service.auth import AuthenticationError
from zetbros_service.google_identity_proof import (IdentityProof, IdentityProofProfile, COOKIE,
    create_identity_proof_app, load_identity_proof_profile)

ROOT=pathlib.Path(__file__).resolve().parents[1]


class ProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        google_fixtures.GoogleAuthTests.setUpClass();cls.key=google_fixtures.GoogleAuthTests.key;cls.jwk=google_fixtures.GoogleAuthTests.jwk
    def setUp(self):
        self.now=int(time.time());self.clock=lambda:self.now
        self.profile=IdentityProofProfile(enabled=True,client_id=CID,origin=ORIGIN)
        self.keys=google_fixtures.Keys(self.jwk)
        self.app=create_identity_proof_app(self.profile,key_source=self.keys,clock=self.clock)
        self.client=TestClient(self.app,base_url=ORIGIN)
    def tearDown(self):self.client.close()
    def token(self,expected_nonce,**changes):
        claims={'iss':'https://accounts.google.com','aud':CID,'sub':SUB,'iat':int(time.time())-1,
            'exp':int(time.time())+300,'nonce':expected_nonce,'email':'fictional@untrusted.invalid',
            'role':'reviewer','user_metadata':{'role':'admin'}}
        claims.update(changes)
        return jwt.encode(claims,self.key,algorithm='RS256',headers={'kid':'mock-google','typ':'JWT'})
    def bootstrap(self):
        r=self.client.get('/identity/bootstrap');self.assertEqual(r.status_code,200);return r.json()
    def prove(self,challenge,token=None,**header_changes):
        headers={'Origin':ORIGIN,'X-Proof-CSRF':challenge['csrf']}|header_changes
        return self.client.post('/identity/proof',headers=headers,json={'credential':token or self.token(challenge['nonce'])})

    def test_verified_unknown_subject_returns_only_proof_and_never_session_or_enrollment(self):
        challenge=self.bootstrap();result=self.prove(challenge)
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.json(),{'verified':True,'subject':SUB,'authority':'none'})
        self.assertNotIn('fictional@',result.text);self.assertNotIn('credential',result.text)
        self.assertNotIn('zetbros-review=',result.headers['set-cookie'])
        self.assertIsNone(self.client.cookies.get(COOKIE))
        self.assertFalse(hasattr(self.app.state,'service'));self.assertFalse(hasattr(self.app.state.identity_proof,'sessions'))
        self.assertEqual(self.app.state.identity_proof.challenges,{})

    def test_profile_and_startup_need_no_sub_agent_mail_keys_or_ledger(self):
        with patch('zetbros_service.pilot_runtime.build_service',side_effect=AssertionError('service')), \
             patch('zetbros_service.google_public_keys.GooglePublicKeys.get_jwks',side_effect=AssertionError('network')):
            app=create_identity_proof_app(self.profile,key_source=self.keys)
            with TestClient(app,base_url=ORIGIN) as client:
                self.assertEqual(client.get('/healthz').status_code,200)
                self.assertEqual(client.get('/readyz').status_code,503)
                self.assertEqual(client.get('/identity').status_code,200)
        self.assertEqual(self.keys.calls,[])
        self.assertNotIn('reviewer_subjects',self.profile.model_dump())

    def test_review_agent_mail_and_audit_routes_are_absent_with_any_authority(self):
        for path in ('/review','/review/auth/google','/review/api/audit','/v1/proposals','/v1/audit','/v1/source/imap-v1:1:1'):
            self.assertEqual(self.client.get(path).status_code,503,path)
            self.assertEqual(self.client.post(path,json={}).status_code,503,path)
        self.assertEqual(self.client.get('/identity',headers={'Authorization':'Bearer FICTIONAL'}).status_code,403)
        self.assertEqual(self.client.get('/v1/audit',headers={'Authorization':'Bearer FICTIONAL'}).status_code,403)

    def test_nonce_signature_audience_issuer_subject_and_times_must_verify(self):
        for changes in ({'nonce':'wrong'},{'aud':'other'},{'iss':'https://attacker.invalid'},
                        {'sub':''},{'sub':'mail@untrusted.invalid'},{'sub':'x'*256},
                        {'exp':int(time.time())-1},{'azp':'other'},{'aud':[CID]},
                        {'iat':int(time.time())+600},{'exp':int(time.time())+7200}):
            self.app.state.identity_proof.rates.clear()
            challenge=self.bootstrap();response=self.prove(challenge,self.token(challenge['nonce'],**changes))
            self.assertEqual(response.status_code,401,changes);self.assertNotIn('fictional@',response.text)
        self.app.state.identity_proof.rates.clear();challenge=self.bootstrap()
        token=self.token(challenge['nonce']);bad=token.rsplit('.',1)[0]+'.AAAA'
        self.assertEqual(self.prove(challenge,bad).status_code,401)

    def test_success_failure_cancel_and_expiry_make_challenges_single_use(self):
        challenge=self.bootstrap();cookie=self.client.cookies.get(COOKIE)
        self.assertEqual(self.prove(challenge).status_code,200)
        self.client.cookies.set(COOKIE,cookie,domain='review.owner-runtime.net',path='/identity')
        self.assertEqual(self.prove(challenge).status_code,401)
        challenge=self.bootstrap();cookie=self.client.cookies.get(COOKIE)
        self.assertEqual(self.prove(challenge,self.token('wrong')).status_code,401)
        self.client.cookies.set(COOKIE,cookie,domain='review.owner-runtime.net',path='/identity')
        self.assertEqual(self.prove(challenge).status_code,401)
        challenge=self.bootstrap();cookie=self.client.cookies.get(COOKIE)
        self.assertEqual(self.client.post('/identity/cancel',headers={'Origin':ORIGIN,'X-Proof-CSRF':challenge['csrf']}).status_code,200)
        self.client.cookies.set(COOKIE,cookie,domain='review.owner-runtime.net',path='/identity');self.assertEqual(self.prove(challenge).status_code,401)
        challenge=self.bootstrap();self.now+=121
        self.assertEqual(self.prove(challenge).status_code,401)

    def test_new_challenge_restart_and_clock_rollback_discard_old_proof(self):
        first=self.bootstrap();cookie=self.client.cookies.get(COOKIE);second=self.bootstrap()
        self.assertEqual(len(self.app.state.identity_proof.challenges),1)
        with self.assertRaises(AuthenticationError):self.app.state.identity_proof.prove(cookie,first['csrf'],self.token(first['nonce']))
        restarted=IdentityProof(self.profile,key_source=self.keys,clock=self.clock)
        with self.assertRaises(AuthenticationError):restarted.prove(self.client.cookies.get(COOKIE),second['csrf'],self.token(second['nonce']))
        self.now-=1;self.assertEqual(self.prove(second).status_code,503)

    def test_challenge_must_still_be_current_when_verification_returns(self):
        challenge=self.bootstrap()
        verifier=self.app.state.identity_proof.verifier
        original=verifier.verify_identity
        def delayed(token,nonce):
            subject=original(token,nonce);self.now+=121;return subject
        with patch.object(verifier,'verify_identity',side_effect=delayed):
            self.assertEqual(self.prove(challenge).status_code,401)

    def test_capture_window_closes_all_routes_and_never_reopens_itself(self):
        self.bootstrap();self.now+=self.profile.capture_window_seconds
        for path in ('/identity','/identity/bootstrap','/identity/assets/identity.js','/healthz'):
            response=self.client.get(path);self.assertEqual(response.status_code,503)
            self.assertEqual(response.json()['error'],'identity_capture_window_closed')
        self.assertEqual(self.app.state.identity_proof.challenges,{})
        with self.assertRaises(AuthenticationError):self.app.state.identity_proof.bootstrap('owner-browser')
        self.now-=30  # A wall-clock correction cannot reopen this process.
        self.assertEqual(self.client.get('/identity').status_code,503)

    def test_monotonic_limit_cannot_be_extended_by_wall_clock_correction(self):
        elapsed=[10.0]
        proof=IdentityProof(self.profile,key_source=self.keys,clock=self.clock,monotonic=lambda:elapsed[0])
        proof.bootstrap('owner-browser')
        elapsed[0]+=self.profile.capture_window_seconds
        with self.assertRaises(AuthenticationError):proof.bootstrap('owner-browser')
        elapsed[0]-=30
        with self.assertRaises(AuthenticationError):proof.bootstrap('owner-browser')

    def test_public_shell_navigation_is_allowed_but_cross_site_proof_fetch_is_refused(self):
        self.assertEqual(self.client.get('/identity',headers={'Sec-Fetch-Site':'cross-site'}).status_code,200)
        self.assertEqual(self.client.get('/identity/bootstrap',headers={'Sec-Fetch-Site':'cross-site'}).status_code,403)

    def test_csrf_cookie_origin_cors_queries_and_duplicate_fields_fail_closed(self):
        challenge=self.bootstrap()
        self.assertEqual(self.client.post('/identity/proof',json={'credential':self.token(challenge['nonce'])}).status_code,403)
        self.assertEqual(self.prove(challenge,Origin='https://attacker.invalid').status_code,403)
        self.assertEqual(self.client.options('/identity/proof',headers={'Origin':ORIGIN}).status_code,403)
        self.assertEqual(self.client.get('/identity/bootstrap?token=FICTIONAL').status_code,400)
        self.client.cookies.clear();challenge=self.bootstrap()
        cookie=self.client.cookies.get(COOKIE)
        self.assertEqual(self.client.post('/identity/proof',headers={'Origin':ORIGIN,'X-Proof-CSRF':challenge['csrf'],
            'Cookie':COOKIE+'='+cookie+'; '+COOKIE+'='+cookie},json={'credential':self.token(challenge['nonce'])}).status_code,401)
        self.client.cookies.clear();challenge=self.bootstrap()
        duplicate='{"credential":"FICTIONAL","credential":"FICTIONAL"}'
        self.assertEqual(self.client.post('/identity/proof',headers={'Origin':ORIGIN,'X-Proof-CSRF':challenge['csrf'],
            'Content-Type':'application/json'},content=duplicate).status_code,401)

    def test_wrong_csrf_consumes_challenge_and_missing_cookie_never_proves(self):
        challenge=self.bootstrap();cookie=self.client.cookies.get(COOKIE)
        self.assertEqual(self.prove(challenge,**{'X-Proof-CSRF':'wrong'}).status_code,401)
        self.client.cookies.set(COOKIE,cookie,domain='review.owner-runtime.net',path='/identity')
        self.assertEqual(self.prove(challenge).status_code,401)
        challenge=self.bootstrap();self.client.cookies.clear()
        self.assertEqual(self.prove(challenge).status_code,401)

    def test_non_ascii_cancel_csrf_is_sanitized_and_never_server_error(self):
        challenge=self.bootstrap()
        response=self.client.post('/identity/cancel',headers=[('Origin',ORIGIN),
            (b'X-Proof-CSRF',b'non-ascii-\xff')])
        self.assertEqual(response.status_code,401)
        self.assertEqual(response.json(),{'error':'identity_proof_failed','authority':'none'})

    def test_peer_bounds_ignore_raw_forwarding_and_expire_without_persistence(self):
        for _ in range(self.profile.bootstrap_per_peer_per_minute):
            self.assertEqual(self.client.get('/identity/bootstrap',headers={'X-Forwarded-For':'spoofed'}).status_code,200)
        self.assertEqual(self.client.get('/identity/bootstrap',headers={'X-Forwarded-For':'other'}).status_code,401)
        self.assertLessEqual(len(self.app.state.identity_proof.challenges),self.profile.max_challenges)
        self.now+=60;self.assertEqual(self.client.get('/identity/bootstrap').status_code,200)


class ProofConfigTests(unittest.TestCase):
    def test_proposed_unit_is_isolated_temporary_and_has_no_ledger_or_secret_dependency(self):
        unit=(ROOT/'deploy/zetbros-identity-proof.service.example').read_text()
        for required in ('DynamicUser=yes','User=zetbros-proof','RuntimeMaxSec=600','Restart=no',
                         '--port 8082','--forwarded-allow-ips 127.0.0.1','MemoryMax=80M','TasksMax=16',
                         'InaccessiblePaths=-/srv/zetbros/pilot -/etc/zetbros/pilot -/run/credentials'):
            self.assertIn(required,unit)
        for absent in ('LoadCredential','ReadWritePaths','Group=zetbros-pilot','[Install]','--port 8081'):
            self.assertNotIn(absent,unit)

    def test_disabled_example_and_invalid_profile_cannot_activate(self):
        profile=IdentityProofProfile.model_validate_json((ROOT/'deploy/identity-proof.disabled.example.json').read_text())
        with self.assertRaises(ValueError):IdentityProof(profile)
        for change in ({'enabled':False,'client_id':'FAKE'},{'enabled':True},
                       {'enabled':True,'client_id':CID,'origin':'http://localhost:8082'},
                       {'enabled':True,'client_id':CID,'origin':ORIGIN,'reviewer_subjects':[SUB]}):
            with self.assertRaises(ValueError):IdentityProofProfile.model_validate_json(json.dumps(change))

    def test_config_loader_regular_bounded_enabled_and_explicit_path_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'profile.json'
            profile=IdentityProofProfile(enabled=True,client_id=CID,origin=ORIGIN)
            path.write_text(profile.model_dump_json())
            with patch.dict(os.environ,{'ZETBROS_IDENTITY_PROOF_CONFIG_FILE':str(path)}):
                self.assertEqual(load_identity_proof_profile(),profile)
                path.write_text('x'*16385)
                with self.assertRaises(RuntimeError):load_identity_proof_profile()
                path.unlink();path.symlink_to(pathlib.Path(directory)/'missing')
                with self.assertRaises(RuntimeError):load_identity_proof_profile()
            with patch.dict(os.environ,{'ZETBROS_IDENTITY_PROOF_CONFIG_FILE':''}):
                with self.assertRaises(RuntimeError):load_identity_proof_profile()
