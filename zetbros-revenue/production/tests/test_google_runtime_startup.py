"""Fail-closed Google runtime setup and the unchanged pending-identity stage."""
import json
import pathlib
import unittest
from unittest.mock import patch

import test_disabled_pilot_walkthrough as fixtures
from test_google_reviewer_auth import CID, SUB, ORIGIN
from zetbros_service.config import Grant, PendingPilotConfig
from zetbros_service.google_auth_runtime import load_google_review_profile
from zetbros_service.google_reviewer_auth import GoogleReviewerProfile, AgentOnlyVerifier
from zetbros_service.models import canonical
from zetbros_service.pilot_runtime import initialize_new_pilot, build_service, stage_without_ledger

ROOT=pathlib.Path(__file__).resolve().parents[1]


class GoogleStartupTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ConfiguredDisabledLedgerTests('test_disabled_init_check_and_app_use_mail_store_without_any_factory')
        self.fixture.setUp()
        self.profile_path=self.fixture.fixture.fixture.fixture.root/'google-reviewer.json'
        self.profile=GoogleReviewerProfile(enabled=True,client_id=CID,origin=ORIGIN,reviewer_subjects=(SUB,))
        self.profile_path.write_text(canonical(self.profile))
        settings=self.fixture.settings
        agents=tuple(grant for grant in settings.principals if grant.role=='agent')
        self.settings=settings.model_copy(update={'review_origin':ORIGIN,'reviewer_auth':'google_oidc',
            'google_reviewer_profile_file':str(self.profile_path),
            'principals':agents+(Grant(subject=SUB,client_id=CID,role='reviewer'),)})
    def tearDown(self):self.fixture.tearDown()

    def test_disabled_google_runtime_initializes_without_keys_network_factory_or_credentials(self):
        with patch('zetbros_service.google_public_keys.GooglePublicKeys.get_jwks',side_effect=AssertionError('Google network')), \
             patch('zetbros_service.pilot_runtime.LiveSessionFactory',side_effect=AssertionError('mail factory')), \
             patch('zetbros_service.live_session_factory.SystemdCredentialSource.read',side_effect=AssertionError('credential')):
            store=initialize_new_pilot(self.settings)
            service=build_service(self.settings,factory_builder=lambda _: (_ for _ in ()).throw(AssertionError('factory')))
        self.assertTrue(store.healthy());self.assertIsInstance(service.verifier,AgentOnlyVerifier)
        self.assertTrue(service.google_reviewer_auth.profile.enabled)
        self.assertIsNone(service.prepared_transport)
        self.assertFalse(service.readiness()['live_delivery_ready'])
        self.assertEqual(service.google_reviewer_auth.id_tokens.keys,{})
        with self.assertRaises(FileExistsError):initialize_new_pilot(self.settings)

    def test_disabled_missing_or_mismatched_google_profile_cannot_pin_new_ledger(self):
        bad=(GoogleReviewerProfile(),self.profile.model_copy(update={'origin':'https://another.owner-runtime.net'}),
             self.profile.model_copy(update={'reviewer_subjects':('another-approved-sub',)}))
        for profile in bad:
            self.profile_path.write_text(canonical(profile))
            with self.assertRaises(RuntimeError):initialize_new_pilot(self.settings)
            self.assertFalse(pathlib.Path(self.settings.database_path).exists())
        self.profile_path.unlink()
        with self.assertRaises(RuntimeError):initialize_new_pilot(self.settings)
        self.assertFalse(pathlib.Path(self.settings.database_path).exists())

    def test_profile_requires_regular_bounded_file_exact_mapping_and_separate_agent_resource(self):
        target=self.profile_path.with_suffix('.original');self.profile_path.rename(target)
        self.profile_path.symlink_to(target)
        with self.assertRaises(RuntimeError):load_google_review_profile(self.settings)
        self.profile_path.unlink();self.profile_path.write_text('x'*16385)
        with self.assertRaises(RuntimeError):load_google_review_profile(self.settings)
        self.profile_path.write_text(canonical(self.profile))
        for change in ({'issuer':'https://accounts.google.com'},{'audience':CID},
                       {'principals':(Grant(subject=SUB,client_id=CID,role='agent'),)}):
            with self.assertRaises(RuntimeError):load_google_review_profile(self.settings.model_copy(update=change))

    def test_pending_example_remains_identity_free_and_does_not_select_google_login(self):
        data=json.loads((ROOT/'deploy/pilot-runtime.disabled.example.json').read_text())
        data['database_path']=str(self.profile_path.with_suffix('.absent'))
        pending=PendingPilotConfig.model_validate_json(json.dumps(data))
        self.assertFalse(stage_without_ledger(pending)['ledger_created'])
        for change in ({'reviewer_auth':'google_oidc'},{'google_reviewer_profile_file':str(self.profile_path)},
                       {'review_origin':ORIGIN},{'principals':[]}):
            with self.assertRaises(ValueError):PendingPilotConfig.model_validate_json(json.dumps(data|change))
        example=GoogleReviewerProfile.model_validate_json((ROOT/'deploy/google-reviewer.disabled.example.json').read_text())
        self.assertFalse(example.enabled);self.assertIsNone(example.client_id)
        self.assertIsNone(example.origin);self.assertFalse(example.reviewer_subjects)
