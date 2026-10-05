"""Exact disabled deployment walkthrough: no identity fabrication or provider I/O."""
from __future__ import annotations
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import test_pilot_runtime as fixtures
from zetbros_service.config import PendingPilotConfig, Settings
from zetbros_service.models import canonical
from zetbros_service.pilot_runtime import (build_service, configured_pilot_app,
    initialize_new_pilot, pending_status, stage_without_ledger)
from zetbros_service.store import Store

ROOT=pathlib.Path(__file__).resolve().parents[1]


class PendingWalkthroughTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temp.name)
        data=json.loads((ROOT/'deploy/pilot-runtime.disabled.example.json').read_text())
        data['database_path']=str(self.root/'not-created.sqlite3')
        self.config=PendingPilotConfig.model_validate_json(json.dumps(data))
        self.path=self.root/'service.json';self.path.write_text(canonical(self.config))
        self.env=os.environ|{'ZETBROS_CONFIG_FILE':str(self.path)}
    def tearDown(self):self.temp.cleanup()
    def cli(self,*args):
        return subprocess.run([sys.executable,'-m','zetbros_service.pilot',*args],cwd=ROOT,env=self.env,
            text=True,capture_output=True,timeout=10)

    def test_exact_shipped_disabled_config_init_check_start_never_creates_ledger(self):
        before=set(self.root.iterdir())
        initialized=self.cli('init');checked=self.cli('check')
        self.assertEqual(initialized.returncode,0,initialized.stderr)
        self.assertEqual(checked.returncode,0,checked.stderr)
        self.assertFalse(json.loads(initialized.stdout)['ledger_created'])
        self.assertEqual(json.loads(checked.stdout)['identity_configuration'],'pending')
        with patch.dict(os.environ,{'ZETBROS_CONFIG_FILE':str(self.path)}), \
             patch('zetbros_service.pilot_runtime.Verifier',side_effect=AssertionError('identity consulted')), \
             patch('zetbros_service.pilot_runtime.LiveSessionFactory',side_effect=AssertionError('factory constructed')), \
             patch('zetbros_service.live_session_factory.SystemdCredentialSource.read',side_effect=AssertionError('credential read')):
            app=configured_pilot_app()
            with TestClient(app) as client:
                self.assertEqual(client.get('/healthz').status_code,200)
                ready=client.get('/readyz');self.assertEqual(ready.status_code,503)
                self.assertEqual(ready.json()['delivery'],'disabled')
                self.assertEqual(client.post('/v1/proposals',json={'body':'must not be processed'}).status_code,503)
                self.assertEqual(client.get('/review').status_code,503)
        self.assertEqual(set(self.root.iterdir()),before)
        self.assertFalse(pathlib.Path(self.config.database_path).exists())

    def test_pending_schema_rejects_identity_fields_and_live_flags(self):
        data=self.config.model_dump(mode='json')
        for changes in ({'issuer':'https://placeholder.invalid'}, {'tenant_id':'REPLACE_CUSTOMER'},
                        {'principals':[]}, {'outbound_adapter':'private_spacemail'}, {'source_adapter':'private_spacemail'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                PendingPilotConfig.model_validate_json(json.dumps(data|changes))
        self.assertFalse(pathlib.Path(self.config.database_path).exists())

    def test_pending_init_refuses_existing_and_dangling_symlink_paths_without_change(self):
        path=pathlib.Path(self.config.database_path);path.write_bytes(b'EXISTING_NOT_A_LEDGER')
        with self.assertRaises(FileExistsError):stage_without_ledger(self.config)
        self.assertEqual(path.read_bytes(),b'EXISTING_NOT_A_LEDGER')
        path.unlink();path.symlink_to(self.root/'absent-target')
        with self.assertRaises(FileExistsError):stage_without_ledger(self.config)
        self.assertTrue(path.is_symlink())

    def test_pending_maintenance_entrypoint_cannot_create_or_restore_a_ledger(self):
        for args in (['init'],['check'],['backup','--destination',str(self.root/'backup.sqlite3')],
                     ['restore','--source',str(self.root/'old.sqlite3')]):
            result=subprocess.run([sys.executable,'-m','zetbros_service.maintenance',*args],cwd=ROOT,
                env=self.env,text=True,capture_output=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('identity pending',result.stderr)
            self.assertFalse(pathlib.Path(self.config.database_path).exists())
            self.assertFalse((self.root/'backup.sqlite3').exists())

    def test_default_disabled_unit_has_no_credential_dependency(self):
        unit=(ROOT/'deploy/zetbros-pilot.service.example').read_text()
        self.assertNotIn('LoadCredential',unit)
        self.assertIn('MemoryMax=128M',unit)
        self.assertIn('configured_pilot_app',unit)
        self.assertIn('LoadCredential=spacemail-password:',(ROOT/'deploy/zetbros-pilot.live.service.example').read_text())


class ConfiguredDisabledLedgerTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.PilotAssemblyTests('test_new_absent_pilot_ledger_initializes_without_provider_io')
        self.fixture.setUp()
        data=self.fixture.settings.model_dump(mode='json')
        data['database_path']=str(self.fixture.fixture.fixture.root/'data/disabled-new.sqlite3')
        data['outbound_adapter']='disabled'
        self.settings=Settings.model_validate_json(json.dumps(data))
        self.profile=self.fixture.configured_profile.model_copy(update={'outbound':'disabled'})
        self.fixture.profile_file.write_text(canonical(self.profile))
    def tearDown(self):self.fixture.tearDown()

    def test_disabled_init_check_and_app_use_mail_store_without_any_factory(self):
        def forbidden(profile):raise AssertionError('disabled factory constructed')
        with patch('zetbros_service.live_session_factory.SystemdCredentialSource.read',side_effect=AssertionError('credential read')):
            store=initialize_new_pilot(self.settings,factory_builder=forbidden)
            self.assertTrue(store.healthy())
            service=build_service(self.settings,factory_builder=forbidden)
        self.assertEqual(service.store.workflow_contract,'mail_exact_wire_v1')
        self.assertIsNone(service.prepared_transport)
        self.assertEqual(service.readiness()['delivery'],'disabled')
        self.assertFalse(service.readiness()['live_delivery_ready'])
        with self.assertRaises(Exception):service.read_source('imap-v1:123:42')
        with self.assertRaises(RuntimeError):Store(self.settings)

    def test_bad_public_jwks_or_placeholder_identity_never_writes_new_ledger(self):
        path=pathlib.Path(self.settings.database_path)
        data=self.settings.model_dump(mode='json');data['jwks_file']=str(self.fixture.fixture.fixture.root/'missing-jwks.json')
        with self.assertRaises(Exception):initialize_new_pilot(Settings.model_validate_json(json.dumps(data)))
        self.assertFalse(path.exists())
        data=self.settings.model_dump(mode='json');data['tenant_id']='REPLACE_CUSTOMER_ID'
        with self.assertRaises(RuntimeError):initialize_new_pilot(Settings.model_validate_json(json.dumps(data)))
        self.assertFalse(path.exists())

    def test_later_live_configuration_preserves_same_identity_and_mail_ledger(self):
        store=initialize_new_pilot(self.settings)
        original_identity=self.settings.identity_digest
        live=self.settings.model_copy(update={'outbound_adapter':'private_spacemail'})
        self.fixture.profile_file.write_text(canonical(self.profile.model_copy(update={'outbound':'owner_approved_live'})))
        service=build_service(live,factory_builder=self.fixture.factory)
        self.assertEqual(service.settings.identity_digest,original_identity)
        self.assertEqual(service.store.path,store.path)
        self.assertEqual(self.fixture.credentials.calls,0)
        with self.assertRaises(FileExistsError):initialize_new_pilot(live)

    def test_actual_mailbox_cannot_change_behind_an_unchanged_account_label(self):
        initialize_new_pilot(self.settings)
        changed=self.profile.model_copy(update={'mailbox_address':'another-mailbox@owner-runtime.net',
            'approved_aliases':(self.profile.binding.sender_address,)})
        self.fixture.profile_file.write_text(canonical(changed))
        with self.assertRaises(RuntimeError):build_service(self.settings)

    def test_private_maintenance_init_cannot_create_the_wrong_action_ledger(self):
        path=self.fixture.fixture.fixture.root/'private-disabled-service.json'
        path.write_text(canonical(self.settings))
        result=subprocess.run([sys.executable,'-m','zetbros_service.maintenance','init'],cwd=ROOT,
            env=os.environ|{'ZETBROS_CONFIG_FILE':str(path)},text=True,capture_output=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('action-only initialization is refused',result.stderr)
        self.assertFalse(pathlib.Path(self.settings.database_path).exists())

    def test_resolved_identity_cannot_rebind_existing_disabled_ledger(self):
        initialize_new_pilot(self.settings)
        changed=self.settings.model_copy(update={'audience':'another-owner-audience'})
        with self.assertRaises(RuntimeError):build_service(changed)
        with self.assertRaises(FileExistsError):initialize_new_pilot(changed)
