"""Live-capable assembly tested exclusively with injected fakes; no sockets/secrets."""
from __future__ import annotations

import asyncio
import io
import json
import os
import pathlib
import socket
import ssl
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import test_mail_integration as integration
import test_private_wire_bridge as bridge
from zetbros_service.config import PendingPilotConfig, Settings
from zetbros_service.live_session_factory import (BoundedResolver, ConnectionGuard, LiveSessionFactory,
    PinnedConnector, SystemdCredentialSource, _PinnedImap, _PinnedSmtp)
from zetbros_service.models import canonical
from zetbros_service.pilot_runtime import build_service, initialize_new_pilot
from zetbros_service.private_wire_bridge import PrivateBridgeProfile
from zetbros_service.spacemail_contract import ContractError

LIVE = bridge.PROFILE.model_copy(update={"outbound": "owner_approved_live"})


class SyntheticCredential:
    def __init__(self): self.calls = 0
    def read(self): self.calls += 1; return "SYNTHETIC_NOT_A_PROVIDER_SECRET"


class FakeAuthenticatedSmtp(bridge.FakeSmtp):
    def __init__(self, host, port, guard, connector):
        super().__init__(); self.host, self.port = host, port; guard.register(self.sock)
        self.logins = []
    def ehlo(self): return 250, b"AUTH LOGIN"
    def login(self, username, password): self.logins.append((username, password)); return 235, b"ok"
    def close(self): self.sock.close()


class FakeAuthenticatedImap(bridge.FakeImap):
    def __init__(self, host, port, guard, connector):
        super().__init__(); self.host, self.port = host, port; guard.register(self.sock)
        self.logins = []; self.selections = []
    def login(self, username, password): self.logins.append((username, password)); return "OK", [b"ok"]
    def select(self, folder, readonly=True): self.selections.append((folder, readonly)); return "OK", [b"1"]
    def shutdown(self): self.sock.close()


class FactoryTests(unittest.TestCase):
    def factory(self, **kwargs):
        self.credential = SyntheticCredential()
        return LiveSessionFactory(LIVE, credentials=self.credential,
            smtp_client=FakeAuthenticatedSmtp, imap_client=FakeAuthenticatedImap, **kwargs)

    def test_factory_authenticates_fixed_account_readonly_folder_and_isolated_sessions(self):
        factory = self.factory()
        with patch("socket.socket", side_effect=AssertionError("socket attempted")):
            with factory.smtp(LIVE, deadline=time.monotonic()+2, timeout=1) as bound:
                self.assertEqual((bound.session.host, bound.session.port), ("mail.spacemail.com", 465))
                self.assertEqual(bound.session.logins[0][0], LIVE.mailbox_address)
                self.assertIsNone(bound.session.password)
                first = bound.session
            with factory.imap(LIVE, folder="INBOX", readonly=True, deadline=time.monotonic()+2, timeout=1) as bound:
                self.assertEqual(bound.session.selections, [('"INBOX"', True)])
                self.assertIsNot(bound.session, first)
        self.assertEqual(self.credential.calls, 2)

    def test_disabled_or_mismatched_profile_refuses_before_credential(self):
        with self.assertRaises(ContractError): LiveSessionFactory(bridge.PROFILE)
        factory = self.factory()
        other = LIVE.model_copy(update={"mailbox_address": "other@example.invalid", "approved_aliases": (LIVE.binding.sender_address,)})
        with self.assertRaises(ContractError):
            with factory.smtp(other, deadline=time.monotonic()+2, timeout=1): pass
        with self.assertRaises(ContractError):
            with factory.imap(LIVE, folder="Trash", readonly=False, deadline=time.monotonic()+2, timeout=1): pass
        self.assertEqual(self.credential.calls, 0)

    def test_smtp_auth_ack_must_be_positive_for_the_fresh_account_session(self):
        class Refused(FakeAuthenticatedSmtp):
            def login(self,username,password):return 535,b"SYNTHETIC_AUTH_CANARY"
        factory=LiveSessionFactory(LIVE,credentials=SyntheticCredential(),smtp_client=Refused,
            imap_client=FakeAuthenticatedImap)
        with self.assertRaises(ContractError) as error:
            with factory.smtp(LIVE,deadline=time.monotonic()+2,timeout=1):self.fail("refused session yielded")
        self.assertNotIn("CANARY",str(error.exception))

    def test_protected_credential_missing_unsafe_symlink_oversize_and_no_echo(self):
        source = SystemdCredentialSource()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ContractError): source.read()
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / source.name
            path.write_text("SYNTHETIC_NOT_A_PROVIDER_SECRET"); path.chmod(0o600)
            with patch.dict(os.environ, {"CREDENTIALS_DIRECTORY": root}):
                self.assertEqual(source.read(), "SYNTHETIC_NOT_A_PROVIDER_SECRET")
                path.chmod(0o644)
                with self.assertRaises(ContractError) as error: source.read()
                self.assertNotIn("SYNTHETIC", str(error.exception))
                path.chmod(0o600); path.write_bytes(b"x"*513)
                with self.assertRaises(ContractError): source.read()
                path.unlink(); path.symlink_to(pathlib.Path(root)/"missing")
                with self.assertRaises(ContractError): source.read()

    def test_resolver_wait_and_worker_count_are_bounded(self):
        release = threading.Event(); calls=[]
        def lookup(*args, **kwargs):
            calls.append(args); release.wait(1); return []
        resolver = BoundedResolver(lookup=lookup)
        for _ in range(3):
            with self.assertRaises(ContractError): resolver.resolve("mail.spacemail.com",465,0.01)
        self.assertEqual(len(calls),1); release.set()
        with self.assertRaises(ContractError): resolver.resolve("attacker.invalid",465,0.01)

    def test_resolver_cached_addresses_remain_correct_for_each_port(self):
        resolver = BoundedResolver(lookup=lambda *a,**k: [(socket.AF_INET,socket.SOCK_STREAM,6,"",("203.0.113.1",0))])
        self.assertEqual(resolver.resolve("mail.spacemail.com",465,1)[0][-1][1],465)
        self.assertEqual(resolver.resolve("mail.spacemail.com",993,1)[0][-1][1],993)

    def test_pinned_connector_registers_tls_before_handshake_without_actual_socket(self):
        calls=[]
        class FakeNet(bridge.FakeSocket):
            def connect(self, location): calls.append(("connect",location))
            def do_handshake(self): calls.append(("handshake",self in guard.sockets))
        raw, secured = FakeNet(), FakeNet()
        class Context:
            verify_mode=ssl.CERT_REQUIRED;check_hostname=True;minimum_version=ssl.TLSVersion.TLSv1_2
            def wrap_socket(self,sock,**kwargs): calls.append(("wrap",kwargs));return secured
        resolver=BoundedResolver(lookup=lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,"",("203.0.113.1",0))])
        connector=PinnedConnector(resolver=resolver,socket_factory=lambda *a:raw,tls_context=Context)
        guard=ConnectionGuard(time.monotonic()+2,1)
        try:
            self.assertIs(connector.connect("mail.spacemail.com",465,guard),secured)
            self.assertEqual(calls[-1],("handshake",True))
            self.assertFalse(calls[1][1]["do_handshake_on_connect"])
        finally:guard.close()

    def test_real_smtp_subclass_uses_no_local_dns_with_fake_io(self):
        class FakeIo(bridge.FakeSocket):
            def makefile(self, mode): return io.BytesIO(b"220 ready\r\n")
        class Connector:
            def connect(self,host,port,guard): return guard.register(FakeIo())
        guard=ConnectionGuard(time.monotonic()+2,1)
        try:
            with patch("socket.getfqdn",side_effect=AssertionError("local DNS")), patch("socket.gethostbyname",side_effect=AssertionError("local DNS")):
                client=_PinnedSmtp("mail.spacemail.com",465,guard,Connector())
                self.assertEqual(client.local_hostname,"[127.0.0.1]");client.close()
        finally:guard.close()

    def test_imap_debug_cache_never_retains_login_or_provider_text(self):
        imap=object.__new__(_PinnedImap)
        imap._cmd_log={};imap._cmd_log_idx=0;imap._cmd_log_len=10
        imap._log(b"LOGIN user SYNTHETIC_NOT_A_PROVIDER_SECRET")
        imap._mesg("SYNTHETIC_NOT_A_PROVIDER_SECRET")
        self.assertEqual(imap._cmd_log,{})

    def test_smtp_aggregate_multiline_and_imap_literal_limits_precede_materialization(self):
        guard=ConnectionGuard(time.monotonic()+2,1)
        try:
            smtp=object.__new__(_PinnedSmtp);smtp._guard=guard;smtp.sock=bridge.FakeSocket()
            smtp.file=io.BytesIO(b"250-more\r\n"*33)
            with self.assertRaises(ContractError):smtp.getreply()
            imap=object.__new__(_PinnedImap);imap._guard=guard;imap.sock=bridge.FakeSocket();imap._literal_count=0
            class NoRead:
                def read(self,n):raise AssertionError("oversize allocation attempted")
            imap.file=NoRead()
            with self.assertRaises(ContractError):imap.read(65537)
        finally:guard.close()


class PilotAssemblyTests(unittest.TestCase):
    def setUp(self):
        integration.MailIntegrationTests.setUpClass()
        self.fixture=integration.MailIntegrationTests("test_exact_claim_gate_and_wire_no_retry_survive_restart")
        self.fixture.setUp()
        self.profile_file=self.fixture.fixture.root/"private-profile.json"
        self.configured_profile=LIVE.model_copy(update={
            "binding":LIVE.binding.model_copy(update={"sender_address":"support@owner-runtime.net"}),
            "mailbox_address":"support@owner-runtime.net"})
        self.profile_file.write_text(canonical(self.configured_profile))
        data=self.fixture.fixture.settings.model_dump(mode="json")
        data.update(sender_address="support@owner-runtime.net",issuer="https://identity.owner-runtime.net",
            database_path=str(self.fixture.fixture.root/"data/runtime-pilot.sqlite3"))
        data.update(source_adapter="private_spacemail",outbound_adapter="private_spacemail",
            private_bridge_profile_file=str(self.profile_file),review_origin="http://127.0.0.1:8081")
        self.settings=Settings.model_validate_json(json.dumps(data))
        self.credentials=SyntheticCredential();self.sessions=[]
        def smtp(host,port,guard,connector):
            obj=FakeAuthenticatedSmtp(host,port,guard,connector);self.sessions.append(obj);return obj
        self.factory=lambda profile:LiveSessionFactory(profile,credentials=self.credentials,
            smtp_client=smtp,imap_client=FakeAuthenticatedImap)
        initialize_new_pilot(self.settings,factory_builder=self.factory,clock=self.fixture.fixture.clock)
    def tearDown(self):self.fixture.tearDown()

    def test_operator_runtime_is_assembled_without_startup_connections_or_credentials(self):
        with patch("socket.socket",side_effect=AssertionError("socket attempted")):
            service=build_service(self.settings,factory_builder=self.factory,clock=self.fixture.fixture.clock)
        self.assertEqual(self.credentials.calls,0)
        self.assertFalse(service.readiness()["live_delivery_ready"])
        proposal=service.propose(self.fixture.input(),integration.base.AGENT,"pilot-proposal")
        service.store.decide(proposal["id"],proposal["digest"],"approve",integration.base.REVIEWER,"pilot-review")
        async def run():
            # asyncio's own AF_UNIX wake-up socketpair is local IPC, not a
            # provider/listening network socket; the injected clients never open one.
            await service.worker_once();await service.worker_once()
        with patch("socket.create_connection",side_effect=AssertionError("provider socket attempted")):
            asyncio.run(run())
        final=service.store.get(proposal["id"])
        self.assertEqual(final["execution"]["state"],"accepted")
        self.assertEqual(len([x for session in self.sessions for x in session.calls if x[0]=="send"]),1)

    def test_blocked_source_does_not_block_async_review_or_health_loop(self):
        service=build_service(self.settings,factory_builder=self.factory,clock=self.fixture.fixture.clock)
        proposal=service.propose(self.fixture.input(),integration.base.AGENT,"pilot-responsive")
        service.store.decide(proposal["id"],proposal["digest"],"approve",integration.base.REVIEWER,"pilot-review")
        entered=threading.Event();release=threading.Event();original=service.source.get
        def blocked(source_id):
            entered.set()
            if not release.wait(2):raise AssertionError("source worker not released")
            return original(source_id)
        async def run():
            with patch.object(service.source,"get",side_effect=blocked):
                task=asyncio.create_task(service.worker_once())
                self.assertTrue(await asyncio.to_thread(entered.wait,1))
                progressed=asyncio.Event()
                async def heartbeat():await asyncio.sleep(0);progressed.set()
                await asyncio.wait_for(heartbeat(),0.2)
                self.assertTrue(progressed.is_set())
                release.set();await task
        asyncio.run(run())
        self.assertEqual(service.store.get(proposal["id"])["execution"]["state"],"accepted")

    def test_new_ledger_initializer_refuses_existing_path_without_credentials_or_network(self):
        with self.assertRaises(FileExistsError):initialize_new_pilot(self.settings,factory_builder=self.factory)
        self.assertEqual(self.credentials.calls,0)
        self.assertEqual(self.sessions,[])

    def test_new_absent_pilot_ledger_initializes_without_provider_io(self):
        data=self.settings.model_dump(mode="json");data["database_path"]=str(self.fixture.fixture.root/"data/new-pilot.sqlite3")
        settings=Settings.model_validate_json(json.dumps(data))
        with patch("socket.socket",side_effect=AssertionError("socket attempted")):
            store=initialize_new_pilot(settings,factory_builder=self.factory)
        self.assertTrue(store.healthy());self.assertEqual(self.credentials.calls,0)
        with self.assertRaises(FileExistsError):initialize_new_pilot(settings,factory_builder=self.factory)

    def test_shipped_default_runtime_stays_disabled_and_reads_no_profile(self):
        # The fixture already has a mail ledger, so base runtime refuses it;
        # importantly no factory/profile or credential is consulted on this path.
        with patch("zetbros_service.pilot_runtime.load_profile",side_effect=AssertionError("profile read")), self.assertRaises(RuntimeError):
            build_service(self.fixture.fixture.settings,factory_builder=self.factory)
        self.assertEqual(self.credentials.calls,0)

    def test_proposed_unit_limits_and_shipped_runtime_remain_dormant(self):
        root=pathlib.Path(__file__).resolve().parents[1]
        settings=PendingPilotConfig.model_validate_json((root/'deploy/pilot-runtime.disabled.example.json').read_text())
        self.assertEqual((settings.source_adapter,settings.outbound_adapter),('disabled','disabled'))
        unit=(root/'deploy/zetbros-pilot.service.example').read_text()
        for value in ('User=zetbros-pilot','MemoryHigh=96M','MemoryMax=128M','MemorySwapMax=0','CPUQuota=25%',
                      'TasksMax=32','--host 127.0.0.1 --port 8081 --workers 1'):
            self.assertIn(value,unit)
        self.assertNotIn('LoadCredential=',unit)
        self.assertNotIn('SYNTHETIC_NOT_A_PROVIDER_SECRET',unit)

    def test_profile_drift_and_unpaired_adapters_fail_closed(self):
        self.profile_file.write_text(canonical(self.configured_profile.model_copy(update={"binding":self.configured_profile.binding.model_copy(update={"tenant_id":"other-customer"})})))
        with self.assertRaises(RuntimeError):build_service(self.settings,factory_builder=self.factory)
        data=self.settings.model_dump(mode="json");data["source_adapter"]="customer_staged_snapshot"
        with self.assertRaises(ValueError):Settings.model_validate_json(json.dumps(data))
        self.assertEqual(self.credentials.calls,0)
