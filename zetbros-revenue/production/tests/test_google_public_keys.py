"""Fixed public JWKS stdlib transport exercised with fake DNS/TLS/socket I/O."""
import io
import json
import socket
import time
import unittest
from unittest.mock import patch

from zetbros_service.google_public_keys import GooglePublicKeys
from zetbros_service.google_reviewer_auth import GOOGLE_JWKS_URL
from zetbros_service.spacemail_contract import ContractError


class FakeSocket:
    def __init__(self,response):self.response=response;self.sent=[];self.closed=False;self.calls=[]
    def settimeout(self,value):self.calls.append(('timeout',value))
    def connect(self,location):self.calls.append(('connect',location))
    def do_handshake(self):self.calls.append(('handshake',))
    def sendall(self,value):self.sent.append(value)
    def makefile(self,*args):return io.BytesIO(self.response)
    def shutdown(self,*args):self.closed=True
    def close(self):self.closed=True


class Context:
    def __init__(self,sock):self.sock=sock;self.calls=[]
    def wrap_socket(self,raw,**kwargs):self.calls.append(kwargs);return self.sock


class GooglePublicKeysTests(unittest.TestCase):
    def fetch(self,body,status='200 OK',extra='',declared=None):
        size=len(body) if declared is None else declared
        response=('HTTP/1.1 '+status+'\r\nContent-Length: '+str(size)+'\r\n'+extra+'\r\n').encode()+body
        raw,secured=FakeSocket(b''),FakeSocket(response);context=Context(secured)
        calls=[]
        def lookup(host,port,**kwargs):
            calls.append((host,port,kwargs));return [(socket.AF_INET,socket.SOCK_STREAM,6,'',('203.0.113.2',443))]
        source=GooglePublicKeys(lookup=lookup,socket_factory=lambda *args:raw)
        with patch('zetbros_service.google_public_keys.verified_tls_context',return_value=context):
            result=source.get_jwks(GOOGLE_JWKS_URL)
        self.assertTrue(raw.closed);self.assertTrue(secured.closed)
        self.assertEqual(calls[0][:2],('www.googleapis.com',443))
        self.assertEqual(context.calls,[{'server_hostname':'www.googleapis.com','do_handshake_on_connect':False}])
        wire=b''.join(secured.sent)
        self.assertIn(b'GET /oauth2/v3/certs HTTP/1.1',wire)
        self.assertNotIn(b'Authorization:',wire);self.assertNotIn(b'Cookie:',wire)
        return result

    def test_fixed_public_get_uses_fake_tls_and_cleans_up(self):
        self.assertEqual(self.fetch(b'{"keys":[]}'),{'keys':[]})

    def test_status_redirect_duplicate_and_oversize_responses_fail_closed(self):
        cases=[(b'{}','302 Found','Location: https://attacker.invalid/\r\n',None),
               (b'{}','200 OK','Location: https://attacker.invalid/\r\n',None),
               (b'{"keys":[],"keys":[]}','200 OK','',None),
               (b'{}','200 OK','',65537),(b'NaN','200 OK','',None)]
        for body,status,headers,length in cases:
            with self.subTest(body=body,status=status),self.assertRaises(ContractError):
                self.fetch(body,status,headers,length)

    def test_arbitrary_urls_are_rejected_before_dns_or_socket(self):
        source=GooglePublicKeys(lookup=lambda *a,**k:(_ for _ in ()).throw(AssertionError('DNS')),
            socket_factory=lambda *a:(_ for _ in ()).throw(AssertionError('socket')))
        with self.assertRaises(ContractError):source.get_jwks('https://attacker.invalid/keys')

    def test_stalled_dns_is_one_job_and_times_out_without_socket(self):
        import threading
        release=threading.Event();calls=[]
        def lookup(*args,**kwargs):calls.append(args);release.wait(1);return []
        source=GooglePublicKeys(lookup=lookup,socket_factory=lambda *args:(_ for _ in ()).throw(AssertionError('socket')))
        try:
            for _ in range(2):
                with self.assertRaises(ContractError):source.resolve(.01)
            self.assertEqual(len(calls),1)
        finally:release.set()
