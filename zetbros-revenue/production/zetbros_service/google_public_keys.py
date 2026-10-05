"""Fixed public Google JWKS transport, dormant until an enabled login verifies.

No API access tokens, user credentials, discovery URLs or redirects are used.
Offline tests replace this source. DNS waits and socket/response allocation are
bounded with an absolute guard; no network is used by a disabled profile.
"""
import http.client
import json
import socket
import ssl
import threading
import time
from .google_reviewer_auth import GOOGLE_JWKS_URL
from .live_session_factory import ConnectionGuard
from .spacemail_contract import verified_tls_context, ContractError


class GooglePublicKeys:
    def __init__(self, *, lookup=socket.getaddrinfo, socket_factory=socket.socket, monotonic=time.monotonic):
        self.lookup,self.socket_factory,self.monotonic=lookup,socket_factory,monotonic
        self.lock=threading.Lock();self.job=None;self.addresses=None;self.cached_at=0

    def resolve(self,timeout):
        with self.lock:
            if self.job is None or (self.job.is_set() and self.monotonic()-self.cached_at>60):
                event=threading.Event();self.job=event
                def run():
                    addresses=[]
                    try:
                        for family,kind,proto,_,location in self.lookup('www.googleapis.com',443,type=socket.SOCK_STREAM):
                            if family in (socket.AF_INET,socket.AF_INET6) and kind==socket.SOCK_STREAM:
                                addresses.append((family,kind,proto,location))
                        self.addresses=addresses[:8]
                    except Exception:self.addresses=[]
                    finally:self.cached_at=self.monotonic();event.set()
                threading.Thread(target=run,daemon=True,name='zetbros-google-jwks-dns').start()
            event=self.job
        if not event.wait(timeout) or not self.addresses:raise ContractError('google_keys_unavailable')
        return self.addresses

    def get_jwks(self,url):
        if url!=GOOGLE_JWKS_URL:raise ContractError('fixed_google_keys_required')
        guard=ConnectionGuard(self.monotonic()+3,1,monotonic=self.monotonic)
        connection=None
        try:
            context=verified_tls_context();secured=None
            for family,kind,proto,location in self.resolve(guard.remaining()):
                try:
                    raw=guard.register(self.socket_factory(family,kind,proto));raw.connect(location)
                    secured=guard.register(context.wrap_socket(raw,server_hostname='www.googleapis.com',do_handshake_on_connect=False))
                    secured.do_handshake();guard.remaining();break
                except Exception:
                    try:raw.close()
                    except Exception:pass
                    guard.remaining();secured=None
            if secured is None:raise ValueError
            connection=http.client.HTTPSConnection('www.googleapis.com',timeout=guard.remaining(),context=context)
            connection.sock=secured
            connection.request('GET','/oauth2/v3/certs',headers={'Accept':'application/json','Connection':'close'})
            response=connection.getresponse()
            if response.status!=200 or response.getheader('Location') is not None:raise ValueError
            declared=response.getheader('Content-Length')
            if declared is not None and (not declared.isdecimal() or int(declared)>65536):raise ValueError
            raw=response.read(65537)
            if len(raw)>65536:raise ValueError
            def unique(pairs):
                out={}
                for key,value in pairs:
                    if key in out:raise ValueError
                    out[key]=value
                return out
            data=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError()))
            guard.remaining();return data
        except Exception:raise ContractError('google_keys_unavailable') from None
        finally:
            if connection is not None:
                try:connection.close()
                except Exception:pass
            guard.close()
