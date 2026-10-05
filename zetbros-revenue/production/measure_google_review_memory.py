"""One bounded cloud sample of mock Google login and concurrent cookie review."""
import concurrent.futures
import json
import pathlib
import resource
import sys
import threading
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
from test_google_reviewer_auth import GoogleAuthTests


def status():
    out={}
    for line in pathlib.Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('VmRSS','VmHWM','Threads'):out[key]=int(value.strip().split()[0])
    return out


def main():
    GoogleAuthTests.setUpClass()
    fixture=GoogleAuthTests('test_dedicated_login_creates_only_short_secure_opaque_reviewer_cookie')
    fixture.setUp();stop=threading.Event();peak_threads=0
    def sample():
        nonlocal peak_threads
        while not stop.wait(.002):peak_threads=max(peak_threads,status()['Threads'])
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    try:
        with patch('zetbros_service.google_public_keys.GooglePublicKeys.get_jwks',side_effect=AssertionError('network')), \
             patch('zetbros_service.live_session_factory.SystemdCredentialSource.read',side_effect=AssertionError('credential')), \
             fixture.client as client:
            fixture.login();startup=status()
            def call(index):return client.get(fixture.path()).status_code,client.get('/healthz').status_code
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results=list(pool.map(call,range(24)))
            assert all(result==(200,200) for result in results)
            after=status()
        stop.set();sampler.join()
        peak=max(after['VmHWM'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(json.dumps({'sample':'mock Google ID-token login then eight concurrent ASGI cookie-review and health callers',
            'peak_rss_kib':peak,'startup_rss_kib':startup['VmRSS'],'sampled_peak_threads':peak_threads,
            'public_keys':'fictional in-memory RSA fixture only','provider_connections':'not_performed',
            'real_credentials':'not_read','listener_sockets':'not_opened',
            'under_memory_high_96m':peak<98304,'under_memory_max_128m':peak<131072,
            'under_tasks_max_32':peak_threads<=32,'cloud_sample_not_target_host_acceptance':True},indent=2,sort_keys=True))
        return 0 if peak<98304 and peak_threads<=32 else 1
    finally:stop.set();sampler.join();fixture.tearDown()


if __name__=='__main__':raise SystemExit(main())
