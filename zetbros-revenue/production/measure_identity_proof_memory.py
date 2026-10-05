"""One bounded mock identity-proof cloud sample, no ledger or actual Google I/O."""
import concurrent.futures
import json
import pathlib
import resource
import sys
import threading

ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
import test_google_identity_proof as fixtures


def status():
    out={}
    for line in pathlib.Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('VmRSS','VmHWM','Threads'):out[key]=int(value.strip().split()[0])
    return out


def main():
    fixtures.ProofTests.setUpClass()
    fixture=fixtures.ProofTests('test_verified_unknown_subject_returns_only_proof_and_never_session_or_enrollment')
    fixture.setUp();stop=threading.Event();peak_threads=0
    def sample():
        nonlocal peak_threads
        while not stop.wait(.002):peak_threads=max(peak_threads,status()['Threads'])
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    try:
        with fixture.client as client:
            challenge=fixture.bootstrap();assert fixture.prove(challenge).status_code==200
            startup=status()
            def call(index):return client.get('/identity').status_code,client.get('/healthz').status_code
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results=list(pool.map(call,range(16)))
            assert all(result==(200,200) for result in results)
            after=status()
        stop.set();sampler.join()
        peak=max(after['VmHWM'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(json.dumps({'sample':'mock signed Google identity proof then four concurrent ASGI shell/health callers',
            'peak_rss_kib':peak,'startup_rss_kib':startup['VmRSS'],'sampled_peak_threads':peak_threads,
            'ledger_service_sessions_or_grants':'none','provider_connections':'not_performed',
            'real_credentials':'not_read','listener_sockets':'not_opened',
            'under_memory_high_64m':peak<65536,'under_memory_max_80m':peak<81920,
            'under_tasks_max_16':peak_threads<=16,'cloud_sample_not_target_host_acceptance':True},indent=2,sort_keys=True))
        return 0 if peak<65536 and peak_threads<=16 else 1
    finally:stop.set();sampler.join();fixture.tearDown()


if __name__=='__main__':raise SystemExit(main())
