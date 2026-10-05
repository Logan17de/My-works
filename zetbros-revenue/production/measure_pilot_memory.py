"""Offline resource sample in a separate process; fake providers and ASGI only."""
from __future__ import annotations
import asyncio
import concurrent.futures
import hashlib
import json
import pathlib
import resource
import sys
import threading
import time
from fastapi.testclient import TestClient

ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
import test_pilot_runtime as fixture_module
from zetbros_service.api import create_app
from zetbros_service.pilot_runtime import build_service
from zetbros_service.spacemail_contract import verified_tls_context


def status():
    values={}
    for line in pathlib.Path('/proc/self/status').read_text().splitlines():
        key, _, value=line.partition(':')
        if key in ('VmRSS','VmHWM','Threads'):values[key]=int(value.strip().split()[0])
    return values


def main():
    fixture=fixture_module.PilotAssemblyTests('test_operator_runtime_is_assembled_without_startup_connections_or_credentials')
    fixture.setUp()
    peak_threads=0
    stop=threading.Event()
    def sample():
        nonlocal peak_threads
        while not stop.wait(0.005):peak_threads=max(peak_threads,status()['Threads'])
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    started=time.monotonic()
    try:
        # Include retained verified CA contexts even though no TLS connection is made.
        tls_contexts=[verified_tls_context(),verified_tls_context()]
        service=build_service(fixture.settings,factory_builder=fixture.factory,clock=fixture.fixture.fixture.clock)
        app=create_app(service,run_worker=False)
        with TestClient(app,base_url='http://127.0.0.1:8081') as client:
            startup=status()
            agent=fixture.fixture.fixture.headers()
            record=client.get('/v1/source/imap-v1:123:42',headers=agent).json()
            def propose(index):
                request={'operation_key':f'memory-pilot-operation-{index:04d}',
                    'source_id':record['source']['source_id'],'source_version':record['source']['version'],
                    'source_fingerprint':record['source_fingerprint'],'body':'A'*16000}
                response=client.post('/v1/proposals',headers=agent,json=request)
                if response.status_code!=201:raise RuntimeError('offline proposal sample failed')
                return response.json()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                proposals=list(pool.map(propose,range(8)))
            reviewer=fixture.fixture.fixture.headers(reviewer=True)
            for proposal in proposals:
                shown=client.get('/review/api/proposals/'+proposal['id'],headers=reviewer)
                if shown.status_code!=200:raise RuntimeError('offline review sample failed')
                result=client.post('/review/api/proposals/'+proposal['id']+'/decision',
                    headers=reviewer|{'Origin':'http://127.0.0.1:8081'},
                    json={'digest':shown.json()['digest'],'decision':'approve'})
                if result.status_code!=200:raise RuntimeError('offline decision sample failed')
            asyncio.run(service.worker_once())
            for proposal in proposals:
                if service.store.get(proposal['id'])['execution']['state']!='accepted':raise RuntimeError('offline result sample failed')
            after=status()
        stop.set();sampler.join()
        peak_kib=max(after['VmHWM'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report={'sample':'one separate cloud Python process,8 concurrent ASGI proposal requests then8 fake SMTP/IMAP operations',
            'provider_connections':'not_performed','listener_sockets':'not_opened','real_credentials':'not_read',
            'event_loop_ipc':'standard local asyncio wakeup socketpair only',
            'startup_rss_kib':startup['VmRSS'],'peak_rss_kib':peak_kib,'after_rss_kib':after['VmRSS'],
            'sampled_peak_threads_including_benchmark_callers':peak_threads,
            'proposed_memory_high_kib':98304,'proposed_memory_max_kib':131072,
            'under_memory_high':peak_kib<98304,'under_memory_max':peak_kib<131072,
            'under_tasks_max_32':peak_threads<=32,'seconds':round(time.monotonic()-started,3),
            'cloud_sample_not_target_host_or_cgroup_acceptance':True}
        print(json.dumps(report,indent=2,sort_keys=True))
        return 0 if report['under_memory_max'] and report['under_tasks_max_32'] else 1
    finally:
        stop.set();sampler.join();fixture.tearDown()


if __name__=='__main__':raise SystemExit(main())
