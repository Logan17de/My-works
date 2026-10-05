"""Exact disabled init/check/app sample: no verifier, ledger or provider calls."""
import concurrent.futures
import json
import os
import pathlib
import resource
import tempfile
import threading
from unittest.mock import patch
from fastapi.testclient import TestClient
from zetbros_service.config import PendingPilotConfig
from zetbros_service.models import canonical
from zetbros_service.pilot_runtime import configured_pilot_app, initialize_new_pilot, pending_status

ROOT=pathlib.Path(__file__).resolve().parent


def status():
    out={}
    for line in pathlib.Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('VmRSS','VmHWM','Threads'):out[key]=int(value.strip().split()[0])
    return out


def main():
    with tempfile.TemporaryDirectory() as directory:
        root=pathlib.Path(directory)
        data=json.loads((ROOT/'deploy/pilot-runtime.disabled.example.json').read_text())
        data['database_path']=str(root/'never-created.sqlite3')
        config=PendingPilotConfig.model_validate_json(json.dumps(data))
        path=root/'service.json';path.write_text(canonical(config))
        stop=threading.Event();peak_threads=0
        def sample():
            nonlocal peak_threads
            while not stop.wait(0.002):peak_threads=max(peak_threads,status()['Threads'])
        sampler=threading.Thread(target=sample,daemon=True);sampler.start()
        try:
            with patch.dict(os.environ,{'ZETBROS_CONFIG_FILE':str(path)}), \
                 patch('zetbros_service.pilot_runtime.Verifier',side_effect=AssertionError('verifier called')), \
                 patch('zetbros_service.live_session_factory.SystemdCredentialSource.read',side_effect=AssertionError('secret read')):
                initialized=initialize_new_pilot(config)
                assert initialized['ledger_created'] is False
                app=configured_pilot_app()
                with TestClient(app) as client:
                    startup=status()
                    def call(index):
                        return client.get('/healthz').status_code,client.get('/readyz').status_code
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                        results=list(pool.map(call,range(16)))
                    assert all(result==(200,503) for result in results)
                    assert client.post('/v1/proposals',json={}).status_code==503
                    after=status()
            assert not pathlib.Path(config.database_path).exists()
            stop.set();sampler.join()
            peak=max(after['VmHWM'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            print(json.dumps({'sample':'shipped pending-identity disabled init and8 concurrent ASGI health/readiness callers',
                'peak_rss_kib':peak,'startup_rss_kib':startup['VmRSS'],'sampled_peak_threads':peak_threads,
                'ledger_created':False,'verifier_calls':0,'provider_connections':'not_performed',
                'real_credentials':'not_read','listener_sockets':'not_opened',
                'under_memory_high_96m':peak<98304,'under_memory_max_128m':peak<131072,
                'under_tasks_max_32':peak_threads<=32,'cloud_sample_not_target_host_acceptance':True},indent=2,sort_keys=True))
            return 0 if peak<131072 and peak_threads<=32 else 1
        finally:stop.set();sampler.join()


if __name__=='__main__':raise SystemExit(main())
