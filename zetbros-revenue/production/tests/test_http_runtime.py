"""Actual loopback HTTP subprocess/restart test; outbound remains disabled."""
import json
import os
import socket
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

import httpx

import test_service as fixtures


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ServiceTests.setUpClass.__func__(cls)
    setUp=fixtures.ServiceTests.setUp
    tearDown=fixtures.ServiceTests.tearDown
    write_record=fixtures.ServiceTests.write_record
    token=fixtures.ServiceTests.token
    headers=fixtures.ServiceTests.headers
    input=fixtures.ServiceTests.input

    def launch(self, port):
        config=self.root/"config.json"
        config.write_text(self.settings.model_dump_json())
        process=subprocess.Popen([sys.executable,"-m","uvicorn","zetbros_service.api:configured_app","--factory","--host","127.0.0.1","--port",str(port),"--no-access-log","--log-level","info"],
                                 env=os.environ|{"ZETBROS_CONFIG_FILE":str(config)},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.addCleanup(self.stop,process)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}",timeout=2,trust_env=False) as client:
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    self.fail("HTTP service failed before readiness")
                try:
                    if client.get("/healthz").status_code==200: return process
                except httpx.HTTPError: pass
                time.sleep(0.05)
        self.fail("HTTP service did not become ready")

    def stop(self, process):
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        if process.stdout and not process.stdout.closed:
            process._zetbros_stdout=process.stdout.read().decode(errors="replace")
            process.stdout.close()
        if process.stderr and not process.stderr.closed:
            process._zetbros_stderr=process.stderr.read().decode(errors="replace")
            process.stderr.close()

    def test_loopback_http_authenticated_workflow_graceful_shutdown_and_restart(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1",0)); port=reserved.getsockname()[1]
        process=self.launch(port)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}",timeout=2,trust_env=False) as client:
            self.assertEqual(client.get("/v1/source/message-001").status_code,401)
            self.assertFalse(client.get("/readyz").json()["live_delivery_ready"])
            response=client.post("/v1/proposals",headers=self.headers(),json=self.input().model_dump(mode="json"))
            self.assertEqual(response.status_code,201)
            proposal=response.json()
            preview=client.get(f'/v1/proposals/{proposal["id"]}',headers=self.headers(True)).json()
            self.assertEqual(preview["payload"]["body"],self.input().body)
            response=client.post(f'/v1/reviews/{proposal["id"]}/decision',headers=self.headers(True),json={"digest":preview["digest"],"decision":"approve"})
            self.assertEqual(response.status_code,200)
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                result=client.get(f'/v1/proposals/{proposal["id"]}',headers=self.headers()).json()
                if result["execution"] and result["execution"]["state"]=="blocked": break
                time.sleep(0.05)
            self.assertEqual(result["execution"]["result"]["error_class"],"delivery_disabled")
            self.assertIsNone(result["consumed_at"])
        self.stop(process)
        self.assertIn(process.returncode,(0,-signal.SIGTERM))
        self.assertIn("Application shutdown complete.",process._zetbros_stderr)
        restarted=self.launch(port)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}",timeout=2,trust_env=False) as client:
            result=client.get(f'/v1/proposals/{proposal["id"]}',headers=self.headers()).json()
            self.assertEqual(result["execution"]["state"],"blocked")
            events=client.get("/v1/audit",headers=self.headers(True)).json()["events"]
            self.assertEqual(sum(e["event_type"]=="result" for e in events),1)
        self.stop(restarted)
        self.assertIn(restarted.returncode,(0,-signal.SIGTERM))
        self.assertIn("Application shutdown complete.",restarted._zetbros_stderr)
