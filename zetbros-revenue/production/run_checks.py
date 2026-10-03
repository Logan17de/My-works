"""Run release-control tests and write public-safe source/runtime provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))


class RecordedResult(unittest.TextTestResult):
    def startTest(self,test):
        self.started=time.monotonic()
        super().startTest(test)
    def record(self,test,status):
        self.records.append({"test":test.id(),"outcome":status,"duration_seconds":round(time.monotonic()-self.started,6)})
    def addSuccess(self,test):
        self.record(test,"passed"); super().addSuccess(test)
    def addFailure(self,test,err):
        self.record(test,"failed"); super().addFailure(test,err)
    def addError(self,test,err):
        self.record(test,"error"); super().addError(test,err)
    def addSkip(self,test,reason):
        self.record(test,"skipped"); super().addSkip(test,reason)
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.records=[]


def main():
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
    output=stream.getvalue()
    print(output)
    sources={}
    for path in sorted(ROOT.rglob("*")):
        relative=path.relative_to(ROOT)
        if path.is_file() and not any(part in ("evidence","__pycache__",".venv",".git") for part in relative.parts):
            sources[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    packages={}
    for name in ("fastapi","pydantic","uvicorn","PyJWT","cryptography","httpx","starlette","anyio","idna","sniffio","certifi","click","h11","annotated-types","pydantic_core","typing_extensions","typing-inspection","cffi","pycparser","annotated-doc"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    report={"recorded_at":datetime.now(timezone.utc).isoformat(),"base_repository":"Logan17de/My-works",
      "base_pull_request":4,"base_remote_commit":"7b207534173da9ed7a46ca3ce894b84b7adb3569",
      "runtime":{"python":platform.python_version(),"sqlite":__import__('sqlite3').sqlite_version,"packages":packages},
      "source_sha256":sources,"source_content_revision":hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
      "tests_run":result.testsRun,"passed":sum(r["outcome"]=="passed" for r in result.records),"failed":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"tests":result.records,
      "test_transport":"controlled fictional adapter; runtime HTTP uses disabled transport",
      "real_mail_source":"blocked_unavailable","real_identity_provider":"unrun_existing_customer_configuration_needed",
      "real_customer_deployment":"unrun_not_authorized_or_available","real_delivery":"disabled_unrun",
      "actual_model_or_mcp_client":"unrun","public_deployment":"not_performed","provider_credentials":"not_read_or_created",
      "scope":"single-customer support-reply control service, local SQLite durable volume, no HA claim"}
    (ROOT/"evidence").mkdir(exist_ok=True)
    (ROOT/"evidence/test-results.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (ROOT/"evidence/unit-test-output.txt").write_text(output)
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    raise SystemExit(main())
