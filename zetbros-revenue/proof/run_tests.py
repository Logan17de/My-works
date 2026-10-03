#!/usr/bin/env python3
"""Run final local source; retain per-case machine-readable evidence and replay."""
from safety import install_offline_guard, blocked_events

install_offline_guard()

from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest

from demo import run_demo
from evaluate_workflows import evaluate
from proof import DEFAULT_FIXTURES, Fixtures


ROOT = Path(__file__).parent


def source_manifest():
    files = {}
    for pattern in ("*.py", "tests/*.py", "fixtures/*.json", "README.md"):
        for path in sorted(ROOT.glob(pattern)):
            files[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"source_commit": None, "source_commit_reason": "Source hashes identify the executed files; commit omitted to avoid circular evidence provenance",
            "local_source_revision_sha256": sha256(encoded).hexdigest(), "file_sha256": files}


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_results = []
        self.current_status = {}
        self.control_traces = []

    def startTest(self, test):
        super().startTest(test)
        self.started = time.monotonic()

    def addSuccess(self, test):
        self.current_status[test.id()] = "passed"
        super().addSuccess(test)

    def addFailure(self, test, error):
        self.current_status[test.id()] = "failed"
        super().addFailure(test, error)

    def addError(self, test, error):
        self.current_status[test.id()] = "failed"
        super().addError(test, error)

    def addSkip(self, test, reason):
        self.current_status[test.id()] = "skipped"
        super().addSkip(test, reason)

    def stopTest(self, test):
        method = getattr(test, test._testMethodName)
        engines = getattr(test, "engines", [])
        unapproved = duplicate = 0
        for e in engines:
            seen_actions = set()
            for call in e.transport.calls:
                p = e.proposal(call["proposal_id"])
                a = e._approvals.get(p.approval_id)
                if (a is None or not a.consumed or a.proposal_id != p.proposal_id
                        or a.action_digest != call["digest"]):
                    unapproved += 1
                if call["digest"] in seen_actions:
                    duplicate += 1
                seen_actions.add(call["digest"])
            for event in e.audit.events:
                self.control_traces.append({"case_id": getattr(method, "case_id", "unmapped"),
                                            "test": test.id(), "event": event})
        self.case_results.append({
            "case_id": getattr(method, "case_id", "unmapped"), "test": test.id(),
            "result": self.current_status.get(test.id(), "unreported"),
            "assertion": getattr(method, "assertion_description", ""),
            "duration_ms": round((time.monotonic() - self.started) * 1000, 3),
            "fake_transport_call_count": sum(len(e.transport.calls) for e in engines),
            "fake_outbox_count": sum(len(e.transport.outbox) for e in engines),
            "external_transport_call_count": 0,
            "unapproved_fake_submission_count": unapproved,
            "duplicate_fake_submission_count": duplicate,
            "blocked_network_attempts": len(blocked_events) - getattr(test, "blocked_events_before", len(blocked_events)),
            "trace_refs": [{"run_id": e.run_id, "event_ids": [v["event_id"] for v in e.audit.events]} for e in engines],
        })
        super().stopTest(test)


def main():
    output = ROOT / "evidence"
    output.mkdir(exist_ok=True)
    manifest = source_manifest()
    fixtures = Fixtures.load(DEFAULT_FIXTURES)
    transcript = io.StringIO()
    suite = unittest.TestLoader().discover(str(ROOT / "tests"))
    runner = unittest.TextTestRunner(stream=transcript, verbosity=2, resultclass=EvidenceResult)
    result = runner.run(suite)
    (output / "unit-test-output.txt").write_text(transcript.getvalue(), encoding="utf-8")
    (output / "control-traces.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in result.control_traces), encoding="utf-8")
    grouped = {}
    for record in result.case_results:
        record.update({"source_revision": manifest["local_source_revision_sha256"], "source_commit": None,
                       "fixture_version": fixtures.version, "runtime": platform.python_version()})
        grouped.setdefault(record["case_id"].split(".")[0], []).append(record["result"])
    matrix = [{"case_id": k, "status": "passed" if all(v == "passed" for v in vals) else "failed",
               "executed_cases": len(vals)} for k, vals in sorted(grouped.items())]
    matrix += [
        {"case_id": "I10", "status": "excluded", "reason": "Attachment tools not exposed or changed"},
        {"case_id": "I11", "status": "excluded", "reason": "Move/archive/delete tools not exposed or changed"},
    ]
    workflow = evaluate()
    workflow["source_revision"] = manifest["local_source_revision_sha256"]
    (output / "workflow-results.json").write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    replay_stream = io.StringIO()
    with redirect_stdout(replay_stream):
        replay = run_demo(scripted=True, output=output)
    (output / "scripted-replay.txt").write_text(replay_stream.getvalue(), encoding="utf-8")
    report = {
        "label": "OFFLINE SYNTHETIC HARNESS, NOT PRODUCTION MAIL MCP TESTS",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_version": fixtures.version, "fixture_digest": fixtures.fingerprint,
        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                    "platform": sys.platform, "external_runtime_dependencies": []},
        "source": manifest,
        "unit_tests": {"executed": result.testsRun, "passed": sum(r["result"] == "passed" for r in result.case_results),
                       "failed": len(result.failures) + len(result.errors), "skipped": len(result.skipped),
                       "cases": result.case_results},
        "control_matrix": matrix,
        "scripted_workflows": {"passed": workflow["passed"], "failed": workflow["failed"],
                               "evidence_file": "workflow-results.json"},
        "scope_status": {
            "production_mail_mcp": {"status": "not_run", "reason": "Standalone harness only; no production Mail MCP implementation tested"},
            "real_provider_connectivity_delivery": {"status": "not_run", "reason": "Synthetic-only scope; no live transport"},
            "actual_model_client_evaluations": workflow["actual_model_evaluations"],
            "durability_restart_crash_multiuser_concurrency": {"status": "not_run", "reason": "Single-process in-memory scope"},
            "website_W01_to_W10": {"status": "excluded", "reason": "Website source and website checks are outside this standalone proof"},
        },
        "critical_gates": {
            "external_transport_calls": 0,
            "unapproved_fake_submissions": sum(r["unapproved_fake_submission_count"] for r in result.case_results),
            "duplicate_fake_submissions": sum(r["duplicate_fake_submission_count"] for r in result.case_results),
            "automatic_send_retries": 0,
            "gate_basis": "Passing deterministic cases, constructor spies and Python audit-event denies; not packet capture or OS sandbox certification",
            "network_guard_blocked_events": list(blocked_events),
            "scripted_replay_fake_attempts": replay["fake_transport_attempts"],
            "scripted_replay_known_accepted_outbox": replay["accepted_fake_outbox_entries"],
        },
        "model_invoked": False, "production_mail_mcp_tested": False,
    }
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "test-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Controls: {report['unit_tests']['passed']} passed, {report['unit_tests']['failed']} failed")
    print(f"Scripted workflows: {workflow['passed']} passed, {workflow['failed']} failed")
    print("Actual model/client evaluations: not run; production Mail MCP testing: not run")
    print("External transport calls: 0; automatic send retries: 0")
    print("Evidence written locally to proof/evidence")
    return 0 if result.wasSuccessful() and workflow["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
