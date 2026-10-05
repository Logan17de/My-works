"""Record the offline owner-interface increment without rewriting prior evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    from run_checks import RecordedResult
    output = io.StringIO()
    discovered = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    excluded = []
    def socket_free(suite):
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                yield from socket_free(test)
            elif test.id().startswith("test_http_runtime."):
                excluded.append(test.id())
            else:
                yield test
    suite = unittest.TestSuite(socket_free(discovered))
    result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=RecordedResult).run(suite)
    text = output.getvalue()
    print(text)
    evidence = ROOT / "evidence" / "mail-integration"
    evidence.mkdir(exist_ok=True)
    base = json.loads((evidence / "base-manifest.json").read_text())
    unchanged, changed = {}, []
    for relative, expected in base["files"].items():
        raw = (ROOT / relative).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()
        unchanged[relative] = actual == expected
        if actual != expected:
            changed.append(relative)
    hashes = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if path.is_file() and not any(part in ("evidence", "__pycache__", ".venv", ".git") for part in relative.parts):
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    packages = {}
    for name in ("fastapi", "pydantic", "PyJWT", "httpx", "starlette", "cryptography", "uvicorn"):
        packages[name] = importlib.metadata.version(name)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "base_repository": "Logan17de/My-works",
        "base_remote_commit": "7fb50e5138fe05ad4e002eab3182f9c50be4a8c9",
        "base_git_blob_sha": base["files"], "unchanged_base_files": unchanged,
        "modified_base_files": changed,
        "runtime": {"python": platform.python_version(), "sqlite": __import__("sqlite3").sqlite_version, "packages": packages},
        "tests_run": result.testsRun,
        "passed": sum(record["outcome"] == "passed" for record in result.records),
        "failed": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "tests": result.records, "source_sha256": hashes,
        "source_content_revision": hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "configured_delivery": "disabled_no_activation_configuration",
        "mail_test_ports": "fictional_owner_shaped_read_and_prepared_wire_objects",
        "excluded_loopback_tests": excluded,
        "existing_http_regression": "passed_in_prior_combined_run_not_rerun_on_final_revision",
        "prior_combined_evidence": "prior-combined-test-results.json",
        "current_run_network": "socket_free_tests_only",
        "provider_connections": "not_performed", "provider_credentials": "not_read_or_created",
        "owning_source_interface": "inspected_not_vendored_or_republished",
        "human_exact_wire_preview": "offline_api_and_immutable_ledger_verified",
        "live_account_endpoint_attestation": "unverified",
        "live_prepared_wire_bridge": "not_implemented",
        "live_identity_and_customer_deployment": "unrun",
        "real_delivery": "disabled_unrun", "buyer_ready": False,
        "supplied_private_archive_artifacts": "not_opened_or_published",
    }
    (evidence / "test-results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (evidence / "unit-test-output.txt").write_text(text)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
