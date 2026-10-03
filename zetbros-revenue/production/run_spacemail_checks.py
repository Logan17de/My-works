"""Record dormant provider tests without overwriting the published base evidence."""
from __future__ import annotations

import hashlib
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
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=RecordedResult).run(suite)
    text = output.getvalue()
    print(text)
    evidence = ROOT / "evidence" / "spacemail"
    evidence.mkdir(exist_ok=True)
    base = json.loads((evidence / "base-manifest.json").read_text())
    base_matches = {}
    for relative, expected in base["files"].items():
        raw = (ROOT / relative).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()
        base_matches[relative] = actual == expected
    hashes = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if path.is_file() and not any(part in ("evidence", "__pycache__", ".venv", ".git") for part in relative.parts):
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "base_repository": "Logan17de/My-works",
        "base_remote_commit": "d19b6d4228e295296ae043a4efc47845cb50daa8",
        "base_files_unchanged": all(base_matches.values()),
        "base_git_blob_sha": base["files"],
        "base_blob_matches": base_matches,
        "python": platform.python_version(),
        "sqlite": __import__("sqlite3").sqlite_version,
        "tests_run": result.testsRun,
        "passed": sum(record["outcome"] == "passed" for record in result.records),
        "failed": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "tests": result.records,
        "source_sha256": hashes,
        "source_content_revision": hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "delivery": "disabled",
        "test_provider": "fictional injected IMAP/SMTP transcripts only",
        "real_provider_connections": "not_performed",
        "provider_credentials": "not_read_or_created",
        "real_connector_source": "blocked_unavailable",
        "human_wire_preview_integration": "blocked_not_wired_into_existing_api_or_ledger",
        "real_customer_deployment": "unrun",
        "real_identity_provider": "unrun",
        "real_delivery": "disabled_unrun",
        "buyer_ready": False,
    }
    (evidence / "test-results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (evidence / "unit-test-output.txt").write_text(text)
    return 0 if result.wasSuccessful() and all(base_matches.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
