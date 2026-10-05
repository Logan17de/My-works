"""Record dormant Google reviewer/agent boundaries and unchanged disabled staging offline."""
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
            elif test.id().startswith("test_http_runtime.") or test.id().endswith("test_mocked_local_chromium_interrupted_repeated_and_racing_flows"):
                excluded.append(test.id())
            else:
                yield test
    suite = unittest.TestSuite(socket_free(discovered))
    result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=RecordedResult).run(suite)
    text = output.getvalue()
    print(text)
    evidence = ROOT / "evidence" / "google-reviewer-auth"
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
        "base_remote_commit": "6381c4e7eb0a5864e9c3c8996a63044dd1fd3d71",
        "base_git_blob_sha": base["files"], "unchanged_base_files": unchanged,
        "modified_base_files": changed,
        "runtime": {"python": platform.python_version(), "sqlite": __import__("sqlite3").sqlite_version, "packages": packages},
        "tests_run": result.testsRun,
        "passed": sum(record["outcome"] == "passed" for record in result.records),
        "failed": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "tests": result.records, "source_sha256": hashes,
        "source_content_revision": hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "configured_delivery": "default_pending_identity_stage_no_ledger_factory_credential_or_worker",
        "pending_init": "successfully_stages_without_identity_pin_or_ledger_creation",
        "configured_disabled_init": "validates_public_JWKS_profile_and_identity_before_new_exact_wire_ledger",
        "default_unit_credentials": "no_LoadCredential_dependency",
        "identity_rebinding": "refused_no_migration",
        "live_factory": "own_pinned_stdlib_implementation_verified_with_fake_connector_clients_and_synthetic_fixture_only",
        "review_ui": "ASGI_and_Node_bearer_and_Google_state_harness_verified",
        "google_id_token_login": "mocked_public_RSA_JWKS_exact_audience_issuer_sub_nonce_expiry_only",
        "google_api_credentials": "no_client_secret_access_or_refresh_token_used",
        "reviewer_session": "short_process_local_opaque_cookie_CSRF_and_exact_origin",
        "agent_boundary": "separate_pinned_external_access_token_resource_agent_only",
        "google_setup": "disabled_example_null_identity_actual_origin_client_sub_and_browser_acceptance_pending",
        "google_public_key_transport": "fake_DNS_TLS_and_socket_interfaces_no_actual_network",
        "bootstrap_availability": "bounded_peer_throttle_raw_XFF_ignored_distributed_DoS_and_verified_proxy_acceptance_required",
        "browser_pixels": "Chromium_launch_blocked_by_sandbox_socket_restriction_no_escalation",
        "credential_source": "systemd_fixed_name_implementation_no_actual_credential_read",
        "event_loop_ipc": "standard_local_asyncio_socketpair_only_no_provider_or_listener_network_socket",
        "mail_test_ports": "fictional_stdlib_shaped_smtp_imap_sessions_no_actual_sockets",
        "excluded_loopback_tests": excluded,
        "existing_http_regression": "passed_in_prior_combined_run_not_rerun_on_final_revision",
        "prior_combined_evidence": "../mail-integration/prior-combined-test-results.json",
        "current_run_network": "socket_free_tests_only",
        "provider_connections": "not_performed", "provider_credentials": "not_read_or_created",
        "owning_source_interface": "inspected_not_vendored_or_republished",
        "human_exact_wire_preview": "authenticated_ASGI_UI_composite_preview_and_immutable_ledger_verified",
        "live_account_endpoint_attestation": "unverified",
        "private_factory": "implemented_owner_credential_and_identity_configuration_required",
        "tenant_profile": "one_customer_account_and_local_ledger_per_process_no_shared_multitenant_claim",
        "total_deadline": "offline_fake_socket_watchdog_verified_real_socket_interrupt_unrun",
        "live_prepared_wire_bridge": "own_factory_runtime_and_exact_wire_bridge_implemented_actual_provider_and_deployment_unrun",
        "live_identity_and_customer_deployment": "unrun",
        "real_delivery": "disabled_unrun", "buyer_ready": False,
        "supplied_private_archive_artifacts": "not_opened_or_published",
    }
    (evidence / "test-results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (evidence / "unit-test-output.txt").write_text(text)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
