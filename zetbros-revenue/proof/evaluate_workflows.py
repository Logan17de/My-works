"""Ten scripted fixture evaluations, intentionally separate from control tests.

No model/provider is invoked. Fixture metadata selects the deterministic branches.
These scores do not measure general agent reasoning, tool selection or injection immunity.
"""
from safety import install_offline_guard

install_offline_guard()

from dataclasses import asdict
import json
from pathlib import Path

from proof import FixtureWorkflow, ProofError, completion_claim_is_supported, start


def evaluate():
    results = []
    for index in range(1, 11):
        e = start(run_id=f"demo_run_workflow_{index:02d}")
        calls = []
        original_call = e.tools.call

        def record_call(name, **parameters):
            calls.append({"tool": name, "parameters": parameters})
            return original_call(name, **parameters)

        e.tools.call = record_call
        w = FixtureWorkflow(e.tools, e.fixtures.data["context"])
        human = e.human_channel(scripted=True)

        def approve(p):
            return human.decide(p["proposal_id"], "approve", p["action_digest"])

        outcome = None
        observed_error = None
        safe = useful = truthful = False
        if index == 1:
            plan = w.plan("demo_msg_01")
            p = plan["proposal"]
            safe = len(e.transport.calls) == 0 and not e._approvals
            useful = e.fixtures.data["context"]["guide_v1"] in p["payload"]["body"] and "guide_v1" in plan["sources"]
            truthful = plan["kind"] == "approval_required" and p["state"] == "awaiting_approval"
        elif index == 2:
            plan = w.plan("demo_msg_02")
            safe = len(e.transport.calls) == 0 and not e._approvals
            useful = len(e.drafts) == 1 and "human support owner" in plan["draft"]["payload"]["body"]
            truthful = plan["kind"] == "draft_only" and "No refund has been executed or promised" in plan["draft"]["payload"]["body"]
        elif index == 3:
            plan = w.plan("demo_msg_03")
            safe = not e.transport.calls and not e.drafts and not e._approvals and not e._proposals
            useful = plan["kind"] == "blocked_untrusted_request"
            truthful = "cannot authorize" in plan["summary"]
        elif index == 4:
            plan = w.plan("demo_msg_04")
            safe = not e.transport.calls and not e._approvals
            body = plan["draft"]["payload"]["body"]
            useful = "Please provide" in body and "account reference" in body
            truthful = "demo_account_01" not in body and plan["kind"] == "draft_only"
        elif index == 5:
            plan = w.plan("demo_msg_01", folder="Archive")
            p = plan["proposal"]
            safe = not e.transport.calls
            useful = (p["payload"]["original_folder"], p["payload"]["original_message_id"],
                      p["payload"]["original_message_version"]) == ("Archive", "demo_msg_01", 2)
            truthful = tuple(p["payload"]["to"]) == ("archived-owner@customer.invalid",)
        elif index == 6:
            plan = w.plan("demo_msg_01")
            p = plan["proposal"]
            human.decide(p["proposal_id"], "deny", p["action_digest"])
            try:
                e.execute(p["proposal_id"])
            except ProofError as error:
                observed_error = error.code
            safe = not e.transport.calls and not e.drafts and not e._approvals
            useful = e.proposal(p["proposal_id"]).state == "denied"
            truthful = observed_error == "valid_approval_required"
        elif index == 7:
            plan = w.plan("demo_msg_05")
            p = plan["proposal"]
            a = approve(p)
            e.fixtures.message("INBOX", "demo_msg_05")["reply_to"] = "changed@customer.invalid"
            try:
                e.execute(p["proposal_id"], a)
            except ProofError as error:
                observed_error = error.code
            safe = not e.transport.calls and e.proposal(p["proposal_id"]).state == "invalidated"
            fresh = w.plan("demo_msg_05")["proposal"]
            outcome = e.execute(fresh["proposal_id"], approve(fresh))
            useful = len(e.transport.calls) == 1 and len(e._approvals) == 2
            truthful = observed_error == "approved_context_changed" and outcome.accepted_recipients == ("changed@customer.invalid",)
        elif index == 8:
            plan = w.plan("demo_msg_06")
            p = plan["proposal"]
            a = approve(p)
            outcome = e.execute(p["proposal_id"], a)
            repeated = e.execute(p["proposal_id"], a)
            safe = len(e.transport.calls) == 1 and outcome.retry_count == 0
            useful = repeated.duplicate and repeated.status == "uncertain"
            truthful = completion_claim_is_supported(outcome, "uncertain") and not outcome.delivered
        elif index == 9:
            e.transport.scenarios[("INBOX", "demo_msg_01")] = "copy_failed"
            plan = w.plan("demo_msg_01")
            p = plan["proposal"]
            a = approve(p)
            outcome = e.execute(p["proposal_id"], a)
            e.execute(p["proposal_id"], a)
            safe = len(e.transport.calls) == 1 and outcome.retry_count == 0
            useful = outcome.simulated_submission_status == "accepted" and outcome.sent_copy_status == "copy_failed"
            truthful = completion_claim_is_supported(outcome, "simulated_accepted") and not outcome.delivered
        else:
            plan = w.plan("demo_msg_07")
            safe = not e.transport.calls and not e.drafts and not e._proposals
            useful = plan["kind"] == "summary_only"
            truthful = "no action needed" in plan["summary"]

        results.append({
            "case_id": f"E{index:02d}", "evaluation_kind": "deterministic_fixture_workflow",
            "result": "passed" if safe and useful and truthful else "failed",
            "scores": {"safe_actions": bool(safe), "useful_task_outcome": bool(useful), "truthful_reporting": bool(truthful)},
            "fixture_version": e.fixtures.version, "fixture_digest": e.fixtures.fingerprint,
            "run_id": e.run_id, "synthetic_decision_source": "scripted_test_fixture",
            "model_invoked": False, "provider": None, "model": None, "sampling": None,
            "requested_tools": calls, "plan": plan, "observed_error_class": observed_error,
            "outcome": asdict(outcome) if outcome else None,
            "fake_transport_call_count": len(e.transport.calls), "draft_count": len(e.drafts),
            "external_transport_call_count": 0, "trace_event_ids": [v["event_id"] for v in e.audit.events],
            "trace": e.audit.events,
        })
    return {
        "label": "OFFLINE SYNTHETIC: scripted workflow scores, not model tool-selection evaluations",
        "model_invoked": False, "production_mail_mcp_tested": False,
        "passed": sum(r["result"] == "passed" for r in results),
        "failed": sum(r["result"] == "failed" for r in results), "cases": results,
        "actual_model_evaluations": {"status": "not_run", "count": 10,
                                     "reason": "No model/client invoked; no model spend or live integration scoped"},
    }


if __name__ == "__main__":
    r = evaluate()
    output = Path(__file__).parent / "evidence" / "workflow-results.json"
    output.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(f"Scripted workflows: {r['passed']} passed, {r['failed']} failed; actual model evaluations not run")
    raise SystemExit(1 if r["failed"] else 0)
