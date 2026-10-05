#!/usr/bin/env python3
"""Runnable local approval demonstration, with an explicitly scripted replay option."""
from safety import install_offline_guard

install_offline_guard()

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from proof import DEFAULT_FIXTURES, FixtureWorkflow, ProofError, start


LABEL = "OFFLINE SYNTHETIC ONLY: deterministic workflow; fake in-memory transport; no real Mail MCP or delivery."


def show(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def get_decision(channel, preview, scripted_decision=None):
    print("EXACT SIMULATED ACTION PREVIEW")
    show(preview)
    token = preview["action_digest"]
    if scripted_decision is None:
        print(f"Local operator: type 'approve {token}', 'deny', or 'cancel'")
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "cancel"
        if raw == "approve " + token:
            decision = "approve"
        elif raw in {"deny", "cancel"}:
            decision = raw
        else:
            print("Unrecognized input: canceling this proposal, with no simulation")
            decision = "cancel"
    else:
        print(f"SCRIPTED TEST FIXTURE DECISION: {scripted_decision}; this is not a human sign-off")
        decision = scripted_decision
    return channel.decide(preview["proposal_id"], decision, token)


def run_demo(*, mode="synthetic", fixtures_path=DEFAULT_FIXTURES, scripted=False, output=None):
    print(LABEL)
    print("Fixture clock is fixed; approval expiry is demonstrated in the deterministic tests")
    e = start(mode=mode, fixtures_path=fixtures_path)
    workflow = FixtureWorkflow(e.tools, e.fixtures.data["context"])
    human = e.human_channel(scripted=scripted)
    observations = []

    guide = workflow.plan("demo_msg_01")
    p = guide["proposal"]
    print(f"Pending approval: fake transport calls={len(e.transport.calls)}, fake outbox={len(e.transport.outbox)}")
    decision = get_decision(human, p, "approve" if scripted else None)
    if isinstance(decision, str):
        result = e.execute(p["proposal_id"], decision)
        duplicate = e.execute(p["proposal_id"], decision)
        show(asdict(result))
        print("Duplicate request returns the stored result; no additional transport call")
        observations.append({"scenario": "guide", "outcome": asdict(result), "replay": asdict(duplicate)})
    else:
        show(decision)
        observations.append({"scenario": "guide", "decision": decision})

    denied = workflow.plan("demo_msg_05")["proposal"]
    d = get_decision(human, denied, "deny" if scripted else None)
    if isinstance(d, str):
        outcome = e.execute(denied["proposal_id"], d)
        observations.append({"scenario": "reply_to", "outcome": asdict(outcome)})
    else:
        try:
            e.execute(denied["proposal_id"])
        except ProofError as error:
            print(f"Denied/canceled execution blocked: {error.code}")
        observations.append({"scenario": "denial", "decision": d})

    for mid in ("demo_msg_02", "demo_msg_03", "demo_msg_04", "demo_msg_07"):
        plan = workflow.plan(mid)
        show({"fixture": mid, "kind": plan["kind"], "summary": plan["summary"]})
        observations.append({"scenario": mid, "kind": plan["kind"], "summary": plan["summary"]})

    unknown = workflow.plan("demo_msg_06")["proposal"]
    a = get_decision(human, unknown, "approve" if scripted else None)
    if isinstance(a, str):
        result = e.execute(unknown["proposal_id"], a)
        receipt = e.execute(unknown["proposal_id"], a)
        show(asdict(result))
        print("Uncertain means unknown. Stored uncertainty is returned on replay; there is no resend")
        observations.append({"scenario": "uncertain", "outcome": asdict(result), "replay": asdict(receipt)})
    else:
        observations.append({"scenario": "uncertain", "decision": a})

    summary = {
        "label": LABEL, "fixture_version": e.fixtures.version,
        "decision_source": "scripted_test_fixture" if scripted else "human_console",
        "model_invoked": False, "production_mail_mcp_tested": False,
        "transport_kind": "in_memory_fake", "fake_transport_attempts": len(e.transport.calls),
        "accepted_fake_outbox_entries": len(e.transport.outbox), "fake_drafts": len(e.drafts),
        "automatic_send_retries": 0, "external_transport_calls": 0,
        "observations": observations,
    }
    print("SYNTHETIC REPLAY SUMMARY")
    show(summary)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        e.audit.save(output / "synthetic-trace.jsonl")
        (output / "replay-results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=LABEL)
    parser.add_argument("--mode", default="synthetic", help="Only synthetic is accepted")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--scripted-replay", action="store_true",
                        help="Use scripted fixture decisions, never claim human approval")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        run_demo(mode=args.mode, fixtures_path=args.fixtures, scripted=args.scripted_replay, output=args.output)
    except ProofError as error:
        print("Stopped safely: " + error.code, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
