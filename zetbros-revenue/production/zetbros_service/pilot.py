"""Operator CLI: init only an absent new pilot ledger, or check without Mail I/O."""
import argparse
import json
from .config import PendingPilotConfig, load_settings
from .pilot_runtime import build_service, initialize_new_pilot, pending_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check"))
    args = parser.parse_args()
    settings = load_settings()
    if args.command == "init":
        initialized = initialize_new_pilot(settings)
        result = (initialized if isinstance(initialized, dict) else
                  {"new_ledger_initialized": initialized.healthy(), "provider_connections": "not_performed"})
    else:
        result = pending_status() if isinstance(settings, PendingPilotConfig) else build_service(settings).readiness()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
