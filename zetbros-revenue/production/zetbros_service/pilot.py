"""Operator CLI: init only an absent new pilot ledger, or check without Mail I/O."""
import argparse
import json
from .config import load_settings
from .pilot_runtime import build_service, initialize_new_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check"))
    args = parser.parse_args()
    settings = load_settings()
    if args.command == "init":
        store = initialize_new_pilot(settings)
        result = {"new_ledger_initialized": store.healthy(), "provider_connections": "not_performed"}
    else:
        result = build_service(settings).readiness()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
