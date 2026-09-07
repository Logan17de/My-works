#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qtm.config import dump_config, load_config


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/colab_smoke.json", help="JSON config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override any dotted config key; repeatable")


def model_info(cfg: dict) -> dict:
    m = cfg["model"]
    h, i = int(m["hidden_size"]), int(m["expert_intermediate_size"])
    l, e, s = int(m["num_layers"]), int(m["num_experts"]), int(m["num_shared_experts"])
    per_expert = 3 * h * i
    routed = l * e * per_expert
    shared = l * s * per_expert
    router = l * h * e
    token_io = 2 * int(m["vocab_size"]) * h
    return {
        "expert_params_each": per_expert,
        "routed_expert_params": routed,
        "shared_expert_params": shared,
        "router_params": router,
        "embedding_plus_lm_head_params": token_io,
        "routed_bf16_gib": routed * 2 / 1024**3,
        "routed_one_byte_gib": routed / 1024**3,
        "one_layer_routed_bf16_gib": e * per_expert * 2 / 1024**3,
        "one_layer_routed_one_byte_gib": e * per_expert / 1024**3,
        "rope": f"partial RoPE, theta={m['rope_theta']}, fraction={m['partial_rotary_factor']}",
        "router": "softmax + Top-K only",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3.8-inspired threshold MoE single-GPU research trainer")
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("show-config", "info", "probe-data", "init-experts", "validate-layerwise", "train", "plot"):
        sp = sub.add_parser(name)
        add_common(sp)
        if name == "probe-data":
            sp.add_argument("--samples", type=int, default=3)
        if name == "validate-layerwise":
            sp.add_argument("--device", default="cuda")

    args = p.parse_args()
    cfg = load_config(args.config, args.set)

    if args.command == "show-config":
        print(dump_config(cfg))
        return
    if args.command == "info":
        print(json.dumps(model_info(cfg), indent=2))
        return
    if args.command == "probe-data":
        from qtm.data import probe_dataset
        print(json.dumps(probe_dataset(cfg["data"], n=args.samples), indent=2, ensure_ascii=False))
        return
    if args.command == "init-experts":
        from qtm.experts import ExpertStore
        m, t = cfg["model"], cfg["training"]
        root = Path(t["work_dir"]) / t["run_name"] / "expert_store"
        store = ExpertStore(
            root, num_layers=int(m["num_layers"]), num_experts=int(m["num_experts"]),
            hidden_size=int(m["hidden_size"]), intermediate_size=int(m["expert_intermediate_size"]),
            storage=t["expert_store"], init_std=float(m["initializer_range"]), seed=int(t["seed"]),
        )
        def progress(idx, total):
            print(f"\rInitializing expert layer {idx + 1}/{total}", end="", flush=True)
        store.initialize_all(expert_chunk=int(t["expert_update_chunk"]), progress=progress)
        print(f"\nExpert store ready: {root}")
        return
    if args.command == "validate-layerwise":
        from qtm.validate import validate_layerwise_recompute
        result = validate_layerwise_recompute(args.device)
        print(json.dumps(result, indent=2))
        worst = max(result.values())
        if worst > 5e-2:
            raise SystemExit(f"Validation failed: max relative error {worst}")
        return
    if args.command == "plot":
        from qtm.logging_utils import TrainingLogger
        t = cfg["training"]
        logger = TrainingLogger(Path(t["output_dir"]) / t["run_name"])
        logger.plot()
        print(logger.plot_path)
        return
    if args.command == "train":
        from qtm.trainer import LayerwiseTrainer
        trainer = LayerwiseTrainer(cfg)
        trainer.train()
        return


if __name__ == "__main__":
    main()
