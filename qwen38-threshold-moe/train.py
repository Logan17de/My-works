#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qtm.config import dump_config, load_config


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/colab_smoke.json", help="JSON config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override any dotted config key; repeatable")

    # First-class CLI aliases for the knobs we change most during experiments.
    parser.add_argument("--run-name")
    parser.add_argument("--save-every", type=int, metavar="N", help="Save checkpoint every N optimizer steps; 0 disables periodic saves")
    parser.add_argument("--plot-every", type=int, metavar="N", help="Refresh plots every N steps; 0 disables during training")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", metavar="PATH|latest|none")
    parser.add_argument("--output-dir", help="Logs/checkpoints directory")
    parser.add_argument("--work-dir", help="Expert masters and optimizer-state directory")
    parser.add_argument("--drive-root", help="Convenience root: sets output-dir=<root>/runs and work-dir=<root>/work")
    parser.add_argument("--snapshot-experts", action=argparse.BooleanOptionalAction, default=None, help="Include an exact expert-store snapshot in checkpoints")

    parser.add_argument("--cache-layers", type=int, help="Number of routed-expert layers kept resident on GPU")
    parser.add_argument("--superbatch", type=int, help="Sequences processed while a layer is resident")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--dataset", help="Dataset alias/name, e.g. k2-txt360-v2 or hf:ORG/NAME")
    parser.add_argument("--dataset-config", help="HF dataset subset/config name")
    parser.add_argument("--shuffle-buffer", type=int)

    parser.add_argument("--lr", type=float, help="Resident-parameter learning rate")
    parser.add_argument("--expert-lr", type=float, help="Routed-expert Adafactor learning rate")
    parser.add_argument("--expert-grad-dtype", choices=["int8", "bf16", "fp32"])
    parser.add_argument("--expert-compute-dtype", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--warmup-tokens", type=int, help="Threshold curriculum warmup tokens; -1 keeps masking disabled")
    parser.add_argument("--imbalance-threshold", type=float, help="Trigger gap, e.g. 0.30")
    parser.add_argument("--block-fraction", type=float, help="Fraction of hottest experts blocked when threshold fires")
    parser.add_argument("--threshold", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable threshold masking")


def _append_override(overrides: list[str], key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    overrides.append(f"{key}={text}")


def apply_cli_aliases(args: argparse.Namespace) -> list[str]:
    overrides = list(args.set)
    mapping = {
        "run_name": "training.run_name",
        "save_every": "training.save_every_steps",
        "plot_every": "training.plot_every_steps",
        "max_steps": "training.max_steps",
        "resume": "training.resume",
        "output_dir": "training.output_dir",
        "work_dir": "training.work_dir",
        "snapshot_experts": "training.snapshot_experts",
        "cache_layers": "training.expert_cache_layers",
        "superbatch": "training.layer_superbatch",
        "microbatch": "training.microbatch_size",
        "seq_len": "data.sequence_length",
        "dataset": "data.source",
        "dataset_config": "data.config_name",
        "shuffle_buffer": "data.shuffle_buffer",
        "lr": "training.learning_rate",
        "expert_lr": "training.expert_learning_rate",
        "expert_grad_dtype": "training.gradient_transfer_dtype",
        "expert_compute_dtype": "training.expert_compute_dtype",
        "warmup_tokens": "threshold.warmup_tokens",
        "imbalance_threshold": "threshold.imbalance_threshold",
        "block_fraction": "threshold.block_fraction",
        "threshold": "threshold.enabled",
    }
    for attr, key in mapping.items():
        _append_override(overrides, key, getattr(args, attr, None))

    if getattr(args, "drive_root", None):
        root = str(Path(args.drive_root))
        _append_override(overrides, "training.output_dir", str(Path(root) / "runs"))
        _append_override(overrides, "training.work_dir", str(Path(root) / "work"))
    return overrides


def model_info(cfg: dict) -> dict:
    m, t, d, th = cfg["model"], cfg["training"], cfg["data"], cfg["threshold"]
    h, i = int(m["hidden_size"]), int(m["expert_intermediate_size"])
    l, e, s = int(m["num_layers"]), int(m["num_experts"]), int(m["num_shared_experts"])
    per_expert = 3 * h * i
    routed = l * e * per_expert
    shared = l * s * per_expert
    router = l * h * e
    token_io = 2 * int(m["vocab_size"]) * h
    return {
        "architecture": {
            "layers": l,
            "hidden": h,
            "routed_experts_per_layer": e,
            "shared_experts_per_layer": s,
            "top_k": int(m["top_k"]),
            "expert_params_each": per_expert,
            "routed_expert_params": routed,
            "shared_expert_params": shared,
            "router_params": router,
            "embedding_plus_lm_head_params": token_io,
            "rope": f"partial RoPE, theta={m['rope_theta']}, fraction={m['partial_rotary_factor']}",
            "router": "softmax + Top-K only",
        },
        "memory": {
            "routed_bf16_gib": routed * 2 / 1024**3,
            "routed_one_byte_gib": routed / 1024**3,
            "one_layer_routed_bf16_gib": e * per_expert * 2 / 1024**3,
            "one_layer_routed_one_byte_gib": e * per_expert / 1024**3,
            "expert_cache_layers": int(t["expert_cache_layers"]),
            "expert_compute_dtype": t["expert_compute_dtype"],
            "expert_gradient_transfer": t["gradient_transfer_dtype"],
        },
        "training": {
            "run_name": t["run_name"],
            "max_steps": int(t["max_steps"]),
            "save_every_steps": int(t["save_every_steps"]),
            "plot_every_steps": int(t["plot_every_steps"]),
            "resume": t.get("resume", "none"),
            "output_dir": t["output_dir"],
            "work_dir": t["work_dir"],
            "layer_superbatch": int(t["layer_superbatch"]),
            "microbatch_size": int(t["microbatch_size"]),
        },
        "data": {
            "source": d["source"],
            "config_name": d.get("config_name"),
            "sequence_length": int(d["sequence_length"]),
            "streaming": bool(d.get("streaming", True)),
        },
        "threshold": {
            "enabled": bool(th["enabled"]),
            "warmup_tokens": int(th["warmup_tokens"]),
            "imbalance_threshold": float(th["imbalance_threshold"]),
            "block_fraction": float(th["block_fraction"]),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3.8-inspired threshold MoE single-GPU research trainer")
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("show-config", "info", "inspect", "probe-data", "init-experts", "validate-layerwise", "train", "plot"):
        sp = sub.add_parser(name)
        add_common(sp)
        if name == "probe-data":
            sp.add_argument("--samples", type=int, default=3)
        if name == "validate-layerwise":
            sp.add_argument("--device", default="cuda")

    args = p.parse_args()
    cfg = load_config(args.config, apply_cli_aliases(args))

    if args.command == "show-config":
        print(dump_config(cfg))
        return
    if args.command in ("info", "inspect"):
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
