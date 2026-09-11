from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from config import ModelConfig
from model import ContinualCellTransformerV2, config_from_v1, migrate_v1_state
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sample_batch(ids: list[int], batch_size: int, seq_len: int, device: torch.device):
    if not ids:
        raise ValueError("Dataset produced zero tokens.")
    if len(ids) < seq_len + 2:
        ids = ids * ((seq_len + 2) // len(ids) + 1)
    starts = torch.randint(0, len(ids) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(ids[start : start + seq_len]) for start in starts])
    y = torch.stack([torch.tensor(ids[start + 1 : start + seq_len + 1]) for start in starts])
    return x.long().to(device), y.long().to(device)


@torch.no_grad()
def evaluate(model, ids, batch_size, seq_len, device, batches=20):
    was_training = model.training
    model.eval()
    losses = []
    stable_counts = None
    plastic_counts = None
    gate_sums = None
    gate_maxes = None

    for _ in range(batches):
        x, y = sample_batch(ids, batch_size, seq_len, device)
        output = model(x, labels=y)
        losses.append(float(output["loss"]))
        if stable_counts is None:
            stable_counts = [torch.zeros_like(item.stable_counts) for item in output["route_stats"]]
            plastic_counts = [torch.zeros_like(item.plastic_counts) for item in output["route_stats"]]
            gate_sums = [0.0 for _ in output["route_stats"]]
            gate_maxes = [0.0 for _ in output["route_stats"]]
        for index, item in enumerate(output["route_stats"]):
            stable_counts[index] += item.stable_counts
            plastic_counts[index] += item.plastic_counts
            gate_sums[index] += float(item.plastic_gate_mean)
            gate_maxes[index] = max(gate_maxes[index], float(item.plastic_gate_max))

    route_summary = []
    if stable_counts is not None:
        for index, block in enumerate(model.cell_blocks()):
            stable_top = torch.topk(
                stable_counts[index], k=min(5, block.stable_pool.active_count)
            ).indices.tolist()
            plastic_active = block.plastic_pool.active_count
            plastic_top = (
                torch.topk(
                    plastic_counts[index][:plastic_active],
                    k=min(5, plastic_active),
                ).indices.tolist()
                if plastic_active
                else []
            )
            route_summary.append(
                {
                    "cell_block": index,
                    "stable_top": stable_top,
                    "plastic_top": plastic_top,
                    "plastic_gate_mean": gate_sums[index] / batches,
                    "plastic_gate_max": gate_maxes[index],
                }
            )

    model.train(was_training)
    return sum(losses) / len(losses), route_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Continual Cell Transformer V2.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file")
    parser.add_argument("--retention-file")
    parser.add_argument("--out-dir", required=True)

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume-v1", help="Import a V1 checkpoint as the frozen stable path.")
    source.add_argument("--resume-v2", help="Continue from a V2 checkpoint.")

    parser.add_argument("--mode", choices=("base", "continual"), default="continual")
    parser.add_argument("--allocate-plastic-cells", type=int, default=8)
    parser.add_argument("--plastic-capacity", type=int, default=64)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--plastic-lr", type=float, default=5e-4)
    parser.add_argument("--adapter-lr", type=float, default=5e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-4)
    parser.add_argument("--gate-penalty", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--add-token", action="append", default=[])
    parser.add_argument("--auto-add-words", action="store_true")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--stable-cells", type=int, default=64)
    parser.add_argument("--stable-top-k", type=int, default=8)
    parser.add_argument("--plastic-top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def build_optimizer(model: ContinualCellTransformerV2, args: argparse.Namespace):
    if args.mode == "base":
        for parameter in model.parameters():
            parameter.requires_grad = True
        for block in model.cell_blocks():
            for parameter in block.plastic_pool.parameters():
                parameter.requires_grad = False
            for parameter in block.plastic_adapter.parameters():
                parameter.requires_grad = False
            block.plastic_gate_bias.requires_grad = False
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.base_lr,
            weight_decay=args.weight_decay,
        )

    model.freeze_for_continual_learning()
    plastic_bank = []
    adapter = []
    gate = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "plastic_pool" in name:
            plastic_bank.append(parameter)
        elif "plastic_adapter" in name:
            adapter.append(parameter)
        elif "plastic_gate_bias" in name:
            gate.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": plastic_bank, "lr": args.plastic_lr},
            {"params": adapter, "lr": args.adapter_lr},
            {"params": gate, "lr": args.gate_lr},
        ],
        weight_decay=args.weight_decay,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_text = Path(args.train_file).read_text(encoding="utf-8")
    eval_text = Path(args.eval_file).read_text(encoding="utf-8") if args.eval_file else train_text
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    previous_step = 0
    source_checkpoint = None
    if args.resume_v1:
        source_checkpoint = load_checkpoint(args.resume_v1)
        tokenizer = DynamicByteTokenizer.from_dict(source_checkpoint["tokenizer"])
        v1_state = source_checkpoint["model_state"]
        active_counts = []
        for layer_index in source_checkpoint["model_config"].get("cell_layers", (1, 3)):
            mask = v1_state[f"blocks.{layer_index}.pool.active_mask"]
            active_counts.append(int(mask.sum()))
        stable_cells = min(active_counts)
        config = config_from_v1(
            source_checkpoint["model_config"],
            stable_cells=stable_cells,
            plastic_capacity=args.plastic_capacity,
        )
        model = ContinualCellTransformerV2(config)
        migrate_v1_state(model, v1_state)
        previous_step = int(source_checkpoint.get("step", 0))
        print(f"migrated V1 stable path with {stable_cells} cells per cell block")
    elif args.resume_v2:
        source_checkpoint = load_checkpoint(args.resume_v2)
        tokenizer = DynamicByteTokenizer.from_dict(source_checkpoint["tokenizer"])
        config = ModelConfig.from_dict(source_checkpoint["model_config"])
        model = ContinualCellTransformerV2(config)
        model.load_state_dict(source_checkpoint["model_state"], strict=True)
        previous_step = int(source_checkpoint.get("step", 0))
    else:
        if args.mode != "base":
            raise ValueError("Continual mode requires --resume-v1 or --resume-v2.")
        tokenizer = DynamicByteTokenizer()
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            cell_layers=tuple(index for index in (1, 3) if index < args.layers)
            or (args.layers - 1,),
            stable_cells=args.stable_cells,
            stable_top_k=args.stable_top_k,
            plastic_capacity=args.plastic_capacity,
            plastic_top_k=args.plastic_top_k,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = ContinualCellTransformerV2(config)

    old_vocab = tokenizer.vocab_size
    new_tokens = list(args.add_token)
    if args.auto_add_words:
        new_tokens += tokenizer.discover_tokens(train_text)
    tokenizer.add_tokens(new_tokens)
    if tokenizer.vocab_size != old_vocab:
        if args.mode == "continual":
            raise ValueError(
                "V2 continual mode currently forbids vocabulary resize because the frozen LM head "
                "cannot learn new output rows. Train token insertion separately."
            )
        model.resize_vocabulary(tokenizer.vocab_size)

    if args.seq_len > model.config.max_seq_len:
        raise ValueError("seq-len exceeds checkpoint max_seq_len.")

    model.to(device)
    train_ids = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    eval_ids = tokenizer.encode(eval_text, add_bos=True, add_eos=True)

    if args.mode == "continual":
        plastic_counts = [block.plastic_pool.active_count for block in model.cell_blocks()]
        if all(count == 0 for count in plastic_counts):
            model.eval()
            seed_x, _ = sample_batch(train_ids, args.batch_size, args.seq_len, device)
            seeds = model(seed_x)["allocation_seeds"]
            allocated = model.allocate_plastic_once(args.allocate_plastic_cells, seeds)
            print("one-time plastic allocation:", allocated)
        elif args.allocate_plastic_cells:
            print("plastic bank already allocated; no additional cells were added:", plastic_counts)

    optimizer = build_optimizer(model, args)

    initial_eval, initial_routes = evaluate(
        model, eval_ids, args.batch_size, args.seq_len, device, args.eval_batches
    )
    print(f"initial eval loss={initial_eval:.4f} ppl={math.exp(min(initial_eval, 20)):.2f}")
    print("initial routes:", json.dumps(initial_routes))

    retention_ids = None
    retention_before = None
    retention_routes_before = None
    if args.retention_file:
        retention_text = Path(args.retention_file).read_text(encoding="utf-8")
        retention_ids = tokenizer.encode(retention_text, add_bos=True, add_eos=True)
        retention_before, retention_routes_before = evaluate(
            model,
            retention_ids,
            args.batch_size,
            args.seq_len,
            device,
            args.eval_batches,
        )
        print(f"old-task loss before={retention_before:.4f}")
        print("old-task routes before:", json.dumps(retention_routes_before))

    best_eval = float("inf")
    no_improvement = 0
    final_step = previous_step
    final_eval = initial_eval
    model.train()

    for local_step in range(1, args.steps + 1):
        step = previous_step + local_step
        final_step = step
        x, y = sample_batch(train_ids, args.batch_size, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x, labels=y)
        ce_loss = output["loss"]
        total_loss = ce_loss + args.gate_penalty * output["plastic_gate_mean"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.plastic_parameters() if args.mode == "continual" else model.parameters(),
            args.grad_clip,
        )
        optimizer.step()

        if local_step == 1 or local_step % args.log_interval == 0:
            print(
                f"step={step} ce={float(ce_loss.detach()):.4f} "
                f"total={float(total_loss.detach()):.4f} "
                f"ppl={math.exp(min(float(ce_loss.detach()), 20)):.2f} "
                f"plastic_gate={float(output['plastic_gate_mean'].detach()):.4f} "
                f"stable={output['stable_cells']} plastic={output['plastic_cells']}"
            )

        if local_step % args.eval_interval == 0 or local_step == args.steps:
            final_eval, routes = evaluate(
                model, eval_ids, args.batch_size, args.seq_len, device, args.eval_batches
            )
            print(f"eval step={step} loss={final_eval:.4f} ppl={math.exp(min(final_eval, 20)):.2f}")
            print("routes:", json.dumps(routes))

            if final_eval < best_eval - args.early_stop_min_delta:
                best_eval = final_eval
                no_improvement = 0
                torch.save(
                    {
                        "architecture_version": 2,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "model_config": model.config.to_dict(),
                        "tokenizer": tokenizer.to_dict(),
                        "step": step,
                        "train_args": vars(args),
                    },
                    out_dir / "best_checkpoint.pt",
                )
            else:
                no_improvement += 1
                if args.early_stop_patience > 0 and no_improvement >= args.early_stop_patience:
                    print(f"early stopping at step {step}; best eval={best_eval:.4f}")
                    break
            model.train()

    retention_after = None
    retention_routes_after = None
    if retention_ids is not None:
        retention_after, retention_routes_after = evaluate(
            model,
            retention_ids,
            args.batch_size,
            args.seq_len,
            device,
            args.eval_batches,
        )
        print(
            f"old-task loss after={retention_after:.4f}; "
            f"delta={retention_after - retention_before:+.4f}"
        )
        print("old-task routes after:", json.dumps(retention_routes_after))

    payload = {
        "architecture_version": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": final_step,
        "train_args": vars(args),
    }
    torch.save(payload, out_dir / "checkpoint.pt")
    tokenizer.save(out_dir / "tokenizer.json")
    summary = {
        "architecture_version": 2,
        "step": final_step,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "retention_before": retention_before,
        "retention_after": retention_after,
        "retention_delta": (
            retention_after - retention_before
            if retention_after is not None and retention_before is not None
            else None
        ),
        "initial_routes": initial_routes,
        "retention_routes_before": retention_routes_before,
        "retention_routes_after": retention_routes_after,
        "stable_cells": [block.stable_pool.active_count for block in model.cell_blocks()],
        "plastic_cells": [block.plastic_pool.active_count for block in model.cell_blocks()],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("saved", out_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
