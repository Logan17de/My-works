from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


ARCHITECTURE_VERSION = ContinualCellTransformer.ARCHITECTURE_VERSION


def load_checkpoint(path: str | Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    version = int(checkpoint.get("architecture_version", 0))
    if version != ARCHITECTURE_VERSION:
        raise ValueError(
            f"Checkpoint architecture_version={version} is incompatible with "
            f"shared-population V{ARCHITECTURE_VERSION}. Retrain the base task "
            "with the current code; banked V2/V3 checkpoints are intentionally unsupported."
        )
    return checkpoint


def sample_batch(
    ids: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not ids:
        raise ValueError("Dataset produced zero tokens.")
    if len(ids) < seq_len + 2:
        ids = ids * ((seq_len + 2) // len(ids) + 1)

    starts = torch.randint(0, len(ids) - seq_len - 1, (batch_size,))
    x = torch.stack(
        [torch.tensor(ids[start : start + seq_len]) for start in starts]
    ).long()
    y = torch.stack(
        [torch.tensor(ids[start + 1 : start + seq_len + 1]) for start in starts]
    ).long()
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model: ContinualCellTransformer,
    ids: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    batches: int,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        x, y = sample_batch(ids, batch_size, seq_len, device)
        losses.append(float(model(x, labels=y)["loss"]))
    model.train(was_training)
    return sum(losses) / len(losses)


@torch.no_grad()
def collect_growth_seeds(
    model: ContinualCellTransformer,
    ids: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    batches: int,
) -> list[torch.Tensor]:
    was_training = model.training
    model.eval()
    per_pool: list[list[torch.Tensor]] = [[] for _ in model.pools()]
    for _ in range(max(1, batches)):
        x, _ = sample_batch(ids, batch_size, seq_len, device)
        result = model(x)
        for index, seed in enumerate(result["growth_seeds"]):
            per_pool[index].append(seed)
    model.train(was_training)
    return [torch.stack(items).mean(dim=0) for items in per_pool]


def build_optimizer(
    model: ContinualCellTransformer,
    config: TrainConfig,
) -> torch.optim.AdamW:
    groups: dict[str, list[torch.nn.Parameter]] = {
        "embedding": [],
        "attention": [],
        "ffn": [],
        "cells": [],
        "other": [],
    }

    for name, parameter in model.named_parameters():
        if name.startswith("token_embedding") or name.startswith("lm_head"):
            groups["embedding"].append(parameter)
        elif ".pool." in name:
            groups["cells"].append(parameter)
        elif ".attn." in name:
            groups["attention"].append(parameter)
        elif ".ffn." in name:
            groups["ffn"].append(parameter)
        else:
            groups["other"].append(parameter)

    learning_rates = {
        "embedding": config.embedding_lr,
        "attention": config.attention_lr,
        "ffn": config.ffn_lr,
        "cells": config.cell_lr,
        "other": config.other_lr,
    }

    parameter_groups = []
    for name, parameters in groups.items():
        if not parameters:
            continue
        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rates[name],
                # Cell tensors contain dormant and consolidated rows. Decoupled
                # weight decay would move them even after gradient masking.
                "weight_decay": 0.0 if name == "cells" else config.weight_decay,
                "group_name": name,
            }
        )
    return torch.optim.AdamW(parameter_groups)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file")
    parser.add_argument("--retention-file")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume")

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)

    parser.add_argument("--add-token", action="append", default=[])
    parser.add_argument("--auto-add-words", action="store_true")

    parser.add_argument("--allocate-cells", type=int, default=0)
    parser.add_argument("--allocation-seed-batches", type=int, default=8)
    parser.add_argument("--enable-growth", action="store_true")
    parser.add_argument("--growth-warmup", type=int, default=100)
    parser.add_argument("--growth-patience", type=int, default=40)
    parser.add_argument("--growth-cells", type=int, default=8)
    parser.add_argument("--growth-loss-floor", type=float, default=1.0)
    parser.add_argument("--growth-coverage-ceiling", type=float, default=1.0)
    parser.add_argument("--max-growth-events", type=int, default=1)
    parser.add_argument("--growth-cooldown", type=int, default=100)

    parser.add_argument("--consolidate-active-cells", action="store_true")
    parser.add_argument("--embedding-lr", type=float, default=1e-5)
    parser.add_argument("--attention-lr", type=float, default=2e-5)
    parser.add_argument("--ffn-lr", type=float, default=5e-5)
    parser.add_argument("--cell-lr", type=float, default=2e-4)
    parser.add_argument("--other-lr", type=float, default=5e-5)
    parser.add_argument("--consolidated-cell-scale", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--retention-replay-weight", type=float, default=0.0)
    parser.add_argument("--plastic-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--retention-output-penalty", type=float, default=0.0)
    parser.add_argument("--restore-optimizer", action="store_true")

    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--initial-cells", type=int, default=64)
    parser.add_argument("--threshold-temperature", type=float, default=0.08)
    parser.add_argument("--initial-threshold", type=float, default=0.10)
    parser.add_argument("--new-cell-threshold", type=float, default=0.00)
    parser.add_argument("--recurrent-steps", type=int, default=2)
    parser.add_argument("--recurrent-fan-in", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: ContinualCellTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: DynamicByteTokenizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "architecture_version": ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "train_args": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_text = Path(args.train_file).read_text(encoding="utf-8")
    eval_text = (
        Path(args.eval_file).read_text(encoding="utf-8")
        if args.eval_file
        else train_text
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    previous_step = 0
    checkpoint = None
    structural_change = False
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
        old_vocab = tokenizer.vocab_size
        new_tokens = list(args.add_token)
        if args.auto_add_words:
            new_tokens.extend(tokenizer.discover_tokens(train_text))
        tokenizer.add_tokens(new_tokens)

        model = ContinualCellTransformer(
            ModelConfig.from_dict(checkpoint["model_config"])
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if tokenizer.vocab_size > old_vocab:
            model.resize_vocabulary(tokenizer.vocab_size)
            structural_change = True
        previous_step = int(checkpoint.get("step", 0))
    else:
        tokenizer = DynamicByteTokenizer()
        new_tokens = list(args.add_token)
        if args.auto_add_words:
            new_tokens.extend(tokenizer.discover_tokens(train_text))
        tokenizer.add_tokens(new_tokens)

        cell_layers = tuple(
            index for index in (1, 3) if index < args.layers
        ) or (args.layers - 1,)
        model = ContinualCellTransformer(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                d_model=args.d_model,
                n_layers=args.layers,
                n_heads=args.heads,
                d_ff=args.d_ff,
                max_seq_len=args.seq_len,
                cell_layers=cell_layers,
                max_cells=args.max_cells,
                initial_active_cells=args.initial_cells,
                threshold_temperature=args.threshold_temperature,
                initial_threshold=args.initial_threshold,
                new_cell_threshold=args.new_cell_threshold,
                recurrent_steps=args.recurrent_steps,
                recurrent_fan_in=args.recurrent_fan_in,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )

    if args.seq_len > model.config.max_seq_len:
        raise ValueError(
            f"--seq-len={args.seq_len} exceeds model max_seq_len="
            f"{model.config.max_seq_len}."
        )

    model.to(device)
    train_ids = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    eval_ids = tokenizer.encode(eval_text, add_bos=True, add_eos=True)

    retention_ids = None
    if args.retention_file:
        retention_text = Path(args.retention_file).read_text(encoding="utf-8")
        retention_ids = tokenizer.encode(
            retention_text,
            add_bos=True,
            add_eos=True,
        )

    if args.allocate_cells > 0:
        seeds = collect_growth_seeds(
            model,
            train_ids,
            min(args.batch_size, 32),
            args.seq_len,
            device,
            args.allocation_seed_batches,
        )
        model.eval()
        probe_x, _ = sample_batch(
            train_ids,
            min(args.batch_size, 16),
            args.seq_len,
            device,
        )
        before_logits = model(probe_x)["logits"].detach().clone()
        allocation = model.allocate_cells(args.allocate_cells, seeds)
        after_logits = model(probe_x)["logits"].detach()
        allocation_drift = float((after_logits - before_logits).abs().max())
        structural_change = True
        print("one-time allocation:", allocation)
        print("pool summaries:", model.pool_summaries())
        print(f"zero-impact verification max_logit_drift={allocation_drift:.3e}")
        if allocation_drift > 1e-5:
            raise RuntimeError(
                "Cell allocation changed existing outputs before training."
            )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        embedding_lr=args.embedding_lr,
        attention_lr=args.attention_lr,
        ffn_lr=args.ffn_lr,
        cell_lr=args.cell_lr,
        other_lr=args.other_lr,
        consolidated_cell_scale=args.consolidated_cell_scale,
        retention_replay_weight=args.retention_replay_weight,
        plastic_sparsity_weight=args.plastic_sparsity_weight,
        retention_output_penalty=args.retention_output_penalty,
        enable_growth=args.enable_growth,
        growth_warmup_steps=args.growth_warmup,
        growth_patience=args.growth_patience,
        growth_cells=args.growth_cells,
        growth_loss_floor=args.growth_loss_floor,
        growth_coverage_ceiling=args.growth_coverage_ceiling,
        max_growth_events=args.max_growth_events,
        growth_cooldown_steps=args.growth_cooldown,
        seed=args.seed,
    )
    optimizer = build_optimizer(model, train_config)
    if (
        args.resume
        and args.restore_optimizer
        and not structural_change
        and checkpoint is not None
        and "optimizer_state" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("restored optimizer state")
    elif args.resume:
        print("started a fresh optimizer for continual learning")

    initial_eval = evaluate(
        model,
        eval_ids,
        args.batch_size,
        args.seq_len,
        device,
        args.eval_batches,
    )
    print(
        f"initial eval loss={initial_eval:.4f} "
        f"ppl={math.exp(min(initial_eval, 20)):.2f}"
    )

    retention_before = None
    if retention_ids is not None:
        retention_before = evaluate(
            model,
            retention_ids,
            args.batch_size,
            args.seq_len,
            device,
            args.eval_batches,
        )
        print(f"old-task loss before={retention_before:.4f}")

    loss_ema: float | None = None
    low_coverage_steps = 0
    growth_events = 0
    last_growth_step = -10**9
    best_eval = initial_eval
    no_improvement = 0
    final_eval = initial_eval
    completed_steps = 0

    model.train()
    for local_step in range(1, args.steps + 1):
        completed_steps = local_step
        global_step = previous_step + local_step
        x, y = sample_batch(
            train_ids,
            args.batch_size,
            args.seq_len,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        result = model(x, labels=y)
        task_loss = result["loss"]
        total_loss = task_loss
        replay_loss = None
        replay_result = None

        if retention_ids is not None and args.retention_replay_weight > 0.0:
            old_x, old_y = sample_batch(
                retention_ids,
                args.batch_size,
                args.seq_len,
                device,
            )
            replay_result = model(old_x, labels=old_y)
            replay_loss = replay_result["loss"]
            total_loss = total_loss + args.retention_replay_weight * replay_loss

        if args.plastic_sparsity_weight > 0.0:
            activity_source = replay_result if replay_result is not None else result
            total_loss = (
                total_loss
                + args.plastic_sparsity_weight
                * activity_source["plastic_activity_mean"]
            )

        if replay_result is not None and args.retention_output_penalty > 0.0:
            total_loss = (
                total_loss
                + args.retention_output_penalty
                * replay_result["plastic_output_rms"].square()
            )

        total_loss.backward()
        model.mask_cell_gradients(args.consolidated_cell_scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.advance_maturity()

        task_loss_value = float(task_loss.detach())
        total_loss_value = float(total_loss.detach())
        coverage = float(result["coverage"].detach())
        loss_ema = (
            task_loss_value
            if loss_ema is None
            else 0.98 * loss_ema + 0.02 * task_loss_value
        )

        growth_ready = (
            args.enable_growth
            and growth_events < args.max_growth_events
            and local_step >= args.growth_warmup
            and local_step - last_growth_step >= args.growth_cooldown
        )
        if (
            growth_ready
            and loss_ema > args.growth_loss_floor
            and coverage < args.growth_coverage_ceiling
        ):
            low_coverage_steps += 1
        else:
            low_coverage_steps = 0

        if growth_ready and low_coverage_steps >= args.growth_patience:
            allocated = model.allocate_cells(
                args.growth_cells,
                result["growth_seeds"],
            )
            growth_events += 1
            last_growth_step = local_step
            low_coverage_steps = 0
            print(
                f"growth event={growth_events} step={global_step} "
                f"allocated={allocated} loss_ema={loss_ema:.4f} "
                f"coverage={coverage:.3f}"
            )

        if local_step == 1 or local_step % args.log_interval == 0:
            replay_text = (
                ""
                if replay_loss is None
                else f" replay_loss={float(replay_loss.detach()):.4f}"
            )
            print(
                f"step={global_step} task_loss={task_loss_value:.4f} "
                f"total_loss={total_loss_value:.4f}{replay_text} "
                f"ppl={math.exp(min(task_loss_value, 20)):.2f} "
                f"coverage={coverage:.3f} "
                f"plastic_activity={float(result['plastic_activity_mean'].detach()):.3f} "
                f"plastic_output={float(result['plastic_output_rms'].detach()):.6f} "
                f"pools={result['pool_summaries']}"
            )

        if local_step % args.eval_interval == 0 or local_step == args.steps:
            final_eval = evaluate(
                model,
                eval_ids,
                args.batch_size,
                args.seq_len,
                device,
                args.eval_batches,
            )
            print(
                f"eval step={global_step} loss={final_eval:.4f} "
                f"ppl={math.exp(min(final_eval, 20)):.2f}"
            )

            if final_eval < best_eval - args.early_stop_min_delta:
                best_eval = final_eval
                no_improvement = 0
                save_checkpoint(
                    out_dir / "best_checkpoint.pt",
                    model,
                    optimizer,
                    tokenizer,
                    global_step,
                    args,
                )
            else:
                no_improvement += 1

            if (
                args.early_stop_patience > 0
                and no_improvement >= args.early_stop_patience
            ):
                print(
                    f"early stopping after {no_improvement} evaluations "
                    "without meaningful improvement"
                )
                break

    if args.consolidate_active_cells:
        model.consolidate_active_cells()
        print("Consolidated all currently active cells.")

    retention_after = None
    if retention_ids is not None:
        retention_after = evaluate(
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

    final_step = previous_step + completed_steps
    save_checkpoint(
        out_dir / "checkpoint.pt",
        model,
        optimizer,
        tokenizer,
        final_step,
        args,
    )
    tokenizer.save(out_dir / "tokenizer.json")

    summary = {
        "architecture_version": ARCHITECTURE_VERSION,
        "step": final_step,
        "vocab_size": tokenizer.vocab_size,
        "pool_summaries": model.pool_summaries(),
        "train_file": args.train_file,
        "eval_file": args.eval_file or args.train_file,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "retention_file": args.retention_file,
        "retention_before": retention_before,
        "retention_after": retention_after,
        "growth_events": growth_events,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("saved", out_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
