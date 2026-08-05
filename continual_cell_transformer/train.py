from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sample_batch(
    ids: list[int],
    batch: int,
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not ids:
        raise ValueError("Dataset produced zero tokens.")
    if len(ids) < length + 2:
        ids = ids * ((length + 2) // len(ids) + 1)
    starts = torch.randint(0, len(ids) - length - 1, (batch,))
    x = torch.stack(
        [torch.tensor(ids[start : start + length]) for start in starts]
    ).long().to(device)
    y = torch.stack(
        [torch.tensor(ids[start + 1 : start + length + 1]) for start in starts]
    ).long().to(device)
    return x, y


@torch.no_grad()
def evaluate(
    model: ContinualCellTransformer,
    ids: list[int],
    batch: int,
    length: int,
    device: torch.device,
    batches: int = 20,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        x, y = sample_batch(ids, batch, length, device)
        losses.append(float(model(x, labels=y)["loss"]))
    model.train(was_training)
    return sum(losses) / len(losses)


@torch.no_grad()
def collect_allocation_context(
    model: ContinualCellTransformer,
    ids: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    batches: int,
) -> tuple[list[torch.Tensor], list[list[int]]]:
    was_training = model.training
    model.eval()
    seed_lists: list[list[torch.Tensor]] = [[] for _ in model.pools()]
    parent_counts: list[Counter[int]] = [Counter() for _ in model.pools()]

    for _ in range(max(1, batches)):
        x, _ = sample_batch(ids, batch_size, seq_len, device)
        result = model(x)
        for index, seed in enumerate(result["growth_seeds"]):
            seed_lists[index].append(seed)
        for index, parent_ids in enumerate(result["growth_parent_ids"]):
            parent_counts[index].update(parent_ids)

    model.train(was_training)
    seeds = [
        torch.stack(values).mean(dim=0)
        for values in seed_lists
    ]
    parents = [
        [
            cell_id
            for cell_id, _ in counts.most_common(
                model.config.recurrent_fan_in
            )
        ]
        for counts in parent_counts
    ]
    return seeds, parents


def optimizer_for(
    model: ContinualCellTransformer,
    config: TrainConfig,
) -> torch.optim.AdamW:
    groups = {
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
        weight_decay = 0.0 if name == "cells" else config.weight_decay
        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rates[name],
                "weight_decay": weight_decay,
                "group_name": name,
            }
        )
    return torch.optim.AdamW(parameter_groups)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument(
        "--eval-file",
        help="Held-out file used for periodic validation. Defaults to training data.",
    )
    parser.add_argument("--retention-file")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--reset-optimizer", action="store_true")

    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)

    parser.add_argument("--add-token", action="append", default=[])
    parser.add_argument("--auto-add-words", action="store_true")

    parser.add_argument("--allocate-cells", type=int, default=0)
    parser.add_argument("--allocate-new-bank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--allocation-seed-batches", type=int, default=8)

    parser.add_argument("--enable-growth", action="store_true")
    parser.add_argument("--growth-warmup", type=int, default=100)
    parser.add_argument("--growth-patience", type=int, default=40)
    parser.add_argument("--growth-cells", type=int, default=8)
    parser.add_argument("--growth-loss-floor", type=float, default=1.5)
    parser.add_argument("--growth-active-fraction", type=float, default=0.20)
    parser.add_argument("--max-growth-events", type=int, default=1)
    parser.add_argument("--growth-cooldown", type=int, default=100)

    parser.add_argument("--seal-active-cells", action="store_true")
    parser.add_argument("--embedding-lr", type=float, default=1e-5)
    parser.add_argument("--attention-lr", type=float, default=2e-5)
    parser.add_argument("--ffn-lr", type=float, default=5e-5)
    parser.add_argument("--cell-lr", type=float, default=2e-4)
    parser.add_argument("--other-lr", type=float, default=5e-5)
    parser.add_argument("--mature-cell-scale", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--retention-replay-weight", type=float, default=0.0)
    parser.add_argument("--retention-output-penalty", type=float, default=0.0)
    parser.add_argument("--activity-sparsity-weight", type=float, default=0.0)

    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--initial-cells", type=int, default=64)
    parser.add_argument("--recurrent-steps", type=int, default=2)
    parser.add_argument("--recurrent-fan-in", type=int, default=8)
    parser.add_argument("--initial-cell-threshold", type=float, default=0.15)
    parser.add_argument("--new-cell-threshold", type=float, default=0.05)
    parser.add_argument("--cell-maturity-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


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
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    previous_step = 0
    structural_change = False
    migrated_checkpoint = False

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
        old_vocab = tokenizer.vocab_size

        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(train_text)
        tokenizer.add_tokens(tokens)

        model_config = ModelConfig.from_dict(checkpoint["model_config"])
        if args.cell_maturity_steps is not None:
            model_config.cell_maturity_steps = args.cell_maturity_steps
        model = ContinualCellTransformer(model_config)
        missing, unexpected = model.load_compatible_state_dict(
            checkpoint["model_state"]
        )
        if missing or unexpected:
            migrated_checkpoint = True
            structural_change = True
            print(
                "checkpoint migration:",
                f"missing={missing}",
                f"unexpected={unexpected}",
            )
            print(
                "Warning: bank/top-k checkpoints do not preserve their old "
                "routing exactly under V4. For controlled results, retrain "
                "the base task with V4 before continual learning."
            )

        if tokenizer.vocab_size > old_vocab:
            model.resize_vocabulary(tokenizer.vocab_size)
            structural_change = True
        previous_step = int(checkpoint.get("step", 0))
    else:
        tokenizer = DynamicByteTokenizer()
        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(train_text)
        tokenizer.add_tokens(tokens)

        cell_layers = tuple(
            index for index in (1, 3) if index < args.layers
        ) or (args.layers - 1,)
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            cell_layers=cell_layers,
            max_cells=args.max_cells,
            initial_active_cells=args.initial_cells,
            recurrent_steps=args.recurrent_steps,
            recurrent_fan_in=args.recurrent_fan_in,
            initial_cell_threshold=args.initial_cell_threshold,
            new_cell_threshold=args.new_cell_threshold,
            cell_maturity_steps=(
                args.cell_maturity_steps
                if args.cell_maturity_steps is not None
                else 2_000
            ),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = ContinualCellTransformer(model_config)

    if args.seq_len > model.config.max_seq_len:
        raise ValueError(
            f"--seq-len={args.seq_len} exceeds "
            f"max_seq_len={model.config.max_seq_len}."
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

    allocation_count = max(args.allocate_cells, args.allocate_new_bank)
    if args.allocate_new_bank > 0:
        print(
            "Deprecated --allocate-new-bank interpreted as --allocate-cells; "
            "V4 uses one shared population."
        )

    if allocation_count > 0:
        seeds, parents = collect_allocation_context(
            model=model,
            ids=train_ids,
            batch_size=min(args.batch_size, 32),
            seq_len=args.seq_len,
            device=device,
            batches=args.allocation_seed_batches,
        )
        allocated = model.allocate_cells(
            allocation_count,
            seeds,
            parents,
        )
        structural_change = True
        print(f"zero-impact cell allocation={allocated}")
        print(f"active cells={[pool.active_count for pool in model.pools()]}")

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
        mature_cell_scale=args.mature_cell_scale,
        retention_replay_weight=args.retention_replay_weight,
        retention_output_penalty=args.retention_output_penalty,
        activity_sparsity_weight=args.activity_sparsity_weight,
        enable_growth=args.enable_growth,
        growth_warmup_steps=args.growth_warmup,
        growth_patience=args.growth_patience,
        growth_cells=args.growth_cells,
        growth_loss_floor=args.growth_loss_floor,
        growth_active_fraction=args.growth_active_fraction,
        max_growth_events=args.max_growth_events,
        growth_cooldown_steps=args.growth_cooldown,
        seed=args.seed,
    )
    optimizer = optimizer_for(model, train_config)

    if (
        args.resume
        and not structural_change
        and not args.reset_optimizer
        and checkpoint is not None
        and "optimizer_state" in checkpoint
    ):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("restored optimizer state")
        except Exception as error:
            print("optimizer state restarted:", error)
    elif args.resume:
        print("optimizer state restarted after migration/allocation/resize")

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
        f"ppl={math.exp(min(initial_eval, 20)):.2f} "
        f"file={args.eval_file or args.train_file}"
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

    loss_ema = None
    low_capacity_steps = 0
    growth_events = 0
    last_growth_step = -10**9
    completed_steps = 0
    final_eval = initial_eval
    best_eval = initial_eval
    evaluations_without_improvement = 0

    model.train()
    for local_step in range(1, args.steps + 1):
        step = previous_step + local_step
        completed_steps = local_step

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
        old_result = None
        if retention_ids is not None and args.retention_replay_weight > 0:
            old_x, old_y = sample_batch(
                retention_ids,
                args.batch_size,
                args.seq_len,
                device,
            )
            old_result = model(old_x, labels=old_y)
            replay_loss = old_result["loss"]
            total_loss = (
                total_loss
                + args.retention_replay_weight * replay_loss
            )

        if (
            old_result is not None
            and args.retention_output_penalty > 0
        ):
            total_loss = (
                total_loss
                + args.retention_output_penalty
                * old_result["plastic_output_rms"].square()
            )

        if args.activity_sparsity_weight > 0:
            sparsity_source = old_result if old_result is not None else result
            total_loss = (
                total_loss
                + args.activity_sparsity_weight
                * sparsity_source["plastic_gate_mean"]
            )

        total_loss.backward()
        model.mask_cell_gradients(args.mature_cell_scale)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()
        model.advance_maturity()

        task_loss_value = float(task_loss.detach())
        total_loss_value = float(total_loss.detach())
        replay_loss_value = (
            None if replay_loss is None else float(replay_loss.detach())
        )
        active_fraction = float(result["active_fraction"].detach())
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
            and active_fraction >= args.growth_active_fraction
        ):
            low_capacity_steps += 1
        else:
            low_capacity_steps = 0

        if growth_ready and low_capacity_steps >= args.growth_patience:
            allocated = model.allocate_cells(
                args.growth_cells,
                result["growth_seeds"],
                result["growth_parent_ids"],
            )
            growth_events += 1
            last_growth_step = local_step
            low_capacity_steps = 0
            print(
                f"growth event={growth_events} allocated={allocated} "
                f"step={step} loss_ema={loss_ema:.4f} "
                f"active_fraction={active_fraction:.3f}"
            )
            optimizer = optimizer_for(model, train_config)

        if local_step == 1 or local_step % args.log_interval == 0:
            replay_text = (
                ""
                if replay_loss_value is None
                else f" replay_loss={replay_loss_value:.4f}"
            )
            old_output = (
                None
                if old_result is None
                else float(old_result["plastic_output_rms"].detach())
            )
            print(
                f"step={step} task_loss={task_loss_value:.4f} "
                f"total_loss={total_loss_value:.4f}{replay_text} "
                f"ppl={math.exp(min(task_loss_value, 20)):.2f} "
                f"active_fraction={active_fraction:.3f} "
                f"mean_gate={float(result['mean_gate'].detach()):.4f} "
                f"plastic_gate={float(result['plastic_gate_mean'].detach()):.4f} "
                f"plastic_output={float(result['plastic_output_rms'].detach()):.6f} "
                f"old_plastic_output={old_output} "
                f"cells={result['active_cells']} "
                f"active_ids={result['active_cell_ids']}"
            )

        if (
            local_step % args.eval_interval == 0
            or local_step == args.steps
        ):
            final_eval = evaluate(
                model,
                eval_ids,
                args.batch_size,
                args.seq_len,
                device,
                args.eval_batches,
            )
            print(
                f"eval step={step} loss={final_eval:.4f} "
                f"ppl={math.exp(min(final_eval, 20)):.2f}"
            )

            if final_eval < best_eval - args.early_stop_min_delta:
                best_eval = final_eval
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1

            if (
                args.early_stop_patience > 0
                and evaluations_without_improvement
                >= args.early_stop_patience
            ):
                print(
                    f"early stop at step={step}; "
                    f"best_eval={best_eval:.4f}"
                )
                break

    if args.seal_active_cells:
        model.seal_active_cells()

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
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": final_step,
        "train_args": vars(args),
        "architecture_version": 4,
    }
    torch.save(payload, out / "checkpoint.pt")
    tokenizer.save(out / "tokenizer.json")

    summary = {
        "step": final_step,
        "architecture_version": 4,
        "vocab_size": tokenizer.vocab_size,
        "active_cells": [pool.active_count for pool in model.pools()],
        "train_file": args.train_file,
        "eval_file": args.eval_file or args.train_file,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "retention_file": args.retention_file,
        "retention_before": retention_before,
        "retention_after": retention_after,
        "growth_events": growth_events,
        "migrated_checkpoint": migrated_checkpoint,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("saved", out / "checkpoint.pt")


if __name__ == "__main__":
    main()
