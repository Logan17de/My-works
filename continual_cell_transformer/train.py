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
def collect_growth_seeds(
    model: ContinualCellTransformer,
    ids: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    batches: int,
) -> list[torch.Tensor]:
    """Average several new-task contexts before initializing a new bank."""
    was_training = model.training
    model.eval()
    per_pool: list[list[torch.Tensor]] = [
        [] for _ in model.pools()
    ]
    for _ in range(max(1, batches)):
        x, _ = sample_batch(ids, batch_size, seq_len, device)
        result = model(x)
        for pool_index, seed in enumerate(result["growth_seeds"]):
            per_pool[pool_index].append(seed)
    model.train(was_training)
    return [
        torch.stack(seeds).mean(dim=0)
        for seeds in per_pool
    ]


def optimizer_for(
    model: ContinualCellTransformer,
    cfg: TrainConfig,
) -> torch.optim.AdamW:
    groups = {
        "embedding": [],
        "attention": [],
        "ffn": [],
        "cells": [],
        "router": [],
        "other": [],
    }

    for name, parameter in model.named_parameters():
        if name.startswith("token_embedding") or name.startswith("lm_head"):
            groups["embedding"].append(parameter)
        elif ".pool.bank_gate_" in name:
            groups["router"].append(parameter)
        elif ".pool." in name:
            groups["cells"].append(parameter)
        elif ".attn." in name:
            groups["attention"].append(parameter)
        elif ".ffn." in name:
            groups["ffn"].append(parameter)
        else:
            groups["other"].append(parameter)

    learning_rates = {
        "embedding": cfg.embedding_lr,
        "attention": cfg.attention_lr,
        "ffn": cfg.ffn_lr,
        "cells": cfg.cell_lr,
        "router": cfg.router_lr,
        "other": cfg.other_lr,
    }

    parameter_groups = []
    for name, parameters in groups.items():
        if not parameters:
            continue
        group_weight_decay = (
            0.0 if name in {"cells", "router"} else cfg.weight_decay
        )
        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rates[name],
                "weight_decay": group_weight_decay,
                "group_name": name,
            }
        )

    return torch.optim.AdamW(parameter_groups)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument(
        "--eval-file",
        help="Held-out file used for periodic validation. Defaults to the training file.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--retention-file")

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)

    parser.add_argument("--add-token", action="append", default=[])
    parser.add_argument("--auto-add-words", action="store_true")

    parser.add_argument("--allocate-new-bank", type=int, default=0)
    parser.add_argument("--allocation-seed-batches", type=int, default=8)
    parser.add_argument("--do-not-seal-existing-banks", action="store_true")

    parser.add_argument("--enable-growth", action="store_true")
    parser.add_argument("--growth-warmup", type=int, default=100)
    parser.add_argument("--growth-patience", type=int, default=40)
    parser.add_argument("--growth-cells", type=int, default=8)
    parser.add_argument("--growth-confidence", type=float, default=0.42)
    parser.add_argument("--growth-loss-floor", type=float, default=1.5)
    parser.add_argument("--max-growth-events", type=int, default=1)
    parser.add_argument("--growth-cooldown", type=int, default=100)

    parser.add_argument("--seal-active-cells", action="store_true")
    parser.add_argument("--embedding-lr", type=float, default=1e-5)
    parser.add_argument("--attention-lr", type=float, default=2e-5)
    parser.add_argument("--ffn-lr", type=float, default=5e-5)
    parser.add_argument("--cell-lr", type=float, default=2e-4)
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument("--other-lr", type=float, default=5e-5)
    parser.add_argument("--mature-cell-scale", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--retention-replay-weight",
        type=float,
        default=0.0,
        help="Add supervised old-task loss during new-task training.",
    )
    parser.add_argument(
        "--gate-sparsity-weight",
        type=float,
        default=0.0,
        help="Penalize plastic-bank activation, preferably on retention batches.",
    )

    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--initial-cells", type=int, default=64)
    parser.add_argument("--top-k-cells", type=int, default=8)
    parser.add_argument("--max-cell-banks", type=int, default=8)
    parser.add_argument("--new-bank-adapter-scale", type=float, default=1.0)
    parser.add_argument("--bank-context-scale", type=float, default=0.25)
    parser.add_argument("--bank-gate-temperature", type=float, default=0.70)
    parser.add_argument("--new-bank-gate-bias", type=float, default=-2.0)
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

    previous_step = 0
    structural_change = False
    checkpoint = None

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
        old_vocab = tokenizer.vocab_size

        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(train_text)
        tokenizer.add_tokens(tokens)

        config = ModelConfig.from_dict(checkpoint["model_config"])
        model = ContinualCellTransformer(config)
        missing, unexpected = model.load_compatible_state_dict(
            checkpoint["model_state"]
        )
        if missing or unexpected:
            print(
                f"checkpoint migration: missing={missing} "
                f"unexpected={unexpected}"
            )
            structural_change = True

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

        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            cell_layers=cell_layers,
            max_cells=args.max_cells,
            initial_active_cells=args.initial_cells,
            top_k_cells=args.top_k_cells,
            max_cell_banks=args.max_cell_banks,
            new_bank_adapter_scale=args.new_bank_adapter_scale,
            bank_context_scale=args.bank_context_scale,
            bank_gate_temperature=args.bank_gate_temperature,
            new_bank_gate_bias=args.new_bank_gate_bias,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = ContinualCellTransformer(config)

    if args.seq_len > model.config.max_seq_len:
        raise ValueError(
            f"--seq-len={args.seq_len} exceeds "
            f"model max_seq_len={model.config.max_seq_len}."
        )

    model.to(device)
    train_ids = tokenizer.encode(
        train_text,
        add_bos=True,
        add_eos=True,
    )
    eval_ids = tokenizer.encode(
        eval_text,
        add_bos=True,
        add_eos=True,
    )

    retention_ids = None
    if args.retention_file:
        retention_text = Path(args.retention_file).read_text(
            encoding="utf-8"
        )
        retention_ids = tokenizer.encode(
            retention_text,
            add_bos=True,
            add_eos=True,
        )

    if args.allocate_new_bank > 0:
        seeds = collect_growth_seeds(
            model=model,
            ids=train_ids,
            batch_size=min(args.batch_size, 32),
            seq_len=args.seq_len,
            device=device,
            batches=args.allocation_seed_batches,
        )
        allocation = model.allocate_new_bank(
            args.allocate_new_bank,
            seeds,
            not args.do_not_seal_existing_banks,
        )
        structural_change = True
        print(f"one-time bank allocation={allocation}")
        print(f"bank summaries={model.bank_summaries()}")

    train_config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        embedding_lr=args.embedding_lr,
        attention_lr=args.attention_lr,
        ffn_lr=args.ffn_lr,
        cell_lr=args.cell_lr,
        router_lr=args.router_lr,
        other_lr=args.other_lr,
        mature_cell_scale=args.mature_cell_scale,
        retention_replay_weight=args.retention_replay_weight,
        gate_sparsity_weight=args.gate_sparsity_weight,
        enable_growth=args.enable_growth,
        growth_warmup_steps=args.growth_warmup,
        growth_patience=args.growth_patience,
        growth_cells=args.growth_cells,
        growth_confidence=args.growth_confidence,
        growth_loss_floor=args.growth_loss_floor,
        max_growth_events=args.max_growth_events,
        growth_cooldown_steps=args.growth_cooldown,
        seed=args.seed,
    )
    optimizer = optimizer_for(model, train_config)

    if (
        args.resume
        and not structural_change
        and checkpoint is not None
        and "optimizer_state" in checkpoint
    ):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("restored optimizer state")
        except Exception as error:
            print("optimizer state restarted:", error)
    elif args.resume:
        print(
            "optimizer state restarted after migration, vocabulary, "
            "or bank structural change"
        )

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
    low_confidence = 0
    growth_events = 0
    last_growth_step = -10**9
    final_eval = initial_eval
    completed_local_steps = 0

    best_eval = initial_eval
    evaluations_without_improvement = 0
    model.train()

    for local_step in range(1, args.steps + 1):
        step = previous_step + local_step
        completed_local_steps = local_step

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

        if (
            retention_ids is not None
            and args.retention_replay_weight > 0.0
        ):
            old_x, old_y = sample_batch(
                retention_ids,
                args.batch_size,
                args.seq_len,
                device,
            )
            replay_result = model(old_x, labels=old_y)
            replay_loss = replay_result["loss"]
            total_loss = (
                total_loss
                + args.retention_replay_weight * replay_loss
            )

        if args.gate_sparsity_weight > 0.0:
            gate_source = (
                replay_result
                if replay_result is not None
                else result
            )
            total_loss = (
                total_loss
                + args.gate_sparsity_weight
                * gate_source["plastic_gate_mean"]
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
            None
            if replay_loss is None
            else float(replay_loss.detach())
        )
        confidence = float(result["cell_confidence"])
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
            and confidence < args.growth_confidence
        ):
            low_confidence += 1
        else:
            low_confidence = 0

        if growth_ready and low_confidence >= args.growth_patience:
            allocated = model.allocate_new_bank(
                args.growth_cells,
                result["growth_seeds"],
                True,
            )
            growth_events += 1
            last_growth_step = local_step
            low_confidence = 0
            print(
                f"growth event={growth_events} "
                f"allocation={allocated} step={step} "
                f"loss_ema={loss_ema:.4f} "
                f"confidence={confidence:.3f}"
            )
            print(f"bank summaries={model.bank_summaries()}")

        if local_step == 1 or local_step % args.log_interval == 0:
            replay_text = (
                ""
                if replay_loss_value is None
                else f" replay_loss={replay_loss_value:.4f}"
            )
            replay_gates = (
                None
                if replay_result is None
                else replay_result["bank_gate_means"]
            )
            print(
                f"step={step} task_loss={task_loss_value:.4f} "
                f"total_loss={total_loss_value:.4f}"
                f"{replay_text} "
                f"ppl={math.exp(min(task_loss_value, 20)):.2f} "
                f"confidence={confidence:.3f} "
                f"banks={result['cell_banks']} "
                f"task_gates={result['bank_gate_means']} "
                f"old_gates={replay_gates} "
                f"top_ids={result['bank_top_ids']}"
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
                f"ppl={math.exp(min(final_eval, 20)):.2f} "
                f"file={args.eval_file or args.train_file}"
            )

            if args.early_stop_patience > 0:
                if (
                    final_eval
                    < best_eval - args.early_stop_min_delta
                ):
                    best_eval = final_eval
                    evaluations_without_improvement = 0
                else:
                    evaluations_without_improvement += 1

                if (
                    evaluations_without_improvement
                    >= args.early_stop_patience
                ):
                    print(
                        f"early stopping at step={step}; "
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

    final_step = previous_step + completed_local_steps
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": final_step,
        "train_args": vars(args),
        "architecture_version": 3,
    }
    torch.save(payload, out / "checkpoint.pt")
    tokenizer.save(out / "tokenizer.json")

    summary = {
        "step": final_step,
        "architecture_version": 3,
        "vocab_size": tokenizer.vocab_size,
        "active_cells": [
            pool.active_count for pool in model.pools()
        ],
        "cell_banks": model.bank_summaries(),
        "train_file": args.train_file,
        "eval_file": args.eval_file or args.train_file,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "retention_file": args.retention_file,
        "retention_before": retention_before,
        "retention_after": retention_after,
        "retention_replay_weight": args.retention_replay_weight,
        "growth_events": growth_events,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("saved", out / "checkpoint.pt")


if __name__ == "__main__":
    main()
