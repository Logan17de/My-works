from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer

ARCHITECTURE_VERSION = ContinualCellTransformer.ARCHITECTURE_VERSION
MATH_PATTERN = re.compile(r"<Q>\s*([^\n]+?)\s*=\s*\n<A>\s*(-?\d+)")


def load_checkpoint(path: str | Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    version = int(checkpoint.get("architecture_version", 0))
    if version != ARCHITECTURE_VERSION:
        raise ValueError(
            f"Checkpoint V{version} is incompatible with V{ARCHITECTURE_VERSION}; "
            "retrain the base task with the current architecture."
        )
    return checkpoint


def sample_batch(
    ids: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not ids:
        raise ValueError("Dataset produced zero tokens")
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
def evaluate_loss(
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
        result = model(x, labels=y, adaptive_inference=False)
        losses.append(float(result["loss"]))
    model.train(was_training)
    return sum(losses) / len(losses)


def parse_math_examples(text: str) -> list[tuple[str, str]]:
    return [(question.strip(), answer) for question, answer in MATH_PATTERN.findall(text)]


@torch.no_grad()
def evaluate_math_mastery(
    model: ContinualCellTransformer,
    tokenizer: DynamicByteTokenizer,
    examples: list[tuple[str, str]],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, float | int | list[dict[str, str]]]:
    was_training = model.training
    model.eval()
    correct = 0
    mistakes: list[dict[str, str]] = []

    for question, expected in examples:
        prompt = f"<Q> {question} =\n<A>"
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(
            generated[0, len(prompt_ids) :].tolist()
        )
        for marker in ("<END>", "<Q>", "\n"):
            if marker in completion:
                completion = completion.split(marker, 1)[0]
        match = re.search(r"-?\d+", completion.strip())
        predicted = match.group(0) if match else ""
        if predicted == expected:
            correct += 1
        elif len(mistakes) < 10:
            mistakes.append(
                {
                    "question": question,
                    "expected": expected,
                    "predicted": predicted or "[none]",
                }
            )

    model.train(was_training)
    total = len(examples)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(1, total),
        "mistakes": mistakes,
    }


def build_optimizer(
    model: ContinualCellTransformer,
    config: TrainConfig,
) -> torch.optim.AdamW:
    groups = {
        key: []
        for key in (
            "embedding",
            "attention",
            "ffn",
            "cells",
            "halt",
            "other",
        )
    }
    for name, parameter in model.named_parameters():
        if name.startswith("token_embedding") or name.startswith("lm_head"):
            key = "embedding"
        elif ".attn." in name:
            key = "attention"
        elif ".ffn." in name:
            key = "ffn"
        elif ".cells." in name:
            key = "cells"
        elif "halt_" in name:
            key = "halt"
        else:
            key = "other"
        groups[key].append(parameter)

    rates = {
        "embedding": config.embedding_lr,
        "attention": config.attention_lr,
        "ffn": config.ffn_lr,
        "cells": config.cell_lr,
        "halt": config.halt_lr,
        "other": config.other_lr,
    }
    return torch.optim.AdamW(
        [
            {
                "params": parameters,
                "lr": rates[key],
                "weight_decay": (
                    0.0
                    if key in {"cells", "halt"}
                    else config.weight_decay
                ),
            }
            for key, parameters in groups.items()
            if parameters
        ]
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-file")
    parser.add_argument("--retention-file")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--mastery-stop", action="store_true")
    parser.add_argument("--mastery-accuracy", type=float, default=1.0)
    parser.add_argument("--mastery-patience", type=int, default=2)
    parser.add_argument("--mastery-max-new-tokens", type=int, default=8)

    parser.add_argument("--add-token", action="append", default=[])
    parser.add_argument("--auto-add-words", action="store_true")
    parser.add_argument("--allocate-cells", type=int, default=0)
    parser.add_argument("--grow-micro", type=int, default=0)

    parser.add_argument("--embedding-lr", type=float, default=1e-6)
    parser.add_argument("--attention-lr", type=float, default=2e-6)
    parser.add_argument("--ffn-lr", type=float, default=5e-6)
    parser.add_argument("--cell-lr", type=float, default=5e-4)
    parser.add_argument("--halt-lr", type=float, default=1e-4)
    parser.add_argument("--other-lr", type=float, default=1e-6)
    parser.add_argument("--consolidated-cell-scale", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--depth-penalty", type=float, default=0.01)
    parser.add_argument("--routing-loss-weight", type=float, default=0.05)
    parser.add_argument("--retention-replay-weight", type=float, default=0.0)
    parser.add_argument("--plastic-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--retention-output-penalty", type=float, default=0.0)

    parser.add_argument("--enable-cell-growth", action="store_true")
    parser.add_argument("--cell-growth-warmup", type=int, default=100)
    parser.add_argument("--cell-growth-patience", type=int, default=40)
    parser.add_argument("--cell-growth-count", type=int, default=2)
    parser.add_argument("--cell-growth-loss-floor", type=float, default=1.0)
    parser.add_argument("--cell-growth-coverage-floor", type=float, default=0.12)
    parser.add_argument("--max-cell-growth-events", type=int, default=4)
    parser.add_argument("--growth-cooldown", type=int, default=100)
    parser.add_argument("--growth-min-delta", type=float, default=1e-3)

    parser.add_argument("--enable-micro-growth", action="store_true")
    parser.add_argument("--micro-growth-patience", type=int, default=30)
    parser.add_argument("--micro-growth-count", type=int, default=1)
    parser.add_argument("--micro-growth-saturation", type=float, default=0.80)
    parser.add_argument("--micro-growth-min-score", type=float, default=0.05)
    parser.add_argument("--micro-growth-max-cells", type=int, default=4)
    parser.add_argument("--micro-growth-fallback-cells", type=int, default=1)
    parser.add_argument("--max-micro-growth-events", type=int, default=8)
    parser.add_argument("--consolidate-active-cells", action="store_true")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--halt-threshold", type=float, default=0.90)
    parser.add_argument("--halt-temperature", type=float, default=1.0)
    parser.add_argument("--halt-bias", type=float, default=-1.5)

    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--initial-cells", type=int, default=64)
    parser.add_argument("--threshold-temperature", type=float, default=0.10)
    parser.add_argument("--initial-threshold", type=float, default=0.25)
    parser.add_argument("--new-cell-threshold", type=float, default=0.20)
    parser.add_argument("--target-active-fraction", type=float, default=0.15)
    parser.add_argument("--max-micro-neurons", type=int, default=16)
    parser.add_argument("--initial-micro-neurons", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def zero_impact(
    model: ContinualCellTransformer,
    inputs: torch.Tensor,
    mutation,
    label: str,
):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        before = model(
            inputs,
            adaptive_inference=False,
        )["logits"].detach().clone()
        result = mutation()
        after = model(
            inputs,
            adaptive_inference=False,
        )["logits"].detach()
    drift = float((after - before).abs().max())
    model.train(was_training)
    print(f"{label}: {result}; max_logit_drift={drift:.3e}")
    if drift > 1e-5:
        raise RuntimeError(f"{label} was not zero-impact")
    return result


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
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_step = 0

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
        old_vocab = tokenizer.vocab_size
        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(train_text)
        tokenizer.add_tokens(tokens)

        model = ContinualCellTransformer(
            ModelConfig.from_dict(checkpoint["model_config"])
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if tokenizer.vocab_size > old_vocab:
            model.resize_vocabulary(tokenizer.vocab_size)
        previous_step = int(checkpoint.get("step", 0))
    else:
        tokenizer = DynamicByteTokenizer()
        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(train_text)
        tokenizer.add_tokens(tokens)

        model = ContinualCellTransformer(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                d_model=args.d_model,
                n_heads=args.heads,
                d_ff=args.d_ff,
                max_seq_len=args.seq_len,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                halt_threshold=args.halt_threshold,
                halt_temperature=args.halt_temperature,
                halt_bias=args.halt_bias,
                max_cells=args.max_cells,
                initial_active_cells=args.initial_cells,
                threshold_temperature=args.threshold_temperature,
                initial_threshold=args.initial_threshold,
                new_cell_threshold=args.new_cell_threshold,
                target_active_fraction=args.target_active_fraction,
                max_micro_neurons=args.max_micro_neurons,
                initial_micro_neurons=args.initial_micro_neurons,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
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

    mastery_examples = parse_math_examples(eval_text) if args.mastery_stop else []
    if args.mastery_stop and not mastery_examples:
        raise ValueError(
            "--mastery-stop requires an eval file containing <Q>/<A> math examples"
        )

    probe, _ = sample_batch(
        train_ids,
        min(args.batch_size, 16),
        args.seq_len,
        device,
    )
    if args.allocate_cells:
        seed = model(probe, adaptive_inference=False)["growth_seed"]
        zero_impact(
            model,
            probe,
            lambda: model.allocate_cells(args.allocate_cells, seed),
            "cell allocation",
        )
    if args.grow_micro:
        zero_impact(
            model,
            probe,
            lambda: model.grow_micro_neurons(args.grow_micro),
            "micro growth",
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
        halt_lr=args.halt_lr,
        other_lr=args.other_lr,
        consolidated_cell_scale=args.consolidated_cell_scale,
        depth_penalty=args.depth_penalty,
        routing_loss_weight=args.routing_loss_weight,
        retention_replay_weight=args.retention_replay_weight,
        plastic_sparsity_weight=args.plastic_sparsity_weight,
        retention_output_penalty=args.retention_output_penalty,
        enable_cell_growth=args.enable_cell_growth,
        cell_growth_warmup=args.cell_growth_warmup,
        cell_growth_patience=args.cell_growth_patience,
        cell_growth_count=args.cell_growth_count,
        cell_growth_loss_floor=args.cell_growth_loss_floor,
        cell_growth_coverage_floor=args.cell_growth_coverage_floor,
        max_cell_growth_events=args.max_cell_growth_events,
        growth_cooldown=args.growth_cooldown,
        enable_micro_growth=args.enable_micro_growth,
        micro_growth_patience=args.micro_growth_patience,
        micro_growth_count=args.micro_growth_count,
        micro_growth_saturation=args.micro_growth_saturation,
        max_micro_growth_events=args.max_micro_growth_events,
        seed=args.seed,
    )
    optimizer = build_optimizer(model, train_config)

    initial_eval = evaluate_loss(
        model,
        eval_ids,
        args.batch_size,
        args.seq_len,
        device,
        args.eval_batches,
    )
    retention_before = None
    if retention_ids is not None:
        retention_before = evaluate_loss(
            model,
            retention_ids,
            args.batch_size,
            args.seq_len,
            device,
            args.eval_batches,
        )
    print(f"initial eval={initial_eval:.4f}")
    if retention_before is not None:
        print(f"old-task before={retention_before:.4f}")

    best_eval = initial_eval
    stale_evals = 0
    mastery_streak = 0
    plateau_steps = 0
    plateau_best = float("inf")
    cell_events = 0
    micro_events = 0
    loss_ema = None
    last_growth = -10**9
    completed = 0
    model.train()

    for local_step in range(1, args.steps + 1):
        completed = local_step
        step = previous_step + local_step
        x, y = sample_batch(
            train_ids,
            args.batch_size,
            args.seq_len,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        result = model(x, labels=y, adaptive_inference=False)
        total_loss = (
            result["loss"]
            + args.depth_penalty * result["expected_depth"]
            + args.routing_loss_weight * result["routing_loss"]
            + args.plastic_sparsity_weight * result["plastic_activity"]
        )

        if retention_ids is not None and args.retention_replay_weight > 0:
            old_x, old_y = sample_batch(
                retention_ids,
                args.batch_size,
                args.seq_len,
                device,
            )
            old_result = model(
                old_x,
                labels=old_y,
                adaptive_inference=False,
            )
            total_loss = (
                total_loss
                + args.retention_replay_weight * old_result["loss"]
                + args.retention_output_penalty
                * old_result["plastic_output_rms"].square()
            )

        total_loss.backward()
        model.update_growth_signals()
        model.mask_cell_gradients(args.consolidated_cell_scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.advance_maturity()

        current_loss = float(result["loss"].detach())
        loss_ema = (
            current_loss
            if loss_ema is None
            else 0.98 * loss_ema + 0.02 * current_loss
        )
        if loss_ema < plateau_best - args.growth_min_delta:
            plateau_best = loss_ema
            plateau_steps = 0
        else:
            plateau_steps += 1

        cell_growth_ready = (
            args.enable_cell_growth
            and cell_events < args.max_cell_growth_events
            and local_step >= args.cell_growth_warmup
            and local_step - last_growth >= args.growth_cooldown
        )
        if (
            cell_growth_ready
            and plateau_steps >= args.cell_growth_patience
            and loss_ema > args.cell_growth_loss_floor
            and float(result["active_fraction"])
            >= args.cell_growth_coverage_floor
        ):
            zero_impact(
                model,
                x,
                lambda: model.allocate_cells(
                    args.cell_growth_count,
                    result["growth_seed"],
                ),
                f"cell growth {cell_events + 1}",
            )
            cell_events += 1
            last_growth = local_step
            plateau_steps = 0
            plateau_best = loss_ema

        if (
            args.enable_micro_growth
            and micro_events < args.max_micro_growth_events
            and plateau_steps >= args.micro_growth_patience
        ):
            growth_result = zero_impact(
                model,
                x,
                lambda: model.intelligent_capacity_growth(
                    micro_count=args.micro_growth_count,
                    outer_cell_count=args.micro_growth_fallback_cells,
                    seed=result["growth_seed"],
                    max_candidate_cells=args.micro_growth_max_cells,
                    minimum_score=args.micro_growth_min_score,
                    minimum_saturation=args.micro_growth_saturation,
                ),
                f"intelligent growth {micro_events + 1}",
            )
            if growth_result.get("grown"):
                micro_events += 1
            plateau_steps = 0
            plateau_best = loss_ema

        if local_step == 1 or local_step % args.log_interval == 0:
            print(
                f"step={step} loss={current_loss:.4f} "
                f"total={float(total_loss.detach()):.4f} "
                f"depth={float(result['expected_depth'].detach()):.2f} "
                f"route_coverage={float(result['route_coverage'].detach()):.3f} "
                f"active_fraction={float(result['active_fraction'].detach()):.3f} "
                f"mean_active_cells={float(result['mean_active_cells'].detach()):.2f} "
                f"effective_cells={float(result['effective_cell_fraction'].detach()):.3f} "
                f"routing_loss={float(result['routing_loss'].detach()):.4f} "
                f"micro_util={float(result['micro_utilization'].detach()):.3f} "
                f"micro_capacity={float(result['micro_capacity_fraction'].detach()):.3f} "
                f"pool={model.pool_summary()} "
                f"growth_candidates={model.growth_diagnostics(4)}"
            )

        if local_step % args.eval_interval == 0 or local_step == args.steps:
            eval_loss = evaluate_loss(
                model,
                eval_ids,
                args.batch_size,
                args.seq_len,
                device,
                args.eval_batches,
            )
            print(f"eval step={step} loss={eval_loss:.4f}")

            if eval_loss < best_eval - args.early_stop_min_delta:
                best_eval = eval_loss
                stale_evals = 0
            else:
                stale_evals += 1

            if args.mastery_stop:
                mastery = evaluate_math_mastery(
                    model,
                    tokenizer,
                    mastery_examples,
                    device,
                    args.mastery_max_new_tokens,
                )
                accuracy = float(mastery["accuracy"])
                mastery_streak = (
                    mastery_streak + 1
                    if accuracy >= args.mastery_accuracy
                    else 0
                )
                print(
                    f"mastery={mastery['correct']}/{mastery['total']} "
                    f"accuracy={accuracy:.4f} "
                    f"streak={mastery_streak}/{args.mastery_patience} "
                    f"mistakes={mastery['mistakes']}"
                )
                if mastery_streak >= args.mastery_patience:
                    print(f"mastery reached at step={step}")
                    break

            if (
                args.early_stop_patience
                and stale_evals >= args.early_stop_patience
            ):
                print(f"loss early stop at step={step}")
                break

    if args.consolidate_active_cells:
        model.consolidate_active_cells()

    retention_after = None
    if retention_ids is not None:
        retention_after = evaluate_loss(
            model,
            retention_ids,
            args.batch_size,
            args.seq_len,
            device,
            args.eval_batches,
        )
        print(
            f"old-task after={retention_after:.4f}; "
            f"delta={retention_after - retention_before:+.4f}"
        )

    payload = {
        "architecture_version": ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": previous_step + completed,
        "train_args": vars(args),
    }
    torch.save(payload, output_dir / "checkpoint.pt")
    tokenizer.save(output_dir / "tokenizer.json")
    summary = {
        "architecture_version": ARCHITECTURE_VERSION,
        "step": previous_step + completed,
        "best_eval": best_eval,
        "retention_before": retention_before,
        "retention_after": retention_after,
        "pool": model.pool_summary(),
        "cell_growth_events": cell_events,
        "micro_growth_events": micro_events,
        "growth_candidates": model.growth_diagnostics(8),
        "mastery_streak": mastery_streak,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("saved", output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
