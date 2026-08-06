from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ModelConfig, TrainConfig
from mastery_eval import evaluate_math_mastery
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer
from train import arguments, build_optimizer, load_checkpoint, zero_impact


OBJECTIVE_VERSION = "math_answer_only_v1"
MATH_RECORD_PATTERN = re.compile(
    r"<Q>\s*([^\n]+?)\s*=\s*\n<A>\s*(-?\d+)\s*\n<END>",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class MathRecord:
    question: str
    answer: str


@dataclass(frozen=True)
class EncodedMathRecord:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    loss_weights: tuple[float, ...]


def parse_math_records(text: str) -> list[MathRecord]:
    records = [
        MathRecord(question=question.strip(), answer=answer.strip())
        for question, answer in MATH_RECORD_PATTERN.findall(text)
    ]
    if not records:
        raise ValueError(
            "No math records found. Expected '<Q> ... =\\n<A> answer\\n<END>'."
        )
    return records


def encode_math_records(
    records: list[MathRecord],
    tokenizer: DynamicByteTokenizer,
    max_seq_len: int,
) -> list[EncodedMathRecord]:
    encoded: list[EncodedMathRecord] = []

    for record in records:
        prompt = f"<Q> {record.question} =\n<A> "
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        answer_ids = tokenizer.encode(record.answer)
        control_ids = tokenizer.encode("\n<END>\n")

        full_ids = (
            prompt_ids
            + answer_ids
            + control_ids
            + [tokenizer.eos_token_id]
        )
        input_ids = full_ids[:-1]
        next_ids = full_ids[1:]

        if len(input_ids) > max_seq_len:
            raise ValueError(
                f"Math example requires {len(input_ids)} tokens but "
                f"max_seq_len={max_seq_len}. Increase --seq-len."
            )

        # The first answer token is predicted from the final prompt token.
        answer_start = len(prompt_ids) - 1
        answer_end = answer_start + len(answer_ids)
        control_end = answer_end + len(control_ids)

        labels = [tokenizer.pad_token_id] * len(next_ids)
        weights = [0.0] * len(next_ids)

        # Give the arithmetic answer the strongest signal.
        for index in range(answer_start, answer_end):
            labels[index] = next_ids[index]
            weights[index] = 4.0

        # Teach a clean terminator, but do not let formatting dominate.
        for index in range(answer_end, control_end):
            labels[index] = next_ids[index]
            weights[index] = 0.25

        # Explicit per-example EOS prevents the model from continuing into
        # another question template.
        eos_index = control_end
        labels[eos_index] = next_ids[eos_index]
        weights[eos_index] = 1.0

        encoded.append(
            EncodedMathRecord(
                input_ids=tuple(input_ids),
                labels=tuple(labels),
                loss_weights=tuple(weights),
            )
        )

    return encoded


def math_batch(
    records: list[EncodedMathRecord],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("Cannot batch an empty math dataset")

    selected = torch.randint(0, len(records), (batch_size,)).tolist()
    batch_records = [records[index] for index in selected]
    length = max(len(record.input_ids) for record in batch_records)

    inputs = torch.full(
        (batch_size, length),
        pad_token_id,
        dtype=torch.long,
    )
    labels = torch.full(
        (batch_size, length),
        pad_token_id,
        dtype=torch.long,
    )
    weights = torch.zeros((batch_size, length), dtype=torch.float32)

    for row, record in enumerate(batch_records):
        size = len(record.input_ids)
        inputs[row, :size] = torch.tensor(record.input_ids)
        labels[row, :size] = torch.tensor(record.labels)
        weights[row, :size] = torch.tensor(record.loss_weights)

    return inputs.to(device), labels.to(device), weights.to(device)


def weighted_answer_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    token_loss = F.cross_entropy(
        logits.flatten(0, 1),
        labels.flatten(),
        ignore_index=pad_token_id,
        reduction="none",
    ).view_as(labels)
    denominator = weights.sum().clamp_min(1.0)
    return (token_loss * weights).sum() / denominator


@torch.no_grad()
def evaluate_math_loss(
    model: ContinualCellTransformer,
    records: list[EncodedMathRecord],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_weight = 0.0

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        length = max(len(record.input_ids) for record in chunk)

        inputs = torch.full(
            (len(chunk), length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        labels = torch.full_like(inputs, pad_token_id)
        weights = torch.zeros(
            (len(chunk), length),
            dtype=torch.float32,
            device=device,
        )

        for row, record in enumerate(chunk):
            size = len(record.input_ids)
            inputs[row, :size] = torch.tensor(
                record.input_ids,
                dtype=torch.long,
                device=device,
            )
            labels[row, :size] = torch.tensor(
                record.labels,
                dtype=torch.long,
                device=device,
            )
            weights[row, :size] = torch.tensor(
                record.loss_weights,
                dtype=torch.float32,
                device=device,
            )

        result = model(inputs, adaptive_inference=False)
        token_loss = F.cross_entropy(
            result["logits"].flatten(0, 1),
            labels.flatten(),
            ignore_index=pad_token_id,
            reduction="none",
        ).view_as(labels)
        total_loss += float((token_loss * weights).sum())
        total_weight += float(weights.sum())

    model.train(was_training)
    return total_loss / max(1.0, total_weight)


def make_train_config(args) -> TrainConfig:
    return TrainConfig(
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


def build_model_and_tokenizer(args, train_text: str):
    previous_step = 0

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        objective = checkpoint.get("training_objective")
        if objective != OBJECTIVE_VERSION:
            raise ValueError(
                f"Checkpoint objective is {objective!r}, expected "
                f"{OBJECTIVE_VERSION!r}. Retrain addition with train_math.py."
            )

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
        return model, tokenizer, previous_step

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
    return model, tokenizer, previous_step


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
    retention_text = (
        Path(args.retention_file).read_text(encoding="utf-8")
        if args.retention_file
        else None
    )

    train_records_raw = parse_math_records(train_text)
    eval_records_raw = parse_math_records(eval_text)
    retention_records_raw = (
        parse_math_records(retention_text)
        if retention_text is not None
        else None
    )

    model, tokenizer, previous_step = build_model_and_tokenizer(
        args,
        train_text,
    )
    model.to(device)

    train_records = encode_math_records(
        train_records_raw,
        tokenizer,
        model.config.max_seq_len,
    )
    eval_records = encode_math_records(
        eval_records_raw,
        tokenizer,
        model.config.max_seq_len,
    )
    retention_records = (
        encode_math_records(
            retention_records_raw,
            tokenizer,
            model.config.max_seq_len,
        )
        if retention_records_raw is not None
        else None
    )

    print(
        f"objective={OBJECTIVE_VERSION} "
        f"train_examples={len(train_records)} "
        f"eval_examples={len(eval_records)} "
        f"max_example_tokens={max(len(row.input_ids) for row in train_records)}"
    )

    probe, _, _ = math_batch(
        train_records,
        min(args.batch_size, 16),
        tokenizer.pad_token_id,
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

    train_config = make_train_config(args)
    optimizer = build_optimizer(model, train_config)

    initial_eval = evaluate_math_loss(
        model,
        eval_records,
        args.batch_size,
        tokenizer.pad_token_id,
        device,
    )
    retention_before = (
        evaluate_math_loss(
            model,
            retention_records,
            args.batch_size,
            tokenizer.pad_token_id,
            device,
        )
        if retention_records is not None
        else None
    )
    print(f"initial answer-only eval={initial_eval:.4f}")
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

    mastery_examples = [
        (record.question, record.answer)
        for record in eval_records_raw
    ]

    for local_step in range(1, args.steps + 1):
        completed = local_step
        step = previous_step + local_step

        x, labels, loss_weights = math_batch(
            train_records,
            args.batch_size,
            tokenizer.pad_token_id,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        result = model(x, adaptive_inference=False)
        task_loss = weighted_answer_loss(
            result["logits"],
            labels,
            loss_weights,
            tokenizer.pad_token_id,
        )
        total_loss = (
            task_loss
            + args.depth_penalty * result["expected_depth"]
            + args.routing_loss_weight * result["routing_loss"]
            + args.plastic_sparsity_weight * result["plastic_activity"]
        )

        if (
            retention_records is not None
            and args.retention_replay_weight > 0
        ):
            old_x, old_labels, old_weights = math_batch(
                retention_records,
                args.batch_size,
                tokenizer.pad_token_id,
                device,
            )
            old_result = model(old_x, adaptive_inference=False)
            old_task_loss = weighted_answer_loss(
                old_result["logits"],
                old_labels,
                old_weights,
                tokenizer.pad_token_id,
            )
            total_loss = (
                total_loss
                + args.retention_replay_weight * old_task_loss
                + args.retention_output_penalty
                * old_result["plastic_output_rms"].square()
            )

        total_loss.backward()
        model.update_growth_signals()
        model.mask_cell_gradients(args.consolidated_cell_scale)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()
        model.advance_maturity()

        current_loss = float(task_loss.detach())
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
                f"step={step} answer_loss={current_loss:.4f} "
                f"total={float(total_loss.detach()):.4f} "
                f"depth={float(result['expected_depth'].detach()):.2f} "
                f"active_fraction={float(result['active_fraction'].detach()):.3f} "
                f"mean_active_cells={float(result['mean_active_cells'].detach()):.2f} "
                f"effective_cells={float(result['effective_cell_fraction'].detach()):.3f} "
                f"micro_util={float(result['micro_utilization'].detach()):.3f} "
                f"micro_capacity={float(result['micro_capacity_fraction'].detach()):.3f} "
                f"pool={model.pool_summary()}"
            )

        if local_step % args.eval_interval == 0 or local_step == args.steps:
            eval_loss = evaluate_math_loss(
                model,
                eval_records,
                args.batch_size,
                tokenizer.pad_token_id,
                device,
            )
            print(f"eval step={step} answer-only loss={eval_loss:.4f}")

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
                    f"average_generated_depth="
                    f"{float(mastery['average_generated_depth']):.2f} "
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

    retention_after = (
        evaluate_math_loss(
            model,
            retention_records,
            args.batch_size,
            tokenizer.pad_token_id,
            device,
        )
        if retention_records is not None
        else None
    )
    if retention_after is not None:
        print(
            f"old-task after={retention_after:.4f}; "
            f"delta={retention_after - retention_before:+.4f}"
        )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture_version": ContinualCellTransformer.ARCHITECTURE_VERSION,
        "training_objective": OBJECTIVE_VERSION,
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
        "architecture_version": ContinualCellTransformer.ARCHITECTURE_VERSION,
        "training_objective": OBJECTIVE_VERSION,
        "step": previous_step + completed,
        "best_answer_only_eval": best_eval,
        "retention_before": retention_before,
        "retention_after": retention_after,
        "pool": model.pool_summary(),
        "cell_growth_events": cell_events,
        "micro_growth_events": micro_events,
        "growth_candidates": model.growth_diagnostics(8),
        "mastery_streak": mastery_streak,
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("saved", output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
