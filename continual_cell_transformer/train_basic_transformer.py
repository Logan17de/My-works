from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from basic_transformer import BasicTransformer
from config import ModelConfig
from mastery_eval import evaluate_math_mastery
from math_objective_v2 import OBJECTIVE_VERSION, encode_math_records
from tokenizer import DynamicByteTokenizer
from train_math import (
    evaluate_math_loss,
    math_batch,
    parse_math_records,
    weighted_answer_loss,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a plain fixed-depth Transformer on the same arithmetic objective as V7."
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)

    parser.add_argument("--mastery-stop", action="store_true")
    parser.add_argument("--mastery-accuracy", type=float, default=1.0)
    parser.add_argument("--mastery-patience", type=int, default=2)
    parser.add_argument("--mastery-max-new-tokens", type=int, default=4)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def checkpoint_payload(
    model: BasicTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: DynamicByteTokenizer,
    args: argparse.Namespace,
    step: int,
    parameter_count: int,
    eval_loss: float,
) -> dict:
    return {
        "model_type": BasicTransformer.MODEL_TYPE,
        "architecture_version": BasicTransformer.ARCHITECTURE_VERSION,
        "training_objective": OBJECTIVE_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "num_layers": args.layers,
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "eval_loss": float(eval_loss),
        "train_args": vars(args),
        "parameter_count": parameter_count,
    }


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_text = Path(args.train_file).read_text(encoding="utf-8")
    eval_text = Path(args.eval_file).read_text(encoding="utf-8")
    train_raw = parse_math_records(train_text)
    eval_raw = parse_math_records(eval_text)

    tokenizer = DynamicByteTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        min_depth=1,
        max_depth=args.layers,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = BasicTransformer(config, num_layers=args.layers).to(device)

    train_records = encode_math_records(train_raw, tokenizer, args.seq_len)
    eval_records = encode_math_records(eval_raw, tokenizer, args.seq_len)
    mastery_examples = [(row.question, row.answer) for row in eval_raw]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_count = model.parameter_count()
    print(
        f"model=basic_transformer objective={OBJECTIVE_VERSION} "
        f"layers={args.layers} d_model={args.d_model} heads={args.heads} d_ff={args.d_ff} "
        f"parameters={parameter_count:,} device={device}"
    )
    print(
        f"train_examples={len(train_records)} eval_examples={len(eval_records)} "
        f"max_example_tokens={max(len(row.input_ids) for row in train_records)}"
    )

    initial_eval = evaluate_math_loss(
        model,
        eval_records,
        args.batch_size,
        tokenizer.pad_token_id,
        device,
    )
    print(f"initial answer-only eval={initial_eval:.4f}")

    best_eval = initial_eval
    best_step = 0
    final_eval = initial_eval
    stale_evals = 0
    mastery_streak = 0
    completed = 0

    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            tokenizer,
            args,
            step=0,
            parameter_count=parameter_count,
            eval_loss=initial_eval,
        ),
        output_dir / "best_checkpoint.pt",
    )
    print(f"best checkpoint step=0 eval={initial_eval:.4f}")

    model.train()

    for step in range(1, args.steps + 1):
        completed = step
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
        task_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_interval == 0:
            print(f"step={step} answer_loss={float(task_loss.detach()):.4f}")

        if step % args.eval_interval == 0 or step == args.steps:
            eval_loss = evaluate_math_loss(
                model,
                eval_records,
                args.batch_size,
                tokenizer.pad_token_id,
                device,
            )
            final_eval = eval_loss
            print(f"eval step={step} answer-only loss={eval_loss:.4f}")

            if eval_loss < best_eval - args.early_stop_min_delta:
                best_eval = eval_loss
                best_step = step
                stale_evals = 0
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        tokenizer,
                        args,
                        step=step,
                        parameter_count=parameter_count,
                        eval_loss=eval_loss,
                    ),
                    output_dir / "best_checkpoint.pt",
                )
                print(f"NEW BEST step={step} eval={eval_loss:.4f} -> best_checkpoint.pt")
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
                args.early_stop_patience > 0
                and stale_evals >= args.early_stop_patience
            ):
                print(f"loss early stop at step={step}")
                break

    payload = checkpoint_payload(
        model,
        optimizer,
        tokenizer,
        args,
        step=completed,
        parameter_count=parameter_count,
        eval_loss=final_eval,
    )
    torch.save(payload, output_dir / "checkpoint.pt")
    tokenizer.save(output_dir / "tokenizer.json")

    overfit_delta = float(final_eval - best_eval)
    overfit_percent = 100.0 * overfit_delta / max(abs(best_eval), 1e-12)
    summary = {
        "model_type": BasicTransformer.MODEL_TYPE,
        "architecture_version": BasicTransformer.ARCHITECTURE_VERSION,
        "training_objective": OBJECTIVE_VERSION,
        "step": completed,
        "best_step": best_step,
        "best_answer_only_eval": best_eval,
        "final_answer_only_eval": final_eval,
        "overfit_delta": overfit_delta,
        "overfit_percent": overfit_percent,
        "mastery_streak": mastery_streak,
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "parameter_count": parameter_count,
        "num_layers": args.layers,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("saved final", output_dir / "checkpoint.pt")
    print("saved best ", output_dir / "best_checkpoint.pt")
    print(
        f"best_eval={best_eval:.4f} at step={best_step}; "
        f"final_eval={final_eval:.4f}; "
        f"overfit_delta={overfit_delta:+.4f} ({overfit_percent:+.2f}%)"
    )


if __name__ == "__main__":
    main()
