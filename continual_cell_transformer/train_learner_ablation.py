from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from config import ModelConfig
from learner_variant_transformer import LearnerVariantTransformer, VARIANTS
from math_objective_v2 import OBJECTIVE_VERSION, encode_math_records
from plain_math_compat import parse_math_records
from tokenizer import DynamicByteTokenizer
from train_math import evaluate_math_loss, math_batch, weighted_answer_loss


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train with learners enabled, but evaluate the same weights both "
            "with and without learner contributions at every eval interval."
        )
    )
    parser.add_argument("--variant", choices=VARIANTS, default="post_ffn_all")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--learner-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--learner-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def checkpoint_payload(
    model: LearnerVariantTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: DynamicByteTokenizer,
    args: argparse.Namespace,
    step: int,
    learner_on_loss: float,
    learner_off_loss: float,
) -> dict:
    gain = float(learner_off_loss - learner_on_loss)
    gain_percent = 100.0 * gain / max(abs(learner_off_loss), 1e-12)
    return {
        "model_type": LearnerVariantTransformer.MODEL_TYPE,
        "architecture_version": LearnerVariantTransformer.ARCHITECTURE_VERSION,
        "training_objective": OBJECTIVE_VERSION,
        "experiment": "learner_on_off_ablation",
        "variant": args.variant,
        "learner_dim": args.learner_dim,
        "num_layers": args.layers,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": int(step),
        "eval_loss_learner_on": float(learner_on_loss),
        "eval_loss_learner_off": float(learner_off_loss),
        "learner_gain": gain,
        "learner_gain_percent": gain_percent,
        "train_args": vars(args),
        "parameter_count": model.parameter_count(),
        "learner_parameter_count": model.learner_parameter_count(),
    }


@torch.no_grad()
def evaluate_on_off(
    model: LearnerVariantTransformer,
    eval_records,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Evaluate identical weights with learner residuals enabled and bypassed."""
    model.set_learners_enabled(True)
    learner_on = evaluate_math_loss(
        model,
        eval_records,
        batch_size,
        pad_token_id,
        device,
    )

    model.set_learners_enabled(False)
    learner_off = evaluate_math_loss(
        model,
        eval_records,
        batch_size,
        pad_token_id,
        device,
    )

    # Training always continues with the learner enabled.
    model.set_learners_enabled(True)

    gain = float(learner_off - learner_on)
    gain_percent = 100.0 * gain / max(abs(learner_off), 1e-12)
    return float(learner_on), float(learner_off), gain, gain_percent


def append_history(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


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
    model = LearnerVariantTransformer(
        config,
        num_layers=args.layers,
        variant=args.variant,
        learner_dim=args.learner_dim,
    ).to(device)
    model.set_learners_enabled(True)

    train_records = encode_math_records(train_raw, tokenizer, args.seq_len)
    eval_records = encode_math_records(eval_raw, tokenizer, args.seq_len)

    learner_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if ".learner." in name:
            learner_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": learner_parameters,
                "lr": args.learner_lr,
                "weight_decay": args.weight_decay,
            },
        ]
    )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "eval_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    print(
        f"experiment=learner_on_off_ablation variant={args.variant} "
        f"objective={OBJECTIVE_VERSION} layers={args.layers} "
        f"d_model={args.d_model} heads={args.heads} d_ff={args.d_ff} "
        f"learner_dim={args.learner_dim} parameters={model.parameter_count():,} "
        f"learner_parameters={model.learner_parameter_count():,} device={device}"
    )
    print(
        f"train_examples={len(train_records)} eval_examples={len(eval_records)} "
        f"max_example_tokens={max(len(row.input_ids) for row in train_records)}"
    )
    print(
        "learner_gain = loss_without_learner - loss_with_learner; "
        "positive means the learner helps"
    )

    on_loss, off_loss, gain, gain_percent = evaluate_on_off(
        model,
        eval_records,
        args.batch_size,
        tokenizer.pad_token_id,
        device,
    )
    print(
        f"eval step=0 learner_ON={on_loss:.4f} learner_OFF={off_loss:.4f} "
        f"gain={gain:+.4f} ({gain_percent:+.2f}%)"
    )
    append_history(
        history_path,
        {
            "step": 0,
            "learner_on_loss": on_loss,
            "learner_off_loss": off_loss,
            "learner_gain": gain,
            "learner_gain_percent": gain_percent,
        },
    )

    best_on_loss = on_loss
    best_on_step = 0
    best_on_paired_off_loss = off_loss
    best_off_loss = off_loss
    best_off_step = 0
    best_off_paired_on_loss = on_loss
    final_on_loss = on_loss
    final_off_loss = off_loss
    stale_evals = 0
    completed = 0

    model.set_learners_enabled(True)
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            tokenizer,
            args,
            0,
            on_loss,
            off_loss,
        ),
        output_dir / "best_checkpoint.pt",
    )

    model.train()
    for step in range(1, args.steps + 1):
        completed = step
        model.set_learners_enabled(True)
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
            print(f"step={step} train_loss_learner_ON={float(task_loss.detach()):.4f}")

        if step % args.eval_interval == 0 or step == args.steps:
            on_loss, off_loss, gain, gain_percent = evaluate_on_off(
                model,
                eval_records,
                args.batch_size,
                tokenizer.pad_token_id,
                device,
            )
            final_on_loss = on_loss
            final_off_loss = off_loss

            print(
                f"eval step={step} learner_ON={on_loss:.4f} "
                f"learner_OFF={off_loss:.4f} "
                f"gain={gain:+.4f} ({gain_percent:+.2f}%)"
            )
            append_history(
                history_path,
                {
                    "step": step,
                    "learner_on_loss": on_loss,
                    "learner_off_loss": off_loss,
                    "learner_gain": gain,
                    "learner_gain_percent": gain_percent,
                },
            )

            if on_loss < best_on_loss - args.early_stop_min_delta:
                best_on_loss = on_loss
                best_on_step = step
                best_on_paired_off_loss = off_loss
                stale_evals = 0
                model.set_learners_enabled(True)
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        tokenizer,
                        args,
                        step,
                        on_loss,
                        off_loss,
                    ),
                    output_dir / "best_checkpoint.pt",
                )
                print(
                    f"NEW BEST learner_ON step={step} loss={on_loss:.4f}; "
                    f"same weights learner_OFF={off_loss:.4f}"
                )
            else:
                stale_evals += 1

            if off_loss < best_off_loss - args.early_stop_min_delta:
                best_off_loss = off_loss
                best_off_step = step
                best_off_paired_on_loss = on_loss

            if (
                args.early_stop_patience > 0
                and stale_evals >= args.early_stop_patience
            ):
                print(f"loss early stop at step={step}")
                break

    model.set_learners_enabled(True)
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            tokenizer,
            args,
            completed,
            final_on_loss,
            final_off_loss,
        ),
        output_dir / "checkpoint.pt",
    )
    tokenizer.save(output_dir / "tokenizer.json")

    best_paired_gain = best_on_paired_off_loss - best_on_loss
    best_paired_gain_percent = (
        100.0
        * best_paired_gain
        / max(abs(best_on_paired_off_loss), 1e-12)
    )
    final_gain = final_off_loss - final_on_loss
    final_gain_percent = (
        100.0 * final_gain / max(abs(final_off_loss), 1e-12)
    )

    summary = {
        "experiment": "learner_on_off_ablation",
        "model_type": LearnerVariantTransformer.MODEL_TYPE,
        "variant": args.variant,
        "training_objective": OBJECTIVE_VERSION,
        "step": completed,
        "best_learner_on_step": best_on_step,
        "best_learner_on_loss": best_on_loss,
        "learner_off_loss_at_best_on_step": best_on_paired_off_loss,
        "learner_gain_at_best_on_step": best_paired_gain,
        "learner_gain_percent_at_best_on_step": best_paired_gain_percent,
        "best_learner_off_step": best_off_step,
        "best_learner_off_loss": best_off_loss,
        "learner_on_loss_at_best_off_step": best_off_paired_on_loss,
        "final_learner_on_loss": final_on_loss,
        "final_learner_off_loss": final_off_loss,
        "final_learner_gain": final_gain,
        "final_learner_gain_percent": final_gain_percent,
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "parameter_count": model.parameter_count(),
        "learner_parameter_count": model.learner_parameter_count(),
        "num_layers": args.layers,
        "learner_dim": args.learner_dim,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nAblation summary")
    print(
        f"best learner_ON: step={best_on_step} loss={best_on_loss:.4f}\n"
        f"same weights OFF:             loss={best_on_paired_off_loss:.4f}\n"
        f"learner gain:                 {best_paired_gain:+.4f} "
        f"({best_paired_gain_percent:+.2f}%)"
    )
    print(
        f"best learner_OFF: step={best_off_step} loss={best_off_loss:.4f} "
        f"(paired ON={best_off_paired_on_loss:.4f})"
    )
    print(
        f"final learner_ON={final_on_loss:.4f} "
        f"learner_OFF={final_off_loss:.4f} "
        f"gain={final_gain:+.4f} ({final_gain_percent:+.2f}%)"
    )
    print("saved", output_dir / "best_checkpoint.pt")
    print("saved", output_dir / "checkpoint.pt")
    print("saved", history_path)


if __name__ == "__main__":
    main()
