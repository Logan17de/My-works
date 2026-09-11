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
        description="Train one all-layer learner-placement Transformer variant."
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=250)
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
    eval_loss: float,
) -> dict:
    return {
        "model_type": LearnerVariantTransformer.MODEL_TYPE,
        "architecture_version": LearnerVariantTransformer.ARCHITECTURE_VERSION,
        "training_objective": OBJECTIVE_VERSION,
        "variant": args.variant,
        "learner_dim": args.learner_dim,
        "num_layers": args.layers,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": int(step),
        "eval_loss": float(eval_loss),
        "train_args": vars(args),
        "parameter_count": model.parameter_count(),
        "learner_parameter_count": model.learner_parameter_count(),
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
    model = LearnerVariantTransformer(
        config,
        num_layers=args.layers,
        variant=args.variant,
        learner_dim=args.learner_dim,
    ).to(device)

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

    print(
        f"model=learner_variant variant={args.variant} "
        f"objective={OBJECTIVE_VERSION} layers={args.layers} "
        f"d_model={args.d_model} heads={args.heads} d_ff={args.d_ff} "
        f"learner_dim={args.learner_dim} parameters={model.parameter_count():,} "
        f"learner_parameters={model.learner_parameter_count():,} device={device}"
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
    best_eval = initial_eval
    best_step = 0
    final_eval = initial_eval
    stale_evals = 0
    completed = 0

    torch.save(
        checkpoint_payload(model, optimizer, tokenizer, args, 0, initial_eval),
        output_dir / "best_checkpoint.pt",
    )
    print(f"initial held-out eval={initial_eval:.4f}")
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
            print(f"eval step={step} held-out answer-only loss={eval_loss:.4f}")

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
                        step,
                        eval_loss,
                    ),
                    output_dir / "best_checkpoint.pt",
                )
                print(
                    f"NEW BEST step={step} eval={eval_loss:.4f} "
                    "-> best_checkpoint.pt"
                )
            else:
                stale_evals += 1

            if (
                args.early_stop_patience > 0
                and stale_evals >= args.early_stop_patience
            ):
                print(f"loss early stop at step={step}")
                break

    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            tokenizer,
            args,
            completed,
            final_eval,
        ),
        output_dir / "checkpoint.pt",
    )
    tokenizer.save(output_dir / "tokenizer.json")

    overfit_delta = float(final_eval - best_eval)
    overfit_percent = 100.0 * overfit_delta / max(abs(best_eval), 1e-12)
    summary = {
        "model_type": LearnerVariantTransformer.MODEL_TYPE,
        "variant": args.variant,
        "training_objective": OBJECTIVE_VERSION,
        "step": completed,
        "best_step": best_step,
        "best_answer_only_eval": best_eval,
        "final_answer_only_eval": final_eval,
        "overfit_delta": overfit_delta,
        "overfit_percent": overfit_percent,
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

    print("saved final", output_dir / "checkpoint.pt")
    print("saved best ", output_dir / "best_checkpoint.pt")
    print(
        f"best_eval={best_eval:.4f} at step={best_step}; "
        f"final_eval={final_eval:.4f}; "
        f"overfit_delta={overfit_delta:+.4f} ({overfit_percent:+.2f}%)"
    )


if __name__ == "__main__":
    main()
