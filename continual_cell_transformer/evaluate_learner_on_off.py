from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from learner_variant_transformer import LearnerVariantTransformer
from mastery_eval import evaluate_math_mastery
from math_objective_v2 import encode_math_records
from plain_math_compat import parse_math_records
from tokenizer import DynamicByteTokenizer
from train_math import evaluate_math_loss


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("model_type") != LearnerVariantTransformer.MODEL_TYPE:
        raise ValueError("Checkpoint is not a learner_variant_transformer")

    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = LearnerVariantTransformer(
        config,
        num_layers=int(checkpoint["num_layers"]),
        variant=str(checkpoint["variant"]),
        learner_dim=int(checkpoint["learner_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    raw_records = parse_math_records(
        Path(args.eval_file).read_text(encoding="utf-8")
    )
    examples = [(row.question, row.answer) for row in raw_records]
    encoded = encode_math_records(raw_records, tokenizer, config.max_seq_len)

    results = {}
    for label, enabled in (("ON", True), ("OFF", False)):
        model.set_learners_enabled(enabled)
        loss = evaluate_math_loss(
            model,
            encoded,
            args.batch_size,
            tokenizer.pad_token_id,
            device,
        )
        mastery = evaluate_math_mastery(
            model,
            tokenizer,
            examples,
            device,
            args.max_new_tokens,
        )
        results[label] = {
            "loss": float(loss),
            "correct": int(mastery["correct"]),
            "total": int(mastery["total"]),
            "accuracy": float(mastery["accuracy"]),
            "mistakes": mastery["mistakes"],
        }

    model.set_learners_enabled(True)

    gain = results["OFF"]["loss"] - results["ON"]["loss"]
    gain_percent = 100.0 * gain / max(abs(results["OFF"]["loss"]), 1e-12)
    accuracy_delta = 100.0 * (
        results["ON"]["accuracy"] - results["OFF"]["accuracy"]
    )

    print(
        f"variant={checkpoint['variant']} learner_dim={checkpoint['learner_dim']} "
        f"step={checkpoint.get('step', '?')}"
    )
    print()
    print(f"{'state':10} {'loss':>10} {'correct':>12} {'accuracy':>10}")
    print("-" * 48)
    for label in ("ON", "OFF"):
        row = results[label]
        print(
            f"learner_{label:<3} {row['loss']:10.4f} "
            f"{row['correct']:5d}/{row['total']:<5d} "
            f"{100.0 * row['accuracy']:9.2f}%"
        )

    print()
    print(
        f"learner loss gain={gain:+.4f} ({gain_percent:+.2f}%); "
        f"exact accuracy delta={accuracy_delta:+.2f} percentage points"
    )

    if results["OFF"]["mistakes"]:
        print("\nFirst mistakes with learner OFF:")
        for row in results["OFF"]["mistakes"][:5]:
            print(
                {
                    "question": row["question"],
                    "expected": row["expected"],
                    "predicted": row["predicted"],
                }
            )


if __name__ == "__main__":
    main()
