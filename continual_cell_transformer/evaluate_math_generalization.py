from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from config import ModelConfig
from mastery_eval import evaluate_math_mastery
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


MATH_PATTERN = re.compile(r"<Q>\s*([^\n]+?)\s*=\s*\n<A>\s*(-?\d+)")


def load_checkpoint(path: str | Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    version = int(checkpoint.get("architecture_version", 0))
    expected = ContinualCellTransformer.ARCHITECTURE_VERSION
    if version != expected:
        raise ValueError(
            f"Checkpoint V{version} is incompatible with V{expected}"
        )
    return checkpoint


def load_examples(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    examples = [
        (question.strip(), answer.strip())
        for question, answer in MATH_PATTERN.findall(text)
    ]
    if not examples:
        raise ValueError(f"No arithmetic examples found in {path}")
    return examples


def classify(results: dict[str, dict]) -> list[str]:
    notes: list[str] = []
    seen = float(results["seen_canonical"]["accuracy"])
    format_accuracy = float(results["seen_format"]["accuracy"])
    heldout = float(results["heldout_operand"]["accuracy"])
    heldout_format = float(results["heldout_operand_format"]["accuracy"])
    two_digit = float(results["two_digit"]["accuracy"])

    if seen >= 0.95 and heldout < 0.50:
        notes.append(
            "Seen facts are strong but held-out-operand transfer is weak: "
            "the behavior is dominated by memorization."
        )
    elif heldout >= 0.80:
        notes.append(
            "Held-out-operand accuracy is strong: this is evidence of "
            "reusable arithmetic structure rather than pure lookup."
        )
    else:
        notes.append(
            "Held-out-operand transfer is partial: the model shows some "
            "structure learning, but not a reliable arithmetic rule."
        )

    if seen - format_accuracy > 0.10:
        notes.append(
            "Accuracy drops substantially under unseen whitespace: the model "
            "is still sensitive to surface form."
        )
    else:
        notes.append(
            "Unseen whitespace causes little degradation on familiar facts."
        )

    if heldout - heldout_format > 0.10:
        notes.append(
            "Formatting and operand generalization do not combine reliably."
        )

    if two_digit < 0.50:
        notes.append(
            "Two-digit extrapolation is weak; single-digit success should not "
            "be described as general arithmetic learning."
        )
    elif two_digit >= 0.80:
        notes.append(
            "Two-digit extrapolation is strong evidence of a reusable rule."
        )
    else:
        notes.append("Two-digit extrapolation is partial.")

    return notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--operation",
        choices=("addition", "multiplication"),
        required=True,
    )
    parser.add_argument("--heldout-digit", type=int, default=7)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    args = parser.parse_args()

    if not 0 <= args.heldout_digit <= 9:
        raise ValueError("--heldout-digit must be between 0 and 9")

    data_dir = Path(args.data_dir)
    prefix = args.operation
    files = {
        "seen_canonical": data_dir / f"{prefix}_seen_canonical_eval.txt",
        "seen_format": data_dir / f"{prefix}_seen_format_eval.txt",
        "heldout_operand": (
            data_dir / f"{prefix}_heldout_digit{args.heldout_digit}_eval.txt"
        ),
        "heldout_operand_format": (
            data_dir
            / f"{prefix}_heldout_digit{args.heldout_digit}_format_eval.txt"
        ),
        "notation": data_dir / f"{prefix}_notation_eval.txt",
        "two_digit": data_dir / f"{prefix}_two_digit_eval.txt",
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing generalization datasets. Run "
            "generate_math_generalization_datasets.py first:\n- "
            + "\n- ".join(missing)
        )

    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    model = ContinualCellTransformer(
        ModelConfig.from_dict(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    results: dict[str, dict] = {}
    for name, path in files.items():
        examples = load_examples(path)
        results[name] = evaluate_math_mastery(
            model,
            tokenizer,
            examples,
            device,
            args.max_new_tokens,
        )

    print(f"Generalization report: {args.operation}")
    print(f"Held-out operand digit: {args.heldout_digit}")
    print()
    print(f"{'split':28} {'correct':>10} {'accuracy':>10} {'avg_depth':>10}")
    print("-" * 62)
    for name, result in results.items():
        correct = f"{result['correct']}/{result['total']}"
        accuracy = 100.0 * float(result["accuracy"])
        depth = float(result["average_generated_depth"])
        print(f"{name:28} {correct:>10} {accuracy:9.2f}% {depth:10.3f}")

    print("\nInterpretation:")
    for note in classify(results):
        print(f"- {note}")

    for name, result in results.items():
        if not result["mistakes"]:
            continue
        print(f"\nFirst mistakes: {name}")
        for row in result["mistakes"][:3]:
            print(
                {
                    "question": row["question"],
                    "expected": row["expected"],
                    "predicted": row["predicted"],
                    "raw_completion": row["raw_completion"],
                }
            )


if __name__ == "__main__":
    main()
