from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from basic_transformer import BasicTransformer
from config import ModelConfig
from evaluate_math_generalization import classify, load_examples
from mastery_eval import evaluate_math_mastery
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if checkpoint.get("model_type") != BasicTransformer.MODEL_TYPE:
        raise ValueError(
            f"Checkpoint model_type={checkpoint.get('model_type')!r}; "
            f"expected {BasicTransformer.MODEL_TYPE!r}"
        )
    version = int(checkpoint.get("architecture_version", 0))
    if version != BasicTransformer.ARCHITECTURE_VERSION:
        raise ValueError(
            f"Basic Transformer checkpoint V{version} is incompatible with "
            f"V{BasicTransformer.ARCHITECTURE_VERSION}"
        )
    return checkpoint


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
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not 0 <= args.heldout_digit <= 9:
        raise ValueError("--heldout-digit must be between 0 and 9")
    if args.limit_per_split < 0:
        raise ValueError("--limit-per-split must be non-negative")

    data_dir = Path(args.data_dir)
    prefix = args.operation
    files = {
        "seen_canonical": data_dir / f"{prefix}_seen_canonical_eval.txt",
        "seen_format": data_dir / f"{prefix}_seen_format_eval.txt",
        "heldout_operand": (
            data_dir / f"{prefix}_heldout_digit{args.heldout_digit}_eval.txt"
        ),
        "heldout_operand_format": (
            data_dir / f"{prefix}_heldout_digit{args.heldout_digit}_format_eval.txt"
        ),
        "notation": data_dir / f"{prefix}_notation_eval.txt",
        "two_digit": data_dir / f"{prefix}_two_digit_eval.txt",
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing generalization datasets:\n- " + "\n- ".join(missing))

    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    config = ModelConfig.from_dict(checkpoint["model_config"])
    layers = int(checkpoint["num_layers"])
    model = BasicTransformer(config, num_layers=layers)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(
        f"model=basic_transformer layers={layers} "
        f"parameters={int(checkpoint.get('parameter_count', model.parameter_count())):,} "
        f"device={device}",
        flush=True,
    )

    results: dict[str, dict] = {}
    total_start = time.perf_counter()
    for split_index, (name, path) in enumerate(files.items()):
        examples = load_examples(path)
        original_count = len(examples)
        if args.limit_per_split and len(examples) > args.limit_per_split:
            rng = random.Random(args.seed + split_index)
            examples = rng.sample(examples, args.limit_per_split)

        print(
            f"evaluating {name}: {len(examples)}/{original_count} examples...",
            flush=True,
        )
        split_start = time.perf_counter()
        results[name] = evaluate_math_mastery(
            model,
            tokenizer,
            examples,
            device,
            args.max_new_tokens,
        )
        elapsed = time.perf_counter() - split_start
        print(f"finished {name} in {elapsed:.1f}s", flush=True)

    print(f"\nGeneralization report: basic Transformer / {args.operation}")
    print(f"Held-out operand digit: {args.heldout_digit}")
    print(f"Layers: {layers}")
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

    print(f"\ntotal evaluation time={time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
