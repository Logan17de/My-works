from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from config import ModelConfig
from mastery_eval import evaluate_math_mastery
from ordered_specialist_transformer import OrderedSpecialistTransformer
from plain_math_compat import parse_math_records
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def route_name(question: str) -> str:
    operators = [character for character in question if character in "+*"]
    mapping = {"+": "Add", "*": "Multiply"}
    return "->".join(mapping[operator] for operator in operators)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("model_type") != OrderedSpecialistTransformer.MODEL_TYPE:
        raise ValueError(
            f"Expected {OrderedSpecialistTransformer.MODEL_TYPE!r}, got "
            f"{checkpoint.get('model_type')!r}"
        )

    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = OrderedSpecialistTransformer(
        config,
        num_layers=int(checkpoint["num_layers"]),
        plus_token_id=int(checkpoint["plus_token_id"]),
        multiply_token_id=int(checkpoint["multiply_token_id"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    records = parse_math_records(Path(args.eval_file).read_text(encoding="utf-8"))
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in records:
        groups[route_name(row.question)].append((row.question, row.answer))

    total_correct = 0
    total_examples = 0
    weighted_depth = 0.0
    results = []

    for route in ("Add->Add", "Multiply->Multiply", "Add->Multiply", "Multiply->Add"):
        examples = groups.get(route, [])
        if not examples:
            continue
        result = evaluate_math_mastery(
            model,
            tokenizer,
            examples,
            device,
            args.max_new_tokens,
        )
        total_correct += int(result["correct"])
        total_examples += int(result["total"])
        weighted_depth += (
            float(result["average_generated_depth"]) * int(result["total"])
        )
        results.append((route, result))

    print(
        f"model={OrderedSpecialistTransformer.MODEL_TYPE} "
        f"checkpoint_step={int(checkpoint.get('step', -1))}"
    )
    print("semantics=left-to-right operator appearance order")
    print()
    print(f"{'route':22} {'correct':>10} {'accuracy':>10}")
    print("-" * 46)
    for route, result in results:
        print(
            f"{route:22} "
            f"{result['correct']:>4}/{result['total']:<5} "
            f"{100.0 * float(result['accuracy']):9.2f}%"
        )

    overall = total_correct / max(1, total_examples)
    print("-" * 46)
    print(
        f"{'OVERALL':22} {total_correct:>4}/{total_examples:<5} "
        f"{100.0 * overall:9.2f}%"
    )
    print(
        "average_generated_depth=",
        f"{weighted_depth / max(1, total_examples):.3f}",
    )

    mistakes_printed = 0
    for route, result in results:
        for mistake in result["mistakes"]:
            if mistakes_printed >= 10:
                break
            print(
                {
                    "route": route,
                    "question": mistake["question"],
                    "expected": mistake["expected"],
                    "predicted": mistake["predicted"],
                    "raw_completion": mistake["raw_completion"],
                }
            )
            mistakes_printed += 1
        if mistakes_printed >= 10:
            break


if __name__ == "__main__":
    main()
