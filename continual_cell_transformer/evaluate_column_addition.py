from __future__ import annotations

import argparse
from pathlib import Path

import torch

from column_addition_eval import generate_procedure
from column_addition_objective import (
    OBJECTIVE_VERSION,
    PROCEDURE_PATTERN,
    parse_question,
    parse_records,
    procedure_for,
)
from config import ModelConfig
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


SPLITS = {
    "seen": "column_addition_seen_eval.txt",
    "validation_unseen": "column_addition_validation_eval.txt",
    "test_unseen": "column_addition_test_eval.txt",
    "test_unseen_format": "column_addition_test_format_eval.txt",
    "test_carry": "column_addition_test_carry_eval.txt",
    "test_no_carry": "column_addition_test_no_carry_eval.txt",
}
FIELDS = (
    "ones_total",
    "ones_digit",
    "ones_carry",
    "tens_total",
    "tens_digit",
    "tens_carry",
    "answer",
)


def load_checkpoint(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_split(
    model: ContinualCellTransformer,
    tokenizer: DynamicByteTokenizer,
    records,
    device: torch.device,
    max_new_tokens: int,
) -> dict:
    final_correct = 0
    procedure_correct = 0
    parsed = 0
    field_correct = {field: 0 for field in FIELDS}
    depth_sum = 0.0
    generated_tokens = 0
    mistakes: list[dict] = []

    for record in records:
        a, b = parse_question(record.question)
        expected_procedure = procedure_for(a, b)
        expected_match = PROCEDURE_PATTERN.fullmatch(expected_procedure)
        assert expected_match is not None
        expected_fields = expected_match.groupdict()

        generated = generate_procedure(
            model,
            tokenizer,
            record.question,
            device,
            max_new_tokens,
            adaptive_inference=True,
        )
        fields = generated["fields"]
        if fields is not None:
            parsed += 1
            for field in FIELDS:
                if fields[field] == expected_fields[field]:
                    field_correct[field] += 1
            if fields["answer"] == record.answer:
                final_correct += 1
        if generated["procedure"] == expected_procedure:
            procedure_correct += 1

        trace = generated["trace"]
        depth_sum += sum(float(row["used_depth"]) for row in trace)
        generated_tokens += len(trace)

        if (
            (fields is None or fields["answer"] != record.answer)
            and len(mistakes) < 5
        ):
            mistakes.append(
                {
                    "question": record.question,
                    "expected_answer": record.answer,
                    "predicted_answer": fields["answer"] if fields else "[none]",
                    "expected_procedure": expected_procedure,
                    "raw_completion": repr(generated["raw"]),
                }
            )

    total = len(records)
    return {
        "total": total,
        "parsed": parsed,
        "final_correct": final_correct,
        "procedure_correct": procedure_correct,
        "field_correct": field_correct,
        "average_depth": depth_sum / max(1, generated_tokens),
        "mistakes": mistakes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum examples per split; use 0 for every example.",
    )
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    objective = checkpoint.get("training_objective")
    if objective != OBJECTIVE_VERSION:
        raise ValueError(
            f"Checkpoint objective is {objective!r}; expected {OBJECTIVE_VERSION!r}"
        )

    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    model = ContinualCellTransformer(
        ModelConfig.from_dict(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    data_dir = Path(args.data_dir)
    results: dict[str, dict] = {}
    for split, filename in SPLITS.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        records = parse_records(path.read_text(encoding="utf-8"))
        if args.limit > 0:
            records = records[: args.limit]
        results[split] = evaluate_split(
            model,
            tokenizer,
            records,
            device,
            args.max_new_tokens,
        )

    print("Procedural column-addition report")
    print(f"checkpoint objective: {objective}")
    print()
    print(
        f"{'split':22} {'final':>10} {'procedure':>11} "
        f"{'parsed':>10} {'avg_depth':>10}"
    )
    print("-" * 70)
    for split, result in results.items():
        total = result["total"]
        final = 100.0 * result["final_correct"] / max(1, total)
        procedure = 100.0 * result["procedure_correct"] / max(1, total)
        parsed = 100.0 * result["parsed"] / max(1, total)
        print(
            f"{split:22} {final:9.2f}% {procedure:10.2f}% "
            f"{parsed:9.2f}% {result['average_depth']:10.3f}"
        )

    print("\nSub-step accuracy on the unseen test split:")
    test = results["test_unseen"]
    for field in FIELDS:
        accuracy = 100.0 * test["field_correct"][field] / max(1, test["total"])
        print(f"- {field:12}: {accuracy:6.2f}%")

    print("\nInterpretation:")
    seen = results["seen"]["final_correct"] / max(1, results["seen"]["total"])
    unseen = results["test_unseen"]["final_correct"] / max(
        1, results["test_unseen"]["total"]
    )
    procedure = results["test_unseen"]["procedure_correct"] / max(
        1, results["test_unseen"]["total"]
    )
    if seen >= 0.95 and unseen < 0.50:
        print("- Strong seen performance but weak pair transfer: memorization still dominates.")
    elif unseen >= 0.80 and procedure >= 0.70:
        print("- Strong unseen-pair and procedure transfer: evidence of reusable column addition.")
    else:
        print("- Partial unseen-pair transfer: some reusable structure, but not a reliable algorithm yet.")

    for split, result in results.items():
        if not result["mistakes"]:
            continue
        print(f"\nFirst mistakes: {split}")
        for row in result["mistakes"][:3]:
            print(row)


if __name__ == "__main__":
    main()
