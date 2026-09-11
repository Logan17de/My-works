from __future__ import annotations

import argparse
import random
from pathlib import Path


ROOT = Path("data")


def column_procedure(a: int, b: int) -> str:
    """Return a compact, fully checkable two-column addition trace."""
    a_tens, a_ones = divmod(a, 10)
    b_tens, b_ones = divmod(b, 10)

    ones_total = a_ones + b_ones
    ones_digit = ones_total % 10
    carry_ones = ones_total // 10

    tens_total = a_tens + b_tens + carry_ones
    tens_digit = tens_total % 10
    carry_tens = tens_total // 10

    return (
        f"O{a_ones}+{b_ones}={ones_total},D{ones_digit},C{carry_ones}"
        f"|T{a_tens}+{b_tens}+{carry_ones}={tens_total},D{tens_digit},C{carry_tens}"
        f"|H{carry_tens}|A{a + b}"
    )


def record(question: str, procedure: str) -> str:
    return (
        f"<Q> {question} =\n"
        f"<P> {procedure}\n"
        "<END>\n"
    )


def train_forms(a: int, b: int) -> tuple[str, ...]:
    return (
        f"{a} + {b}",
        f"{a}+{b}",
    )


def unseen_format_forms(a: int, b: int) -> tuple[str, ...]:
    return (
        f"{a}  +  {b}",
        f"{a}+ {b}",
        f"{a} +{b}",
        f"{a}\t+\t{b}",
    )


def ordered_pairs(groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for low, high in groups:
        rows.append((low, high))
        if low != high:
            rows.append((high, low))
    return rows


def has_ones_carry(a: int, b: int) -> bool:
    return (a % 10) + (b % 10) >= 10


def write_records(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--seen-eval-limit",
        type=int,
        default=500,
        help="Maximum number of trained ordered pairs in the seen evaluation split.",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if args.train_fraction + args.validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must be less than 1")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    ROOT.mkdir(exist_ok=True)
    rng = random.Random(args.seed)

    # Split unordered pairs so a held-out fact cannot leak through commutativity.
    groups = [
        (a, b)
        for a in range(10, 100)
        for b in range(a, 100)
    ]
    rng.shuffle(groups)

    train_end = int(len(groups) * args.train_fraction)
    validation_end = train_end + int(len(groups) * args.validation_fraction)
    train_groups = groups[:train_end]
    validation_groups = groups[train_end:validation_end]
    test_groups = groups[validation_end:]

    train_pairs = ordered_pairs(train_groups)
    validation_pairs = ordered_pairs(validation_groups)
    test_pairs = ordered_pairs(test_groups)

    base_train = [
        record(form, column_procedure(a, b))
        for a, b in train_pairs
        for form in train_forms(a, b)
    ]
    train_rows: list[str] = []
    for _ in range(args.repeats):
        epoch = base_train.copy()
        rng.shuffle(epoch)
        train_rows.extend(epoch)

    seen_pairs = train_pairs.copy()
    rng.shuffle(seen_pairs)
    seen_pairs = seen_pairs[: min(args.seen_eval_limit, len(seen_pairs))]

    seen_rows = [
        record(f"{a} + {b}", column_procedure(a, b))
        for a, b in seen_pairs
    ]
    validation_rows = [
        record(f"{a} + {b}", column_procedure(a, b))
        for a, b in validation_pairs
    ]
    test_rows = [
        record(f"{a} + {b}", column_procedure(a, b))
        for a, b in test_pairs
    ]
    test_format_rows = [
        record(form, column_procedure(a, b))
        for a, b in test_pairs
        for form in unseen_format_forms(a, b)
    ]
    test_carry_rows = [
        record(f"{a} + {b}", column_procedure(a, b))
        for a, b in test_pairs
        if has_ones_carry(a, b)
    ]
    test_no_carry_rows = [
        record(f"{a} + {b}", column_procedure(a, b))
        for a, b in test_pairs
        if not has_ones_carry(a, b)
    ]

    outputs = {
        "column_addition_train.txt": train_rows,
        "column_addition_seen_eval.txt": seen_rows,
        "column_addition_validation_eval.txt": validation_rows,
        "column_addition_test_eval.txt": test_rows,
        "column_addition_test_format_eval.txt": test_format_rows,
        "column_addition_test_carry_eval.txt": test_carry_rows,
        "column_addition_test_no_carry_eval.txt": test_no_carry_rows,
    }

    for filename, rows in outputs.items():
        write_records(ROOT / filename, rows)

    print("Created procedural column-addition benchmark.")
    print(
        f"unordered groups: train={len(train_groups)} "
        f"validation={len(validation_groups)} test={len(test_groups)}"
    )
    print(
        f"ordered pairs: train={len(train_pairs)} "
        f"validation={len(validation_pairs)} test={len(test_pairs)}"
    )
    for filename, rows in outputs.items():
        path = ROOT / filename
        print(f"- {path}: {len(rows)} examples, {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
