from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable


ROOT = Path("data")


OPERATIONS: dict[str, tuple[str, Callable[[int, int], int]]] = {
    "addition": ("+", lambda a, b: a + b),
    "multiplication": ("*", lambda a, b: a * b),
}


def record(question: str, answer: int) -> str:
    return (
        f"<Q> {question} =\n"
        f"<A> {answer}\n"
        "<END>\n"
    )


def train_forms(a: int, operator: str, b: int) -> list[str]:
    """Surface forms shown during training.

    The arithmetic fact is unchanged, but internal whitespace varies. Test
    formats are intentionally disjoint from these strings.
    """
    return [
        f"{a} {operator} {b}",
        f"{a}{operator}{b}",
        f"{a}  {operator} {b}",
        f"{a} {operator}  {b}",
    ]


def unseen_format_forms(a: int, operator: str, b: int) -> list[str]:
    """Whitespace layouts never emitted by ``train_forms``."""
    return [
        f"{a}   {operator}   {b}",
        f"{a}{operator} {b}",
        f"{a} {operator}{b}",
        f"{a}\t{operator}\t{b}",
    ]


def unseen_notation_forms(a: int, operator: str, b: int) -> list[str]:
    """Equivalent notation beyond whitespace-only perturbations."""
    return [
        f"({a}) {operator} ({b})",
        f"{a:02d} {operator} {b:02d}",
    ]


def write_records(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows), encoding="utf-8")


def build_operation(
    name: str,
    heldout_digit: int,
    repeats: int,
    seed: int,
) -> list[Path]:
    operator, solve = OPERATIONS[name]
    rng = random.Random(seed + (0 if name == "addition" else 1))

    seen_pairs = [
        (a, b)
        for a in range(10)
        for b in range(10)
        if a != heldout_digit and b != heldout_digit
    ]
    heldout_pairs = [
        (a, b)
        for a in range(10)
        for b in range(10)
        if a == heldout_digit or b == heldout_digit
    ]

    # Each seen fact appears in four distinct training formats. Repeating full
    # shuffled epochs keeps the dataset size comparable to the original toy
    # benchmark without repeating one exact byte string 200 times.
    base_train = [
        record(form, solve(a, b))
        for a, b in seen_pairs
        for form in train_forms(a, operator, b)
    ]
    train_rows: list[str] = []
    for _ in range(repeats):
        epoch = base_train.copy()
        rng.shuffle(epoch)
        train_rows.extend(epoch)

    seen_canonical = [
        record(f"{a} {operator} {b}", solve(a, b))
        for a, b in seen_pairs
    ]
    seen_format = [
        record(form, solve(a, b))
        for a, b in seen_pairs
        for form in unseen_format_forms(a, operator, b)
    ]
    heldout_canonical = [
        record(f"{a} {operator} {b}", solve(a, b))
        for a, b in heldout_pairs
    ]
    heldout_format = [
        record(form, solve(a, b))
        for a, b in heldout_pairs
        for form in unseen_format_forms(a, operator, b)
    ]
    notation = [
        record(form, solve(a, b))
        for a, b in seen_pairs
        for form in unseen_notation_forms(a, operator, b)
    ]
    two_digit = [
        record(f"{a} {operator} {b}", solve(a, b))
        for a in range(10, 20)
        for b in range(10, 20)
    ]

    outputs = {
        f"{name}_generalization_train.txt": train_rows,
        f"{name}_seen_canonical_eval.txt": seen_canonical,
        f"{name}_seen_format_eval.txt": seen_format,
        f"{name}_heldout_digit{heldout_digit}_eval.txt": heldout_canonical,
        f"{name}_heldout_digit{heldout_digit}_format_eval.txt": heldout_format,
        f"{name}_notation_eval.txt": notation,
        f"{name}_two_digit_eval.txt": two_digit,
    }

    paths: list[Path] = []
    for filename, rows in outputs.items():
        path = ROOT / filename
        write_records(path, rows)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-digit", type=int, default=7)
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Full shuffled repeats of the four-format training set.",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not 0 <= args.heldout_digit <= 9:
        raise ValueError("--heldout-digit must be between 0 and 9")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    ROOT.mkdir(exist_ok=True)
    created: list[Path] = []
    for name in OPERATIONS:
        created.extend(
            build_operation(
                name=name,
                heldout_digit=args.heldout_digit,
                repeats=args.repeats,
                seed=args.seed,
            )
        )

    print(
        "Generalization suite created with "
        f"operand digit {args.heldout_digit} excluded from training."
    )
    for path in sorted(created):
        text = path.read_text(encoding="utf-8")
        count = text.count("<Q>")
        print(f"- {path}: {count} examples, {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
