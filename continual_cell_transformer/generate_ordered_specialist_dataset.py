from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path


PATTERNS = (("+", "+"), ("*", "*"), ("+", "*"), ("*", "+"))


def left_to_right(a: int, op1: str, b: int, op2: str, c: int) -> int:
    value = a + b if op1 == "+" else a * b
    return value + c if op2 == "+" else value * c


def standard_precedence(a: int, op1: str, b: int, op2: str, c: int) -> int:
    if op1 == "+" and op2 == "*":
        return a + b * c
    if op1 == "*" and op2 == "+":
        return a * b + c
    return left_to_right(a, op1, b, op2, c)


def build_pool(max_operand: int, pattern: tuple[str, str]):
    rows = []
    op1, op2 = pattern
    for a, b, c in itertools.product(range(max_operand + 1), repeat=3):
        answer = left_to_right(a, op1, b, op2, c)

        # For + then *, keep only examples where appearance-order execution
        # actually differs from normal precedence. That prevents the model from
        # succeeding while ignoring the intended specialist order.
        if (
            pattern == ("+", "*")
            and answer == standard_precedence(a, op1, b, op2, c)
        ):
            continue

        rows.append((f"{a} {op1} {b} {op2} {c}", answer))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/ordered_specialist_train.txt")
    parser.add_argument("--eval-file", default="data/ordered_specialist_eval.txt")
    parser.add_argument("--max-operand", type=int, default=19)
    parser.add_argument("--train-per-pattern", type=int, default=500)
    parser.add_argument("--eval-per-pattern", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_rows = []
    eval_rows = []

    for pattern in PATTERNS:
        pool = build_pool(args.max_operand, pattern)
        rng.shuffle(pool)
        needed = args.train_per_pattern + args.eval_per_pattern
        if len(pool) < needed:
            raise ValueError(
                f"Pattern {pattern} has only {len(pool)} examples; need {needed}"
            )
        train_rows.extend(pool[: args.train_per_pattern])
        eval_rows.extend(pool[args.train_per_pattern : needed])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    train_path = Path(args.train_file)
    eval_path = Path(args.eval_file)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    train_path.write_text(
        "".join(f"{expr} = {answer}\n" for expr, answer in train_rows),
        encoding="utf-8",
    )
    eval_path.write_text(
        "".join(f"{expr} = {answer}\n" for expr, answer in eval_rows),
        encoding="utf-8",
    )

    print(
        f"saved train={len(train_rows)} eval={len(eval_rows)} "
        f"patterns={PATTERNS} max_operand={args.max_operand} seed={args.seed}"
    )
    print("semantics: evaluate strictly left-to-right in operator appearance order")


if __name__ == "__main__":
    main()
