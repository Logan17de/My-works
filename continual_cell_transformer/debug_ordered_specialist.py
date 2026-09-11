from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import torch

from config import ModelConfig
from ordered_specialist_transformer import OrderedAttentionLayer, OrderedSpecialistTransformer
from plain_math_compat import parse_math_records
from tokenizer import DynamicByteTokenizer


EXPR = re.compile(r"^\s*(\d+)\s*([+*])\s*(\d+)\s*([+*])\s*(\d+)\s*$")


def left_to_right(question: str) -> int:
    match = EXPR.fullmatch(question)
    if match is None:
        raise ValueError(f"Unexpected question format: {question!r}")
    a, op1, b, op2, c = match.groups()
    value = int(a) + int(b) if op1 == "+" else int(a) * int(b)
    return value + int(c) if op2 == "+" else value * int(c)


def route(question: str) -> str:
    return "".join(character for character in question if character in "+*")


def check_dataset(path: str) -> tuple[set[str], Counter]:
    records = parse_math_records(Path(path).read_text(encoding="utf-8"))
    questions = set()
    counts = Counter()
    for row in records:
        if row.question in questions:
            raise AssertionError(f"Duplicate question in {path}: {row.question}")
        questions.add(row.question)
        expected = left_to_right(row.question)
        if int(row.answer) != expected:
            raise AssertionError(
                f"Wrong target in {path}: {row.question} expected {expected}, got {row.answer}"
            )
        counts[route(row.question)] += 1
    print(path, "examples=", len(records), "routes=", dict(counts))
    return questions, counts


def main() -> None:
    train_questions, train_counts = check_dataset("data/ordered_specialist_train.txt")
    eval_questions, eval_counts = check_dataset("data/ordered_specialist_eval.txt")
    overlap = train_questions & eval_questions
    if overlap:
        raise AssertionError(f"Train/eval overlap: {list(overlap)[:5]}")

    if train_counts != Counter({"++": 200, "**": 200, "+*": 200, "*+": 200}):
        raise AssertionError(f"Unexpected train route balance: {train_counts}")
    if eval_counts != Counter({"++": 50, "**": 50, "+*": 50, "*+": 50}):
        raise AssertionError(f"Unexpected eval route balance: {eval_counts}")

    tokenizer = DynamicByteTokenizer()
    plus_id = tokenizer.encode("+")[0]
    multiply_id = tokenizer.encode("*")[0]

    examples = ["2 + 3 * 4", "2 * 3 + 4"]
    for question in examples:
        ids = tokenizer.encode(f"<Q> {question} =\n<A> ", add_bos=True)
        tensor = torch.tensor([ids], dtype=torch.long)
        masks = OrderedAttentionLayer._ordered_operator_masks(
            tensor,
            plus_id,
            multiply_id,
        )
        routed = []
        for plus_mask, multiply_mask in masks:
            if bool(plus_mask[0, -1]):
                routed.append("Add")
            elif bool(multiply_mask[0, -1]):
                routed.append("Multiply")
        print(question, "-> Shared -> " + " -> ".join(routed))

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        d_ff=512,
        max_seq_len=48,
        dropout=0.0,
        min_depth=1,
        max_depth=8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = OrderedSpecialistTransformer(
        config,
        num_layers=8,
        plus_token_id=plus_id,
        multiply_token_id=multiply_id,
    )
    probe = torch.tensor(
        [
            tokenizer.encode("<Q> 2 + 3 * 4 =\n<A> ", add_bos=True),
            tokenizer.encode("<Q> 2 * 3 + 4 =\n<A> ", add_bos=True),
        ],
        dtype=torch.long,
    )
    result = model(probe)
    if tuple(result["logits"].shape[:2]) != tuple(probe.shape):
        raise AssertionError("Unexpected logits shape")
    print(
        "model smoke passed; parameters=",
        f"{model.parameter_count():,}",
        "ffns=",
        model.ffn_parameter_counts(),
    )
    print("Ordered specialist integrity test passed.")


if __name__ == "__main__":
    main()
