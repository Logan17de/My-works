from __future__ import annotations

from math_objective_v2 import (
    ANSWER_WEIGHT,
    EOS_WEIGHT,
    OBJECTIVE_VERSION,
    encode_math_records,
)
from tokenizer import DynamicByteTokenizer
from train_math import parse_math_records


SAMPLE = """<Q> 3 + 8 =
<A> 11
<END>

<Q> 0 + 2 =
<A> 2
<END>
"""


def supervised_text(tokenizer, record, weight):
    ids = [
        token_id
        for token_id, token_weight in zip(record.labels, record.loss_weights)
        if token_weight == weight
    ]
    return tokenizer.decode(ids, stop_at_eos=False)


def main() -> None:
    tokenizer = DynamicByteTokenizer()
    records = parse_math_records(SAMPLE)
    encoded = encode_math_records(records, tokenizer, max_seq_len=64)

    assert len(records) == 2
    assert records[0].question == "3 + 8"
    assert records[0].answer == "11"

    first = encoded[0]
    answer_target = supervised_text(tokenizer, first, ANSWER_WEIGHT)
    assert answer_target == "11", answer_target

    eos_targets = [
        token_id
        for token_id, weight in zip(first.labels, first.loss_weights)
        if weight == EOS_WEIGHT
    ]
    assert eos_targets == [tokenizer.eos_token_id], eos_targets

    supervised_ids = [
        token_id
        for token_id, weight in zip(first.labels, first.loss_weights)
        if weight > 0.0 and token_id != tokenizer.eos_token_id
    ]
    supervised_non_eos = tokenizer.decode(supervised_ids, stop_at_eos=False)
    assert "\n" not in supervised_non_eos
    assert "<END>" not in supervised_non_eos
    assert supervised_non_eos == "11"

    zero_weight_count = sum(weight == 0.0 for weight in first.loss_weights)
    assert zero_weight_count > 0
    assert all(
        label == tokenizer.pad_token_id
        for label, weight in zip(first.labels, first.loss_weights)
        if weight == 0.0
    )

    answer_weight_total = sum(
        weight for weight in first.loss_weights if weight == ANSWER_WEIGHT
    )
    eos_weight_total = sum(
        weight for weight in first.loss_weights if weight == EOS_WEIGHT
    )
    non_answer_fraction = eos_weight_total / (
        answer_weight_total + eos_weight_total
    )
    assert non_answer_fraction < 0.03, non_answer_fraction

    print("Math objective V2 check passed.")
    print("objective:", OBJECTIVE_VERSION)
    print("answer target:", repr(answer_target))
    print("newline supervised: False")
    print("<END> supervised: False")
    print("EOS target weight:", EOS_WEIGHT)
    print("non-answer loss fraction for '11':", f"{non_answer_fraction:.3%}")
    print("masked prompt positions:", zero_weight_count)


if __name__ == "__main__":
    main()
