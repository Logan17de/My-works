from __future__ import annotations

from tokenizer import DynamicByteTokenizer
from train_math import encode_math_records, parse_math_records


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
    assert supervised_text(tokenizer, first, 4.0) == "11"
    assert supervised_text(tokenizer, first, 0.25) == "\n<END>\n"

    eos_targets = [
        token_id
        for token_id, weight in zip(first.labels, first.loss_weights)
        if weight == 1.0
    ]
    assert eos_targets == [tokenizer.eos_token_id]

    zero_weight_count = sum(weight == 0.0 for weight in first.loss_weights)
    assert zero_weight_count > 0
    assert all(
        label == tokenizer.pad_token_id
        for label, weight in zip(first.labels, first.loss_weights)
        if weight == 0.0
    )

    print("Answer-only math objective check passed.")
    print("answer target:", repr(supervised_text(tokenizer, first, 4.0)))
    print("control target:", repr(supervised_text(tokenizer, first, 0.25)))
    print("EOS target id:", eos_targets[0])
    print("masked prompt positions:", zero_weight_count)


if __name__ == "__main__":
    main()
