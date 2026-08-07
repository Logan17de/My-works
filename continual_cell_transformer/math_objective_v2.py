from __future__ import annotations

from tokenizer import DynamicByteTokenizer
from train_math import EncodedMathRecord, MathRecord


OBJECTIVE_VERSION = "math_answer_eos_v2"
ANSWER_WEIGHT = 1.0
EOS_WEIGHT = 0.01


def encode_math_records(
    records: list[MathRecord],
    tokenizer: DynamicByteTokenizer,
    max_seq_len: int,
) -> list[EncodedMathRecord]:
    """Encode complete math examples with no newline or template targets.

    Supervised targets are only:
    1. the answer digits;
    2. a tiny-weight EOS token used only to teach generation to stop.

    The prompt, newline, <END>, and all other formatting receive zero loss.
    """
    encoded: list[EncodedMathRecord] = []

    for record in records:
        prompt = f"<Q> {record.question} =\n<A> "
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        answer_ids = tokenizer.encode(record.answer)

        # No "\n<END>\n" is exposed as a prediction target.
        full_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        input_ids = full_ids[:-1]
        next_ids = full_ids[1:]

        if len(input_ids) > max_seq_len:
            raise ValueError(
                f"Math example requires {len(input_ids)} tokens but "
                f"max_seq_len={max_seq_len}. Increase --seq-len."
            )

        # The first answer token is predicted from the final prompt token.
        answer_start = len(prompt_ids) - 1
        answer_end = answer_start + len(answer_ids)
        eos_index = answer_end

        labels = [tokenizer.pad_token_id] * len(next_ids)
        weights = [0.0] * len(next_ids)

        for index in range(answer_start, answer_end):
            labels[index] = next_ids[index]
            weights[index] = ANSWER_WEIGHT

        # EOS is useful for clean autoregressive generation, but its weight is
        # tiny enough to contribute below 1% for a one-digit answer.
        labels[eos_index] = next_ids[eos_index]
        weights[eos_index] = EOS_WEIGHT

        encoded.append(
            EncodedMathRecord(
                input_ids=tuple(input_ids),
                labels=tuple(labels),
                loss_weights=tuple(weights),
            )
        )

    return encoded
