from __future__ import annotations

import re
from typing import Any

import torch

from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


NUMBER_PATTERN = re.compile(r"-?\d+")
STOP_MARKERS = ("<END>", "<Q>")


def _greedy_generate(
    model: ContinualCellTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    adaptive_inference: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Generate deterministically while retaining per-token depth diagnostics."""
    ids = input_ids
    trace: list[dict[str, Any]] = []

    for _ in range(max_new_tokens):
        result = model(
            ids[:, -model.config.max_seq_len :],
            adaptive_inference=adaptive_inference,
        )
        next_token = result["logits"][:, -1].argmax(dim=-1, keepdim=True)
        trace.append(
            {
                "token_id": int(next_token[0, 0]),
                "used_depth": int(result["used_depth"]),
                "expected_depth": float(result["expected_depth"]),
            }
        )
        ids = torch.cat((ids, next_token), dim=1)

        if torch.all(next_token == model.config.eos_token_id):
            break

    return ids, trace


def _extract_answer(raw_completion: str) -> tuple[str, str]:
    """Return the first integer and the cleaned text used to find it."""
    cleaned = raw_completion
    for marker in STOP_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]

    # Do not discard text merely because a newline appears before the answer.
    match = NUMBER_PATTERN.search(cleaned)
    return (match.group(0) if match else ""), cleaned


@torch.no_grad()
def evaluate_math_mastery(
    model: ContinualCellTransformer,
    tokenizer: DynamicByteTokenizer,
    examples: list[tuple[str, str]],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()

    correct = 0
    mistakes: list[dict[str, Any]] = []
    adaptive_depth_sum = 0.0
    generated_tokens = 0

    for question, expected in examples:
        # This exactly matches the dataset prefix: "<A> {answer}".
        prompt = f"<Q> {question} =\n<A> "
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        generated, trace = _greedy_generate(
            model,
            input_ids,
            max_new_tokens,
            adaptive_inference=True,
        )
        generated_ids = generated[0, len(prompt_ids) :].tolist()
        raw_completion = tokenizer.decode(generated_ids, stop_at_eos=False)
        predicted, cleaned = _extract_answer(raw_completion)

        adaptive_depth_sum += sum(float(row["used_depth"]) for row in trace)
        generated_tokens += len(trace)

        if predicted == expected:
            correct += 1
            continue

        if len(mistakes) < 10:
            # A full-depth retry separates decoding/prompt bugs from halting errors.
            full_generated, full_trace = _greedy_generate(
                model,
                input_ids,
                max_new_tokens,
                adaptive_inference=False,
            )
            full_ids = full_generated[0, len(prompt_ids) :].tolist()
            full_raw = tokenizer.decode(full_ids, stop_at_eos=False)
            full_predicted, _ = _extract_answer(full_raw)

            mistakes.append(
                {
                    "question": question,
                    "expected": expected,
                    "predicted": predicted or "[none]",
                    "raw_completion": repr(raw_completion),
                    "cleaned_completion": repr(cleaned),
                    "generated_token_ids": generated_ids,
                    "adaptive_trace": trace,
                    "full_depth_predicted": full_predicted or "[none]",
                    "full_depth_raw": repr(full_raw),
                    "full_depth_trace": full_trace,
                }
            )

    model.train(was_training)
    total = len(examples)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(1, total),
        "average_generated_depth": adaptive_depth_sum / max(1, generated_tokens),
        "mistakes": mistakes,
    }
