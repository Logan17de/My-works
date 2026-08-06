from __future__ import annotations

from typing import Any

import torch

from column_addition_objective import PROCEDURE_PATTERN, parse_question, procedure_for
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


@torch.no_grad()
def generate_procedure(
    model: ContinualCellTransformer,
    tokenizer: DynamicByteTokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int,
    adaptive_inference: bool = True,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(
        f"<Q> {question} =\n<P> ",
        add_bos=True,
    )
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids: list[int] = []
    trace: list[dict[str, float | int]] = []

    for _ in range(max_new_tokens):
        result = model(
            ids[:, -model.config.max_seq_len :],
            adaptive_inference=adaptive_inference,
        )
        next_token = result["logits"][:, -1].argmax(dim=-1, keepdim=True)
        token_id = int(next_token[0, 0])
        generated_ids.append(token_id)
        trace.append(
            {
                "used_depth": int(result["used_depth"]),
                "expected_depth": float(result["expected_depth"]),
            }
        )
        ids = torch.cat((ids, next_token), dim=1)
        if token_id == tokenizer.eos_token_id:
            break

    raw = tokenizer.decode(generated_ids, stop_at_eos=False)
    match = PROCEDURE_PATTERN.search(raw)
    return {
        "raw": raw,
        "procedure": match.group(0) if match else "",
        "fields": match.groupdict() if match else None,
        "trace": trace,
    }


@torch.no_grad()
def evaluate_mastery(
    model: ContinualCellTransformer,
    tokenizer: DynamicByteTokenizer,
    examples: list[tuple[str, str]],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    final_correct = 0
    procedure_correct = 0
    mistakes: list[dict[str, Any]] = []
    depth_sum = 0.0
    generated_tokens = 0

    for question, expected_answer in examples:
        a, b = parse_question(question)
        expected_procedure = procedure_for(a, b)
        generated = generate_procedure(
            model,
            tokenizer,
            question,
            device,
            max_new_tokens,
            adaptive_inference=True,
        )
        fields = generated["fields"]
        predicted_answer = fields["answer"] if fields else ""
        if predicted_answer == expected_answer:
            final_correct += 1
        if generated["procedure"] == expected_procedure:
            procedure_correct += 1

        trace = generated["trace"]
        depth_sum += sum(float(row["used_depth"]) for row in trace)
        generated_tokens += len(trace)

        if predicted_answer != expected_answer and len(mistakes) < 10:
            mistakes.append(
                {
                    "question": question,
                    "expected": expected_answer,
                    "predicted": predicted_answer or "[none]",
                    "raw_completion": repr(generated["raw"]),
                    "procedure_exact": generated["procedure"] == expected_procedure,
                }
            )

    model.train(was_training)
    total = len(examples)
    return {
        "correct": final_correct,
        "total": total,
        "accuracy": final_correct / max(1, total),
        "procedure_correct": procedure_correct,
        "procedure_accuracy": procedure_correct / max(1, total),
        "average_generated_depth": depth_sum / max(1, generated_tokens),
        "mistakes": mistakes,
    }
