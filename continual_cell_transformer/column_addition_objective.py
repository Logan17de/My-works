from __future__ import annotations

import re
from dataclasses import dataclass

from tokenizer import DynamicByteTokenizer

OBJECTIVE_VERSION = "column_addition_procedure_v1"
RECORD_PATTERN = re.compile(
    r"<Q>\s*([^\n]+?)\s*=\s*\n<P>\s*([^\n]+?)\s*\n<END>",
    flags=re.MULTILINE,
)
PROCEDURE_PATTERN = re.compile(
    r"O(?P<ones_a>\d)\+(?P<ones_b>\d)=(?P<ones_total>\d+),"
    r"D(?P<ones_digit>\d),C(?P<ones_carry>\d)\|"
    r"T(?P<tens_a>\d)\+(?P<tens_b>\d)\+(?P<carry_in>\d)="
    r"(?P<tens_total>\d+),D(?P<tens_digit>\d),C(?P<tens_carry>\d)\|"
    r"H(?P<hundreds>\d)\|A(?P<answer>\d+)"
)
OPERAND_PATTERN = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*$")


@dataclass(frozen=True)
class ProcedureRecord:
    question: str
    procedure: str
    answer: str


@dataclass(frozen=True)
class EncodedProcedureRecord:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    loss_weights: tuple[float, ...]


def procedure_for(a: int, b: int) -> str:
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


def parse_question(question: str) -> tuple[int, int]:
    match = OPERAND_PATTERN.fullmatch(question.replace("\t", " "))
    if match is None:
        raise ValueError(f"Unsupported addition question: {question!r}")
    return int(match.group(1)), int(match.group(2))


def parse_records(text: str) -> list[ProcedureRecord]:
    rows: list[ProcedureRecord] = []
    for question, procedure in RECORD_PATTERN.findall(text):
        procedure = procedure.strip()
        match = PROCEDURE_PATTERN.fullmatch(procedure)
        if match is None:
            raise ValueError(f"Invalid procedure target: {procedure!r}")
        rows.append(
            ProcedureRecord(
                question=question.strip(),
                procedure=procedure,
                answer=match.group("answer"),
            )
        )
    if not rows:
        raise ValueError("No <Q>/<P>/<END> column-addition records found")
    return rows


def encode_records(
    records: list[ProcedureRecord],
    tokenizer: DynamicByteTokenizer,
    max_seq_len: int,
) -> list[EncodedProcedureRecord]:
    encoded: list[EncodedProcedureRecord] = []
    for record in records:
        prompt_ids = tokenizer.encode(
            f"<Q> {record.question} =\n<P> ",
            add_bos=True,
        )
        answer_start = record.procedure.rfind("|A") + 2
        target_ids: list[int] = []
        target_weights: list[float] = []
        for index, character in enumerate(record.procedure):
            ids = tokenizer.encode(character)
            if character.isdigit():
                weight = 2.0 if index >= answer_start else 1.0
            else:
                weight = 0.10
            target_ids.extend(ids)
            target_weights.extend([weight] * len(ids))

        full_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        input_ids = full_ids[:-1]
        next_ids = full_ids[1:]
        if len(input_ids) > max_seq_len:
            raise ValueError(
                f"Example needs {len(input_ids)} tokens; increase --seq-len"
            )

        labels = [tokenizer.pad_token_id] * len(next_ids)
        weights = [0.0] * len(next_ids)
        start = len(prompt_ids) - 1
        for offset, weight in enumerate(target_weights):
            position = start + offset
            labels[position] = next_ids[position]
            weights[position] = weight
        eos_position = start + len(target_ids)
        labels[eos_position] = next_ids[eos_position]
        weights[eos_position] = 0.10

        encoded.append(
            EncodedProcedureRecord(
                input_ids=tuple(input_ids),
                labels=tuple(labels),
                loss_weights=tuple(weights),
            )
        )
    return encoded
