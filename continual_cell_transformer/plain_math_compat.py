from __future__ import annotations

import re

from train_math import MATH_RECORD_PATTERN, MathRecord


RAW_MATH_PATTERN = re.compile(
    r"^\s*(.+?)\s*=\s*(-?\d+)\s*$",
    flags=re.MULTILINE,
)


def parse_math_records(text: str) -> list[MathRecord]:
    """Parse either structured <Q>/<A>/<END> records or `expr = answer` lines."""
    structured = [
        MathRecord(question=question.strip(), answer=answer.strip())
        for question, answer in MATH_RECORD_PATTERN.findall(text)
    ]
    if structured:
        return structured

    records: list[MathRecord] = []
    invalid: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = RAW_MATH_PATTERN.fullmatch(line)
        if match is None:
            invalid.append(line)
            continue
        question, answer = match.groups()
        records.append(
            MathRecord(question=question.strip(), answer=answer.strip())
        )

    if invalid:
        preview = invalid[:3]
        raise ValueError(
            f"Found {len(invalid)} invalid math lines; first examples: {preview}"
        )
    if not records:
        raise ValueError(
            "No math records found. Expected structured <Q>/<A>/<END> records "
            "or one arithmetic equation per line such as '822 + 27 = 849'."
        )
    return records
