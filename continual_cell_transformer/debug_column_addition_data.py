from __future__ import annotations

from pathlib import Path

from column_addition_objective import encode_records, parse_question, parse_records
from tokenizer import DynamicByteTokenizer


def unordered_pairs(path: Path) -> set[tuple[int, int]]:
    records = parse_records(path.read_text(encoding="utf-8"))
    pairs: set[tuple[int, int]] = set()
    for record in records:
        a, b = parse_question(record.question)
        pairs.add((min(a, b), max(a, b)))
    return pairs


def main() -> None:
    data = Path("data")
    train_path = data / "column_addition_train.txt"
    validation_path = data / "column_addition_validation_eval.txt"
    test_path = data / "column_addition_test_eval.txt"

    for path in (train_path, validation_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; run generate_column_addition_datasets.py first"
            )

    train_pairs = unordered_pairs(train_path)
    validation_pairs = unordered_pairs(validation_path)
    test_pairs = unordered_pairs(test_path)

    assert train_pairs.isdisjoint(validation_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert validation_pairs.isdisjoint(test_pairs)

    train_records = parse_records(train_path.read_text(encoding="utf-8"))
    tokenizer = DynamicByteTokenizer()
    encoded = encode_records(train_records[:256], tokenizer, max_seq_len=96)
    max_tokens = max(len(row.input_ids) for row in encoded)

    operand_digits: set[str] = set()
    for record in train_records:
        a, b = parse_question(record.question)
        operand_digits.update(str(a))
        operand_digits.update(str(b))
    assert operand_digits == set("0123456789"), operand_digits

    print("Column-addition data integrity passed.")
    print(
        f"unordered pairs: train={len(train_pairs)} "
        f"validation={len(validation_pairs)} test={len(test_pairs)}"
    )
    print(f"all operand digits present: {sorted(operand_digits)}")
    print(f"sample max tokens: {max_tokens}/96")


if __name__ == "__main__":
    main()
