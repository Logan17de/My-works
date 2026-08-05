from pathlib import Path
import random

ROOT = Path("data")
ROOT.mkdir(exist_ok=True)


def examples(operator: str) -> list[str]:
    rows: list[str] = []
    for a in range(10):
        for b in range(10):
            if operator == "+":
                answer = a + b
            elif operator == "*":
                answer = a * b
            else:
                raise ValueError(f"Unsupported operator: {operator}")

            rows.append(
                f"<Q> {a} {operator} {b} =\n"
                f"<A> {answer}\n"
                "<END>\n"
            )
    return rows


def write_train(
    filename: str,
    operator: str,
    repeats: int = 200,
    seed: int = 17,
) -> None:
    base = examples(operator)
    rng = random.Random(seed)
    output: list[str] = []

    for _ in range(repeats):
        epoch = base.copy()
        rng.shuffle(epoch)
        output.extend(epoch)

    (ROOT / filename).write_text("\n".join(output), encoding="utf-8")


def write_eval(filename: str, operator: str) -> None:
    (ROOT / filename).write_text(
        "\n".join(examples(operator)),
        encoding="utf-8",
    )


write_train("addition_train.txt", "+")
write_eval("addition_eval.txt", "+")
write_train("multiplication_train.txt", "*")
write_eval("multiplication_eval.txt", "*")

print("Created:")
for path in sorted(ROOT.glob("*_*.txt")):
    print(f"- {path}: {path.stat().st_size:,} bytes")
