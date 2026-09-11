from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from config import ModelConfig
from mastery_eval import evaluate_math_mastery
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


MATH_PATTERN = re.compile(r"<Q>\s*([^\n]+?)\s*=\s*\n<A>\s*(-?\d+)")


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint)
    version = int(checkpoint.get("architecture_version", 0))
    expected_version = ContinualCellTransformer.ARCHITECTURE_VERSION
    if version != expected_version:
        raise ValueError(
            f"Checkpoint V{version} is incompatible with V{expected_version}"
        )

    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    model = ContinualCellTransformer(ModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    text = Path(args.eval_file).read_text(encoding="utf-8")
    examples = [
        (question.strip(), answer)
        for question, answer in MATH_PATTERN.findall(text)
    ]
    if not examples:
        raise ValueError("No <Q>/<A> arithmetic examples were found")

    result = evaluate_math_mastery(
        model,
        tokenizer,
        examples,
        device,
        args.max_new_tokens,
    )
    print(
        f"accuracy={result['correct']}/{result['total']} "
        f"({100.0 * float(result['accuracy']):.2f}%)"
    )
    print(
        "average generated-token depth=",
        f"{float(result['average_generated_depth']):.3f}",
    )
    if result["mistakes"]:
        print("mistakes:")
        for row in result["mistakes"]:
            print(row)


if __name__ == "__main__":
    main()
