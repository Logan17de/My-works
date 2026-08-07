from __future__ import annotations

import argparse
from pathlib import Path

import torch

from basic_transformer import BasicTransformer
from config import ModelConfig
from mastery_eval import evaluate_math_mastery
from model import ContinualCellTransformer
from plain_math_compat import parse_math_records
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(checkpoint: dict):
    config = ModelConfig.from_dict(checkpoint["model_config"])
    if checkpoint.get("model_type") == BasicTransformer.MODEL_TYPE:
        model = BasicTransformer(config, num_layers=int(checkpoint["num_layers"]))
        label = f"basic_transformer/{int(checkpoint['num_layers'])}layers"
    else:
        version = int(checkpoint.get("architecture_version", 0))
        if version != ContinualCellTransformer.ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported checkpoint architecture version {version}; "
                f"expected V{ContinualCellTransformer.ARCHITECTURE_VERSION} or a basic Transformer checkpoint."
            )
        model = ContinualCellTransformer(config)
        label = f"continual_cell_transformer/V{version}"
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    model, label = build_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    records = parse_math_records(Path(args.eval_file).read_text(encoding="utf-8"))
    examples = [(row.question, row.answer) for row in records]
    result = evaluate_math_mastery(
        model,
        tokenizer,
        examples,
        device,
        args.max_new_tokens,
    )

    print(f"model={label}")
    print(f"eval_examples={result['total']}")
    print(
        f"accuracy={result['correct']}/{result['total']} "
        f"({100.0 * float(result['accuracy']):.2f}%)"
    )
    print(
        "average_generated_depth=",
        f"{float(result['average_generated_depth']):.3f}",
    )
    if result["mistakes"]:
        print("first mistakes:")
        for row in result["mistakes"]:
            print(
                {
                    "question": row["question"],
                    "expected": row["expected"],
                    "predicted": row["predicted"],
                    "raw_completion": row["raw_completion"],
                }
            )


if __name__ == "__main__":
    main()
