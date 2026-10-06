from __future__ import annotations

import argparse
from pathlib import Path

import torch

from basic_transformer import BasicTransformer
from config import ModelConfig
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("model_type") != BasicTransformer.MODEL_TYPE:
        raise ValueError("Checkpoint is not a basic Transformer checkpoint")
    return checkpoint


def prepare_prompt(text: str) -> str:
    text = text.strip()
    if text.startswith("<Q>"):
        if "<A>" in text:
            return text if text.endswith(" ") else text + " "
        return text + "\n<A> "
    return f"<Q> {text}\n<A> "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    model = BasicTransformer(
        ModelConfig.from_dict(checkpoint["model_config"]),
        num_layers=int(checkpoint["num_layers"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    print(
        "Basic Transformer baseline. "
        f"Layers={model.num_layers}. Type /quit to exit."
    )

    while True:
        try:
            prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in {"/quit", "/exit"}:
            break
        if not prompt.strip():
            continue

        model_prompt = prepare_prompt(prompt)
        ids = tokenizer.encode(model_prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        generated = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(generated[0, len(ids) :].tolist())
        for marker in ("<END>", "<Q>", "\n"):
            if marker in answer:
                answer = answer.split(marker, 1)[0]
        print("Model:", answer.strip() or "[no answer generated]")


if __name__ == "__main__":
    main()
