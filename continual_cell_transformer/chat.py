from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = safe_torch_load(args.checkpoint)
    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = ContinualCellTransformer(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    print("Continual Cell Transformer chat. Type /quit to exit.")
    while True:
        try:
            prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.strip().lower() in {"/quit", "/exit"}:
            break
        if not prompt:
            continue

        ids = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_ids = generated[0, len(ids) :].tolist()
        print("Model:", tokenizer.decode(completion_ids))


if __name__ == "__main__":
    main()
