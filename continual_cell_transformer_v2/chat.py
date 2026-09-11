from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import ContinualCellTransformerV2
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def format_prompt(text: str) -> str:
    text = text.strip()
    if text.startswith("<Q>"):
        return text if "<A>" in text else text + "\n<A>"
    return f"<Q> {text}\n<A>"


@torch.no_grad()
def generate_answer(model, tokenizer, prompt_ids, device, max_new_tokens=24):
    ids = torch.tensor([prompt_ids], device=device)
    generated = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(generated[0, len(prompt_ids) :].tolist())
    stops = [
        position
        for marker in ("<END>", "<Q>")
        if (position := answer.find(marker)) >= 0
    ]
    if stops:
        answer = answer[: min(stops)]
    return answer.replace("<A>", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    if int(checkpoint.get("architecture_version", 0)) != 2:
        raise ValueError("chat.py V2 requires a V2 checkpoint.")
    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    model = ContinualCellTransformerV2(ModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    print("Continual Cell Transformer V2. Type /quit to exit.")
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
        formatted = format_prompt(prompt)
        prompt_ids = tokenizer.encode(formatted, add_bos=True)
        answer = generate_answer(model, tokenizer, prompt_ids, device, args.max_new_tokens)
        print("Model:", answer or "[no answer generated]")


if __name__ == "__main__":
    main()
