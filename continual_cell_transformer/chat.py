from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if int(checkpoint.get("architecture_version", 0)) != ContinualCellTransformer.ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint is not V5")
    return checkpoint


def prepare_prompt(text: str) -> str:
    text = text.strip()
    if text.startswith("<Q>"):
        return text if "<A>" in text else text + "\n<A>"
    return f"<Q> {text}\n<A>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--show-routing", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    model = ContinualCellTransformer(
        ModelConfig.from_dict(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    print("Continual Cell Transformer V5. Type /quit to exit.")
    print("Pool:", model.pool_summary())

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

        if args.show_routing:
            routing = model(input_ids, adaptive_inference=True)
            print("Depth used:", routing["used_depth"])
            print("Expected depth:", float(routing["expected_depth"]))
            print("Halt probabilities:", routing["halt_probs"].squeeze(0).tolist())
            print("Coverage:", float(routing["coverage"]))
            print("Mean active cells:", float(routing["mean_active"]))
            print("Top cells:", routing["top_cell_ids"])
            print("Micro-neuron saturation:", float(routing["micro_saturation"]))

        generated = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(generated[0, len(ids) :].tolist())
        for marker in ("<END>", "<Q>"):
            if marker in answer:
                answer = answer.split(marker, 1)[0]
        print("Model:", answer.strip() or "[no answer generated]")


if __name__ == "__main__":
    main()
