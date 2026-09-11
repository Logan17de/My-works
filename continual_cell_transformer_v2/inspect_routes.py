from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import ModelConfig
from model import ContinualCellTransformerV2
from tokenizer import DynamicByteTokenizer
from train import evaluate


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    model = ContinualCellTransformerV2(ModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    for filename in args.file:
        text = Path(filename).read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        loss, routes = evaluate(
            model, ids, args.batch_size, args.seq_len, device, args.batches
        )
        print(json.dumps({"file": filename, "loss": loss, "routes": routes}, indent=2))


if __name__ == "__main__":
    main()
