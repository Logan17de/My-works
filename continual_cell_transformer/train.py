from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sample_batch(ids: list[int], batch: int, length: int, device: torch.device):
    if len(ids) < length + 2:
        ids = (ids * ((length + 2) // max(1, len(ids)) + 1))
    starts = torch.randint(0, len(ids) - length - 1, (batch,))
    x = torch.stack([torch.tensor(ids[s:s + length]) for s in starts]).long().to(device)
    y = torch.stack([torch.tensor(ids[s + 1:s + length + 1]) for s in starts]).long().to(device)
    return x, y


@torch.no_grad()
def evaluate(model, ids, batch, length, device, batches=20):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = sample_batch(ids, batch, length, device)
        losses.append(float(model(x, y)["loss"]))
    model.train(was_training)
    return sum(losses) / len(losses)


def optimizer_for(model: ContinualCellTransformer, cfg: TrainConfig):
    groups = {"embedding": [], "attention": [], "ffn": [], "cells": [], "other": []}
    for name, p in model.named_parameters():
        if name.startswith("token_embedding") or name.startswith("lm_head"):
            groups["embedding"].append(p)
        elif ".pool." in name:
            groups["cells"].append(p)
        elif ".attn." in name:
            groups["attention"].append(p)
        elif ".ffn." in name:
            groups["ffn"].append(p)
        else:
            groups["other"].append(p)
    lrs = {
        "embedding": cfg.embedding_lr,
        "attention": cfg.attention_lr,
        "ffn": cfg.ffn_lr,
        "cells": cfg.cell_lr,
        "other": cfg.other_lr,
    }
    return torch.optim.AdamW([
        {"params": params, "lr": lrs[name], "weight_decay": cfg.weight_decay}
        for name, params in groups.items() if params
    ])


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume")
    p.add_argument("--retention-file")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--add-token", action="append", default=[])
    p.add_argument("--auto-add-words", action="store_true")
    p.add_argument("--enable-growth", action="store_true")
    p.add_argument("--growth-patience", type=int, default=40)
    p.add_argument("--growth-cells", type=int, default=8)
    p.add_argument("--growth-confidence", type=float, default=0.42)
    p.add_argument("--growth-loss-floor", type=float, default=1.5)
    p.add_argument("--seal-active-cells", action="store_true")
    p.add_argument("--embedding-lr", type=float, default=1e-5)
    p.add_argument("--attention-lr", type=float, default=2e-5)
    p.add_argument("--ffn-lr", type=float, default=5e-5)
    p.add_argument("--cell-lr", type=float, default=2e-4)
    p.add_argument("--mature-cell-scale", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--max-cells", type=int, default=256)
    p.add_argument("--initial-cells", type=int, default=64)
    p.add_argument("--top-k-cells", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def main():
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = Path(args.train_file).read_text(encoding="utf-8")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    previous_step = 0
    resized = False
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        tokenizer = DynamicByteTokenizer.from_dict(ckpt["tokenizer"])
        old_vocab = tokenizer.vocab_size
        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(text)
        tokenizer.add_tokens(tokens)
        cfg = ModelConfig.from_dict(ckpt["model_config"])
        model = ContinualCellTransformer(cfg)
        model.load_state_dict(ckpt["model_state"])
        if tokenizer.vocab_size > old_vocab:
            model.resize_vocabulary(tokenizer.vocab_size)
            resized = True
        previous_step = int(ckpt.get("step", 0))
    else:
        tokenizer = DynamicByteTokenizer()
        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(text)
        tokenizer.add_tokens(tokens)
        cell_layers = tuple(i for i in (1, 3) if i < args.layers) or (args.layers - 1,)
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            cell_layers=cell_layers,
            max_cells=args.max_cells,
            initial_active_cells=args.initial_cells,
            top_k_cells=args.top_k_cells,
        )
        model = ContinualCellTransformer(cfg)

    model.to(device)
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        embedding_lr=args.embedding_lr,
        attention_lr=args.attention_lr,
        ffn_lr=args.ffn_lr,
        cell_lr=args.cell_lr,
        mature_cell_scale=args.mature_cell_scale,
    )
    opt = optimizer_for(model, train_cfg)
    if args.resume and not resized:
        try:
            opt.load_state_dict(ckpt["optimizer_state"])
        except Exception as error:
            print("Optimizer state restarted:", error)

    train_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    retention_ids = None
    before = None
    if args.retention_file:
        retention_text = Path(args.retention_file).read_text(encoding="utf-8")
        retention_ids = tokenizer.encode(retention_text, add_bos=True, add_eos=True)
        before = evaluate(model, retention_ids, args.batch_size, args.seq_len, device)
        print(f"old-task loss before={before:.4f}")

    loss_ema = None
    low_confidence = 0
    model.train()
    for local_step in range(1, args.steps + 1):
        step = previous_step + local_step
        x, y = sample_batch(train_ids, args.batch_size, args.seq_len, device)
        opt.zero_grad(set_to_none=True)
        result = model(x, y)
        result["loss"].backward()
        model.mask_cell_gradients(args.mature_cell_scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.advance_maturity()

        loss = float(result["loss"])
        confidence = float(result["cell_confidence"])
        loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss
        if args.enable_growth and loss_ema > args.growth_loss_floor and confidence < args.growth_confidence:
            low_confidence += 1
        else:
            low_confidence = 0
        if args.enable_growth and low_confidence >= args.growth_patience:
            print("allocated", model.allocate_cells(args.growth_cells, result["growth_seeds"]))
            low_confidence = 0

        if local_step == 1 or local_step % 20 == 0:
            print(f"step={step} loss={loss:.4f} ppl={math.exp(min(loss, 20)):.2f} confidence={confidence:.3f} cells={result['active_cells']}")
        if local_step % args.eval_interval == 0 or local_step == args.steps:
            val = evaluate(model, train_ids, args.batch_size, args.seq_len, device)
            print(f"eval step={step} loss={val:.4f} ppl={math.exp(min(val, 20)):.2f}")

    if args.seal_active_cells:
        model.seal_active_cells()

    after = evaluate(model, retention_ids, args.batch_size, args.seq_len, device) if retention_ids else None
    if after is not None:
        print(f"old-task loss after={after:.4f}; delta={after - before:+.4f}")

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": previous_step + args.steps,
    }
    torch.save(payload, out / "checkpoint.pt")
    tokenizer.save(out / "tokenizer.json")
    (out / "summary.json").write_text(json.dumps({
        "step": payload["step"],
        "vocab_size": tokenizer.vocab_size,
        "active_cells": [p.active_count for p in model.pools()],
        "retention_before": before,
        "retention_after": after,
    }, indent=2), encoding="utf-8")
    print("saved", out / "checkpoint.pt")


if __name__ == "__main__":
    main()
