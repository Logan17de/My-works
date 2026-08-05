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
    if not ids:
        raise ValueError("Dataset produced zero tokens.")
    if len(ids) < length + 2:
        ids = ids * ((length + 2) // len(ids) + 1)
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
        losses.append(float(model(x, labels=y)["loss"]))
    model.train(was_training)
    return sum(losses) / len(losses)


def optimizer_for(model: ContinualCellTransformer, cfg: TrainConfig):
    groups = {"embedding": [], "attention": [], "ffn": [], "cells": [], "other": []}
    for name, parameter in model.named_parameters():
        if name.startswith("token_embedding") or name.startswith("lm_head"):
            groups["embedding"].append(parameter)
        elif ".pool." in name:
            groups["cells"].append(parameter)
        elif ".attn." in name:
            groups["attention"].append(parameter)
        elif ".ffn." in name:
            groups["ffn"].append(parameter)
        else:
            groups["other"].append(parameter)
    learning_rates = {
        "embedding": cfg.embedding_lr,
        "attention": cfg.attention_lr,
        "ffn": cfg.ffn_lr,
        "cells": cfg.cell_lr,
        "other": cfg.other_lr,
    }
    return torch.optim.AdamW([
        {
            "params": parameters,
            "lr": learning_rates[name],
            "weight_decay": cfg.weight_decay,
        }
        for name, parameters in groups.items()
        if parameters
    ])


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", help="Held-out file used for periodic validation. Defaults to the training file.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--retention-file")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--add-token", action="append", default=[])
    parser.add_argument("--auto-add-words", action="store_true")
    parser.add_argument("--enable-growth", action="store_true")
    parser.add_argument("--growth-warmup", type=int, default=100)
    parser.add_argument("--growth-patience", type=int, default=40)
    parser.add_argument("--growth-cells", type=int, default=8)
    parser.add_argument("--growth-confidence", type=float, default=0.42)
    parser.add_argument("--growth-loss-floor", type=float, default=1.5)
    parser.add_argument("--seal-active-cells", action="store_true")
    parser.add_argument("--embedding-lr", type=float, default=1e-5)
    parser.add_argument("--attention-lr", type=float, default=2e-5)
    parser.add_argument("--ffn-lr", type=float, default=5e-5)
    parser.add_argument("--cell-lr", type=float, default=2e-4)
    parser.add_argument("--other-lr", type=float, default=5e-5)
    parser.add_argument("--mature-cell-scale", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--initial-cells", type=int, default=64)
    parser.add_argument("--top-k-cells", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main():
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_text = Path(args.train_file).read_text(encoding="utf-8")
    eval_text = (
        Path(args.eval_file).read_text(encoding="utf-8")
        if args.eval_file
        else train_text
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    previous_step = 0
    resized = False
    checkpoint = None

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
        old_vocab = tokenizer.vocab_size
        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(train_text)
        tokenizer.add_tokens(tokens)

        cfg = ModelConfig.from_dict(checkpoint["model_config"])
        model = ContinualCellTransformer(cfg)
        model.load_state_dict(checkpoint["model_state"])
        if tokenizer.vocab_size > old_vocab:
            model.resize_vocabulary(tokenizer.vocab_size)
            resized = True
        previous_step = int(checkpoint.get("step", 0))
    else:
        tokenizer = DynamicByteTokenizer()
        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(train_text)
        tokenizer.add_tokens(tokens)

        cell_layers = tuple(index for index in (1, 3) if index < args.layers) or (args.layers - 1,)
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
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = ContinualCellTransformer(cfg)

    if args.seq_len > model.config.max_seq_len:
        raise ValueError(
            f"--seq-len={args.seq_len} exceeds model max_seq_len={model.config.max_seq_len}."
        )

    model.to(device)
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        embedding_lr=args.embedding_lr,
        attention_lr=args.attention_lr,
        ffn_lr=args.ffn_lr,
        cell_lr=args.cell_lr,
        other_lr=args.other_lr,
        mature_cell_scale=args.mature_cell_scale,
        enable_growth=args.enable_growth,
        growth_warmup_steps=args.growth_warmup,
        growth_patience=args.growth_patience,
        growth_cells=args.growth_cells,
        growth_confidence=args.growth_confidence,
        growth_loss_floor=args.growth_loss_floor,
        seed=args.seed,
    )
    optimizer = optimizer_for(model, train_cfg)

    if args.resume and not resized and checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("restored optimizer state")
        except Exception as error:
            print("optimizer state restarted:", error)
    elif resized:
        print("vocabulary expanded; optimizer state restarted because parameter shapes changed")

    train_ids = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    eval_ids = tokenizer.encode(eval_text, add_bos=True, add_eos=True)

    initial_eval = evaluate(
        model,
        eval_ids,
        args.batch_size,
        args.seq_len,
        device,
        batches=args.eval_batches,
    )
    print(
        f"initial eval loss={initial_eval:.4f} "
        f"ppl={math.exp(min(initial_eval, 20)):.2f} "
        f"file={args.eval_file or args.train_file}"
    )

    retention_ids = None
    retention_before = None
    if args.retention_file:
        retention_text = Path(args.retention_file).read_text(encoding="utf-8")
        retention_ids = tokenizer.encode(retention_text, add_bos=True, add_eos=True)
        retention_before = evaluate(
            model,
            retention_ids,
            args.batch_size,
            args.seq_len,
            device,
            batches=args.eval_batches,
        )
        print(f"old-task loss before={retention_before:.4f}")

    loss_ema = None
    low_confidence = 0
    final_eval = initial_eval
    model.train()

    for local_step in range(1, args.steps + 1):
        step = previous_step + local_step
        x, y = sample_batch(train_ids, args.batch_size, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        result = model(x, labels=y)
        result["loss"].backward()
        model.mask_cell_gradients(args.mature_cell_scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.advance_maturity()

        loss = float(result["loss"])
        confidence = float(result["cell_confidence"])
        loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss

        growth_ready = local_step >= args.growth_warmup
        if (
            args.enable_growth
            and growth_ready
            and loss_ema > args.growth_loss_floor
            and confidence < args.growth_confidence
        ):
            low_confidence += 1
        else:
            low_confidence = 0

        if args.enable_growth and low_confidence >= args.growth_patience:
            allocated = model.allocate_cells(args.growth_cells, result["growth_seeds"])
            print(
                f"allocated={allocated} step={step} "
                f"loss_ema={loss_ema:.4f} confidence={confidence:.3f}"
            )
            low_confidence = 0

        if local_step == 1 or local_step % args.log_interval == 0:
            print(
                f"step={step} loss={loss:.4f} "
                f"ppl={math.exp(min(loss, 20)):.2f} "
                f"confidence={confidence:.3f} cells={result['active_cells']}"
            )

        if local_step % args.eval_interval == 0 or local_step == args.steps:
            final_eval = evaluate(
                model,
                eval_ids,
                args.batch_size,
                args.seq_len,
                device,
                batches=args.eval_batches,
            )
            print(
                f"eval step={step} loss={final_eval:.4f} "
                f"ppl={math.exp(min(final_eval, 20)):.2f} "
                f"file={args.eval_file or args.train_file}"
            )

    if args.seal_active_cells:
        model.seal_active_cells()

    retention_after = None
    if retention_ids is not None:
        retention_after = evaluate(
            model,
            retention_ids,
            args.batch_size,
            args.seq_len,
            device,
            batches=args.eval_batches,
        )
        print(
            f"old-task loss after={retention_after:.4f}; "
            f"delta={retention_after - retention_before:+.4f}"
        )

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": previous_step + args.steps,
        "train_args": vars(args),
    }
    torch.save(payload, out / "checkpoint.pt")
    tokenizer.save(out / "tokenizer.json")

    summary = {
        "step": payload["step"],
        "vocab_size": tokenizer.vocab_size,
        "active_cells": [pool.active_count for pool in model.pools()],
        "train_file": args.train_file,
        "eval_file": args.eval_file or args.train_file,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "retention_file": args.retention_file,
        "retention_before": retention_before,
        "retention_after": retention_after,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("saved", out / "checkpoint.pt")


if __name__ == "__main__":
    main()
