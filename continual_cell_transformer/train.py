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


def sample_batch(
    ids: list[int],
    batch: int,
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not ids:
        raise ValueError("Dataset produced zero tokens.")
    if len(ids) < length + 2:
        ids = ids * ((length + 2) // len(ids) + 1)
    starts = torch.randint(
        0,
        len(ids) - length - 1,
        (batch,),
    )
    x = torch.stack(
        [torch.tensor(ids[start : start + length]) for start in starts]
    ).long()
    y = torch.stack(
        [
            torch.tensor(
                ids[start + 1 : start + length + 1]
            )
            for start in starts
        ]
    ).long()
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model: ContinualCellTransformer,
    ids: list[int],
    batch: int,
    length: int,
    device: torch.device,
    batches: int = 20,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        x, y = sample_batch(ids, batch, length, device)
        losses.append(
            float(model(x, labels=y)["loss"].item())
        )
    model.train(was_training)
    return sum(losses) / len(losses)


def optimizer_for(
    model: ContinualCellTransformer,
    config: TrainConfig,
) -> torch.optim.AdamW:
    groups: dict[str, list[torch.nn.Parameter]] = {
        "embedding": [],
        "attention": [],
        "ffn": [],
        "cells": [],
        "other": [],
    }
    for name, parameter in model.named_parameters():
        if (
            name.startswith("token_embedding")
            or name.startswith("lm_head")
        ):
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
        "embedding": config.embedding_lr,
        "attention": config.attention_lr,
        "ffn": config.ffn_lr,
        "cells": config.cell_lr,
        "other": config.other_lr,
    }
    return torch.optim.AdamW(
        [
            {
                "params": parameters,
                "lr": learning_rates[name],
                "weight_decay": config.weight_decay,
                "group_name": name,
            }
            for name, parameters in groups.items()
            if parameters
        ]
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument(
        "--eval-file",
        help=(
            "Held-out file used for periodic validation. "
            "Defaults to the training file."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--retention-file")

    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--log-cell-routing", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--add-token", action="append", default=[])
    parser.add_argument("--auto-add-words", action="store_true")

    parser.add_argument(
        "--allocate-cells",
        type=int,
        default=0,
        help=(
            "Allocate this many cells per pool into one new routing bank "
            "before training. Existing banks are sealed first."
        ),
    )

    parser.add_argument("--enable-growth", action="store_true")
    parser.add_argument("--growth-warmup", type=int, default=100)
    parser.add_argument("--growth-patience", type=int, default=40)
    parser.add_argument("--growth-cells", type=int, default=8)
    parser.add_argument("--growth-confidence", type=float, default=0.42)
    parser.add_argument("--growth-loss-floor", type=float, default=1.5)
    parser.add_argument("--max-growth-events", type=int, default=1)
    parser.add_argument("--growth-cooldown", type=int, default=100)

    parser.add_argument("--seal-active-cells", action="store_true")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Set embedding, attention, FFN and other learning rates to zero. "
            "Only unsealed cell banks and their adapters learn."
        ),
    )
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
    parser.add_argument("--max-cell-banks", type=int, default=8)
    parser.add_argument(
        "--new-bank-adapter-scale",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--bank-context-scale",
        type=float,
        default=0.25,
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    train_text = Path(args.train_file).read_text(
        encoding="utf-8"
    )
    eval_text = (
        Path(args.eval_file).read_text(encoding="utf-8")
        if args.eval_file
        else train_text
    )
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_step = 0
    resized = False
    architecture_migrated = False
    checkpoint = None

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        tokenizer = DynamicByteTokenizer.from_dict(
            checkpoint["tokenizer"]
        )
        old_vocab = tokenizer.vocab_size
        tokens = list(args.add_token)
        if args.auto_add_words:
            tokens += tokenizer.discover_tokens(train_text)
        tokenizer.add_tokens(tokens)

        model_config = ModelConfig.from_dict(
            checkpoint["model_config"]
        )
        model = ContinualCellTransformer(model_config)
        missing, unexpected = model.load_compatible_state_dict(
            checkpoint["model_state"]
        )
        architecture_migrated = bool(missing or unexpected)
        if missing:
            print("checkpoint upgraded; initialized V2 fields:", missing)
        if unexpected:
            print("ignored unexpected checkpoint fields:", unexpected)

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

        cell_layers = tuple(
            index
            for index in (1, 3)
            if index < args.layers
        ) or (args.layers - 1,)
        model_config = ModelConfig(
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
            max_cell_banks=args.max_cell_banks,
            new_bank_adapter_scale=args.new_bank_adapter_scale,
            bank_context_scale=args.bank_context_scale,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = ContinualCellTransformer(model_config)

    if args.seq_len > model.config.max_seq_len:
        raise ValueError(
            f"--seq-len={args.seq_len} exceeds model "
            f"max_seq_len={model.config.max_seq_len}."
        )

    model.to(device)
    train_ids = tokenizer.encode(
        train_text,
        add_bos=True,
        add_eos=True,
    )
    eval_ids = tokenizer.encode(
        eval_text,
        add_bos=True,
        add_eos=True,
    )

    retention_ids: list[int] | None = None
    retention_pre_allocation: float | None = None
    retention_before: float | None = None
    if args.retention_file:
        retention_text = Path(args.retention_file).read_text(
            encoding="utf-8"
        )
        retention_ids = tokenizer.encode(
            retention_text,
            add_bos=True,
            add_eos=True,
        )
        retention_pre_allocation = evaluate(
            model,
            retention_ids,
            args.batch_size,
            args.seq_len,
            device,
            batches=args.eval_batches,
        )
        print(
            "old-task loss before allocation="
            f"{retention_pre_allocation:.4f}"
        )

    allocated_at_start: list[dict[str, object]] = []
    if args.allocate_cells > 0:
        seed_x, _ = sample_batch(
            train_ids,
            args.batch_size,
            args.seq_len,
            device,
        )
        with torch.no_grad():
            seed_result = model(seed_x)
        allocated_at_start = model.allocate_new_bank(
            count=args.allocate_cells,
            seeds=seed_result["growth_seeds"],
            seal_existing=True,
        )
        print("allocated one new routing bank:", allocated_at_start)
        architecture_migrated = True

    if retention_ids is not None:
        retention_before = evaluate(
            model,
            retention_ids,
            args.batch_size,
            args.seq_len,
            device,
            batches=args.eval_batches,
        )
        if args.allocate_cells > 0:
            print(
                "old-task loss after allocation="
                f"{retention_before:.4f}; allocation delta="
                f"{retention_before - retention_pre_allocation:+.4f}"
            )
        else:
            print(f"old-task loss before={retention_before:.4f}")

    embedding_lr = (
        0.0 if args.freeze_backbone else args.embedding_lr
    )
    attention_lr = (
        0.0 if args.freeze_backbone else args.attention_lr
    )
    ffn_lr = 0.0 if args.freeze_backbone else args.ffn_lr
    other_lr = (
        0.0 if args.freeze_backbone else args.other_lr
    )

    train_config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        embedding_lr=embedding_lr,
        attention_lr=attention_lr,
        ffn_lr=ffn_lr,
        cell_lr=args.cell_lr,
        other_lr=other_lr,
        mature_cell_scale=args.mature_cell_scale,
        enable_growth=args.enable_growth,
        growth_warmup_steps=args.growth_warmup,
        growth_patience=args.growth_patience,
        growth_cells=args.growth_cells,
        growth_confidence=args.growth_confidence,
        growth_loss_floor=args.growth_loss_floor,
        max_growth_events=args.max_growth_events,
        growth_cooldown_steps=args.growth_cooldown,
        seed=args.seed,
    )
    optimizer = optimizer_for(model, train_config)

    can_restore_optimizer = (
        args.resume
        and checkpoint is not None
        and not resized
        and not architecture_migrated
        and args.allocate_cells == 0
    )
    if can_restore_optimizer:
        try:
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )
            print("restored optimizer state")
        except Exception as error:
            print("optimizer state restarted:", error)
    elif args.resume:
        print(
            "optimizer state restarted for vocabulary, V2 migration, "
            "or new-bank allocation"
        )

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
    print("routing banks:", model.bank_summaries())

    loss_ema: float | None = None
    low_confidence = 0
    growth_events = 0
    growth_cooldown_until = 0
    final_eval = initial_eval
    best_eval = initial_eval
    bad_eval_count = 0
    final_step = previous_step
    model.train()

    for local_step in range(1, args.steps + 1):
        step = previous_step + local_step
        final_step = step
        x, y = sample_batch(
            train_ids,
            args.batch_size,
            args.seq_len,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        result = model(x, labels=y)
        result["loss"].backward()
        model.mask_cell_gradients(args.mature_cell_scale)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()
        model.advance_maturity()

        loss = float(result["loss"].item())
        confidence = float(
            result["cell_confidence"].item()
        )
        loss_ema = (
            loss
            if loss_ema is None
            else 0.98 * loss_ema + 0.02 * loss
        )

        growth_allowed = (
            args.enable_growth
            and growth_events < args.max_growth_events
            and local_step >= args.growth_warmup
            and local_step >= growth_cooldown_until
        )
        if (
            growth_allowed
            and loss_ema > args.growth_loss_floor
            and confidence < args.growth_confidence
        ):
            low_confidence += 1
        else:
            low_confidence = 0

        if (
            growth_allowed
            and low_confidence >= args.growth_patience
        ):
            allocated = model.allocate_new_bank(
                count=args.growth_cells,
                seeds=result["growth_seeds"],
                seal_existing=True,
            )
            any_cells = any(
                bool(item["cells"]) for item in allocated
            )
            print(
                f"growth event={growth_events + 1} "
                f"allocated={allocated} step={step} "
                f"loss_ema={loss_ema:.4f} "
                f"confidence={confidence:.3f}"
            )
            if any_cells:
                growth_events += 1
                growth_cooldown_until = (
                    local_step + args.growth_cooldown
                )
            else:
                growth_events = args.max_growth_events
            low_confidence = 0

        if (
            local_step == 1
            or local_step % args.log_interval == 0
        ):
            message = (
                f"step={step} loss={loss:.4f} "
                f"ppl={math.exp(min(loss, 20)):.2f} "
                f"confidence={confidence:.3f} "
                f"banks={result['cell_banks']}"
            )
            if args.log_cell_routing:
                message += (
                    f" top_ids={result['bank_top_ids']}"
                )
            print(message)

        if (
            local_step % args.eval_interval == 0
            or local_step == args.steps
        ):
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

            if (
                final_eval
                < best_eval - args.early_stop_min_delta
            ):
                best_eval = final_eval
                bad_eval_count = 0
            else:
                bad_eval_count += 1

            if (
                args.early_stop_patience > 0
                and bad_eval_count
                >= args.early_stop_patience
            ):
                print(
                    "early stopping: no held-out improvement "
                    f"for {bad_eval_count} evaluations"
                )
                break

    if args.seal_active_cells:
        model.seal_active_cells()

    retention_after: float | None = None
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
            f"training delta="
            f"{retention_after - retention_before:+.4f}; "
            f"total delta="
            f"{retention_after - retention_pre_allocation:+.4f}"
        )

    payload = {
        "format_version": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": final_step,
        "train_args": vars(args),
    }
    torch.save(payload, output_dir / "checkpoint.pt")
    tokenizer.save(output_dir / "tokenizer.json")

    summary = {
        "format_version": 2,
        "step": final_step,
        "vocab_size": tokenizer.vocab_size,
        "routing_banks": model.bank_summaries(),
        "allocated_at_start": allocated_at_start,
        "growth_events": growth_events,
        "train_file": args.train_file,
        "eval_file": args.eval_file or args.train_file,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "best_eval_loss": best_eval,
        "retention_file": args.retention_file,
        "retention_pre_allocation": retention_pre_allocation,
        "retention_before_training": retention_before,
        "retention_after": retention_after,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("saved", output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
