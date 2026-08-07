from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


MODELS = (
    "basic_transformer",
    "post_ffn_all",
    "standalone_all",
    "pre_activation_all",
    "post_activation_all",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the normal Transformer and all all-layer learner variants sequentially."
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out-root", default="runs/learner_placement_comparison")
    parser.add_argument("--clean", action="store_true")

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--learner-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--learner-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--evaluate-best",
        action="store_true",
        help="Run exact held-out generation on each best checkpoint after training.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def common_args(args: argparse.Namespace, out_dir: Path) -> list[str]:
    return [
        "--train-file", args.train_file,
        "--eval-file", args.eval_file,
        "--out-dir", str(out_dir),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--eval-interval", str(args.eval_interval),
        "--log-interval", str(args.log_interval),
        "--early-stop-patience", str(args.early_stop_patience),
        "--early-stop-min-delta", str(args.early_stop_min_delta),
        "--d-model", str(args.d_model),
        "--heads", str(args.heads),
        "--d-ff", str(args.d_ff),
        "--layers", str(args.layers),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--grad-clip", str(args.grad_clip),
        "--seed", str(args.seed),
    ]


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def print_summary(out_root: Path) -> None:
    rows = []
    for model_name in MODELS:
        path = out_root / model_name / "summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                model_name,
                int(data.get("parameter_count", 0)),
                int(data.get("learner_parameter_count", 0)),
                int(data.get("best_step", 0)),
                float(data.get("best_answer_only_eval", float("nan"))),
                float(data.get("final_answer_only_eval", float("nan"))),
                float(data.get("overfit_percent", float("nan"))),
            )
        )

    print("\nComparison summary")
    print(
        f"{'model':22} {'params':>12} {'learner':>10} "
        f"{'best_step':>10} {'best_eval':>10} {'final_eval':>11} {'overfit':>10}"
    )
    print("-" * 94)
    for name, params, learner, best_step, best_eval, final_eval, overfit in rows:
        print(
            f"{name:22} {params:12,d} {learner:10,d} "
            f"{best_step:10d} {best_eval:10.4f} {final_eval:11.4f} "
            f"{overfit:9.2f}%"
        )


def main() -> None:
    args = arguments()
    out_root = Path(args.out_root)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    for model_name in MODELS:
        out_dir = out_root / model_name
        print("\n" + "=" * 80)
        print(f"TRAINING {model_name}")
        print("=" * 80, flush=True)

        if model_name == "basic_transformer":
            command = [
                python,
                "train_basic_transformer_plain.py",
                *common_args(args, out_dir),
            ]
        else:
            command = [
                python,
                "train_learner_variant.py",
                "--variant", model_name,
                *common_args(args, out_dir),
                "--learner-dim", str(args.learner_dim),
                "--learner-lr", str(args.learner_lr),
            ]

        run(command)

        if args.evaluate_best:
            run(
                [
                    python,
                    "evaluate_plain_math_dataset.py",
                    "--checkpoint", str(out_dir / "best_checkpoint.pt"),
                    "--eval-file", args.eval_file,
                    "--max-new-tokens", str(args.max_new_tokens),
                ]
            )

    print_summary(out_root)


if __name__ == "__main__":
    main()
