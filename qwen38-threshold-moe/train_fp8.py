#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from qtm.config import load_config
from qtm.fp8_trainer import FP8ResidentTrainer


def main() -> None:
    p = argparse.ArgumentParser(description="Full-resident FP8/MXFP8 trainer for the Qwen threshold-MoE experiment")
    p.add_argument("--config", default="configs/full_g4_fp8.json")
    p.add_argument("--drive-root", required=True, help="Persistent Google Drive/cloud root")
    p.add_argument("--resume", default=None, help="latest|none|checkpoint path")
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    t = cfg["training"]
    drive = Path(args.drive_root)
    # Hot training state stays on Colab local disk. Drive is touched only by logs/checkpoint sync.
    t["output_dir"] = str(drive / "runs")
    t["backup_dir"] = str(drive / "work-backup")
    if args.resume is not None:
        t["resume"] = args.resume
    if args.save_every is not None:
        t["save_every_steps"] = args.save_every
    if args.max_steps is not None:
        t["max_steps"] = args.max_steps

    trainer = FP8ResidentTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
