from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


class TrainingLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.plot_path = self.run_dir / "training.png"
        self._header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self.start = time.time()

    def gpu_gb(self) -> tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved

    def live(self, **metrics: Any) -> None:
        alloc, reserved = self.gpu_gb()
        step = metrics.get("step", 0)
        layer = metrics.get("layer", "-")
        loss = metrics.get("loss")
        ppl = math.exp(min(20.0, float(loss))) if loss is not None else float("nan")
        toks = metrics.get("tokens_per_sec", 0.0)
        lr = metrics.get("lr", 0.0)
        phase = metrics.get("phase", "train")
        msg = (
            f"\r[{phase:8}] step {step:6} | layer {str(layer):>3} | "
            f"loss {float(loss) if loss is not None else float('nan'):.4f} | ppl {ppl:.2f} | "
            f"lr {float(lr):.2e} | {float(toks):8.0f} tok/s | GPU {alloc:.1f}/{reserved:.1f} GiB"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    def newline(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    def record(self, row: dict[str, Any]) -> None:
        row = dict(row)
        alloc, reserved = self.gpu_gb()
        row.setdefault("wall_seconds", time.time() - self.start)
        row.setdefault("gpu_allocated_gb", alloc)
        row.setdefault("gpu_reserved_gb", reserved)
        fieldnames = list(row.keys())
        if self._header_written:
            with self.csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                existing = next(reader, [])
            if existing:
                for key in existing:
                    row.setdefault(key, "")
                for key in row:
                    if key not in existing:
                        self._rewrite_with_new_columns(existing + [k for k in row if k not in existing])
                        existing = existing + [k for k in row if k not in existing]
                fieldnames = existing
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())

    def _rewrite_with_new_columns(self, columns: list[str]) -> None:
        rows: list[dict[str, str]] = []
        if self.csv_path.exists() and self.csv_path.stat().st_size:
            with self.csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        tmp = self.csv_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in columns})
        tmp.replace(self.csv_path)

    def plot(self) -> None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return
        import matplotlib.pyplot as plt

        steps, losses = [], []
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    steps.append(int(float(row["step"])))
                    losses.append(float(row["loss"]))
                except (KeyError, ValueError, TypeError):
                    continue
        if not steps:
            return
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(steps, losses)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Training loss")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)


def print_kv(title: str, values: dict[str, Any]) -> None:
    print(title)
    for key, value in values.items():
        print(f"  {key}: {value}")
