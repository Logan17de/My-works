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
        self.routing_plot_path = self.run_dir / "routing.png"
        self._header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self.start = time.time()

    def gpu_gb(self) -> tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved

    @staticmethod
    def _expert_text(stats: dict[str, Any] | None, *, summary: bool) -> str:
        if not stats:
            return ""
        if summary:
            active = float(stats.get("expert_active_mean", 0.0))
            active_min = int(float(stats.get("expert_active_min", 0)))
            gap_mean = 100.0 * float(stats.get("expert_gap_mean", 0.0))
            gap_max = 100.0 * float(stats.get("expert_gap_max", 0.0))
            blocked_layers = int(stats.get("blocked_layers", 0))
            blocked_total = int(stats.get("blocked_experts_total", 0))
            reroute = float(stats.get("reroute_pct", 0.0))
            return (
                f" | E {active:.0f}avg/{active_min}min"
                f" | gap {gap_mean:.1f}/{gap_max:.1f}%"
                f" | block {blocked_layers}L/{blocked_total}E"
                f" | reroute {reroute:.1f}%"
            )
        active = int(stats.get("active", 0))
        total = int(stats.get("total_experts", 0))
        hot_id = int(stats.get("hot_id", -1))
        hot_share = 100.0 * float(stats.get("hot_share", 0.0))
        gap = 100.0 * float(stats.get("gap", 0.0))
        blocked = int(stats.get("blocked", 0))
        reroute = float(stats.get("reroute_pct", 0.0))
        return (
            f" | E {active}/{total}"
            f" | hot e{hot_id}:{hot_share:.1f}%"
            f" | gap {gap:.1f}%"
            f" | block {blocked}"
            f" | reroute {reroute:.1f}%"
        )

    def live(self, **metrics: Any) -> None:
        alloc, reserved = self.gpu_gb()
        step = metrics.get("step", 0)
        layer = metrics.get("layer", "-")
        loss = metrics.get("loss")
        loss_ok = loss is not None and math.isfinite(float(loss))
        loss_text = f"{float(loss):.4f}" if loss_ok else "--"
        ppl_text = f"{math.exp(min(20.0, float(loss))):.2f}" if loss_ok else "--"
        toks = metrics.get("tokens_per_sec")
        toks_text = f"{float(toks):.0f} tok/s" if toks is not None and float(toks) > 0 else "tok/s --"
        lr = metrics.get("lr", 0.0)
        phase = metrics.get("phase", "train")
        expert_stats = metrics.get("expert_stats")
        expert_text = self._expert_text(expert_stats, summary=phase == "train")

        backend = metrics.get("backend")
        backend_text = f" | {backend}" if backend else ""
        cache_capacity = metrics.get("cache_capacity")
        cache_text = f" | cache {int(cache_capacity)}L" if cache_capacity else ""
        loaded_layers = int(metrics.get("loaded_layers", 0) or 0)
        load_gib = float(metrics.get("load_gib", 0.0) or 0.0)
        load_s = float(metrics.get("load_seconds", 0.0) or 0.0)
        load_text = f" | load {loaded_layers}L/{load_gib:.1f}G {load_s:.1f}s" if loaded_layers or load_s > 0 else ""
        compute_s = metrics.get("compute_seconds")
        compute_text = f" | compute {float(compute_s):.1f}s" if compute_s is not None else ""

        msg = (
            f"\r[{phase:8}] step {step:6} | L {str(layer):>5} | "
            f"loss {loss_text} | ppl {ppl_text} | lr {float(lr):.2e} | {toks_text} | "
            f"GPU {alloc:.1f}/{reserved:.1f} GiB"
            f"{cache_text}{load_text}{compute_text}{backend_text}{expert_text}"
        )
        sys.stdout.write(msg + "\033[K")
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
                new_keys = [k for k in row if k not in existing]
                if new_keys:
                    self._rewrite_with_new_columns(existing + new_keys)
                    existing = existing + new_keys
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

    @staticmethod
    def _as_float(row: dict[str, str], key: str) -> float | None:
        try:
            value = row.get(key, "")
            return float(value) if value != "" else None
        except (TypeError, ValueError):
            return None

    def plot(self) -> None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return
        import matplotlib.pyplot as plt

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        steps, losses = [], []
        for row in rows:
            try:
                steps.append(int(float(row["step"])))
                losses.append(float(row["loss"]))
            except (KeyError, ValueError, TypeError):
                continue
        if steps:
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

        r_steps, gap_mean, gap_max, reroute, active = [], [], [], [], []
        for row in rows:
            try:
                step = int(float(row["step"]))
            except (KeyError, ValueError, TypeError):
                continue
            gm = self._as_float(row, "expert_gap_mean")
            gx = self._as_float(row, "expert_gap_max")
            rr = self._as_float(row, "reroute_pct")
            ac = self._as_float(row, "expert_active_mean")
            if gm is None or gx is None or rr is None or ac is None:
                continue
            r_steps.append(step)
            gap_mean.append(gm * 100.0)
            gap_max.append(gx * 100.0)
            reroute.append(rr)
            active.append(ac)
        if r_steps:
            fig = plt.figure(figsize=(11, 6))
            ax = fig.add_subplot(111)
            ax.plot(r_steps, gap_mean, label="routing gap mean %")
            ax.plot(r_steps, gap_max, label="routing gap max %")
            ax.plot(r_steps, reroute, label="threshold reroute %")
            ax.set_xlabel("step")
            ax.set_ylabel("percent")
            ax.set_title("MoE routing experiment")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
            ax2 = ax.twinx()
            ax2.plot(r_steps, active, linestyle="--", label="active experts mean")
            ax2.set_ylabel("active experts / layer")
            fig.tight_layout()
            fig.savefig(self.routing_plot_path, dpi=150)
            plt.close(fig)


def print_kv(title: str, values: dict[str, Any]) -> None:
    print(title)
    for key, value in values.items():
        print(f"  {key}: {value}")
