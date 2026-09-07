from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CheckpointManager:
    def __init__(self, run_dir: str | Path, *, keep_last: int = 2) -> None:
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(1, int(keep_last))

    def latest(self) -> Path | None:
        checkpoints = sorted(self.root.glob("step_*"), key=lambda p: int(p.name.split("_")[-1]))
        return checkpoints[-1] if checkpoints else None

    def _snapshot_experts(self, source: Path, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(target)], check=True)
        except Exception:
            shutil.copytree(source, target)

    def save(
        self,
        *,
        step: int,
        cfg: dict[str, Any],
        model: torch.nn.Module,
        layer_optimizers: list[torch.optim.Optimizer],
        embed_optimizer: torch.optim.Optimizer,
        head_optimizer: torch.optim.Optimizer,
        adafactor_state: dict,
        threshold_state: dict,
        data_state: dict,
        expert_store_root: str | Path,
        snapshot_experts: bool,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        final_dir = self.root / f"step_{step:08d}"
        tmp_dir = self.root / f".step_{step:08d}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), tmp_dir / "resident_model.pt")
        torch.save(
            {
                "layer_optimizers": [opt.state_dict() for opt in layer_optimizers],
                "embed_optimizer": embed_optimizer.state_dict(),
                "head_optimizer": head_optimizer.state_dict(),
                "adafactor": adafactor_state,
            },
            tmp_dir / "optimizers.pt",
        )
        torch.save(threshold_state, tmp_dir / "threshold.pt")
        torch.save(data_state, tmp_dir / "data_state.pt")
        torch.save(
            {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            tmp_dir / "rng.pt",
        )
        (tmp_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        metadata = {
            "step": step,
            "expert_store_root": str(Path(expert_store_root).resolve()),
            "expert_snapshot": bool(snapshot_experts),
            **(extra or {}),
        }
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if snapshot_experts:
            self._snapshot_experts(Path(expert_store_root), tmp_dir / "expert_store")
            metadata["expert_store_root"] = "expert_store"
            (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.replace(final_dir)
        self._prune()
        return final_dir

    def _prune(self) -> None:
        checkpoints = sorted(self.root.glob("step_*"), key=lambda p: int(p.name.split("_")[-1]))
        for old in checkpoints[:-self.keep_last]:
            shutil.rmtree(old, ignore_errors=True)

    @staticmethod
    def load_state(checkpoint: str | Path, device: torch.device) -> dict[str, Any]:
        path = Path(checkpoint)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        expert_path = Path(metadata["expert_store_root"])
        if not expert_path.is_absolute():
            expert_path = path / expert_path
        return {
            "path": path,
            "metadata": metadata,
            "expert_store_root": expert_path,
            "model": torch.load(path / "resident_model.pt", map_location=device),
            "optimizers": torch.load(path / "optimizers.pt", map_location=device),
            "threshold": torch.load(path / "threshold.pt", map_location="cpu"),
            "data_state": torch.load(path / "data_state.pt", map_location="cpu"),
            "rng": torch.load(path / "rng.pt", map_location="cpu"),
        }

    @staticmethod
    def restore_rng(state: dict[str, Any]) -> None:
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
