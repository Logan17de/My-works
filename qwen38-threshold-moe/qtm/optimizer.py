from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .experts import ExpertStore, GradientPacket, _mmap_tensor


class FactoredAdafactorCPU:
    """Momentum-free factored Adafactor for stacked expert matrices.

    States are FP32 and mmap-backed by default. The update is chunked over experts
    so a 500-expert layer does not require a full FP32 copy of its BF16 master.
    """

    def __init__(
        self,
        root: str | Path,
        store: ExpertStore,
        *,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        expert_chunk: int = 8,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.clip_threshold = float(clip_threshold)
        self.weight_decay = float(weight_decay)
        self.expert_chunk = max(1, int(expert_chunk))
        self.step_num = 0
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            manifest.write_text(
                json.dumps({"format": "qtm-adafactor-v1", "dtype": "float32"}, indent=2), encoding="utf-8"
            )

    def _state_paths(self, layer_idx: int, name: str) -> tuple[Path, Path]:
        base = self.root / f"layer_{layer_idx:03d}"
        return base / f"{name}.row.f32", base / f"{name}.col.f32"

    def _state(self, layer_idx: int, name: str, shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        row_path, col_path = self._state_paths(layer_idx, name)
        row = _mmap_tensor(row_path, (shape[0], shape[1]), torch.float32)
        col = _mmap_tensor(col_path, (shape[0], shape[2]), torch.float32)
        return row, col

    @torch.no_grad()
    def _update_matrix(
        self,
        param: torch.Tensor,
        grad: GradientPacket,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        lr: float,
    ) -> None:
        beta2 = self.beta2
        one_minus = 1.0 - beta2
        for start in range(0, param.shape[0], self.expert_chunk):
            end = min(start + self.expert_chunk, param.shape[0])
            g = grad.chunk(start, end, dtype=torch.float32)
            g2 = g.square().add_(self.eps)
            row_mean = g2.mean(dim=-1)
            col_mean = g2.mean(dim=-2)
            row = row_state[start:end]
            col = col_state[start:end]
            row.mul_(beta2).add_(row_mean, alpha=one_minus)
            col.mul_(beta2).add_(col_mean, alpha=one_minus)

            r_factor = (row / row.mean(dim=-1, keepdim=True).clamp_min(self.eps)).rsqrt_()
            c_factor = col.clamp_min(self.eps).rsqrt()
            update = g * r_factor.unsqueeze(-1) * c_factor.unsqueeze(-2)
            rms = update.square().mean(dim=(-2, -1), keepdim=True).sqrt_()
            update.div_(torch.maximum(rms / self.clip_threshold, torch.ones_like(rms)))

            p32 = param[start:end].float()
            if self.weight_decay:
                p32.mul_(1.0 - lr * self.weight_decay)
            p32.add_(update, alpha=-lr)
            param[start:end].copy_(p32.to(param.dtype))
            del g, g2, row_mean, col_mean, r_factor, c_factor, update, rms, p32

    @torch.no_grad()
    def step_layer(
        self,
        layer_idx: int,
        gate_up_grad: GradientPacket,
        down_grad: GradientPacket,
        *,
        lr: float,
    ) -> None:
        layer = self.store.get_layer(layer_idx)
        gu_row, gu_col = self._state(layer_idx, "gate_up", tuple(layer.gate_up.shape))
        dn_row, dn_col = self._state(layer_idx, "down", tuple(layer.down.shape))
        self._update_matrix(layer.gate_up, gate_up_grad, gu_row, gu_col, lr)
        self._update_matrix(layer.down, down_grad, dn_row, dn_col, lr)
        self.step_num += 1

    def state_dict(self) -> dict:
        return {"step_num": self.step_num}

    def load_state_dict(self, state: dict) -> None:
        self.step_num = int(state.get("step_num", 0))


def cosine_lr(step: int, *, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if max_steps <= warmup_steps:
        return base_lr
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(max_steps - warmup_steps)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
