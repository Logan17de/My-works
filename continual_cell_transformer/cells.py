from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


@dataclass
class PoolStats:
    coverage: torch.Tensor
    mean_active: torch.Tensor
    plastic_activity: torch.Tensor
    plastic_output_rms: torch.Tensor
    micro_saturation: torch.Tensor
    active_count: int
    consolidated_count: int
    top_cell_ids: list[int]
    seed: torch.Tensor


class SharedExpandableCellPool(nn.Module):
    """One shared population with threshold routing and growable internal micro-neurons."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c, d, m = config.max_cells, config.d_model, config.max_micro_neurons
        self.max_cells = c
        self.d_model = d
        self.max_micro = m
        self.initial_micro = config.initial_micro_neurons
        self.recurrent_steps = config.recurrent_steps
        self.fan_in = config.recurrent_fan_in
        self.threshold_temperature = config.threshold_temperature
        self.new_cell_threshold = config.new_cell_threshold
        self.maturity_steps = max(1, config.cell_maturity_steps)
        self.micro_hidden_scale = config.micro_hidden_scale

        self.keys = nn.Parameter(torch.randn(c, d) * 0.02)
        self.read_vectors = nn.Parameter(torch.randn(c, d) * 0.02)
        self.thresholds = nn.Parameter(torch.full((c,), config.initial_threshold))
        self.bias = nn.Parameter(torch.zeros(c))
        self.recurrent = nn.Parameter(torch.zeros(c, c))

        self.micro_in = nn.Parameter(torch.randn(c, m) * 0.02)
        self.micro_out = nn.Parameter(torch.randn(c, m, d) * 0.01)
        micro_mask = torch.zeros(c, m, dtype=torch.bool)
        micro_mask[: config.initial_active_cells, : config.initial_micro_neurons] = True
        self.register_buffer("micro_active_mask", micro_mask)

        active = torch.zeros(c, dtype=torch.bool)
        active[: config.initial_active_cells] = True
        self.register_buffer("active_mask", active)
        self.register_buffer("consolidated_mask", torch.zeros(c, dtype=torch.bool))
        self.register_buffer("maturity", torch.zeros(c))
        self.register_buffer("usage_ema", torch.zeros(c))
        self.register_buffer("edge_mask", torch.zeros(c, c, dtype=torch.bool))

        with torch.no_grad():
            self.keys.copy_(F.normalize(self.keys, dim=-1))
            self.read_vectors.copy_(F.normalize(self.read_vectors, dim=-1))
            self._init_edges(torch.arange(config.initial_active_cells))

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def consolidated_count(self) -> int:
        return int((self.active_mask & self.consolidated_mask).sum().item())

    def _init_edges(self, indices: torch.Tensor) -> None:
        for target in indices.tolist():
            fan = min(self.fan_in, indices.numel())
            src = indices[torch.randperm(indices.numel(), device=indices.device)[:fan]]
            self.edge_mask[src, target] = True
            self.recurrent[src, target].normal_(0.0, 0.02)
            self.edge_mask[target, target] = True
            self.recurrent[target, target] = 0.01

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PoolStats]:
        idx = torch.nonzero(self.active_mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            z = torch.zeros_like(x)
            s = x.new_tensor(0.0)
            return z, PoolStats(s, s, s, s, s, 0, 0, [], x.detach().mean((0, 1)))

        nx = F.normalize(x, dim=-1)
        keys = F.normalize(self.keys[idx], dim=-1)
        reads = F.normalize(self.read_vectors[idx], dim=-1)
        scores = torch.einsum("btd,cd->btc", nx, keys) + self.bias[idx]
        soft = torch.sigmoid((scores - self.thresholds[idx]) / self.threshold_temperature)
        hard = (scores > self.thresholds[idx]).to(scores.dtype)
        gates = hard + soft - soft.detach()

        drive = gates * F.silu(torch.einsum("btd,cd->btc", x, reads))
        rec = self.recurrent[idx][:, idx] * self.edge_mask[idx][:, idx].to(x.dtype)
        activity = drive
        for _ in range(self.recurrent_steps):
            activity = F.silu(drive + torch.einsum("btc,cd->btd", activity, rec))

        micro_mask = self.micro_active_mask[idx].to(x.dtype)
        micro_hidden = F.silu(activity.unsqueeze(-1) * self.micro_in[idx][None, None, :, :])
        micro_hidden = micro_hidden * micro_mask[None, None, :, :]
        output = torch.einsum("btcm,cmd->btd", micro_hidden, self.micro_out[idx])
        denom = micro_mask.sum(dim=-1).clamp_min(1).float().sqrt().mean()
        output = output / denom

        plastic = ~self.consolidated_mask[idx]
        if plastic.any():
            p_hidden = micro_hidden * plastic.to(x.dtype)[None, None, :, None]
            p_out = torch.einsum("btcm,cmd->btd", p_hidden, self.micro_out[idx]) / denom
            plastic_activity = hard[..., plastic].mean()
            plastic_output_rms = (p_out.square().mean() + 1e-12).sqrt()
        else:
            plastic_activity = x.new_tensor(0.0)
            plastic_output_rms = x.new_tensor(0.0)

        with torch.no_grad():
            usage = hard.mean((0, 1))
            global_usage = torch.zeros_like(self.usage_ema)
            global_usage[idx] = usage
            self.usage_ema.mul_(0.99).add_(global_usage, alpha=0.01)
            aggregate = hard.sum((0, 1))
            top = idx[aggregate.topk(min(16, aggregate.numel())).indices].tolist()

        coverage = (hard.sum(dim=-1) > 0).float().mean()
        mean_active = hard.sum(dim=-1).float().mean() / max(1, idx.numel())
        micro_saturation = self.micro_active_mask[idx].float().mean()
        return output, PoolStats(
            coverage=coverage,
            mean_active=mean_active,
            plastic_activity=plastic_activity,
            plastic_output_rms=plastic_output_rms,
            micro_saturation=micro_saturation,
            active_count=self.active_count,
            consolidated_count=self.consolidated_count,
            top_cell_ids=top,
            seed=x.detach().mean((0, 1)),
        )

    @torch.no_grad()
    def allocate_cells(self, count: int, seed: torch.Tensor | None = None) -> list[int]:
        dormant = torch.nonzero(~self.active_mask, as_tuple=False).flatten()
        chosen = dormant[: min(count, dormant.numel())]
        if chosen.numel() == 0:
            return []
        if seed is None:
            seed = torch.randn(self.d_model, device=self.keys.device)
        seed = F.normalize(seed.to(self.keys), dim=-1)
        established = torch.nonzero(self.active_mask, as_tuple=False).flatten()
        parents = established[self.usage_ema[established].topk(min(self.fan_in, established.numel())).indices]
        for row in chosen.tolist():
            init = F.normalize(seed + 0.05 * torch.randn_like(seed), dim=-1)
            self.keys[row].copy_(init)
            self.read_vectors[row].copy_(init)
            self.thresholds[row] = self.new_cell_threshold
            self.bias[row] = 0.0
            self.micro_in[row].normal_(0.0, 0.02)
            self.micro_out[row].zero_()
            self.micro_active_mask[row].zero_()
            self.micro_active_mask[row, : self.initial_micro] = True
            self.active_mask[row] = True
            self.consolidated_mask[row] = False
            self.maturity[row] = 0.0
            self.usage_ema[row] = 0.0
            self.edge_mask[parents, row] = True
            self.recurrent[parents, row].normal_(0.0, 0.02)
            self.edge_mask[row, row] = True
            self.recurrent[row, row] = 0.01
        return chosen.tolist()

    @torch.no_grad()
    def grow_micro_neurons(self, count: int, cell_ids: list[int] | None = None) -> dict[int, list[int]]:
        if cell_ids is None:
            active = torch.nonzero(self.active_mask, as_tuple=False).flatten()
            ranked = active[self.usage_ema[active].argsort(descending=True)]
            cell_ids = ranked[: min(8, ranked.numel())].tolist()
        grown: dict[int, list[int]] = {}
        for cell_id in cell_ids:
            free = torch.nonzero(~self.micro_active_mask[cell_id], as_tuple=False).flatten()
            selected = free[: min(count, free.numel())]
            if selected.numel() == 0:
                continue
            self.micro_in[cell_id, selected].normal_(0.0, 0.02)
            self.micro_out[cell_id, selected].zero_()
            self.micro_active_mask[cell_id, selected] = True
            grown[int(cell_id)] = selected.tolist()
        return grown

    def mask_gradients(self, consolidated_scale: float) -> None:
        row_scale = self.active_mask.to(self.keys.dtype) * torch.where(
            self.consolidated_mask,
            torch.full_like(self.maturity, consolidated_scale),
            torch.ones_like(self.maturity),
        )
        for p in (self.keys, self.read_vectors):
            if p.grad is not None:
                p.grad.mul_(row_scale[:, None])
        for p in (self.thresholds, self.bias):
            if p.grad is not None:
                p.grad.mul_(row_scale)
        if self.micro_in.grad is not None:
            self.micro_in.grad.mul_(row_scale[:, None] * self.micro_active_mask.to(row_scale.dtype))
        if self.micro_out.grad is not None:
            self.micro_out.grad.mul_(row_scale[:, None, None] * self.micro_active_mask[:, :, None].to(row_scale.dtype))
        if self.recurrent.grad is not None:
            self.recurrent.grad.mul_(row_scale[None, :] * self.edge_mask.to(row_scale.dtype))

    @torch.no_grad()
    def advance_maturity(self) -> None:
        self.maturity[self.active_mask].add_(1.0 / self.maturity_steps).clamp_(max=1.0)

    @torch.no_grad()
    def consolidate(self) -> None:
        self.consolidated_mask[self.active_mask] = True
        self.maturity[self.active_mask] = 1.0
