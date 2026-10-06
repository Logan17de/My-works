from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


@dataclass
class PoolStats:
    route_coverage: torch.Tensor
    active_fraction: torch.Tensor
    mean_active_cells: torch.Tensor
    effective_cell_fraction: torch.Tensor
    routing_loss: torch.Tensor
    plastic_activity: torch.Tensor
    plastic_output_rms: torch.Tensor
    micro_utilization: torch.Tensor
    micro_capacity_fraction: torch.Tensor
    active_count: int
    consolidated_count: int
    top_cell_ids: list[int]
    seed: torch.Tensor


class SharedExpandableCellPool(nn.Module):
    """Shared threshold-routed cells with growable internal micro-neurons."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        cells, dim, micro = (
            config.max_cells,
            config.d_model,
            config.max_micro_neurons,
        )
        self.max_cells = cells
        self.d_model = dim
        self.max_micro = micro
        self.initial_micro = config.initial_micro_neurons
        self.recurrent_steps = config.recurrent_steps
        self.fan_in = config.recurrent_fan_in
        self.threshold_temperature = config.threshold_temperature
        self.new_cell_threshold = config.new_cell_threshold
        self.target_active_fraction = config.target_active_fraction
        self.maturity_steps = max(1, config.cell_maturity_steps)
        self.micro_hidden_scale = config.micro_hidden_scale

        self.keys = nn.Parameter(torch.randn(cells, dim) * 0.02)
        self.read_vectors = nn.Parameter(torch.randn(cells, dim) * 0.02)
        self.thresholds = nn.Parameter(
            torch.full((cells,), float(config.initial_threshold))
        )
        self.bias = nn.Parameter(torch.zeros(cells))
        self.recurrent = nn.Parameter(torch.zeros(cells, cells))

        self.micro_in = nn.Parameter(torch.randn(cells, micro) * 0.02)
        self.micro_out = nn.Parameter(torch.randn(cells, micro, dim) * 0.01)

        active = torch.zeros(cells, dtype=torch.bool)
        active[: config.initial_active_cells] = True
        self.register_buffer("active_mask", active)
        self.register_buffer("consolidated_mask", torch.zeros(cells, dtype=torch.bool))
        self.register_buffer("maturity", torch.zeros(cells))
        self.register_buffer("edge_mask", torch.zeros(cells, cells, dtype=torch.bool))

        micro_active = torch.zeros(cells, micro, dtype=torch.bool)
        micro_active[
            : config.initial_active_cells,
            : config.initial_micro_neurons,
        ] = True
        self.register_buffer("micro_active_mask", micro_active)
        self.register_buffer(
            "micro_consolidated_mask",
            torch.zeros(cells, micro, dtype=torch.bool),
        )

        self.register_buffer("usage_ema", torch.zeros(cells))
        self.register_buffer("relevance_ema", torch.zeros(cells))
        self.register_buffer("contribution_ema", torch.zeros(cells))
        self.register_buffer("micro_utilization_ema", torch.zeros(cells))
        self.register_buffer("gradient_pressure_ema", torch.zeros(cells))
        self.register_buffer("growth_score_ema", torch.zeros(cells))

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
            sources = indices[
                torch.randperm(indices.numel(), device=indices.device)[:fan]
            ]
            self.edge_mask[sources, target] = True
            self.recurrent[sources, target].normal_(0.0, 0.02)
            self.edge_mask[target, target] = True
            self.recurrent[target, target] = 0.01

    @staticmethod
    def _participation_ratio(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Effective used components divided by available active components."""
        values = values * mask
        numerator = values.sum(dim=-1).square()
        denominator = values.square().sum(dim=-1).clamp_min(1e-12)
        effective = numerator / denominator
        available = mask.sum(dim=-1).clamp_min(1.0)
        utilization = effective / available
        has_signal = values.sum(dim=-1) > 1e-8
        return torch.where(has_signal, utilization, torch.zeros_like(utilization))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PoolStats]:
        indices = torch.nonzero(self.active_mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            zero = torch.zeros_like(x)
            scalar = x.new_tensor(0.0)
            return zero, PoolStats(
                route_coverage=scalar,
                active_fraction=scalar,
                mean_active_cells=scalar,
                effective_cell_fraction=scalar,
                routing_loss=scalar,
                plastic_activity=scalar,
                plastic_output_rms=scalar,
                micro_utilization=scalar,
                micro_capacity_fraction=scalar,
                active_count=0,
                consolidated_count=0,
                top_cell_ids=[],
                seed=x.detach().mean((0, 1)),
            )

        normalized_x = F.normalize(x, dim=-1)
        keys = F.normalize(self.keys[indices], dim=-1)
        reads = F.normalize(self.read_vectors[indices], dim=-1)
        scores = torch.einsum("btd,cd->btc", normalized_x, keys) + self.bias[indices]
        soft = torch.sigmoid(
            (scores - self.thresholds[indices]) / self.threshold_temperature
        )
        hard = (scores > self.thresholds[indices]).to(scores.dtype)
        gates = hard + soft - soft.detach()

        drive = gates * F.silu(torch.einsum("btd,cd->btc", x, reads))
        recurrent = (
            self.recurrent[indices][:, indices]
            * self.edge_mask[indices][:, indices].to(x.dtype)
        )
        activity = drive
        for _ in range(self.recurrent_steps):
            activity = F.silu(
                drive + torch.einsum("btc,cd->btd", activity, recurrent)
            )

        micro_mask = self.micro_active_mask[indices].to(x.dtype)
        micro_hidden = F.silu(
            activity.unsqueeze(-1)
            * self.micro_in[indices][None, None, :, :]
            * self.micro_hidden_scale
        )
        micro_hidden = micro_hidden * micro_mask[None, None, :, :]

        cell_outputs = torch.einsum(
            "btcm,cmd->btcd",
            micro_hidden,
            self.micro_out[indices],
        )
        # Keep normalization independent of growth. Newly activated slots start
        # with zero output, so insertion leaves all existing computation and
        # logits exactly unchanged instead of rescaling the cell by a new
        # sqrt(active_micro_count) denominator.
        per_cell_denom = x.new_full(
            (indices.numel(),),
            float(max(1, self.initial_micro)) ** 0.5,
        )
        cell_outputs = cell_outputs / per_cell_denom[None, None, :, None]
        output = cell_outputs.sum(dim=2)

        plastic_slots = (
            self.micro_active_mask[indices]
            & ~self.micro_consolidated_mask[indices]
        ).to(x.dtype)
        if plastic_slots.any():
            plastic_hidden = micro_hidden * plastic_slots[None, None, :, :]
            plastic_cell_outputs = torch.einsum(
                "btcm,cmd->btcd",
                plastic_hidden,
                self.micro_out[indices],
            ) / per_cell_denom[None, None, :, None]
            plastic_output = plastic_cell_outputs.sum(dim=2)
            plastic_cells = (plastic_slots.sum(dim=-1) > 0).to(x.dtype)
            plastic_activity = (
                hard * plastic_cells[None, None, :]
            ).sum() / plastic_cells.sum().clamp_min(1.0) / hard.shape[0] / hard.shape[1]
            plastic_output_rms = (
                plastic_output.square().mean() + 1e-12
            ).sqrt()
        else:
            plastic_activity = x.new_tensor(0.0)
            plastic_output_rms = x.new_tensor(0.0)

        micro_energy = micro_hidden.detach().abs().mean(dim=(0, 1))
        per_cell_micro_utilization = self._participation_ratio(
            micro_energy,
            micro_mask,
        )

        with torch.no_grad():
            usage = hard.mean((0, 1))
            relevance = (hard * soft).mean((0, 1))
            contribution = cell_outputs.square().mean((0, 1, 3)).sqrt()

            global_usage = torch.zeros_like(self.usage_ema)
            global_relevance = torch.zeros_like(self.relevance_ema)
            global_contribution = torch.zeros_like(self.contribution_ema)
            global_micro_utilization = torch.zeros_like(self.micro_utilization_ema)
            global_usage[indices] = usage
            global_relevance[indices] = relevance
            global_contribution[indices] = contribution
            global_micro_utilization[indices] = per_cell_micro_utilization

            self.usage_ema.mul_(0.99).add_(global_usage, alpha=0.01)
            self.relevance_ema.mul_(0.99).add_(global_relevance, alpha=0.01)
            self.contribution_ema.mul_(0.99).add_(global_contribution, alpha=0.01)
            self.micro_utilization_ema.mul_(0.95).add_(
                global_micro_utilization,
                alpha=0.05,
            )

            aggregate = hard.sum((0, 1))
            top = indices[
                aggregate.topk(min(16, aggregate.numel())).indices
            ].tolist()

        route_coverage = (hard.sum(dim=-1) > 0).float().mean()
        active_fraction = hard.mean()
        mean_active_cells = hard.sum(dim=-1).float().mean()

        soft_sum = soft.sum(dim=-1)
        effective_cells = soft_sum.square() / soft.square().sum(dim=-1).clamp_min(1e-12)
        effective_cell_fraction = (
            effective_cells / max(1, indices.numel())
        ).mean()

        per_token_load = soft.mean(dim=-1)
        load_loss = (
            per_token_load - self.target_active_fraction
        ).square().mean()
        confidence_loss = (soft * (1.0 - soft)).mean()
        routing_loss = load_loss + 0.10 * confidence_loss

        micro_capacity_fraction = micro_mask.mean()
        micro_utilization = per_cell_micro_utilization.mean()

        return output, PoolStats(
            route_coverage=route_coverage,
            active_fraction=active_fraction,
            mean_active_cells=mean_active_cells,
            effective_cell_fraction=effective_cell_fraction,
            routing_loss=routing_loss,
            plastic_activity=plastic_activity,
            plastic_output_rms=plastic_output_rms,
            micro_utilization=micro_utilization,
            micro_capacity_fraction=micro_capacity_fraction,
            active_count=self.active_count,
            consolidated_count=self.consolidated_count,
            top_cell_ids=top,
            seed=x.detach().mean((0, 1)),
        )

    @torch.no_grad()
    def update_growth_signals(self) -> None:
        pressure = torch.zeros_like(self.gradient_pressure_ema)
        if self.micro_out.grad is not None:
            pressure.add_(self.micro_out.grad.square().mean(dim=(1, 2)).sqrt())
        if self.micro_in.grad is not None:
            pressure.add_(self.micro_in.grad.square().mean(dim=1).sqrt())
        if self.keys.grad is not None:
            pressure.add_(self.keys.grad.square().mean(dim=1).sqrt())
        if self.read_vectors.grad is not None:
            pressure.add_(self.read_vectors.grad.square().mean(dim=1).sqrt())
        pressure.mul_(self.active_mask.to(pressure.dtype))
        self.gradient_pressure_ema.mul_(0.95).add_(pressure, alpha=0.05)

        relevance = self._normalize_active(self.relevance_ema)
        contribution = self._normalize_active(self.contribution_ema)
        gradient = self._normalize_active(self.gradient_pressure_ema)
        utilization = self.micro_utilization_ema.clamp(0.0, 1.0)

        raw_score = (
            relevance
            * utilization
            * gradient
            * (0.5 + 0.5 * contribution)
        )
        raw_score.mul_(self.active_mask.to(raw_score.dtype))
        self.growth_score_ema.mul_(0.9).add_(raw_score, alpha=0.1)

    @torch.no_grad()
    def _normalize_active(self, values: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(values)
        active_values = values[self.active_mask]
        if active_values.numel() == 0:
            return result
        result[self.active_mask] = active_values / active_values.max().clamp_min(1e-8)
        return result

    @torch.no_grad()
    def micro_growth_candidates(
        self,
        max_cells: int = 4,
        minimum_score: float = 0.05,
        minimum_saturation: float = 0.75,
    ) -> list[int]:
        has_capacity = self.micro_active_mask.sum(dim=1) < self.max_micro
        eligible = (
            self.active_mask
            & has_capacity
            & (self.micro_utilization_ema >= minimum_saturation)
            & (self.growth_score_ema >= minimum_score)
        )
        ids = torch.nonzero(eligible, as_tuple=False).flatten()
        if ids.numel() == 0:
            return []
        ranked = ids[self.growth_score_ema[ids].argsort(descending=True)]
        return ranked[: min(max_cells, ranked.numel())].tolist()

    @torch.no_grad()
    def growth_diagnostics(self, limit: int = 8) -> list[dict[str, float | int]]:
        ids = torch.nonzero(self.active_mask, as_tuple=False).flatten()
        if ids.numel() == 0:
            return []
        ranked = ids[self.growth_score_ema[ids].argsort(descending=True)]
        rows: list[dict[str, float | int]] = []
        for cell_id in ranked[: min(limit, ranked.numel())].tolist():
            rows.append(
                {
                    "cell": int(cell_id),
                    "score": float(self.growth_score_ema[cell_id]),
                    "relevance": float(self.relevance_ema[cell_id]),
                    "utilization": float(self.micro_utilization_ema[cell_id]),
                    "capacity": float(
                        self.micro_active_mask[cell_id].float().mean()
                    ),
                    "gradient": float(self.gradient_pressure_ema[cell_id]),
                    "contribution": float(self.contribution_ema[cell_id]),
                }
            )
        return rows

    @torch.no_grad()
    def allocate_cells(
        self,
        count: int,
        seed: torch.Tensor | None = None,
    ) -> list[int]:
        dormant = torch.nonzero(~self.active_mask, as_tuple=False).flatten()
        chosen = dormant[: min(count, dormant.numel())]
        if chosen.numel() == 0:
            return []
        if seed is None:
            seed = torch.randn(self.d_model, device=self.keys.device)
        seed = F.normalize(seed.to(self.keys), dim=-1)

        established = torch.nonzero(self.active_mask, as_tuple=False).flatten()
        parent_count = min(self.fan_in, established.numel())
        parents = established[
            self.usage_ema[established].topk(parent_count).indices
        ]

        for row in chosen.tolist():
            initial = F.normalize(
                seed + 0.05 * torch.randn_like(seed),
                dim=-1,
            )
            self.keys[row].copy_(initial)
            self.read_vectors[row].copy_(initial)
            self.thresholds[row] = self.new_cell_threshold
            self.bias[row] = 0.0
            self.micro_in[row].normal_(0.0, 0.02)
            self.micro_out[row].zero_()
            self.micro_active_mask[row].zero_()
            self.micro_active_mask[row, : self.initial_micro] = True
            self.micro_consolidated_mask[row].zero_()
            self.active_mask[row] = True
            self.consolidated_mask[row] = False
            self.maturity[row] = 0.0
            for buffer in (
                self.usage_ema,
                self.relevance_ema,
                self.contribution_ema,
                self.micro_utilization_ema,
                self.gradient_pressure_ema,
                self.growth_score_ema,
            ):
                buffer[row] = 0.0
            self.edge_mask[parents, row] = True
            self.recurrent[parents, row].normal_(0.0, 0.02)
            self.edge_mask[row, row] = True
            self.recurrent[row, row] = 0.01
        return chosen.tolist()

    @torch.no_grad()
    def grow_micro_neurons(
        self,
        count: int,
        cell_ids: list[int] | None = None,
    ) -> dict[int, list[int]]:
        if cell_ids is None:
            cell_ids = self.micro_growth_candidates()
        grown: dict[int, list[int]] = {}
        for cell_id in cell_ids:
            free = torch.nonzero(
                ~self.micro_active_mask[cell_id],
                as_tuple=False,
            ).flatten()
            selected = free[: min(count, free.numel())]
            if selected.numel() == 0:
                continue
            self.micro_in[cell_id, selected].normal_(0.0, 0.02)
            self.micro_out[cell_id, selected].zero_()
            self.micro_active_mask[cell_id, selected] = True
            self.micro_consolidated_mask[cell_id, selected] = False
            grown[int(cell_id)] = selected.tolist()
        return grown

    def mask_gradients(self, consolidated_scale: float) -> None:
        cell_scale = self.active_mask.to(self.keys.dtype) * torch.where(
            self.consolidated_mask,
            torch.full_like(self.maturity, consolidated_scale),
            torch.ones_like(self.maturity),
        )
        for parameter in (self.keys, self.read_vectors):
            if parameter.grad is not None:
                parameter.grad.mul_(cell_scale[:, None])
        for parameter in (self.thresholds, self.bias):
            if parameter.grad is not None:
                parameter.grad.mul_(cell_scale)

        slot_scale = self.micro_active_mask.to(self.keys.dtype) * torch.where(
            self.micro_consolidated_mask,
            torch.full_like(
                self.micro_active_mask,
                consolidated_scale,
                dtype=self.keys.dtype,
            ),
            torch.ones_like(self.micro_active_mask, dtype=self.keys.dtype),
        )
        if self.micro_in.grad is not None:
            self.micro_in.grad.mul_(slot_scale)
        if self.micro_out.grad is not None:
            self.micro_out.grad.mul_(slot_scale[:, :, None])
        if self.recurrent.grad is not None:
            self.recurrent.grad.mul_(
                cell_scale[None, :] * self.edge_mask.to(cell_scale.dtype)
            )

    @torch.no_grad()
    def advance_maturity(self) -> None:
        self.maturity[self.active_mask].add_(
            1.0 / self.maturity_steps
        ).clamp_(max=1.0)

    @torch.no_grad()
    def consolidate(self) -> None:
        self.consolidated_mask[self.active_mask] = True
        self.micro_consolidated_mask[
            self.micro_active_mask
        ] = True
        self.maturity[self.active_mask] = 1.0
