from __future__ import annotations

"""Safe structural growth patches for the shared expandable cell pool.

PyTorch advanced indexing returns a copy for expressions such as
``parameter[row, tensor_indices]``. Calling ``.zero_()`` or ``.normal_()`` on
that result therefore may not modify the original parameter. Structural growth
must use explicit indexed assignment so newly activated micro-neurons really
start with zero output and recurrent edges really receive their initialization.
"""

import torch
import torch.nn.functional as F

from cells import SharedExpandableCellPool


@torch.no_grad()
def _init_edges_safe(
    self: SharedExpandableCellPool,
    indices: torch.Tensor,
) -> None:
    for target in indices.tolist():
        fan = min(self.fan_in, indices.numel())
        sources = indices[
            torch.randperm(indices.numel(), device=indices.device)[:fan]
        ]
        self.edge_mask[sources, target] = True
        self.recurrent[sources, target] = (
            torch.randn(
                sources.numel(),
                device=self.recurrent.device,
                dtype=self.recurrent.dtype,
            )
            * 0.02
        )
        self.edge_mask[target, target] = True
        self.recurrent[target, target] = 0.01


@torch.no_grad()
def _grow_micro_neurons_safe(
    self: SharedExpandableCellPool,
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

        # Explicit assignment is required. ``tensor[row, selected].zero_()``
        # operates on an advanced-indexing copy and can leave random output
        # weights in the real parameter tensor.
        self.micro_in[cell_id, selected] = (
            torch.randn(
                selected.numel(),
                device=self.micro_in.device,
                dtype=self.micro_in.dtype,
            )
            * 0.02
        )
        self.micro_out[cell_id, selected] = 0.0
        self.micro_active_mask[cell_id, selected] = True
        self.micro_consolidated_mask[cell_id, selected] = False
        grown[int(cell_id)] = selected.tolist()

    return grown


@torch.no_grad()
def _allocate_cells_safe(
    self: SharedExpandableCellPool,
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
        self.recurrent[parents, row] = (
            torch.randn(
                parents.numel(),
                device=self.recurrent.device,
                dtype=self.recurrent.dtype,
            )
            * 0.02
        )
        self.edge_mask[row, row] = True
        self.recurrent[row, row] = 0.01

    return chosen.tolist()


def apply_zero_impact_growth_patch() -> None:
    if getattr(
        SharedExpandableCellPool,
        "_safe_structural_growth_patch_v2",
        False,
    ):
        return

    SharedExpandableCellPool._init_edges = _init_edges_safe
    SharedExpandableCellPool.grow_micro_neurons = _grow_micro_neurons_safe
    SharedExpandableCellPool.allocate_cells = _allocate_cells_safe
    SharedExpandableCellPool._safe_structural_growth_patch_v2 = True


apply_zero_impact_growth_patch()
