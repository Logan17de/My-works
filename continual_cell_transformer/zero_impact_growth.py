from __future__ import annotations

"""Zero-impact micro-neuron growth for the V7 shared cell pool.

The pool normalizes each cell by ``sqrt(active_micro_count)``. Activating a
zero-output slot therefore still changes the denominator and rescales every
existing slot. This patch compensates the existing output vectors before the
mask grows, making the represented function unchanged at insertion time.
"""

import math

import torch

from cells import SharedExpandableCellPool


@torch.no_grad()
def _grow_micro_neurons_zero_impact(
    self: SharedExpandableCellPool,
    count: int,
    cell_ids: list[int] | None = None,
) -> dict[int, list[int]]:
    if cell_ids is None:
        cell_ids = self.micro_growth_candidates()

    grown: dict[int, list[int]] = {}
    for cell_id in cell_ids:
        old_mask = self.micro_active_mask[cell_id].clone()
        free = torch.nonzero(~old_mask, as_tuple=False).flatten()
        selected = free[: min(count, free.numel())]
        if selected.numel() == 0:
            continue

        old_count = int(old_mask.sum().item())
        new_count = old_count + int(selected.numel())

        # Forward uses sum(micro_outputs) / sqrt(active_count). Compensating
        # existing output vectors by sqrt(new_count / old_count) exactly
        # preserves their aggregate contribution when the denominator grows.
        if old_count > 0:
            scale = math.sqrt(new_count / old_count)
            self.micro_out[cell_id, old_mask] = (
                self.micro_out[cell_id, old_mask] * scale
            )

        self.micro_in[cell_id, selected].normal_(0.0, 0.02)
        self.micro_out[cell_id, selected].zero_()
        self.micro_active_mask[cell_id, selected] = True
        self.micro_consolidated_mask[cell_id, selected] = False
        grown[int(cell_id)] = selected.tolist()

    return grown


def apply_zero_impact_growth_patch() -> None:
    if getattr(
        SharedExpandableCellPool,
        "_zero_impact_growth_patch_v1",
        False,
    ):
        return

    SharedExpandableCellPool.grow_micro_neurons = (
        _grow_micro_neurons_zero_impact
    )
    SharedExpandableCellPool._zero_impact_growth_patch_v1 = True


apply_zero_impact_growth_patch()
