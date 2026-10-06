from __future__ import annotations

import torch

from config import ModelConfig
from model import ContinualCellTransformer


def tiny_model() -> ContinualCellTransformer:
    config = ModelConfig(
        vocab_size=300,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=16,
        min_depth=2,
        max_depth=6,
        max_cells=16,
        initial_active_cells=4,
        max_micro_neurons=4,
        initial_micro_neurons=2,
    )
    return ContinualCellTransformer(config)


def check_forward_backward() -> None:
    model = tiny_model().train()
    inputs = torch.randint(0, 300, (2, 12))
    labels = torch.randint(0, 300, (2, 12))
    result = model(inputs, labels=labels, adaptive_inference=False)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    model.update_growth_signals()
    model.mask_cell_gradients(0.01)


def check_cumulative_halting() -> None:
    model = tiny_model().eval()
    with torch.no_grad():
        model.halt_head.weight.zero_()
        model.halt_head.bias.fill_(2.2)
        inputs = torch.randint(0, 300, (1, 12))
        result = model(inputs, adaptive_inference=True)
    assert result["used_depth"] == 2, result["used_depth"]
    final_mass = float(result["halt_cumulative"][0, -1])
    assert abs(final_mass - 1.0) < 1e-6, final_mass


def check_zero_impact_micro_growth() -> None:
    model = tiny_model().eval()
    inputs = torch.randint(0, 300, (1, 12))
    with torch.no_grad():
        before = model(inputs, adaptive_inference=False)["logits"].clone()
        model.consolidate_active_cells()
        grown = model.grow_micro_neurons(1, [0])
        after = model(inputs, adaptive_inference=False)["logits"]
    assert grown, grown
    drift = float((after - before).abs().max())
    assert drift <= 1e-6, drift
    summary = model.pool_summary()
    assert summary["plastic_micro"] == 1, summary


def check_metrics_are_distinct() -> None:
    model = tiny_model().eval()
    with torch.no_grad():
        result = model(
            torch.randint(0, 300, (2, 12)),
            adaptive_inference=False,
        )
    assert 0.0 <= float(result["route_coverage"]) <= 1.0
    assert 0.0 <= float(result["active_fraction"]) <= 1.0
    assert 0.0 <= float(result["micro_utilization"]) <= 1.0
    assert 0.0 <= float(result["micro_capacity_fraction"]) <= 1.0


def main() -> None:
    torch.manual_seed(17)
    check_forward_backward()
    check_cumulative_halting()
    check_zero_impact_micro_growth()
    check_metrics_are_distinct()
    print("All V7 debug checks passed.")


if __name__ == "__main__":
    main()
