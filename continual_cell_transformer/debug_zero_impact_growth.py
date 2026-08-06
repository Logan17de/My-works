from __future__ import annotations

import torch

from config import ModelConfig
from growth_safety import zero_impact
from model import ContinualCellTransformer
from zero_impact_growth import apply_zero_impact_growth_patch


def main() -> None:
    torch.manual_seed(17)
    apply_zero_impact_growth_patch()

    model = ContinualCellTransformer(
        ModelConfig(
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
    ).eval()

    # Force strong, nonzero cell activity so the test cannot pass merely
    # because random routing happened to leave the grown cell inactive.
    with torch.no_grad():
        pool = model.block.cells
        pool.thresholds[:4].fill_(-10.0)
        pool.bias[:4].fill_(10.0)
        pool.micro_in[0, :2].fill_(1.0)
        pool.micro_out[0, :2].normal_(0.0, 0.5)
        model.consolidate_active_cells()
        inputs = torch.randint(0, 300, (2, 12))

    def mutation() -> dict[int, list[int]]:
        grown = model.grow_micro_neurons(1, [0])
        assert grown == {0: [2]}, grown
        assert torch.count_nonzero(pool.micro_out[0, 2]).item() == 0, (
            "new micro output was not written as zero"
        )
        assert not bool(pool.micro_consolidated_mask[0, 2])
        return grown

    grown = zero_impact(
        model,
        inputs,
        mutation,
        "forced-active micro growth",
    )

    print("Zero-impact growth regression passed.")
    print("grown:", grown)


if __name__ == "__main__":
    main()
