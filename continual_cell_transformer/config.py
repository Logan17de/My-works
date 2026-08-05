from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1
    rope_base: float = 10_000.0

    # Adaptive recurrent Transformer depth.
    min_depth: int = 2
    max_depth: int = 8
    halt_threshold: float = 0.90
    halt_temperature: float = 1.0
    halt_bias: float = -1.5

    # One shared threshold-routed cell population.
    max_cells: int = 256
    initial_active_cells: int = 64
    recurrent_steps: int = 2
    recurrent_fan_in: int = 8
    cell_residual_scale: float = 0.10
    threshold_temperature: float = 0.08
    initial_threshold: float = 0.10
    new_cell_threshold: float = 0.00
    cell_maturity_steps: int = 500

    # Internal growth inside each cell.
    # Every cell has a fixed reserve of micro-neurons, but only a subset is active.
    max_micro_neurons: int = 16
    initial_micro_neurons: int = 4
    micro_hidden_scale: float = 1.0

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= self.min_depth <= self.max_depth:
            raise ValueError("Require 1 <= min_depth <= max_depth")
        if not 0 < self.halt_threshold <= 1:
            raise ValueError("halt_threshold must be in (0, 1]")
        if not 0 < self.initial_active_cells <= self.max_cells:
            raise ValueError("initial_active_cells must be in [1, max_cells]")
        if not 1 <= self.initial_micro_neurons <= self.max_micro_neurons:
            raise ValueError("initial_micro_neurons must be in [1, max_micro_neurons]")
        if self.recurrent_fan_in <= 0:
            raise ValueError("recurrent_fan_in must be positive")
        if self.threshold_temperature <= 0:
            raise ValueError("threshold_temperature must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(**dict(data))


@dataclass
class TrainConfig:
    batch_size: int = 16
    steps: int = 2_000
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 20
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    embedding_lr: float = 1e-6
    attention_lr: float = 2e-6
    ffn_lr: float = 5e-6
    cell_lr: float = 5e-4
    halt_lr: float = 1e-4
    other_lr: float = 1e-6
    consolidated_cell_scale: float = 0.01

    depth_penalty: float = 0.01
    retention_replay_weight: float = 0.0
    plastic_sparsity_weight: float = 0.0
    retention_output_penalty: float = 0.0

    # Autonomous outer growth.
    enable_cell_growth: bool = False
    cell_growth_warmup: int = 100
    cell_growth_patience: int = 40
    cell_growth_count: int = 2
    cell_growth_loss_floor: float = 1.0
    cell_growth_coverage_floor: float = 0.75
    max_cell_growth_events: int = 4
    growth_cooldown: int = 100

    # Autonomous internal growth.
    enable_micro_growth: bool = False
    micro_growth_patience: int = 30
    micro_growth_count: int = 1
    micro_growth_saturation: float = 0.80
    max_micro_growth_events: int = 8

    seed: int = 17
