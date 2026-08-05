from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1
    rope_base: float = 10_000.0

    # Shared threshold-routed recurrent population.
    cell_layers: tuple[int, ...] = (1, 3)
    max_cells: int = 256
    initial_active_cells: int = 64
    recurrent_steps: int = 2
    recurrent_fan_in: int = 8
    cell_residual_scale: float = 0.10
    cell_match_temperature: float = 0.70
    initial_cell_threshold: float = 0.15
    new_cell_threshold: float = 0.05
    cell_output_normalizer: float = 8.0
    cell_maturity_steps: int = 2_000

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0 < self.initial_active_cells <= self.max_cells:
            raise ValueError("initial_active_cells must be in [1, max_cells].")
        if self.recurrent_fan_in <= 0:
            raise ValueError("recurrent_fan_in must be positive.")
        if self.cell_match_temperature <= 0:
            raise ValueError("cell_match_temperature must be positive.")
        if self.cell_output_normalizer <= 0:
            raise ValueError("cell_output_normalizer must be positive.")
        if self.cell_maturity_steps <= 0:
            raise ValueError("cell_maturity_steps must be positive.")
        if any(layer < 0 or layer >= self.n_layers for layer in self.cell_layers):
            raise ValueError("Every cell_layers index must refer to an existing block.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cell_layers"] = list(self.cell_layers)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        # Ignore fields from older bank-based checkpoints.
        valid = {item.name for item in fields(cls)}
        clean = {key: value for key, value in dict(data).items() if key in valid}
        clean["cell_layers"] = tuple(clean.get("cell_layers", (1, 3)))
        return cls(**clean)


@dataclass
class TrainConfig:
    batch_size: int = 16
    steps: int = 2_000
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    embedding_lr: float = 1e-5
    attention_lr: float = 2e-5
    ffn_lr: float = 5e-5
    cell_lr: float = 2e-4
    other_lr: float = 5e-5
    mature_cell_scale: float = 0.02

    retention_replay_weight: float = 0.0
    retention_output_penalty: float = 0.0
    activity_sparsity_weight: float = 0.0

    enable_growth: bool = False
    growth_warmup_steps: int = 100
    growth_patience: int = 40
    growth_cells: int = 8
    growth_loss_floor: float = 1.5
    growth_active_fraction: float = 0.20
    max_growth_events: int = 1
    growth_cooldown_steps: int = 100

    seed: int = 17
