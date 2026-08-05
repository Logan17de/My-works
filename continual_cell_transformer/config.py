from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 260  # 4 special tokens + 256 byte tokens
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1
    rope_base: float = 10_000.0

    # Insert a recurrent concept-cell pool after attention in these blocks.
    cell_layers: tuple[int, ...] = (1, 3)
    max_cells: int = 256
    initial_active_cells: int = 64
    top_k_cells: int = 8
    recurrent_steps: int = 2
    recurrent_fan_in: int = 8
    cell_residual_scale: float = 0.10
    cell_temperature: float = 0.7
    cell_maturity_steps: int = 500

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0 < self.initial_active_cells <= self.max_cells:
            raise ValueError("initial_active_cells must be in [1, max_cells].")
        if self.top_k_cells <= 0:
            raise ValueError("top_k_cells must be positive.")
        if any(layer < 0 or layer >= self.n_layers for layer in self.cell_layers):
            raise ValueError("Every cell_layers index must refer to an existing block.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cell_layers"] = list(self.cell_layers)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        clean = dict(data)
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

    # Plasticity hierarchy.
    embedding_lr: float = 1e-5
    attention_lr: float = 2e-5
    ffn_lr: float = 5e-5
    cell_lr: float = 2e-4
    other_lr: float = 5e-5
    mature_cell_scale: float = 0.02

    # Optional dynamic recruitment from dormant reserve cells.
    enable_growth: bool = False
    growth_warmup_steps: int = 100
    growth_patience: int = 40
    growth_cells: int = 8
    growth_confidence: float = 0.42
    growth_loss_floor: float = 1.5

    seed: int = 17
