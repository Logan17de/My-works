from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    architecture_version: int = 2
    vocab_size: int = 260
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1
    rope_base: float = 10_000.0

    cell_layers: tuple[int, ...] = (1, 3)
    stable_cells: int = 64
    stable_top_k: int = 8
    plastic_capacity: int = 64
    plastic_top_k: int = 4
    recurrent_steps: int = 2
    recurrent_fan_in: int = 8
    cell_temperature: float = 0.7
    stable_residual_scale: float = 0.10
    plastic_residual_scale: float = 0.10

    adapter_rank: int = 32
    plastic_gate_threshold: float = 0.0  # required plastic-vs-stable similarity margin
    plastic_gate_sharpness: float = 8.0
    plastic_gate_bias_init: float = -4.0

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.architecture_version != 2:
            raise ValueError("This package only supports architecture_version=2.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.stable_cells <= 0 or self.plastic_capacity <= 0:
            raise ValueError("Cell capacities must be positive.")
        if not 0 < self.stable_top_k <= self.stable_cells:
            raise ValueError("stable_top_k must be within stable_cells.")
        if not 0 < self.plastic_top_k <= self.plastic_capacity:
            raise ValueError("plastic_top_k must be within plastic_capacity.")
        if any(index < 0 or index >= self.n_layers for index in self.cell_layers):
            raise ValueError("cell_layers contains an invalid block index.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cell_layers"] = list(self.cell_layers)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        clean = dict(data)
        clean["cell_layers"] = tuple(clean.get("cell_layers", (1, 3)))
        return cls(**clean)
