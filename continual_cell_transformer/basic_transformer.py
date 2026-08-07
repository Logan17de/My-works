from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from config import ModelConfig
from model import CausalSelfAttention, RMSNorm, SwiGLU


class BasicTransformerBlock(nn.Module):
    """Plain pre-norm causal Transformer block: attention -> FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BasicTransformer(nn.Module):
    """Fixed-depth baseline with no cells, growth, consolidation, or halting."""

    ARCHITECTURE_VERSION = 1
    MODEL_TYPE = "basic_transformer"

    def __init__(self, config: ModelConfig, num_layers: int = 8) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.config = config
        self.num_layers = int(num_layers)

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            BasicTransformerBlock(config) for _ in range(self.num_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        adaptive_inference: bool | None = None,
    ) -> dict[str, Any]:
        del labels, adaptive_inference
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.final_norm(x))

        batch = input_ids.size(0)
        expected_depth = torch.full(
            (batch,),
            float(self.num_layers),
            device=input_ids.device,
            dtype=x.dtype,
        ).mean()
        return {
            "logits": logits,
            "used_depth": self.num_layers,
            "expected_depth": expected_depth,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        ids = input_ids
        stop_id = self.config.eos_token_id if eos_token_id is None else eos_token_id

        for _ in range(max_new_tokens):
            result = self(ids[:, -self.config.max_seq_len :])
            next_token = result["logits"][:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat((ids, next_token), dim=1)
            if torch.all(next_token == stop_id):
                break

        self.train(was_training)
        return ids

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
