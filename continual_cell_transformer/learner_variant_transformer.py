from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from model import CausalSelfAttention, RMSNorm


VARIANTS = (
    "post_ffn_all",
    "standalone_all",
    "pre_activation_all",
    "post_activation_all",
)


class BottleneckLearner(nn.Module):
    """Small residual learner used consistently across placement ablations."""

    def __init__(self, input_dim: int, bottleneck_dim: int, output_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


class VariantBlock(nn.Module):
    """One ordinary Transformer block plus one learner placement variant."""

    def __init__(
        self,
        config: ModelConfig,
        variant: str,
        learner_dim: int,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")

        self.variant = variant
        self.learner_enabled = True
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn_up = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.ffn_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if variant in {"post_ffn_all", "standalone_all"}:
            self.learner_norm = RMSNorm(config.d_model)
            self.learner = BottleneckLearner(
                config.d_model,
                learner_dim,
                config.d_model,
            )
        elif variant == "pre_activation_all":
            self.learner = BottleneckLearner(
                config.d_model,
                learner_dim,
                config.d_ff,
            )
        else:  # post_activation_all
            self.learner = BottleneckLearner(
                config.d_ff,
                learner_dim,
                config.d_ff,
            )

    def _ffn(self, h: torch.Tensor) -> torch.Tensor:
        gate, value = self.ffn_up(h).chunk(2, dim=-1)

        if self.variant == "pre_activation_all" and self.learner_enabled:
            gate = gate + self.learner(h)

        z = F.silu(gate) * value

        if self.variant == "post_activation_all" and self.learner_enabled:
            z = z + self.learner(z)

        return self.dropout(self.ffn_down(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_input = x
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self._ffn(self.ffn_norm(x))

        if self.learner_enabled and self.variant == "post_ffn_all":
            x = x + self.dropout(self.learner(self.learner_norm(x)))
        elif self.learner_enabled and self.variant == "standalone_all":
            # True bypass branch: the learner sees the representation that
            # entered this Transformer layer, not the already transformed x.
            x = x + self.dropout(self.learner(self.learner_norm(layer_input)))

        return x


class LearnerVariantTransformer(nn.Module):
    ARCHITECTURE_VERSION = 1
    MODEL_TYPE = "learner_variant_transformer"

    def __init__(
        self,
        config: ModelConfig,
        num_layers: int = 8,
        variant: str = "post_ffn_all",
        learner_dim: int = 32,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if learner_dim < 1:
            raise ValueError("learner_dim must be positive")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")

        self.config = config
        self.num_layers = int(num_layers)
        self.variant = variant
        self.learner_dim = int(learner_dim)
        self.learners_enabled = True

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            VariantBlock(config, variant, learner_dim)
            for _ in range(self.num_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def set_learners_enabled(self, enabled: bool) -> None:
        """Enable or bypass every learner without changing any weights."""
        self.learners_enabled = bool(enabled)
        for layer in self.layers:
            layer.learner_enabled = self.learners_enabled

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

        depth = torch.tensor(
            float(self.num_layers),
            device=input_ids.device,
            dtype=x.dtype,
        )
        return {
            "logits": logits,
            "used_depth": self.num_layers,
            "expected_depth": depth,
            "learners_enabled": self.learners_enabled,
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

    def learner_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if ".learner." in name
        )
