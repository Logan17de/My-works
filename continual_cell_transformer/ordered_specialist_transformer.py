from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from config import ModelConfig
from model import CausalSelfAttention, RMSNorm, SwiGLU


class OrderedAttentionLayer(nn.Module):
    """Attention layer that reuses the model's three global FFN modules.

    Every layer performs:
        attention -> shared FFN -> ordered specialist FFNs

    The three FFN weight sets (shared/add/multiply) are global and reused by
    every layer. Specialist routing is deterministic from the operator symbols
    already visible at each token position, so routing stays causal.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.shared_norm = RMSNorm(config.d_model)
        self.specialist_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _ordered_operator_masks(
        input_ids: torch.Tensor,
        plus_token_id: int,
        multiply_token_id: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return causal per-token masks for each operator occurrence.

        For occurrence k, plus_mask[b, t] is true only when the kth operator in
        row b is '+' and its token position is <= t. multiply_mask is analogous.
        Thus a token never receives a specialist selected by a future operator.
        """
        plus = input_ids.eq(plus_token_id)
        multiply = input_ids.eq(multiply_token_id)
        operator = plus | multiply
        counts = operator.sum(dim=1)
        max_ops = int(counts.max().item()) if input_ids.numel() else 0
        if max_ops == 0:
            return []

        batch, seq = input_ids.shape
        positions = torch.arange(seq, device=input_ids.device).view(1, seq)
        occurrence_index = operator.long().cumsum(dim=1) - 1
        masks: list[tuple[torch.Tensor, torch.Tensor]] = []

        for k in range(max_ops):
            kth = operator & occurrence_index.eq(k)
            kth_plus = kth & plus
            kth_multiply = kth & multiply

            sentinel = torch.full_like(positions.expand(batch, -1), seq)
            kth_positions = torch.where(
                kth,
                positions.expand(batch, -1),
                sentinel,
            ).min(dim=1).values
            visible = positions >= kth_positions.unsqueeze(1)

            row_plus = kth_plus.any(dim=1).unsqueeze(1)
            row_multiply = kth_multiply.any(dim=1).unsqueeze(1)
            masks.append((visible & row_plus, visible & row_multiply))

        return masks

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        shared_ffn: nn.Module,
        add_ffn: nn.Module,
        multiply_ffn: nn.Module,
        plus_token_id: int,
        multiply_token_id: int,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(shared_ffn(self.shared_norm(x)))

        masks = self._ordered_operator_masks(
            input_ids,
            plus_token_id,
            multiply_token_id,
        )
        for plus_mask, multiply_mask in masks:
            h = self.specialist_norm(x)
            # Compute both branches once and select per row/token. The routing
            # masks themselves are fixed from the visible operator sequence.
            add_delta = self.dropout(add_ffn(h))
            multiply_delta = self.dropout(multiply_ffn(h))
            x = (
                x
                + add_delta * plus_mask.unsqueeze(-1).to(add_delta.dtype)
                + multiply_delta
                * multiply_mask.unsqueeze(-1).to(multiply_delta.dtype)
            )

        return x


class OrderedSpecialistTransformer(nn.Module):
    """Transformer with exactly three reusable FFN weight sets.

    FFN 1: shared_ffn     -- always used
    FFN 2: add_ffn        -- used for each '+' in appearance order
    FFN 3: multiply_ffn   -- used for each '*' in appearance order

    Example answer pathways at every Transformer depth:
        2 + 3 * 4  -> shared -> add -> multiply
        2 * 3 + 4  -> shared -> multiply -> add
        2 + 3 + 4  -> shared -> add -> add
    """

    ARCHITECTURE_VERSION = 1
    MODEL_TYPE = "ordered_specialist_transformer"

    def __init__(
        self,
        config: ModelConfig,
        num_layers: int,
        plus_token_id: int,
        multiply_token_id: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if plus_token_id == multiply_token_id:
            raise ValueError("plus and multiply token IDs must differ")

        self.config = config
        self.num_layers = int(num_layers)
        self.plus_token_id = int(plus_token_id)
        self.multiply_token_id = int(multiply_token_id)

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            OrderedAttentionLayer(config) for _ in range(self.num_layers)
        )

        # Exactly three FFN parameter sets, reused across all Transformer layers.
        self.shared_ffn = SwiGLU(config)
        self.add_ffn = SwiGLU(config)
        self.multiply_ffn = SwiGLU(config)

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
            x = layer(
                x,
                input_ids,
                self.shared_ffn,
                self.add_ffn,
                self.multiply_ffn,
                self.plus_token_id,
                self.multiply_token_id,
            )

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

    def ffn_parameter_counts(self) -> dict[str, int]:
        return {
            "shared": sum(p.numel() for p in self.shared_ffn.parameters()),
            "add": sum(p.numel() for p in self.add_ffn.parameters()),
            "multiply": sum(p.numel() for p in self.multiply_ffn.parameters()),
        }
