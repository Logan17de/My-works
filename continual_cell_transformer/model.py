from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from cells import PoolStats, SharedExpandableCellPool


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_seq_len).float()
        angles = torch.repeat_interleave(torch.outer(pos, inv), 2, dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.size(-2)
        cos = self.cos[:length].to(q)[None, None]
        sin = self.sin[:length].to(q)[None, None]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split, (q, k, v))
        q, k = self.rope(q, k)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(attended.transpose(1, 2).contiguous().view(batch, length, dim))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


class RecurrentTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.cell_norm = RMSNorm(config.d_model)
        self.cells = SharedExpandableCellPool(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)
        self.cell_scale = config.cell_residual_scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PoolStats]:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        delta, stats = self.cells(self.cell_norm(x))
        x = x + self.cell_scale * self.dropout(delta)
        x = x + self.ffn(self.ffn_norm(x))
        return x, stats


class ContinualCellTransformer(nn.Module):
    ARCHITECTURE_VERSION = 5

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.block = RecurrentTransformerBlock(config)
        self.halt_norm = RMSNorm(config.d_model)
        self.halt_head = nn.Linear(config.d_model, 1)
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init)
        with torch.no_grad():
            self.halt_head.bias.fill_(config.halt_bias)

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
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")
        if adaptive_inference is None:
            adaptive_inference = not self.training

        x = self.token_embedding(input_ids)
        weighted = torch.zeros_like(x)
        remaining = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        expected_depth = torch.zeros_like(remaining)
        halt_probs: list[torch.Tensor] = []
        stats_per_depth: list[PoolStats] = []
        used_depth = self.config.max_depth

        for depth in range(1, self.config.max_depth + 1):
            x, stats = self.block(x)
            stats_per_depth.append(stats)
            summary = self.halt_norm(x).mean(dim=1)
            probability = torch.sigmoid(
                self.halt_head(summary).squeeze(-1) / self.config.halt_temperature
            )
            if depth < self.config.min_depth:
                probability = torch.zeros_like(probability)
            if depth == self.config.max_depth:
                probability = torch.ones_like(probability)
            halt_probs.append(probability)

            weight = remaining * probability
            weighted = weighted + weight[:, None, None] * x
            expected_depth = expected_depth + weight * depth
            remaining = remaining * (1.0 - probability)

            if (
                adaptive_inference
                and depth >= self.config.min_depth
                and torch.all(probability >= self.config.halt_threshold)
            ):
                used_depth = depth
                break

        logits = self.lm_head(self.final_norm(weighted))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=self.config.pad_token_id,
            )

        last = stats_per_depth[-1]
        return {
            "logits": logits,
            "loss": loss,
            "expected_depth": expected_depth.mean(),
            "used_depth": used_depth,
            "halt_probs": torch.stack(halt_probs, dim=1),
            "coverage": last.coverage,
            "mean_active": last.mean_active,
            "plastic_activity": last.plastic_activity,
            "plastic_output_rms": last.plastic_output_rms,
            "micro_saturation": last.micro_saturation,
            "active_cells": last.active_count,
            "consolidated_cells": last.consolidated_count,
            "top_cell_ids": last.top_cell_ids,
            "growth_seed": last.seed,
        }

    @torch.no_grad()
    def resize_vocabulary(self, new_size: int) -> None:
        old_size = self.config.vocab_size
        if new_size < old_size:
            raise ValueError("Vocabulary shrinking unsupported")
        if new_size == old_size:
            return
        embedding = nn.Embedding(
            new_size,
            self.config.d_model,
            padding_idx=self.config.pad_token_id,
        ).to(self.token_embedding.weight)
        head = nn.Linear(self.config.d_model, new_size, bias=False).to(self.lm_head.weight)
        nn.init.normal_(embedding.weight, 0.0, 0.02)
        nn.init.normal_(head.weight, 0.0, 0.02)
        embedding.weight[:old_size].copy_(self.token_embedding.weight)
        head.weight[:old_size].copy_(self.lm_head.weight)
        self.token_embedding = embedding
        self.lm_head = head
        self.config.vocab_size = new_size

    @torch.no_grad()
    def allocate_cells(self, count: int, seed: torch.Tensor | None = None) -> list[int]:
        return self.block.cells.allocate_cells(count, seed)

    @torch.no_grad()
    def grow_micro_neurons(self, count: int) -> dict[int, list[int]]:
        return self.block.cells.grow_micro_neurons(count)

    def mask_cell_gradients(self, consolidated_scale: float) -> None:
        self.block.cells.mask_gradients(consolidated_scale)

    @torch.no_grad()
    def advance_maturity(self) -> None:
        self.block.cells.advance_maturity()

    @torch.no_grad()
    def consolidate_active_cells(self) -> None:
        self.block.cells.consolidate()

    def pool_summary(self) -> dict[str, int]:
        pool = self.block.cells
        return {
            "active": pool.active_count,
            "consolidated": pool.consolidated_count,
            "reserve": pool.max_cells - pool.active_count,
            "active_micro": int(pool.micro_active_mask.sum().item()),
            "micro_capacity": int(pool.active_mask.sum().item() * pool.max_micro),
        }

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_k: int = 1,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(
                ids[:, -self.config.max_seq_len :],
                adaptive_inference=True,
            )["logits"][:, -1]
            if temperature <= 0 or top_k == 1:
                token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k > 0:
                    cutoff = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                token = torch.multinomial(logits.softmax(-1), 1)
            ids = torch.cat((ids, token), dim=1)
            if eos_token_id is not None and torch.all(token == eos_token_id):
                break
        return ids
