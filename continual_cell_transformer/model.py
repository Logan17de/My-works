from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_seq_len).float()
        angles = torch.repeat_interleave(torch.outer(pos, inv), 2, -1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.size(-2)
        cos = self.cos[:length].to(q)[None, None]
        sin = self.sin[:length].to(q)[None, None]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        split = lambda z: z.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(batch, length, dim))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, -1)
        return self.dropout(self.down(F.silu(gate) * value))


@dataclass
class PoolStats:
    confidence: torch.Tensor
    seed: torch.Tensor
    active_count: int


class RecurrentConceptPool(nn.Module):
    """Sparse recurrent cells with dormant reserve rows and local plasticity."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        n, d = cfg.max_cells, cfg.d_model
        self.top_k = cfg.top_k_cells
        self.steps = cfg.recurrent_steps
        self.fan_in = cfg.recurrent_fan_in
        self.temperature = cfg.cell_temperature
        self.maturity_steps = max(1, cfg.cell_maturity_steps)

        self.keys = nn.Parameter(torch.randn(n, d) * 0.02)
        self.read_vectors = nn.Parameter(torch.randn(n, d) * 0.02)
        self.write_vectors = nn.Parameter(torch.randn(n, d) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n))
        self.recurrent = nn.Parameter(torch.zeros(n, n))

        active = torch.zeros(n, dtype=torch.bool)
        active[: cfg.initial_active_cells] = True
        self.register_buffer("active_mask", active)
        self.register_buffer("maturity", torch.zeros(n))
        self.register_buffer("usage_ema", torch.zeros(n))
        self.register_buffer("edge_mask", torch.zeros(n, n, dtype=torch.bool))

        with torch.no_grad():
            self.keys.copy_(F.normalize(self.keys, dim=-1))
            self.read_vectors.copy_(F.normalize(self.read_vectors, dim=-1))
            for target in range(cfg.initial_active_cells):
                sources = torch.randperm(cfg.initial_active_cells)[: self.fan_in]
                self.edge_mask[sources, target] = True
                self.recurrent[sources, target].normal_(0, 0.02)
                self.edge_mask[target, target] = True
                self.recurrent[target, target] = 0.01

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PoolStats]:
        idx = self.active_mask.nonzero().flatten()
        keys = F.normalize(self.keys[idx], dim=-1)
        reads = F.normalize(self.read_vectors[idx], dim=-1)
        writes = self.write_vectors[idx]

        scores = torch.einsum("btd,cd->btc", F.normalize(x, dim=-1), keys) / self.temperature
        k = min(self.top_k, idx.numel())
        values, local_idx = scores.topk(k, dim=-1)
        weights = values.softmax(-1)
        gates = torch.zeros_like(scores).scatter(-1, local_idx, weights)
        drive = gates * F.silu(torch.einsum("btd,cd->btc", x, reads) + self.bias[idx])

        rec = self.recurrent[idx][:, idx] * self.edge_mask[idx][:, idx].to(x.dtype)
        activity = drive
        for _ in range(self.steps):
            activity = F.silu(drive + torch.einsum("btc,cd->btd", activity, rec))

        output = torch.einsum("btc,cd->btd", activity, writes) / math.sqrt(max(1, k))
        with torch.no_grad():
            use = (gates > 0).float().mean((0, 1))
            global_use = torch.zeros_like(self.usage_ema)
            global_use[idx] = use
            self.usage_ema.mul_(0.99).add_(global_use, alpha=0.01)

        return output, PoolStats(torch.sigmoid(values[..., 0]).mean(), x.detach().mean((0, 1)), int(idx.numel()))

    @torch.no_grad()
    def allocate(self, count: int, seed: torch.Tensor | None = None) -> list[int]:
        dormant = (~self.active_mask).nonzero().flatten()[:count]
        if dormant.numel() == 0:
            return []
        old = self.active_mask.nonzero().flatten()
        seed = F.normalize((seed if seed is not None else torch.randn_like(self.keys[0])).to(self.keys), dim=-1)
        for row in dormant.tolist():
            self.keys[row] = F.normalize(seed + 0.05 * torch.randn_like(seed), dim=-1)
            self.read_vectors[row] = F.normalize(seed + 0.05 * torch.randn_like(seed), dim=-1)
            self.write_vectors[row].normal_(0, 0.01)
            self.bias[row] = 0
            self.maturity[row] = 0
            if old.numel():
                parents = old[self.usage_ema[old].topk(min(self.fan_in, old.numel())).indices]
                self.edge_mask[parents, row] = True
                self.recurrent[parents, row].normal_(0, 0.02)
            self.edge_mask[row, row] = True
            self.recurrent[row, row] = 0.01
            self.active_mask[row] = True
        return dormant.tolist()

    def mask_gradients(self, mature_scale: float) -> None:
        scale = self.active_mask.to(self.keys) * (mature_scale + (1 - mature_scale) * (1 - self.maturity))
        for p in (self.keys, self.read_vectors, self.write_vectors):
            if p.grad is not None:
                p.grad.mul_(scale[:, None])
        if self.bias.grad is not None:
            self.bias.grad.mul_(scale)
        if self.recurrent.grad is not None:
            edge_scale = torch.maximum(scale[:, None], scale[None, :]) * self.edge_mask.to(scale)
            self.recurrent.grad.mul_(edge_scale)

    @torch.no_grad()
    def advance_maturity(self) -> None:
        self.maturity[self.active_mask].add_(1 / self.maturity_steps).clamp_(max=1)

    @torch.no_grad()
    def seal(self) -> None:
        self.maturity[self.active_mask] = 1


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, use_pool: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.pool_norm = RMSNorm(cfg.d_model) if use_pool else None
        self.pool = RecurrentConceptPool(cfg) if use_pool else None
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.pool_scale = cfg.cell_residual_scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PoolStats | None]:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        stats = None
        if self.pool is not None:
            delta, stats = self.pool(self.pool_norm(x))
            x = x + self.pool_scale * self.dropout(delta)
        x = x + self.ffn(self.ffn_norm(x))
        return x, stats


class ContinualCellTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.blocks = nn.ModuleList([Block(cfg, i in cfg.cell_layers) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0, 0.02)

    def pools(self) -> list[RecurrentConceptPool]:
        return [b.pool for b in self.blocks if b.pool is not None]

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")
        x = self.token_embedding(input_ids)
        stats = []
        for block in self.blocks:
            x, item = block(x)
            if item is not None:
                stats.append(item)
        logits = self.lm_head(self.final_norm(x))
        loss = None if labels is None else F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=self.config.pad_token_id
        )
        return {
            "logits": logits,
            "loss": loss,
            "cell_confidence": torch.stack([s.confidence for s in stats]).mean() if stats else logits.new_tensor(1.0),
            "growth_seeds": [s.seed for s in stats],
            "active_cells": [s.active_count for s in stats],
        }

    @torch.no_grad()
    def resize_vocabulary(self, new_size: int) -> None:
        old = self.config.vocab_size
        if new_size < old:
            raise ValueError("Vocabulary shrinking is unsupported")
        if new_size == old:
            return
        emb = nn.Embedding(new_size, self.config.d_model, padding_idx=self.config.pad_token_id).to(self.token_embedding.weight)
        head = nn.Linear(self.config.d_model, new_size, bias=False).to(self.lm_head.weight)
        nn.init.normal_(emb.weight, 0, 0.02)
        nn.init.normal_(head.weight, 0, 0.02)
        emb.weight[:old].copy_(self.token_embedding.weight)
        head.weight[:old].copy_(self.lm_head.weight)
        self.token_embedding, self.lm_head = emb, head
        self.config.vocab_size = new_size

    def mask_cell_gradients(self, mature_scale: float) -> None:
        for pool in self.pools():
            pool.mask_gradients(mature_scale)

    @torch.no_grad()
    def advance_maturity(self) -> None:
        for pool in self.pools():
            pool.advance_maturity()

    @torch.no_grad()
    def seal_active_cells(self) -> None:
        for pool in self.pools():
            pool.seal()

    @torch.no_grad()
    def allocate_cells(self, count: int, seeds: list[torch.Tensor]) -> list[list[int]]:
        return [pool.allocate(count, seed) for pool, seed in zip(self.pools(), seeds)]

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 40, eos_token_id: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.config.max_seq_len:])["logits"][:, -1] / max(temperature, 1e-5)
            if top_k > 0:
                threshold = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            token = torch.multinomial(logits.softmax(-1), 1)
            ids = torch.cat((ids, token), 1)
            if eos_token_id is not None and torch.all(token == eos_token_id):
                break
        return ids
