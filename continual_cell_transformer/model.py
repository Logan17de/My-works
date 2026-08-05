from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

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
        return (
            x
            * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x[..., ::2], x[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.repeat_interleave(torch.outer(positions, inv_freq), 2, dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.size(-2)
        cos = self.cos[:length].to(q)[None, None]
        sin = self.sin[:length].to(q)[None, None]
        return (
            q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
            config.rope_base,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch,
                length,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        q, k = self.rope(q, k)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


@dataclass
class PoolStats:
    confidence: torch.Tensor
    coverage: torch.Tensor
    mean_active: torch.Tensor
    max_active: torch.Tensor
    plastic_activity: torch.Tensor
    plastic_output_rms: torch.Tensor
    active_count: int
    consolidated_count: int
    top_cell_ids: list[int]
    seed: torch.Tensor


class SharedThresholdCellPool(nn.Module):
    """
    One shared append-only population.

    Cells activate independently when contextual similarity crosses each cell's
    learned threshold. There are no task-labelled banks and no global top-k, so
    adding a cell cannot displace an old cell from the route.

    New cells are inserted with exactly zero write vectors. Their initial forward
    contribution is therefore exactly zero, while gradients can immediately train
    those write vectors. Existing cells and existing old-target recurrent edges are
    not modified by allocation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        cells, dim = config.max_cells, config.d_model
        self.max_cells = cells
        self.d_model = dim
        self.recurrent_steps = config.recurrent_steps
        self.recurrent_fan_in = config.recurrent_fan_in
        self.threshold_temperature = config.threshold_temperature
        self.new_cell_threshold = config.new_cell_threshold
        self.maturity_steps = max(1, config.cell_maturity_steps)

        self.keys = nn.Parameter(torch.randn(cells, dim) * 0.02)
        self.read_vectors = nn.Parameter(torch.randn(cells, dim) * 0.02)
        self.write_vectors = nn.Parameter(torch.randn(cells, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(cells))
        self.thresholds = nn.Parameter(
            torch.full((cells,), float(config.initial_threshold))
        )
        self.recurrent = nn.Parameter(torch.zeros(cells, cells))

        active_mask = torch.zeros(cells, dtype=torch.bool)
        active_mask[: config.initial_active_cells] = True
        self.register_buffer("active_mask", active_mask)
        self.register_buffer("consolidated_mask", torch.zeros(cells, dtype=torch.bool))
        self.register_buffer("maturity", torch.zeros(cells))
        self.register_buffer("usage_ema", torch.zeros(cells))
        self.register_buffer("edge_mask", torch.zeros(cells, cells, dtype=torch.bool))

        with torch.no_grad():
            self.keys.copy_(F.normalize(self.keys, dim=-1))
            self.read_vectors.copy_(F.normalize(self.read_vectors, dim=-1))
            self._initialize_base_edges(torch.arange(config.initial_active_cells))

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def consolidated_count(self) -> int:
        return int((self.active_mask & self.consolidated_mask).sum().item())

    def _initialize_base_edges(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        for target in indices.tolist():
            fan_in = min(self.recurrent_fan_in, indices.numel())
            sources = indices[
                torch.randperm(indices.numel(), device=indices.device)[:fan_in]
            ]
            self.edge_mask[sources, target] = True
            self.recurrent[sources, target].normal_(0.0, 0.02)
            self.edge_mask[target, target] = True
            self.recurrent[target, target] = 0.01

    def _straight_through_threshold(
        self,
        scores: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        soft = torch.sigmoid(
            (scores - thresholds) / self.threshold_temperature
        )
        hard = (scores > thresholds).to(dtype=scores.dtype)
        gate = hard + soft - soft.detach()
        return gate, hard, soft

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PoolStats]:
        indices = torch.nonzero(self.active_mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            zero = torch.zeros_like(x)
            scalar_zero = x.new_tensor(0.0)
            return zero, PoolStats(
                confidence=scalar_zero,
                coverage=scalar_zero,
                mean_active=scalar_zero,
                max_active=scalar_zero,
                plastic_activity=scalar_zero,
                plastic_output_rms=scalar_zero,
                active_count=0,
                consolidated_count=0,
                top_cell_ids=[],
                seed=x.detach().mean(dim=(0, 1)),
            )

        keys = F.normalize(self.keys[indices], dim=-1)
        reads = F.normalize(self.read_vectors[indices], dim=-1)
        writes = self.write_vectors[indices]
        thresholds = self.thresholds[indices]

        normalized_x = F.normalize(x, dim=-1)
        scores = torch.einsum("btd,cd->btc", normalized_x, keys)
        gates, hard_gates, soft_gates = self._straight_through_threshold(
            scores,
            thresholds,
        )

        drive = gates * F.silu(
            torch.einsum("btd,cd->btc", x, reads) + self.bias[indices]
        )

        recurrent = (
            self.recurrent[indices][:, indices]
            * self.edge_mask[indices][:, indices].to(dtype=x.dtype)
        )
        activity = drive
        for _ in range(self.recurrent_steps):
            activity = F.silu(
                drive + torch.einsum("btc,cd->btd", activity, recurrent)
            )

        output = (
            torch.einsum("btc,cd->btd", activity, writes)
            / math.sqrt(max(1, self.recurrent_fan_in))
        )

        with torch.no_grad():
            usage = hard_gates.mean(dim=(0, 1))
            global_usage = torch.zeros_like(self.usage_ema)
            global_usage[indices] = usage
            self.usage_ema.mul_(0.99).add_(global_usage, alpha=0.01)

            aggregate = hard_gates.sum(dim=(0, 1))
            top_n = min(12, aggregate.numel())
            top_local = aggregate.topk(top_n).indices
            top_global = indices[top_local].tolist()

        plastic_local = ~self.consolidated_mask[indices]
        if plastic_local.any():
            plastic_activity = soft_gates[..., plastic_local].mean()
            plastic_latent = (
                torch.einsum(
                    "btc,cd->btd",
                    activity[..., plastic_local],
                    writes[plastic_local],
                )
                / math.sqrt(max(1, self.recurrent_fan_in))
            )
            plastic_output_rms = (
                plastic_latent.square().mean() + 1e-12
            ).sqrt()
        else:
            plastic_activity = x.new_tensor(0.0)
            plastic_output_rms = x.new_tensor(0.0)

        max_soft = soft_gates.max(dim=-1).values
        hard_counts = hard_gates.sum(dim=-1)
        return output, PoolStats(
            confidence=max_soft.mean(),
            coverage=max_soft.mean(),
            mean_active=hard_counts.mean(),
            max_active=hard_counts.max(),
            plastic_activity=plastic_activity,
            plastic_output_rms=plastic_output_rms,
            active_count=self.active_count,
            consolidated_count=self.consolidated_count,
            top_cell_ids=top_global,
            seed=x.detach().mean(dim=(0, 1)),
        )

    @torch.no_grad()
    def allocate_cells(
        self,
        count: int,
        seed: torch.Tensor | None = None,
    ) -> list[int]:
        if count <= 0:
            return []
        dormant = torch.nonzero(~self.active_mask, as_tuple=False).flatten()
        if dormant.numel() == 0:
            return []
        chosen = dormant[: min(count, dormant.numel())]
        existing = torch.nonzero(self.active_mask, as_tuple=False).flatten()

        if seed is None:
            seed = torch.randn(self.d_model, device=self.keys.device)
        seed = F.normalize(seed.to(self.keys), dim=-1)

        if existing.numel() > 0:
            similarities = torch.mv(
                F.normalize(self.keys[existing], dim=-1),
                seed,
            )
            parent_count = min(self.recurrent_fan_in, existing.numel())
            parents = existing[similarities.topk(parent_count).indices]
        else:
            parents = existing

        for row in chosen.tolist():
            noise = 0.05 * torch.randn_like(seed)
            initialized = F.normalize(seed + noise, dim=-1)
            self.keys[row].copy_(initialized)
            self.read_vectors[row].copy_(initialized)
            self.write_vectors[row].zero_()
            self.bias[row] = 0.0
            self.thresholds[row] = self.new_cell_threshold
            self.maturity[row] = 0.0
            self.usage_ema[row] = 0.0
            self.consolidated_mask[row] = False
            self.active_mask[row] = True

            if parents.numel() > 0:
                self.edge_mask[parents, row] = True
                self.recurrent[parents, row].normal_(0.0, 0.02)
            self.edge_mask[row, row] = True
            self.recurrent[row, row] = 0.01

        for target in chosen.tolist():
            source_count = min(self.recurrent_fan_in, chosen.numel())
            sources = chosen[
                torch.randperm(chosen.numel(), device=chosen.device)[:source_count]
            ]
            self.edge_mask[sources, target] = True
            self.recurrent[sources, target].normal_(0.0, 0.02)
            self.edge_mask[target, target] = True
            self.recurrent[target, target] = 0.01

        return chosen.tolist()

    @torch.no_grad()
    def consolidate_active_cells(self) -> None:
        self.consolidated_mask[self.active_mask] = True
        self.maturity[self.active_mask] = 1.0

    @torch.no_grad()
    def advance_maturity(self) -> None:
        plastic = self.active_mask & ~self.consolidated_mask
        self.maturity[plastic].add_(1.0 / self.maturity_steps).clamp_(max=1.0)

    def mask_gradients(self, consolidated_scale: float) -> None:
        row_scale = self.active_mask.to(dtype=self.keys.dtype)
        row_scale = row_scale * torch.where(
            self.consolidated_mask,
            torch.full_like(row_scale, consolidated_scale),
            torch.ones_like(row_scale),
        )

        for parameter in (
            self.keys,
            self.read_vectors,
            self.write_vectors,
        ):
            if parameter.grad is not None:
                parameter.grad.mul_(row_scale[:, None])

        for parameter in (self.bias, self.thresholds):
            if parameter.grad is not None:
                parameter.grad.mul_(row_scale)

        if self.recurrent.grad is not None:
            edge_scale = row_scale[None, :] * self.edge_mask.to(row_scale.dtype)
            self.recurrent.grad.mul_(edge_scale)

    def summary(self) -> dict[str, int]:
        return {
            "active": self.active_count,
            "consolidated": self.consolidated_count,
            "plastic": self.active_count - self.consolidated_count,
            "reserve": self.max_cells - self.active_count,
        }


class Block(nn.Module):
    def __init__(self, config: ModelConfig, use_pool: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.pool_norm = RMSNorm(config.d_model) if use_pool else None
        self.pool = SharedThresholdCellPool(config) if use_pool else None
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)
        self.pool_scale = config.cell_residual_scale

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, PoolStats | None]:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        stats = None
        if self.pool is not None and self.pool_norm is not None:
            delta, stats = self.pool(self.pool_norm(x))
            x = x + self.pool_scale * self.dropout(delta)
        x = x + self.ffn(self.ffn_norm(x))
        return x, stats


class ContinualCellTransformer(nn.Module):
    ARCHITECTURE_VERSION = 4

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.blocks = nn.ModuleList(
            [
                Block(config, index in config.cell_layers)
                for index in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def pools(self) -> list[SharedThresholdCellPool]:
        return [block.pool for block in self.blocks if block.pool is not None]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len.")

        x = self.token_embedding(input_ids)
        stats: list[PoolStats] = []
        for block in self.blocks:
            x, item = block(x)
            if item is not None:
                stats.append(item)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=self.config.pad_token_id,
            )

        if stats:
            confidence = torch.stack([item.confidence for item in stats]).mean()
            coverage = torch.stack([item.coverage for item in stats]).mean()
            plastic_activity = torch.stack(
                [item.plastic_activity for item in stats]
            ).mean()
            plastic_output_rms = torch.stack(
                [item.plastic_output_rms for item in stats]
            ).mean()
        else:
            confidence = logits.new_tensor(1.0)
            coverage = logits.new_tensor(1.0)
            plastic_activity = logits.new_tensor(0.0)
            plastic_output_rms = logits.new_tensor(0.0)

        return {
            "logits": logits,
            "loss": loss,
            "cell_confidence": confidence,
            "coverage": coverage,
            "plastic_activity_mean": plastic_activity,
            "plastic_output_rms": plastic_output_rms,
            "growth_seeds": [item.seed for item in stats],
            "pool_summaries": [
                {
                    "active": item.active_count,
                    "consolidated": item.consolidated_count,
                    "plastic": item.active_count - item.consolidated_count,
                    "mean_active_per_token": float(item.mean_active.detach()),
                    "max_active_per_token": int(item.max_active.detach().item()),
                    "top_cell_ids": item.top_cell_ids,
                }
                for item in stats
            ],
        }

    @torch.no_grad()
    def resize_vocabulary(self, new_size: int) -> None:
        old_size = self.config.vocab_size
        if new_size < old_size:
            raise ValueError("Vocabulary shrinking is unsupported.")
        if new_size == old_size:
            return

        embedding = nn.Embedding(
            new_size,
            self.config.d_model,
            padding_idx=self.config.pad_token_id,
        ).to(self.token_embedding.weight)
        head = nn.Linear(
            self.config.d_model,
            new_size,
            bias=False,
        ).to(self.lm_head.weight)
        nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(head.weight, mean=0.0, std=0.02)
        embedding.weight[:old_size].copy_(self.token_embedding.weight)
        head.weight[:old_size].copy_(self.lm_head.weight)
        if self.config.pad_token_id is not None:
            embedding.weight[self.config.pad_token_id].zero_()

        self.token_embedding = embedding
        self.lm_head = head
        self.config.vocab_size = new_size

    @torch.no_grad()
    def allocate_cells(
        self,
        count_per_pool: int,
        seeds: list[torch.Tensor] | None = None,
    ) -> list[list[int]]:
        pools = self.pools()
        seeds = seeds or [None] * len(pools)
        return [
            pool.allocate_cells(count_per_pool, seed)
            for pool, seed in zip(pools, seeds)
        ]

    def mask_cell_gradients(self, consolidated_scale: float) -> None:
        for pool in self.pools():
            pool.mask_gradients(consolidated_scale)

    @torch.no_grad()
    def advance_maturity(self) -> None:
        for pool in self.pools():
            pool.advance_maturity()

    @torch.no_grad()
    def consolidate_active_cells(self) -> None:
        for pool in self.pools():
            pool.consolidate_active_cells()

    def pool_summaries(self) -> list[dict[str, int]]:
        return [pool.summary() for pool in self.pools()]

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        generated = ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]
            logits = self(context)["logits"][:, -1]

            if temperature <= 0.0 or top_k == 1:
                token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k > 0:
                    selected = min(top_k, logits.size(-1))
                    threshold = logits.topk(selected).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                token = torch.multinomial(logits.softmax(dim=-1), 1)

            generated = torch.cat((generated, token), dim=1)
            if eos_token_id is not None and torch.all(token == eos_token_id):
                break
        return generated
