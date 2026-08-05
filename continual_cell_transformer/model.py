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
        inv = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.repeat_interleave(torch.outer(positions, inv), 2, dim=-1)
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
        self.heads = config.n_heads
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
                self.heads,
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
    seed: torch.Tensor
    parent_ids: list[int]
    active_count: int
    active_fraction: torch.Tensor
    active_ids: list[int]
    mean_gate: torch.Tensor
    plastic_gate_mean: torch.Tensor
    plastic_output_rms: torch.Tensor


class ThresholdRecurrentCellPool(nn.Module):
    """
    One shared recurrent population.

    Cells activate independently when their contextual match exceeds their own
    threshold. There is no global top-k competition and no named domain bank.

    New cells:
      * are appended from a dormant reserve;
      * receive keys/read vectors seeded from the new context;
      * start with exactly zero write vectors, so insertion has zero immediate
        effect on old outputs;
      * receive inbound edges from established cells, but no new -> old edges.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        cells, dim = config.max_cells, config.d_model
        self.max_cells = cells
        self.dim = dim
        self.steps = config.recurrent_steps
        self.fan_in = config.recurrent_fan_in
        self.match_temperature = config.cell_match_temperature
        self.new_cell_threshold = config.new_cell_threshold
        self.output_normalizer = config.cell_output_normalizer
        self.maturity_steps = config.cell_maturity_steps

        self.keys = nn.Parameter(torch.randn(cells, dim) * 0.02)
        self.read_vectors = nn.Parameter(torch.randn(cells, dim) * 0.02)
        self.write_vectors = nn.Parameter(torch.randn(cells, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(cells))
        self.thresholds = nn.Parameter(
            torch.full((cells,), config.initial_cell_threshold)
        )
        self.recurrent = nn.Parameter(torch.zeros(cells, cells))

        active = torch.zeros(cells, dtype=torch.bool)
        active[: config.initial_active_cells] = True
        self.register_buffer("active_mask", active)
        self.register_buffer("maturity", torch.zeros(cells))
        self.register_buffer("usage_ema", torch.zeros(cells))
        self.register_buffer(
            "edge_mask",
            torch.zeros(cells, cells, dtype=torch.bool),
        )

        with torch.no_grad():
            self.keys.copy_(F.normalize(self.keys, dim=-1))
            self.read_vectors.copy_(F.normalize(self.read_vectors, dim=-1))
            self._initialize_base_edges(
                torch.arange(config.initial_active_cells)
            )

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    def _initialize_base_edges(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        for target in indices.tolist():
            count = min(self.fan_in, indices.numel())
            order = torch.randperm(indices.numel(), device=indices.device)[:count]
            sources = indices[order]
            self.edge_mask[sources, target] = True
            self.recurrent[sources, target].normal_(0.0, 0.02)
            self.edge_mask[target, target] = True
            self.recurrent[target, target] = 0.01

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PoolStats]:
        indices = torch.nonzero(
            self.active_mask,
            as_tuple=False,
        ).flatten()

        if indices.numel() == 0:
            zero = torch.zeros_like(x)
            scalar = x.new_tensor(0.0)
            return zero, PoolStats(
                seed=x.detach().mean(dim=(0, 1)),
                parent_ids=[],
                active_count=0,
                active_fraction=scalar,
                active_ids=[],
                mean_gate=scalar,
                plastic_gate_mean=scalar,
                plastic_output_rms=scalar,
            )

        normalized_x = F.normalize(x, dim=-1)
        keys = F.normalize(self.keys[indices], dim=-1)
        reads = F.normalize(self.read_vectors[indices], dim=-1)
        writes = self.write_vectors[indices]

        matches = (
            torch.einsum("btd,cd->btc", normalized_x, keys)
            / self.match_temperature
            + self.bias[indices]
        )
        margins = matches - self.thresholds[indices]
        gates = torch.tanh(F.relu(margins))

        read_signal = F.silu(
            torch.einsum("btd,cd->btc", normalized_x, reads)
        )
        drive = gates * read_signal

        recurrent = (
            self.recurrent[indices][:, indices]
            * self.edge_mask[indices][:, indices].to(x.dtype)
        )
        activity = drive
        for _ in range(self.steps):
            spread = torch.einsum("btc,cd->btd", activity, recurrent)
            activity = F.silu(drive + spread)

        output = (
            torch.einsum("btc,cd->btd", activity, writes)
            / math.sqrt(self.output_normalizer)
        )

        plastic_local = self.maturity[indices] < 1.0
        if plastic_local.any():
            plastic_activity = activity * plastic_local.to(activity.dtype)[None, None, :]
            plastic_output = (
                torch.einsum("btc,cd->btd", plastic_activity, writes)
                / math.sqrt(self.output_normalizer)
            )
            plastic_gate_mean = gates[..., plastic_local].mean()
            plastic_output_rms = (
                plastic_output.square().mean() + 1e-12
            ).sqrt()
        else:
            plastic_gate_mean = x.new_tensor(0.0)
            plastic_output_rms = x.new_tensor(0.0)

        with torch.no_grad():
            selected = (gates > 0).float().mean(dim=(0, 1))
            global_usage = torch.zeros_like(self.usage_ema)
            global_usage[indices] = selected
            self.usage_ema.mul_(0.99).add_(global_usage, alpha=0.01)

            aggregate = gates.sum(dim=(0, 1))
            nonzero = torch.nonzero(aggregate > 0, as_tuple=False).flatten()
            if nonzero.numel() > 0:
                count = min(16, nonzero.numel())
                local_top = aggregate.topk(count).indices
                active_ids = indices[local_top].tolist()
            else:
                active_ids = []

            parent_count = min(self.fan_in, indices.numel())
            parent_local = self.usage_ema[indices].topk(parent_count).indices
            parent_ids = indices[parent_local].tolist()

        active_fraction = (gates > 0).float().mean()
        mean_gate = gates.mean()
        return output, PoolStats(
            seed=x.detach().mean(dim=(0, 1)),
            parent_ids=parent_ids,
            active_count=int(indices.numel()),
            active_fraction=active_fraction,
            active_ids=active_ids,
            mean_gate=mean_gate,
            plastic_gate_mean=plastic_gate_mean,
            plastic_output_rms=plastic_output_rms,
        )

    @torch.no_grad()
    def allocate(
        self,
        count: int,
        seed: torch.Tensor | None = None,
        parent_ids: list[int] | None = None,
    ) -> list[int]:
        dormant = torch.nonzero(
            ~self.active_mask,
            as_tuple=False,
        ).flatten()
        if count <= 0 or dormant.numel() == 0:
            return []

        chosen = dormant[: min(count, dormant.numel())]
        if seed is None:
            seed = torch.randn(self.dim, device=self.keys.device)
        seed = F.normalize(seed.to(self.keys), dim=-1)

        established = torch.nonzero(
            self.active_mask,
            as_tuple=False,
        ).flatten()
        valid_parents = [
            int(value)
            for value in (parent_ids or [])
            if 0 <= int(value) < self.max_cells and self.active_mask[int(value)]
        ]
        if not valid_parents and established.numel() > 0:
            count_parents = min(self.fan_in, established.numel())
            top = self.usage_ema[established].topk(count_parents).indices
            valid_parents = established[top].tolist()

        for row in chosen.tolist():
            noise = 0.05 * torch.randn_like(seed)
            initial = F.normalize(seed + noise, dim=-1)
            self.keys[row].copy_(initial)
            self.read_vectors[row].copy_(initial)
            self.write_vectors[row].zero_()
            self.bias[row] = 0.0
            self.thresholds[row] = self.new_cell_threshold
            self.maturity[row] = 0.0
            self.usage_ema[row] = 0.0
            self.active_mask[row] = True

            if valid_parents:
                parents = torch.tensor(
                    valid_parents[: self.fan_in],
                    device=self.keys.device,
                    dtype=torch.long,
                )
                self.edge_mask[parents, row] = True
                self.recurrent[parents, row].normal_(0.0, 0.02)

            self.edge_mask[row, row] = True
            self.recurrent[row, row] = 0.01

        for target in chosen.tolist():
            if chosen.numel() <= 1:
                continue
            count_sources = min(self.fan_in, chosen.numel())
            sources = chosen[
                torch.randperm(chosen.numel(), device=chosen.device)[:count_sources]
            ]
            self.edge_mask[sources, target] = True
            self.recurrent[sources, target].normal_(0.0, 0.02)

        return chosen.tolist()

    def mask_gradients(self, mature_scale: float) -> None:
        active = self.active_mask.to(self.keys.dtype)
        row_scale = active * (
            mature_scale
            + (1.0 - mature_scale) * (1.0 - self.maturity)
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

    @torch.no_grad()
    def advance_maturity(self) -> None:
        increment = 1.0 / self.maturity_steps
        self.maturity[self.active_mask].add_(increment).clamp_(max=1.0)

    @torch.no_grad()
    def seal(self) -> None:
        self.maturity[self.active_mask] = 1.0

    @torch.no_grad()
    def repair_metadata(self) -> None:
        self.thresholds.data.clamp_(-1.0, 2.0)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, use_pool: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.pool_norm = RMSNorm(config.d_model) if use_pool else None
        self.pool = ThresholdRecurrentCellPool(config) if use_pool else None
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
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def pools(self) -> list[ThresholdRecurrentCellPool]:
        return [
            block.pool
            for block in self.blocks
            if block.pool is not None
        ]

    def load_compatible_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[list[str], list[str]]:
        migrated: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            new_name = name.replace(".input_vectors", ".read_vectors")
            new_name = new_name.replace(".output_vectors", ".write_vectors")
            new_name = new_name.replace(".recurrent_mask", ".edge_mask")
            if any(
                token in new_name
                for token in (
                    "bank_adapters",
                    "bank_gate_keys",
                    "bank_gate_bias",
                    "bank_ids",
                    "bank_active",
                    "bank_sealed",
                )
            ):
                continue
            migrated[new_name] = value

        incompatible = self.load_state_dict(migrated, strict=False)
        for pool in self.pools():
            pool.repair_metadata()
        return (
            list(incompatible.missing_keys),
            list(incompatible.unexpected_keys),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        stats: list[PoolStats] = []
        for block in self.blocks:
            x, item = block(x)
            if item is not None:
                stats.append(item)

        logits = self.lm_head(self.final_norm(x))
        loss = (
            None
            if labels is None
            else F.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=self.config.pad_token_id,
            )
        )

        def mean_metric(name: str) -> torch.Tensor:
            if not stats:
                return logits.new_tensor(0.0)
            return torch.stack(
                [getattr(item, name) for item in stats]
            ).mean()

        return {
            "logits": logits,
            "loss": loss,
            "growth_seeds": [item.seed for item in stats],
            "growth_parent_ids": [item.parent_ids for item in stats],
            "active_cells": [item.active_count for item in stats],
            "active_cell_ids": [item.active_ids for item in stats],
            "active_fraction": mean_metric("active_fraction"),
            "mean_gate": mean_metric("mean_gate"),
            "plastic_gate_mean": mean_metric("plastic_gate_mean"),
            "plastic_output_rms": mean_metric("plastic_output_rms"),
        }

    @torch.no_grad()
    def resize_vocabulary(self, new_size: int) -> None:
        old_size = self.config.vocab_size
        if new_size < old_size:
            raise ValueError("Vocabulary shrinking is unsupported")
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
        nn.init.normal_(embedding.weight, 0.0, 0.02)
        nn.init.normal_(head.weight, 0.0, 0.02)
        embedding.weight[:old_size].copy_(self.token_embedding.weight)
        head.weight[:old_size].copy_(self.lm_head.weight)
        self.token_embedding = embedding
        self.lm_head = head
        self.config.vocab_size = new_size

    @torch.no_grad()
    def allocate_cells(
        self,
        count: int,
        seeds: list[torch.Tensor] | None = None,
        parent_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        pools = self.pools()
        seeds = seeds or [None] * len(pools)
        parent_ids = parent_ids or [[] for _ in pools]
        return [
            pool.allocate(count, seed, parents)
            for pool, seed, parents in zip(pools, seeds, parent_ids)
        ]

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
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(
                ids[:, -self.config.max_seq_len :]
            )["logits"][:, -1]
            if temperature <= 0.0 or top_k == 1:
                token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k > 0:
                    threshold = logits.topk(
                        min(top_k, logits.size(-1))
                    ).values[:, -1:]
                    logits = logits.masked_fill(
                        logits < threshold,
                        float("-inf"),
                    )
                token = torch.multinomial(logits.softmax(dim=-1), 1)
            ids = torch.cat((ids, token), dim=1)
            if (
                eos_token_id is not None
                and torch.all(token == eos_token_id)
            ):
                break
        return ids
