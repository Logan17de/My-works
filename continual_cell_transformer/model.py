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
        self, q: torch.Tensor, k: torch.Tensor
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
                batch, length, self.heads, self.head_dim
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
    seed: torch.Tensor
    active_count: int
    bank_counts: dict[int, int]
    bank_top_ids: dict[int, list[int]]
    bank_gate_means: dict[int, torch.Tensor]
    bank_gate_maxes: dict[int, torch.Tensor]


class RecurrentConceptPool(nn.Module):
    """
    Stable append-only routing banks with local token-wise gates.

    Bank 0 is the original route and is always on. Every later bank:
      * competes only inside itself for top-k cells;
      * owns a separate output adapter;
      * owns a separate gate key and bias;
      * adds a gated residual rather than replacing the old route.

    This lets several banks cooperate while preventing a new bank from
    displacing an old bank's selected cells.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        cells, dim = config.max_cells, config.d_model
        self.top_k = config.top_k_cells
        self.steps = config.recurrent_steps
        self.fan_in = config.recurrent_fan_in
        self.temperature = config.cell_temperature
        self.maturity_steps = max(1, config.cell_maturity_steps)
        self.max_banks = config.max_cell_banks
        self.new_bank_adapter_scale = config.new_bank_adapter_scale
        self.bank_context_scale = config.bank_context_scale
        self.bank_gate_temperature = config.bank_gate_temperature
        self.new_bank_gate_bias = config.new_bank_gate_bias

        self.keys = nn.Parameter(torch.randn(cells, dim) * 0.02)
        self.read_vectors = nn.Parameter(torch.randn(cells, dim) * 0.02)
        self.write_vectors = nn.Parameter(torch.randn(cells, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(cells))
        self.recurrent = nn.Parameter(torch.zeros(cells, cells))

        self.bank_adapters = nn.Parameter(torch.zeros(self.max_banks, dim, dim))
        self.bank_gate_keys = nn.Parameter(torch.randn(self.max_banks, dim) * 0.02)
        self.bank_gate_bias = nn.Parameter(
            torch.full((self.max_banks,), self.new_bank_gate_bias)
        )

        active = torch.zeros(cells, dtype=torch.bool)
        active[: config.initial_active_cells] = True
        bank_ids = torch.full((cells,), -1, dtype=torch.long)
        bank_ids[: config.initial_active_cells] = 0
        bank_active = torch.zeros(self.max_banks, dtype=torch.bool)
        bank_active[0] = True

        self.register_buffer("active_mask", active)
        self.register_buffer("bank_ids", bank_ids)
        self.register_buffer("bank_active", bank_active)
        self.register_buffer(
            "bank_sealed",
            torch.zeros(self.max_banks, dtype=torch.bool),
        )
        self.register_buffer("maturity", torch.zeros(cells))
        self.register_buffer("usage_ema", torch.zeros(cells))
        self.register_buffer(
            "edge_mask",
            torch.zeros(cells, cells, dtype=torch.bool),
        )

        with torch.no_grad():
            self.keys.copy_(F.normalize(self.keys, dim=-1))
            self.read_vectors.copy_(F.normalize(self.read_vectors, dim=-1))
            self.bank_gate_keys.copy_(F.normalize(self.bank_gate_keys, dim=-1))

            identity = torch.eye(dim)
            self.bank_adapters[0].copy_(identity)
            self.bank_gate_bias[0] = 20.0
            for bank_id in range(1, self.max_banks):
                self.bank_adapters[bank_id].copy_(
                    identity * self.new_bank_adapter_scale
                )

            self._initialize_bank_edges(
                torch.arange(config.initial_active_cells),
            )

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def active_bank_count(self) -> int:
        return int(self.bank_active.sum().item())

    def _initialize_bank_edges(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        for target in indices.tolist():
            source_count = min(self.fan_in, indices.numel())
            sources = indices[
                torch.randperm(indices.numel(), device=indices.device)[
                    :source_count
                ]
            ]
            self.edge_mask[sources, target] = True
            self.recurrent[sources, target].normal_(0.0, 0.02)
            self.edge_mask[target, target] = True
            self.recurrent[target, target] = 0.01

    def _run_bank(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        keys = F.normalize(self.keys[indices], dim=-1)
        reads = F.normalize(self.read_vectors[indices], dim=-1)
        writes = self.write_vectors[indices]

        scores = (
            torch.einsum(
                "btd,cd->btc",
                F.normalize(x, dim=-1),
                keys,
            )
            / self.temperature
        )
        selected_count = min(self.top_k, indices.numel())
        values, local_indices = scores.topk(selected_count, dim=-1)
        weights = values.softmax(dim=-1)
        gates = torch.zeros_like(scores).scatter(
            dim=-1,
            index=local_indices,
            src=weights,
        )

        drive = gates * F.silu(
            torch.einsum("btd,cd->btc", x, reads) + self.bias[indices]
        )
        recurrent = (
            self.recurrent[indices][:, indices]
            * self.edge_mask[indices][:, indices].to(x.dtype)
        )
        activity = drive
        for _ in range(self.steps):
            activity = F.silu(
                drive + torch.einsum("btc,cd->btd", activity, recurrent)
            )

        output = (
            torch.einsum("btc,cd->btd", activity, writes)
            / math.sqrt(max(1, selected_count))
        )

        with torch.no_grad():
            usage = (gates > 0).float().mean(dim=(0, 1))
            global_usage = torch.zeros_like(self.usage_ema)
            global_usage[indices] = usage
            self.usage_ema.mul_(0.99).add_(global_usage, alpha=0.01)

            aggregate = gates.sum(dim=(0, 1))
            top_n = min(self.top_k, aggregate.numel())
            top_local = aggregate.topk(top_n).indices
            top_global = indices[top_local].tolist()

        confidence = torch.sigmoid(values[..., 0]).mean()
        return output, confidence, top_global

    def _bank_gate(self, x: torch.Tensor, bank_id: int) -> torch.Tensor:
        if bank_id == 0:
            return torch.ones(
                (*x.shape[:2], 1),
                device=x.device,
                dtype=x.dtype,
            )

        key = F.normalize(self.bank_gate_keys[bank_id], dim=-1)
        score = (
            torch.einsum("btd,d->bt", F.normalize(x, dim=-1), key)
            / self.bank_gate_temperature
            + self.bank_gate_bias[bank_id]
        )
        return torch.sigmoid(score).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PoolStats]:
        accumulated = torch.zeros_like(x)
        confidences: list[torch.Tensor] = []
        bank_counts: dict[int, int] = {}
        bank_top_ids: dict[int, list[int]] = {}
        bank_gate_means: dict[int, torch.Tensor] = {}
        bank_gate_maxes: dict[int, torch.Tensor] = {}

        active_banks = torch.nonzero(
            self.bank_active,
            as_tuple=False,
        ).flatten()

        for bank_tensor in active_banks:
            bank_id = int(bank_tensor.item())
            indices = torch.nonzero(
                self.active_mask & (self.bank_ids == bank_id),
                as_tuple=False,
            ).flatten()
            if indices.numel() == 0:
                continue

            bank_input = x
            if bank_id > 0:
                bank_input = x + self.bank_context_scale * accumulated.detach()

            raw_output, confidence, top_ids = self._run_bank(
                bank_input,
                indices,
            )

            if bank_id == 0:
                adapted_output = raw_output
            else:
                adapted_output = torch.einsum(
                    "btd,de->bte",
                    raw_output,
                    self.bank_adapters[bank_id],
                )

            route_gate = self._bank_gate(x, bank_id)
            accumulated = accumulated + route_gate * adapted_output

            confidences.append(confidence)
            bank_counts[bank_id] = int(indices.numel())
            bank_top_ids[bank_id] = top_ids
            bank_gate_means[bank_id] = route_gate.mean()
            bank_gate_maxes[bank_id] = route_gate.max()

        confidence = (
            torch.stack(confidences).mean()
            if confidences
            else x.new_tensor(0.0)
        )
        return accumulated, PoolStats(
            confidence=confidence,
            seed=x.detach().mean(dim=(0, 1)),
            active_count=self.active_count,
            bank_counts=bank_counts,
            bank_top_ids=bank_top_ids,
            bank_gate_means=bank_gate_means,
            bank_gate_maxes=bank_gate_maxes,
        )

    @torch.no_grad()
    def seal_active_banks(self) -> None:
        active_ids = torch.unique(self.bank_ids[self.active_mask])
        active_ids = active_ids[active_ids >= 0]
        if active_ids.numel() > 0:
            self.bank_sealed[active_ids] = True
        self.maturity[self.active_mask] = 1.0

    @torch.no_grad()
    def allocate_new_bank(
        self,
        count: int,
        seed: torch.Tensor | None = None,
        seal_existing: bool = True,
    ) -> dict[str, object]:
        if count <= 0:
            return {"bank_id": None, "cells": []}

        if seal_existing:
            self.seal_active_banks()

        free_banks = torch.nonzero(
            ~self.bank_active,
            as_tuple=False,
        ).flatten()
        dormant = torch.nonzero(
            ~self.active_mask,
            as_tuple=False,
        ).flatten()

        if free_banks.numel() == 0 or dormant.numel() == 0:
            return {"bank_id": None, "cells": []}

        bank_id = int(free_banks[0].item())
        chosen = dormant[: min(count, dormant.numel())]

        if seed is None:
            seed = torch.randn_like(self.keys[0])
        seed = F.normalize(seed.to(self.keys), dim=-1)

        for row in chosen.tolist():
            noise = 0.05 * torch.randn_like(seed)
            self.keys[row].copy_(F.normalize(seed + noise, dim=-1))
            self.read_vectors[row].copy_(F.normalize(seed + noise, dim=-1))
            self.write_vectors[row].normal_(0.0, 0.01)
            self.bias[row] = 0.0
            self.maturity[row] = 0.0
            self.usage_ema[row] = 0.0
            self.active_mask[row] = True
            self.bank_ids[row] = bank_id

        dim = self.bank_adapters.size(-1)
        self.bank_adapters[bank_id].copy_(
            torch.eye(
                dim,
                device=self.bank_adapters.device,
                dtype=self.bank_adapters.dtype,
            )
            * self.new_bank_adapter_scale
        )
        self.bank_gate_keys[bank_id].copy_(seed)
        self.bank_gate_bias[bank_id] = self.new_bank_gate_bias

        self.bank_active[bank_id] = True
        self.bank_sealed[bank_id] = False
        self._initialize_bank_edges(chosen)

        return {
            "bank_id": bank_id,
            "cells": chosen.tolist(),
            "initial_gate_bias": float(self.bank_gate_bias[bank_id].item()),
        }

    def mask_gradients(self, sealed_scale: float) -> None:
        safe_bank_ids = self.bank_ids.clamp_min(0)
        sealed_rows = self.bank_sealed[safe_bank_ids]
        row_scale = self.active_mask.to(self.keys) * torch.where(
            sealed_rows,
            torch.full_like(self.maturity, sealed_scale),
            torch.ones_like(self.maturity),
        )

        for parameter in (
            self.keys,
            self.read_vectors,
            self.write_vectors,
        ):
            if parameter.grad is not None:
                parameter.grad.mul_(row_scale[:, None])

        if self.bias.grad is not None:
            self.bias.grad.mul_(row_scale)

        if self.recurrent.grad is not None:
            same_bank = (
                self.bank_ids[:, None] == self.bank_ids[None, :]
            ) & (self.bank_ids[:, None] >= 0)
            edge_scale = torch.minimum(
                row_scale[:, None],
                row_scale[None, :],
            )
            edge_scale = (
                edge_scale
                * same_bank.to(edge_scale.dtype)
                * self.edge_mask.to(edge_scale.dtype)
            )
            self.recurrent.grad.mul_(edge_scale)

        bank_scale = torch.zeros(
            self.max_banks,
            device=self.keys.device,
            dtype=self.keys.dtype,
        )
        for bank_id in range(1, self.max_banks):
            if not self.bank_active[bank_id]:
                continue
            bank_scale[bank_id] = (
                sealed_scale if self.bank_sealed[bank_id] else 1.0
            )

        if self.bank_adapters.grad is not None:
            self.bank_adapters.grad.mul_(bank_scale[:, None, None])
        if self.bank_gate_keys.grad is not None:
            self.bank_gate_keys.grad.mul_(bank_scale[:, None])
        if self.bank_gate_bias.grad is not None:
            self.bank_gate_bias.grad.mul_(bank_scale)

    @torch.no_grad()
    def advance_maturity(self) -> None:
        unsealed_rows = self.active_mask & ~self.bank_sealed[
            self.bank_ids.clamp_min(0)
        ]
        self.maturity[unsealed_rows].add_(
            1.0 / self.maturity_steps
        ).clamp_(max=1.0)

    @torch.no_grad()
    def repair_bank_metadata(self) -> None:
        """Upgrade pre-bank and pre-gate checkpoints in place."""
        orphaned = self.active_mask & (self.bank_ids < 0)
        self.bank_ids[orphaned] = 0

        self.bank_active.zero_()
        active_ids = torch.unique(self.bank_ids[self.active_mask])
        active_ids = active_ids[
            (active_ids >= 0) & (active_ids < self.max_banks)
        ]
        if active_ids.numel() > 0:
            self.bank_active[active_ids] = True
        elif self.active_mask.any():
            self.bank_active[0] = True
            self.bank_ids[self.active_mask] = 0

        self.bank_gate_bias[0] = 20.0
        for bank_id in range(1, self.max_banks):
            if not self.bank_active[bank_id]:
                self.bank_gate_bias[bank_id] = self.new_bank_gate_bias

    def bank_summary(self) -> dict[int, int]:
        summary: dict[int, int] = {}
        for bank_id in torch.nonzero(
            self.bank_active,
            as_tuple=False,
        ).flatten().tolist():
            count = int(
                (
                    self.active_mask
                    & (self.bank_ids == bank_id)
                ).sum().item()
            )
            if count:
                summary[int(bank_id)] = count
        return summary


class Block(nn.Module):
    def __init__(self, config: ModelConfig, use_pool: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.pool_norm = RMSNorm(config.d_model) if use_pool else None
        self.pool = RecurrentConceptPool(config) if use_pool else None
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)
        self.pool_scale = config.cell_residual_scale

    def forward(
        self, x: torch.Tensor
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

    def pools(self) -> list[RecurrentConceptPool]:
        return [
            block.pool
            for block in self.blocks
            if block.pool is not None
        ]

    def load_compatible_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[list[str], list[str]]:
        incompatible = self.load_state_dict(
            state_dict,
            strict=False,
        )
        for pool in self.pools():
            pool.repair_bank_metadata()
        return (
            list(incompatible.missing_keys),
            list(incompatible.unexpected_keys),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
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

        gate_tensors = [
            gate
            for item in stats
            for bank_id, gate in item.bank_gate_means.items()
            if bank_id > 0
        ]
        plastic_gate_mean = (
            torch.stack(gate_tensors).mean()
            if gate_tensors
            else logits.new_tensor(0.0)
        )

        return {
            "logits": logits,
            "loss": loss,
            "cell_confidence": (
                torch.stack([item.confidence for item in stats]).mean()
                if stats
                else logits.new_tensor(1.0)
            ),
            "growth_seeds": [item.seed for item in stats],
            "active_cells": [item.active_count for item in stats],
            "cell_banks": [item.bank_counts for item in stats],
            "bank_top_ids": [item.bank_top_ids for item in stats],
            "bank_gate_means": [
                {
                    bank_id: float(value.detach().item())
                    for bank_id, value in item.bank_gate_means.items()
                }
                for item in stats
            ],
            "bank_gate_maxes": [
                {
                    bank_id: float(value.detach().item())
                    for bank_id, value in item.bank_gate_maxes.items()
                }
                for item in stats
            ],
            "plastic_gate_mean": plastic_gate_mean,
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

    def mask_cell_gradients(self, sealed_scale: float) -> None:
        for pool in self.pools():
            pool.mask_gradients(sealed_scale)

    @torch.no_grad()
    def advance_maturity(self) -> None:
        for pool in self.pools():
            pool.advance_maturity()

    @torch.no_grad()
    def seal_active_cells(self) -> None:
        for pool in self.pools():
            pool.seal_active_banks()

    @torch.no_grad()
    def allocate_new_bank(
        self,
        count: int,
        seeds: list[torch.Tensor] | None = None,
        seal_existing: bool = True,
    ) -> list[dict[str, object]]:
        pools = self.pools()
        seeds = seeds or [None] * len(pools)
        return [
            pool.allocate_new_bank(
                count=count,
                seed=seed,
                seal_existing=seal_existing,
            )
            for pool, seed in zip(pools, seeds)
        ]

    def bank_summaries(self) -> list[dict[int, int]]:
        return [pool.bank_summary() for pool in self.pools()]

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
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                threshold = logits.topk(
                    min(top_k, logits.size(-1))
                ).values[:, -1:]
                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf"),
                )
            token = torch.multinomial(
                logits.softmax(dim=-1),
                1,
            )
            ids = torch.cat((ids, token), dim=1)
            if (
                eos_token_id is not None
                and torch.all(token == eos_token_id)
            ):
                break
        return ids
