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
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.repeat_interleave(torch.outer(positions, inv), 2, dim=-1)
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
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split, (q, k, v))
        q, k = self.rope(q, k)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(output.transpose(1, 2).contiguous().view(batch, length, dim))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


@dataclass
class BankStats:
    confidence: torch.Tensor
    max_similarity: torch.Tensor
    selection_counts: torch.Tensor
    active_count: int


class CellBank(nn.Module):
    """Independent cell bank. Stable and plastic banks never share top-k competition."""

    def __init__(
        self,
        capacity: int,
        d_model: int,
        top_k: int,
        recurrent_steps: int,
        recurrent_fan_in: int,
        temperature: float,
        initially_active: int,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.d_model = d_model
        self.top_k = top_k
        self.steps = recurrent_steps
        self.fan_in = recurrent_fan_in
        self.temperature = temperature

        self.keys = nn.Parameter(torch.randn(capacity, d_model) * 0.02)
        self.read_vectors = nn.Parameter(torch.randn(capacity, d_model) * 0.02)
        self.write_vectors = nn.Parameter(torch.randn(capacity, d_model) * 0.01)
        self.bias = nn.Parameter(torch.zeros(capacity))
        self.recurrent = nn.Parameter(torch.zeros(capacity, capacity))
        self.register_buffer("edge_mask", torch.zeros(capacity, capacity, dtype=torch.bool))
        self.register_buffer("active_count_tensor", torch.tensor(initially_active, dtype=torch.long))
        self.register_buffer("usage_ema", torch.zeros(capacity))

        with torch.no_grad():
            self.keys.copy_(F.normalize(self.keys, dim=-1))
            self.read_vectors.copy_(F.normalize(self.read_vectors, dim=-1))
            for target in range(initially_active):
                sources = torch.randperm(initially_active)[: min(self.fan_in, initially_active)]
                self.edge_mask[sources, target] = True
                self.recurrent[sources, target].normal_(0, 0.02)
                self.edge_mask[target, target] = True
                self.recurrent[target, target] = 0.01

    @property
    def active_count(self) -> int:
        return int(self.active_count_tensor.item())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, BankStats]:
        count = self.active_count
        if count == 0:
            zero_counts = torch.zeros(self.capacity, device=x.device)
            stats = BankStats(
                confidence=x.new_tensor(0.0),
                max_similarity=x.new_zeros(x.shape[:2]),
                selection_counts=zero_counts,
                active_count=0,
            )
            return torch.zeros_like(x), stats

        keys = F.normalize(self.keys[:count], dim=-1)
        reads = F.normalize(self.read_vectors[:count], dim=-1)
        writes = self.write_vectors[:count]
        cosine = torch.einsum("btd,cd->btc", F.normalize(x, dim=-1), keys)
        scores = cosine / self.temperature
        k = min(self.top_k, count)
        values, indices = scores.topk(k, dim=-1)
        weights = values.softmax(dim=-1)
        gates = torch.zeros_like(scores).scatter(dim=-1, index=indices, src=weights)
        drive = gates * F.silu(torch.einsum("btd,cd->btc", x, reads) + self.bias[:count])

        recurrent = self.recurrent[:count, :count]
        recurrent = recurrent * self.edge_mask[:count, :count].to(x.dtype)
        activity = drive
        for _ in range(self.steps):
            activity = F.silu(drive + torch.einsum("btc,cd->btd", activity, recurrent))

        output = torch.einsum("btc,cd->btd", activity, writes) / math.sqrt(max(1, k))
        with torch.no_grad():
            local_counts = (gates > 0).float().sum(dim=(0, 1))
            global_counts = torch.zeros(self.capacity, device=x.device)
            global_counts[:count] = local_counts
            normalized_use = local_counts / max(1, x.shape[0] * x.shape[1])
            self.usage_ema[:count].mul_(0.99).add_(normalized_use, alpha=0.01)

        stats = BankStats(
            confidence=torch.sigmoid(values[..., 0]).mean(),
            max_similarity=cosine.max(dim=-1).values,
            selection_counts=global_counts,
            active_count=count,
        )
        return output, stats

    @torch.no_grad()
    def allocate_once(self, count: int, seed: torch.Tensor) -> list[int]:
        if self.active_count != 0:
            raise RuntimeError(
                f"Plastic bank already contains {self.active_count} cells. "
                "V2 allocation is intentionally one-time."
            )
        count = min(count, self.capacity)
        if count <= 0:
            return []

        seed = F.normalize(seed.to(self.keys), dim=-1)
        for row in range(count):
            noise = torch.randn_like(seed) * 0.05
            self.keys[row].copy_(F.normalize(seed + noise, dim=-1))
            self.read_vectors[row].copy_(F.normalize(seed + noise, dim=-1))
            self.write_vectors[row].normal_(0, 0.01)
            self.bias[row].zero_()
            sources = torch.randperm(count)[: min(self.fan_in, count)]
            self.edge_mask[sources, row] = True
            self.recurrent[sources, row].normal_(0, 0.02)
            self.edge_mask[row, row] = True
            self.recurrent[row, row] = 0.01
        self.active_count_tensor.fill_(count)
        return list(range(count))


class PlasticAdapter(nn.Module):
    """Small adapter used only by the plastic bank; it cannot alter the stable path."""

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.down.weight, 0, 0.02)
        nn.init.normal_(self.up.weight, 0, 0.005)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(self.norm(x))))


@dataclass
class RouteStats:
    stable_counts: torch.Tensor
    plastic_counts: torch.Tensor
    stable_active: int
    plastic_active: int
    plastic_gate_mean: torch.Tensor
    plastic_gate_max: torch.Tensor
    seed: torch.Tensor


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, use_cells: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.pool_norm = RMSNorm(cfg.d_model) if use_cells else None
        self.stable_pool = (
            CellBank(
                cfg.stable_cells,
                cfg.d_model,
                cfg.stable_top_k,
                cfg.recurrent_steps,
                cfg.recurrent_fan_in,
                cfg.cell_temperature,
                cfg.stable_cells,
            )
            if use_cells
            else None
        )
        self.plastic_pool = (
            CellBank(
                cfg.plastic_capacity,
                cfg.d_model,
                cfg.plastic_top_k,
                cfg.recurrent_steps,
                cfg.recurrent_fan_in,
                cfg.cell_temperature,
                0,
            )
            if use_cells
            else None
        )
        self.plastic_adapter = PlasticAdapter(cfg.d_model, cfg.adapter_rank) if use_cells else None
        self.plastic_gate_bias = (
            nn.Parameter(torch.tensor(cfg.plastic_gate_bias_init)) if use_cells else None
        )
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.stable_scale = cfg.stable_residual_scale
        self.plastic_scale = cfg.plastic_residual_scale
        self.gate_threshold = cfg.plastic_gate_threshold
        self.gate_sharpness = cfg.plastic_gate_sharpness

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RouteStats | None]:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        stats = None
        if self.stable_pool is not None and self.pool_norm is not None:
            stable_source = self.pool_norm(x)
            stable_delta, stable_stats = self.stable_pool(stable_source)
            x = x + self.stable_scale * self.dropout(stable_delta)

            plastic_source = self.pool_norm(x)
            plastic_delta, plastic_stats = self.plastic_pool(plastic_source)
            if plastic_stats.active_count > 0:
                relative_match = (
                    plastic_stats.max_similarity - stable_stats.max_similarity
                )
                gate = torch.sigmoid(
                    self.gate_sharpness
                    * (relative_match - self.gate_threshold)
                    + self.plastic_gate_bias
                ).unsqueeze(-1)
                adapted = self.plastic_adapter(plastic_delta)
                x = x + self.plastic_scale * gate * self.dropout(adapted)
            else:
                gate = x.new_zeros((*x.shape[:2], 1))

            stats = RouteStats(
                stable_counts=stable_stats.selection_counts,
                plastic_counts=plastic_stats.selection_counts,
                stable_active=stable_stats.active_count,
                plastic_active=plastic_stats.active_count,
                plastic_gate_mean=gate.mean(),
                plastic_gate_max=gate.max(),
                seed=plastic_source.detach().mean(dim=(0, 1)),
            )

        x = x + self.ffn(self.ffn_norm(x))
        return x, stats


class ContinualCellTransformerV2(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(
            cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id
        )
        self.blocks = nn.ModuleList(
            [Block(cfg, index in cfg.cell_layers) for index in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0, 0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len.")
        x = self.token_embedding(input_ids)
        route_stats: list[RouteStats] = []
        for block in self.blocks:
            x, stats = block(x)
            if stats is not None:
                route_stats.append(stats)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=self.config.pad_token_id,
            )

        gate_mean = (
            torch.stack([item.plastic_gate_mean for item in route_stats]).mean()
            if route_stats
            else logits.new_tensor(0.0)
        )
        return {
            "logits": logits,
            "loss": loss,
            "plastic_gate_mean": gate_mean,
            "route_stats": route_stats,
            "allocation_seeds": [item.seed for item in route_stats],
            "stable_cells": [item.stable_active for item in route_stats],
            "plastic_cells": [item.plastic_active for item in route_stats],
        }

    def cell_blocks(self) -> list[Block]:
        return [block for block in self.blocks if block.stable_pool is not None]

    @torch.no_grad()
    def allocate_plastic_once(self, count: int, seeds: list[torch.Tensor]) -> list[list[int]]:
        blocks = self.cell_blocks()
        if len(blocks) != len(seeds):
            raise ValueError("Expected one allocation seed per cell block.")
        return [
            block.plastic_pool.allocate_once(count, seed)
            for block, seed in zip(blocks, seeds)
        ]

    def freeze_for_continual_learning(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for block in self.cell_blocks():
            for parameter in block.plastic_pool.parameters():
                parameter.requires_grad = True
            for parameter in block.plastic_adapter.parameters():
                parameter.requires_grad = True
            block.plastic_gate_bias.requires_grad = True

    def plastic_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @torch.no_grad()
    def resize_vocabulary(self, new_size: int) -> None:
        old_size = self.config.vocab_size
        if new_size < old_size:
            raise ValueError("Vocabulary shrinking is unsupported.")
        if new_size == old_size:
            return
        embedding = nn.Embedding(
            new_size, self.config.d_model, padding_idx=self.config.pad_token_id
        ).to(self.token_embedding.weight)
        head = nn.Linear(self.config.d_model, new_size, bias=False).to(self.lm_head.weight)
        nn.init.normal_(embedding.weight, 0, 0.02)
        nn.init.normal_(head.weight, 0, 0.02)
        embedding.weight[:old_size].copy_(self.token_embedding.weight)
        head.weight[:old_size].copy_(self.lm_head.weight)
        self.token_embedding = embedding
        self.lm_head = head
        self.config.vocab_size = new_size

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 24,
        temperature: float = 0.0,
        top_k: int = 1,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = ids[:, -self.config.max_seq_len :]
            logits = self(context)["logits"][:, -1]
            if temperature <= 0 or top_k == 1:
                token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k > 0:
                    threshold = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                token = torch.multinomial(logits.softmax(dim=-1), 1)
            ids = torch.cat((ids, token), dim=1)
            if eos_token_id is not None and torch.all(token == eos_token_id):
                break
        return ids


def config_from_v1(v1_config: dict[str, Any], stable_cells: int, plastic_capacity: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=int(v1_config["vocab_size"]),
        d_model=int(v1_config["d_model"]),
        n_layers=int(v1_config["n_layers"]),
        n_heads=int(v1_config["n_heads"]),
        d_ff=int(v1_config["d_ff"]),
        max_seq_len=int(v1_config["max_seq_len"]),
        dropout=float(v1_config.get("dropout", 0.1)),
        rope_base=float(v1_config.get("rope_base", 10_000.0)),
        cell_layers=tuple(v1_config.get("cell_layers", (1, 3))),
        stable_cells=stable_cells,
        stable_top_k=min(int(v1_config.get("top_k_cells", 8)), stable_cells),
        plastic_capacity=plastic_capacity,
        plastic_top_k=min(4, plastic_capacity),
        recurrent_steps=int(v1_config.get("recurrent_steps", 2)),
        recurrent_fan_in=int(v1_config.get("recurrent_fan_in", 8)),
        cell_temperature=float(v1_config.get("cell_temperature", 0.7)),
        stable_residual_scale=float(v1_config.get("cell_residual_scale", 0.10)),
        pad_token_id=int(v1_config.get("pad_token_id", 0)),
        bos_token_id=int(v1_config.get("bos_token_id", 1)),
        eos_token_id=int(v1_config.get("eos_token_id", 2)),
    )


@torch.no_grad()
def migrate_v1_state(model: ContinualCellTransformerV2, v1_state: dict[str, torch.Tensor]) -> None:
    """Copy a trained V1 model into V2 while preserving the stable path."""
    target = model.state_dict()

    for name, value in v1_state.items():
        if ".pool." in name:
            continue
        if name in target and target[name].shape == value.shape:
            target[name].copy_(value)

    for layer_index in model.config.cell_layers:
        old_prefix = f"blocks.{layer_index}.pool."
        new_prefix = f"blocks.{layer_index}.stable_pool."
        active_mask = v1_state.get(old_prefix + "active_mask")
        if active_mask is None:
            raise KeyError(f"Missing V1 active mask for layer {layer_index}.")
        active_indices = active_mask.nonzero().flatten()
        count = model.config.stable_cells
        if active_indices.numel() < count:
            raise ValueError(
                f"V1 layer {layer_index} has only {active_indices.numel()} active cells; "
                f"V2 expects {count}."
            )
        active_indices = active_indices[:count]

        for field in ("keys", "read_vectors", "write_vectors", "bias"):
            old_value = v1_state[old_prefix + field].index_select(0, active_indices)
            target[new_prefix + field].copy_(old_value)

        old_recurrent = v1_state[old_prefix + "recurrent"]
        old_edges = v1_state[old_prefix + "edge_mask"]
        target[new_prefix + "recurrent"].copy_(
            old_recurrent.index_select(0, active_indices).index_select(1, active_indices)
        )
        target[new_prefix + "edge_mask"].copy_(
            old_edges.index_select(0, active_indices).index_select(1, active_indices)
        )
        if old_prefix + "usage_ema" in v1_state:
            target[new_prefix + "usage_ema"].copy_(
                v1_state[old_prefix + "usage_ema"].index_select(0, active_indices)
            )

    model.load_state_dict(target, strict=True)
