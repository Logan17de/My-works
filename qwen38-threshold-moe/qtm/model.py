from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .experts import ExpertLayerCache, StreamedMoEBlock
from .router import ThresholdController

try:
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet
except Exception as exc:  # pragma: no cover
    Qwen4ExpTextGatedDeltaNet = None
    _QWEN_IMPORT_ERROR = exc
else:
    _QWEN_IMPORT_ERROR = None


class QwenRMSNorm(nn.Module):
    """Qwen zero-centered RMSNorm: output = norm(x) * (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6, group_size: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.group_size = group_size or dim
        if dim % self.group_size != 0:
            raise ValueError("RMSNorm dim must be divisible by group_size")
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        xf = x.float().view(*x.shape[:-1], -1, self.group_size)
        xf = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        xf = xf.reshape_as(x.float())
        return (xf * (1.0 + self.weight.float())).to(orig)


class QwenGatedResidual(nn.Module):
    """Four-branch Gated Residual block following Qwen3.8/Qwen4-Exp."""

    def __init__(self, hidden_size: int, hc_count: int, lowrank: int, eps: float, *, use_combine: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        total = hidden_size * hc_count
        self.norm = QwenRMSNorm(total, eps=eps, group_size=hidden_size)
        self.down = nn.Linear(total, lowrank, bias=False)
        self.up = nn.Linear(lowrank, total, bias=False)
        self.inject = nn.Linear(total, hc_count, bias=False) if use_combine else None

    def forward(self, hyper_input: torch.Tensor):
        normed = self.norm(hyper_input)
        mix = F.silu(self.down(normed) / self.hc_count)
        mix = torch.sigmoid(self.up(mix))
        mix = mix.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (mix * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        if self.inject is None:
            return mixed
        injection_weights = 2.0 * torch.sigmoid(self.inject(normed) / self.hc_count)
        return mixed, hyper_input, injection_weights


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_partial_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    q_rope, q_rest = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rope, k_rest = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rope = q_rope * cos + _rotate_half(q_rope) * sin
    k_rope = k_rope * cos + _rotate_half(k_rope) * sin
    return torch.cat((q_rope, q_rest), dim=-1), torch.cat((k_rope, k_rest), dim=-1)


class QwenGatedAttention(nn.Module):
    """Qwen gated GQA full attention with partial RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = QwenRMSNorm(head_dim, eps=eps)
        self.k_norm = QwenRMSNorm(head_dim, eps=eps)

    def forward(self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q_raw, gate = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim * 2).chunk(2, dim=-1)
        q = self.q_norm(q_raw).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_partial_rope(q, k, cos, sin)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            scale=self.head_dim ** -0.5,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.num_heads * self.head_dim)
        gate = gate.reshape(bsz, seqlen, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)


def _qwen_gdn_config(model_cfg: dict[str, Any]) -> SimpleNamespace:
    layer_types = [
        "full_attention" if (idx + 1) % int(model_cfg["full_attention_interval"]) == 0 else "linear_attention"
        for idx in range(int(model_cfg["num_layers"]))
    ]
    return SimpleNamespace(
        hidden_size=int(model_cfg["hidden_size"]),
        linear_num_value_heads=int(model_cfg["linear_value_heads"]),
        linear_num_key_heads=int(model_cfg["linear_key_heads"]),
        linear_key_head_dim=int(model_cfg["linear_key_head_dim"]),
        linear_value_head_dim=int(model_cfg["linear_value_head_dim"]),
        linear_conv_kernel_dim=int(model_cfg["linear_conv_kernel_dim"]),
        hidden_act="silu",
        rms_norm_eps=float(model_cfg["rms_norm_eps"]),
        output_gate_type="sigmoid",  # Qwen GDN output gate; unrelated to MoE router.
        layer_types=layer_types,
    )


class QTMDecoderLayer(nn.Module):
    def __init__(self, cfg: dict[str, Any], layer_idx: int, controller: ThresholdController) -> None:
        super().__init__()
        hidden = int(cfg["hidden_size"])
        hc = int(cfg["hc_count"])
        eps = float(cfg["rms_norm_eps"])
        self.layer_idx = layer_idx
        self.is_full_attention = (layer_idx + 1) % int(cfg["full_attention_interval"]) == 0
        self.attn_hc = QwenGatedResidual(hidden, hc, int(cfg["hc_lowrank"]), eps)
        self.mlp_hc = QwenGatedResidual(hidden, hc, int(cfg["hc_lowrank"]), eps)
        if self.is_full_attention:
            rotary_dim = int(int(cfg["attention_head_dim"]) * float(cfg["partial_rotary_factor"]))
            self.token_mixer = QwenGatedAttention(
                hidden,
                int(cfg["attention_heads"]),
                int(cfg["kv_heads"]),
                int(cfg["attention_head_dim"]),
                rotary_dim,
                eps,
                float(cfg["attention_dropout"]),
            )
        else:
            if Qwen4ExpTextGatedDeltaNet is None:
                raise ImportError(
                    "transformers with Qwen4-Exp support is required for the exact Gated DeltaNet skeleton"
                ) from _QWEN_IMPORT_ERROR
            self.token_mixer = Qwen4ExpTextGatedDeltaNet(_qwen_gdn_config(cfg), layer_idx)
        self.moe = StreamedMoEBlock(
            layer_idx=layer_idx,
            hidden_size=hidden,
            num_experts=int(cfg["num_experts"]),
            num_shared_experts=int(cfg["num_shared_experts"]),
            top_k=int(cfg["top_k"]),
            intermediate_size=int(cfg["expert_intermediate_size"]),
            controller=controller,
            normalize_topk=bool(cfg["router_normalize_topk"]),
            init_std=float(cfg["initializer_range"]),
        )

    def attach_cache(self, cache: ExpertLayerCache) -> None:
        self.moe.attach_cache(cache)

    def forward(
        self,
        hyper_input: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        record_routing: bool,
    ) -> torch.Tensor:
        x, residual, inject_w = self.attn_hc(hyper_input)
        if self.is_full_attention:
            assert position_embeddings is not None
            x = self.token_mixer(x, position_embeddings)
        else:
            x = self.token_mixer(x, cache_params=None, attention_mask=None)
        hyper_input = residual + (x.unsqueeze(-2) * inject_w.unsqueeze(-1)).flatten(-2)

        x, residual, inject_w = self.mlp_hc(hyper_input)
        x = self.moe(x, record_routing=record_routing)
        return residual + (x.unsqueeze(-2) * inject_w.unsqueeze(-1)).flatten(-2)


class QwenThresholdMoE(nn.Module):
    def __init__(self, cfg: dict[str, Any], controller: ThresholdController) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = int(cfg["hidden_size"])
        self.hc_count = int(cfg["hc_count"])
        self.vocab_size = int(cfg["vocab_size"])
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([QTMDecoderLayer(cfg, i, controller) for i in range(int(cfg["num_layers"]))])
        self.final_mixer = QwenGatedResidual(
            self.hidden_size, self.hc_count, int(cfg["hc_lowrank"]), float(cfg["rms_norm_eps"]), use_combine=False
        )
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self._rope_cache: dict[tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        self._init_resident_weights(float(cfg["initializer_range"]))

    def _init_resident_weights(self, std: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def attach_cache(self, cache: ExpertLayerCache) -> None:
        for layer in self.layers:
            layer.attach_cache(cache)

    def initial_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        return x.repeat(1, 1, self.hc_count)

    def position_embeddings(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seqlen, str(device), dtype)
        if key in self._rope_cache:
            return self._rope_cache[key]
        head_dim = int(self.cfg["attention_head_dim"])
        rotary_dim = int(head_dim * float(self.cfg["partial_rotary_factor"]))
        theta = float(self.cfg["rope_theta"])
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
        positions = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)[None, None, :, :]
        sin = emb.sin().to(dtype)[None, None, :, :]
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def forward_layer(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        *,
        record_routing: bool = False,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        pos = self.position_embeddings(hidden.shape[1], hidden.device, hidden.dtype) if layer.is_full_attention else None
        return layer(hidden, position_embeddings=pos, record_routing=record_routing)

    def final_hidden(self, hyper_input: torch.Tensor) -> torch.Tensor:
        return self.final_mixer(hyper_input)

    def resident_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
