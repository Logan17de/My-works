from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .experts import ExpertLayerCache, ExpertStore
from .model import QwenThresholdMoE
from .router import ThresholdController


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-6)
    return float(((a - b).abs() / denom).max().item())


def validate_layerwise_recompute(device: str = "cuda") -> dict[str, float]:
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    cfg = {
        "tokenizer": "Qwen/Qwen3.8-Flash-Next",
        "vocab_size": 512,
        "hidden_size": 32,
        "num_layers": 1,
        "num_experts": 8,
        "num_shared_experts": 2,
        "top_k": 2,
        "expert_intermediate_size": 64,
        "router_normalize_topk": True,
        "hc_count": 4,
        "hc_lowrank": 8,
        "full_attention_interval": 1,
        "attention_heads": 4,
        "kv_heads": 1,
        "attention_head_dim": 16,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000.0,
        "max_position_embeddings": 128,
        "linear_key_heads": 2,
        "linear_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
    }
    controller = ThresholdController(1, 8, enabled=False, warmup_tokens=-1, imbalance_threshold=.3, block_fraction=.3)
    with tempfile.TemporaryDirectory() as td:
        store = ExpertStore(Path(td) / "experts", num_layers=1, num_experts=8, hidden_size=32, intermediate_size=64, storage="mmap")
        store.initialize_all()
        model = QwenThresholdMoE(cfg, controller).to(dev, dtype=dtype)
        cache = ExpertLayerCache(store, device=dev, capacity=1, compute_dtype=dtype)
        model.attach_cache(cache)
        cache.ensure_window(0, +1)
        layer = model.layers[0]

        torch.manual_seed(123)
        x0 = torch.randn(1, 16, cfg["hidden_size"] * cfg["hc_count"], device=dev, dtype=dtype)
        x = x0.clone().detach().requires_grad_(True)
        out = model.forward_layer(0, x, record_routing=False)
        loss = out.float().square().mean()
        loss.backward()
        direct_x = x.grad.detach().float().cpu()
        direct_resident = [p.grad.detach().float().cpu().clone() if p.grad is not None else None for p in layer.parameters()]
        exp = cache.get(0)
        direct_gu = exp.gate_up.grad.detach().float().cpu().clone()
        direct_dn = exp.down.grad.detach().float().cpu().clone()

        for p in layer.parameters():
            p.grad = None
        exp.zero_grad()

        with torch.no_grad():
            forward_copy = model.forward_layer(0, x0, record_routing=False)
        y = forward_copy.detach().requires_grad_(True)
        proxy_loss = y.float().square().mean()
        upstream = torch.autograd.grad(proxy_loss, y)[0]

        xr = x0.clone().detach().requires_grad_(True)
        recomputed = model.forward_layer(0, xr, record_routing=False)
        torch.autograd.backward(recomputed, upstream)
        recompute_x = xr.grad.detach().float().cpu()
        recompute_resident = [p.grad.detach().float().cpu().clone() if p.grad is not None else None for p in layer.parameters()]
        recompute_gu = exp.gate_up.grad.detach().float().cpu()
        recompute_dn = exp.down.grad.detach().float().cpu()

        resident_err = 0.0
        for a, b in zip(direct_resident, recompute_resident):
            if a is not None and b is not None:
                resident_err = max(resident_err, _max_rel(a, b))
        return {
            "x_grad_max_relative_error": _max_rel(direct_x, recompute_x),
            "resident_grad_max_relative_error": resident_err,
            "expert_gate_up_grad_max_relative_error": _max_rel(direct_gu, recompute_gu),
            "expert_down_grad_max_relative_error": _max_rel(direct_dn, recompute_dn),
        }
