from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "tokenizer": "Qwen/Qwen3.8-Flash-Next",
        "vocab_size": 248320,
        "hidden_size": 256,
        "num_layers": 80,
        "num_experts": 500,
        "num_shared_experts": 4,
        "top_k": 4,
        "expert_intermediate_size": 2048,
        "router_normalize_topk": True,
        "hc_count": 4,
        "hc_lowrank": 32,
        "full_attention_interval": 4,
        "attention_heads": 8,
        "kv_heads": 1,
        "attention_head_dim": 64,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000.0,
        "max_position_embeddings": 262144,
        "linear_key_heads": 4,
        "linear_value_heads": 12,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
    },
    "threshold": {
        "enabled": True,
        "warmup_tokens": -1,
        "imbalance_threshold": 0.30,
        "block_fraction": 0.30,
        "check_every_steps": 1,
    },
    "data": {
        "source": "k2-txt360-v2",
        "config_name": "web-high-medium",
        "split": "train",
        "text_field": None,
        "sequence_length": 8192,
        "streaming": True,
        "shuffle_buffer": 10000,
        "seed": 17,
        "eos_between_documents": True,
        "hf_token_env": "HF_TOKEN",
        "mixture_file": None,
    },
    "training": {
        "run_name": "qtm-run",
        "output_dir": "runs",
        "work_dir": "work",
        "device": "cuda",
        "microbatch_size": 1,
        "layer_superbatch": 1,
        "expert_cache_layers": 1,
        "expert_store": "mmap",
        "expert_compute_dtype": "bf16",
        "gradient_transfer_dtype": "int8",
        "gradient_int8_block_size": 65536,
        "boundary_dtype": "bf16",
        "routed_optimizer": "adafactor",
        "resident_optimizer": "adamw",
        "learning_rate": 2e-4,
        "expert_learning_rate": 2e-4,
        "min_learning_rate": 2e-5,
        "weight_decay": 0.1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-8,
        "adafactor_beta2": 0.999,
        "adafactor_eps": 1e-30,
        "adafactor_clip_threshold": 1.0,
        "expert_update_chunk": 8,
        "grad_clip": 1.0,
        "warmup_steps": 100,
        "max_steps": 1000,
        "save_every_steps": 100,
        "keep_last_checkpoints": 2,
        "plot_every_steps": 10,
        "log_every_steps": 1,
        "lm_chunk_tokens": 256,
        "resume": "none",
        "snapshot_experts": False,
        "seed": 17,
        "torch_compile": False,
    },
}


def _deep_merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower in {"none", "null"}:
            return None
        return value


def apply_override(cfg: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE, got: {expression}")
    dotted, raw = expression.split("=", 1)
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = _parse_scalar(raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = _deep_merge(cfg, payload)
    for item in overrides or []:
        apply_override(cfg, item)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    m = cfg["model"]
    t = cfg["training"]
    th = cfg["threshold"]
    if m["hidden_size"] <= 0 or m["num_layers"] <= 0:
        raise ValueError("hidden_size and num_layers must be positive")
    if m["num_experts"] < m["top_k"]:
        raise ValueError("num_experts must be >= top_k")
    if m["attention_heads"] % m["kv_heads"] != 0:
        raise ValueError("attention_heads must be divisible by kv_heads")
    if m["linear_value_heads"] % m["linear_key_heads"] != 0:
        raise ValueError("linear_value_heads must be divisible by linear_key_heads")
    rotary_dim = int(m["attention_head_dim"] * m["partial_rotary_factor"])
    if rotary_dim % 2:
        raise ValueError("partial RoPE dimension must be even")
    if not 0.0 <= th["block_fraction"] < 1.0:
        raise ValueError("threshold.block_fraction must be in [0,1)")
    if t["expert_cache_layers"] < 1:
        raise ValueError("training.expert_cache_layers must be >= 1")
    if t["layer_superbatch"] < 1 or t["microbatch_size"] < 1:
        raise ValueError("layer_superbatch and microbatch_size must be >= 1")
    if t["gradient_transfer_dtype"] not in {"int8", "bf16", "fp32"}:
        raise ValueError("gradient_transfer_dtype must be int8, bf16, or fp32")
    if t["expert_compute_dtype"] not in {"bf16", "fp16", "fp32"}:
        raise ValueError("expert_compute_dtype must be bf16, fp16, or fp32")


def dump_config(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, indent=2, sort_keys=False)
