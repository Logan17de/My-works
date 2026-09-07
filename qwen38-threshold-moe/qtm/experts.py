from __future__ import annotations

import gc
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .router import SoftmaxTopKRouter, ThresholdController

try:
    from transformers.integrations.moe import grouped_mm_experts_forward
except Exception:  # pragma: no cover - optional fast path
    grouped_mm_experts_forward = None


def resolve_dtype(name: str) -> torch.dtype:
    table = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _create_sized_file(path: Path, nbytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(nbytes)


def _mmap_tensor(path: Path, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    element_size = torch.empty((), dtype=dtype).element_size()
    count = math.prod(shape)
    if not path.exists():
        _create_sized_file(path, count * element_size)
    return torch.from_file(str(path), shared=True, size=count, dtype=dtype).view(shape)


def native_grouped_mm_available(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability(device)
    if hasattr(torch.nn.functional, "grouped_mm"):
        return capability >= (8, 0)
    if not hasattr(torch, "_grouped_mm"):
        return False
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    version = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    if version >= (2, 9):
        return capability >= (8, 0)
    return capability[0] == 9


@dataclass
class CPUExpertLayer:
    gate_up: torch.Tensor
    down: torch.Tensor


@dataclass
class GPUExpertLayer:
    gate_up: torch.Tensor
    down: torch.Tensor
    layer_idx: int

    def zero_grad(self) -> None:
        self.gate_up.grad = None
        self.down.grad = None


class ExpertStore:
    """Authoritative BF16 routed-expert masters, either mmap-backed or RAM resident."""

    def __init__(
        self,
        root: str | Path,
        *,
        num_layers: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        storage: Literal["mmap", "ram"] = "mmap",
        init_std: float = 0.02,
        seed: int = 17,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.storage = storage
        self.init_std = init_std
        self.seed = seed
        self._ram: dict[int, CPUExpertLayer] = {}
        self._manifest = self.root / "manifest.json"
        self._ensure_manifest()

    @property
    def gate_up_shape(self) -> tuple[int, int, int]:
        return (self.num_experts, 2 * self.intermediate_size, self.hidden_size)

    @property
    def down_shape(self) -> tuple[int, int, int]:
        return (self.num_experts, self.hidden_size, self.intermediate_size)

    def _ensure_manifest(self) -> None:
        expected = {
            "format": "qtm-expert-store-v1",
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "dtype": "bfloat16",
            "storage": self.storage,
        }
        if self._manifest.exists():
            actual = json.loads(self._manifest.read_text(encoding="utf-8"))
            for key in ("num_layers", "num_experts", "hidden_size", "intermediate_size", "dtype"):
                if actual.get(key) != expected[key]:
                    raise ValueError(f"Expert store mismatch for {key}: {actual.get(key)} != {expected[key]}")
        else:
            self._manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")

    def _paths(self, layer_idx: int) -> tuple[Path, Path, Path]:
        layer_dir = self.root / f"layer_{layer_idx:03d}"
        return layer_dir / "gate_up.bf16", layer_dir / "down.bf16", layer_dir / ".initialized"

    def layer_initialized(self, layer_idx: int) -> bool:
        if self.storage == "ram":
            return layer_idx in self._ram
        return self._paths(layer_idx)[2].exists()

    @torch.no_grad()
    def initialize_layer(self, layer_idx: int, *, expert_chunk: int = 8) -> None:
        if self.layer_initialized(layer_idx):
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + layer_idx * 1009)
        if self.storage == "ram":
            gate_up = torch.empty(self.gate_up_shape, dtype=torch.bfloat16)
            down = torch.empty(self.down_shape, dtype=torch.bfloat16)
        else:
            gate_path, down_path, marker = self._paths(layer_idx)
            gate_up = _mmap_tensor(gate_path, self.gate_up_shape, torch.bfloat16)
            down = _mmap_tensor(down_path, self.down_shape, torch.bfloat16)
        for start in range(0, self.num_experts, expert_chunk):
            end = min(start + expert_chunk, self.num_experts)
            g = torch.randn(
                (end - start, 2 * self.intermediate_size, self.hidden_size), generator=generator, dtype=torch.float32
            ).mul_(self.init_std).to(torch.bfloat16)
            d = torch.randn(
                (end - start, self.hidden_size, self.intermediate_size), generator=generator, dtype=torch.float32
            ).mul_(self.init_std).to(torch.bfloat16)
            gate_up[start:end].copy_(g)
            down[start:end].copy_(d)
        if self.storage == "ram":
            self._ram[layer_idx] = CPUExpertLayer(gate_up=gate_up, down=down)
        else:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok\n", encoding="utf-8")

    def initialize_all(self, *, expert_chunk: int = 8, progress=None) -> None:
        for layer_idx in range(self.num_layers):
            if progress:
                progress(layer_idx, self.num_layers)
            self.initialize_layer(layer_idx, expert_chunk=expert_chunk)

    def get_layer(self, layer_idx: int) -> CPUExpertLayer:
        if not self.layer_initialized(layer_idx):
            self.initialize_layer(layer_idx)
        if self.storage == "ram":
            return self._ram[layer_idx]
        gate_path, down_path, _ = self._paths(layer_idx)
        return CPUExpertLayer(
            gate_up=_mmap_tensor(gate_path, self.gate_up_shape, torch.bfloat16),
            down=_mmap_tensor(down_path, self.down_shape, torch.bfloat16),
        )

    def bytes_per_layer(self) -> int:
        return (math.prod(self.gate_up_shape) + math.prod(self.down_shape)) * 2


class ExpertLayerCache:
    """Sliding GPU cache over local CPU/NVMe expert masters."""

    def __init__(
        self,
        store: ExpertStore,
        *,
        device: torch.device,
        capacity: int,
        compute_dtype: torch.dtype,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self.store = store
        self.device = device
        self.capacity = min(store.num_layers, max(1, int(capacity)))
        self.compute_dtype = compute_dtype
        self.layers: dict[int, GPUExpertLayer] = {}
        self.progress_callback = progress_callback
        self.last_load_seconds = 0.0
        self.last_loaded_layers = 0
        self.last_loaded_gib = 0.0

    def _load(self, layer_idx: int) -> GPUExpertLayer:
        started = time.time()
        initialized_now = not self.store.layer_initialized(layer_idx)
        cpu = self.store.get_layer(layer_idx)
        gate_up = cpu.gate_up.to(device=self.device, dtype=self.compute_dtype, non_blocking=False).detach().requires_grad_(True)
        down = cpu.down.to(device=self.device, dtype=self.compute_dtype, non_blocking=False).detach().requires_grad_(True)
        layer = GPUExpertLayer(gate_up=gate_up, down=down, layer_idx=layer_idx)
        self.layers[layer_idx] = layer
        elapsed = time.time() - started
        if self.progress_callback:
            self.progress_callback({
                "layer_idx": layer_idx,
                "seconds": elapsed,
                "initialized_now": initialized_now,
                "cached_layers": len(self.layers),
                "capacity": self.capacity,
                "layer_gib": self.store.bytes_per_layer() * (torch.empty((), dtype=self.compute_dtype).element_size() / 2.0) / 1024**3,
            })
        return layer

    def _wanted_window(self, current: int, direction: int) -> list[int]:
        n = self.store.num_layers
        cap = min(self.capacity, n)
        if direction >= 0:
            # Keep a full window near the end instead of shrinking to one layer.
            start = min(max(0, current), max(0, n - cap))
            return list(range(start, min(n, start + cap)))
        # Reverse traversal: keep a full window near the beginning as well.
        end = min(n - 1, max(current, cap - 1))
        start = max(0, end - cap + 1)
        return list(range(start, end + 1))

    def ensure_window(self, current: int, direction: int) -> GPUExpertLayer:
        wanted = self._wanted_window(current, direction)
        wanted_set = set(wanted)
        for idx in list(self.layers):
            if idx not in wanted_set:
                del self.layers[idx]
        started = time.time()
        loaded = 0
        for idx in wanted:
            if idx not in self.layers:
                self._load(idx)
                loaded += 1
        self.last_load_seconds = time.time() - started
        self.last_loaded_layers = loaded
        compute_bytes = self.store.bytes_per_layer() * (torch.empty((), dtype=self.compute_dtype).element_size() / 2.0)
        self.last_loaded_gib = loaded * compute_bytes / 1024**3
        return self.layers[current]

    def get(self, layer_idx: int) -> GPUExpertLayer:
        if layer_idx not in self.layers:
            return self._load(layer_idx)
        return self.layers[layer_idx]

    def clear(self) -> None:
        self.layers.clear()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


@dataclass
class GradientPacket:
    kind: str
    data: torch.Tensor
    scales: torch.Tensor | None = None
    block_size: int | None = None

    def chunk(self, start: int, end: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.kind == "int8":
            assert self.scales is not None and self.block_size is not None
            q = self.data[start:end].to(dtype=torch.float32)
            shape = q.shape
            flat = q.flatten(1)
            n = flat.shape[1]
            blocks = self.scales.shape[1]
            padded_n = blocks * self.block_size
            if padded_n != n:
                flat = F.pad(flat, (0, padded_n - n))
            flat = flat.view(flat.shape[0], blocks, self.block_size)
            flat = flat * self.scales[start:end].to(dtype=torch.float32).unsqueeze(-1)
            return flat.reshape(flat.shape[0], -1)[:, :n].view(shape).to(dtype)
        return self.data[start:end].to(dtype=dtype)


def pack_gradient(
    grad: torch.Tensor,
    mode: str,
    *,
    expert_chunk: int = 8,
    block_size: int = 65536,
) -> GradientPacket:
    if mode == "bf16":
        return GradientPacket("bf16", grad.detach().to(device="cpu", dtype=torch.bfloat16))
    if mode == "fp32":
        return GradientPacket("fp32", grad.detach().to(device="cpu", dtype=torch.float32))
    if mode != "int8":
        raise ValueError(mode)
    shape = tuple(grad.shape)
    if len(shape) != 3:
        raise ValueError("INT8 expert gradient packing expects [experts, rows, cols]")
    block_size = max(256, int(block_size))
    per_expert = shape[1] * shape[2]
    nblocks = math.ceil(per_expert / block_size)
    padded_n = nblocks * block_size
    q = torch.empty(shape, dtype=torch.int8, device="cpu")
    scales = torch.empty((shape[0], nblocks), dtype=torch.float32, device="cpu")
    for start in range(0, shape[0], expert_chunk):
        end = min(start + expert_chunk, shape[0])
        flat = grad[start:end].detach().float().flatten(1)
        if padded_n != per_expert:
            flat = F.pad(flat, (0, padded_n - per_expert))
        blocks = flat.view(end - start, nblocks, block_size)
        scale = blocks.abs().amax(dim=-1).div_(127.0).clamp_min_(1e-12)
        quant = torch.round(blocks / scale.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
        q[start:end].view(end - start, -1).copy_(quant.view(end - start, -1)[:, :per_expert].to("cpu"))
        scales[start:end].copy_(scale.to("cpu"))
        del flat, blocks, quant, scale
    return GradientPacket("int8", q, scales, block_size)


class _ExpertView:
    def __init__(self, weights: GPUExpertLayer, num_experts: int):
        self.gate_up_proj = weights.gate_up
        self.down_proj = weights.down
        self.num_experts = num_experts
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.gate_up_proj_bias = None
        self.down_proj_bias = None
        self.act_fn = F.silu

    def _apply_gate(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


def eager_experts_forward(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
    weights: GPUExpertLayer,
    num_experts: int,
) -> torch.Tensor:
    num_tokens, hidden = hidden_states.shape
    top_k = indices.shape[-1]
    flat_experts = indices.reshape(-1)
    pair_order = torch.argsort(flat_experts)
    sorted_experts = flat_experts[pair_order]
    output = torch.zeros_like(hidden_states)
    if sorted_experts.numel() == 0:
        return output
    unique, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    offset = 0
    for expert_id, count in zip(unique.tolist(), counts.tolist()):
        pairs = pair_order[offset : offset + count]
        token_idx = torch.div(pairs, top_k, rounding_mode="floor")
        slot_idx = pairs.remainder(top_k)
        x = hidden_states.index_select(0, token_idx)
        gate, up = F.linear(x, weights.gate_up[expert_id]).chunk(2, dim=-1)
        y = F.silu(gate) * up
        y = F.linear(y, weights.down[expert_id])
        y = y * routing_weights[token_idx, slot_idx, None]
        output.index_add_(0, token_idx, y.to(output.dtype))
        offset += count
    return output


class SharedExpertBank(nn.Module):
    def __init__(self, num_shared: int, hidden: int, intermediate: int, init_std: float) -> None:
        super().__init__()
        self.num_shared = num_shared
        self.hidden = hidden
        self.intermediate = intermediate
        self.gate_up = nn.Parameter(torch.empty(num_shared, 2 * intermediate, hidden))
        self.down = nn.Parameter(torch.empty(num_shared, hidden, intermediate))
        nn.init.normal_(self.gate_up, mean=0.0, std=init_std)
        nn.init.normal_(self.down, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for idx in range(self.num_shared):
            gate, up = F.linear(x, self.gate_up[idx]).chunk(2, dim=-1)
            y = F.silu(gate) * up
            out.add_(F.linear(y, self.down[idx]))
        return out / float(self.num_shared)


class StreamedMoEBlock(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        hidden_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        intermediate_size: int,
        controller: ThresholdController,
        normalize_topk: bool,
        init_std: float,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.router = SoftmaxTopKRouter(
            hidden_size, num_experts, top_k, layer_idx, controller,
            normalize_topk=normalize_topk, init_std=init_std,
        )
        self.shared = SharedExpertBank(num_shared_experts, hidden_size, intermediate_size, init_std)
        self.cache: ExpertLayerCache | None = None
        self.fast_grouped_mm = True
        self.last_backend = "not-run"
        self.last_backend_error: str | None = None

    def attach_cache(self, cache: ExpertLayerCache) -> None:
        self.cache = cache

    def forward(self, hidden_states: torch.Tensor, *, record_routing: bool = False) -> torch.Tensor:
        if self.cache is None:
            raise RuntimeError("Expert cache not attached")
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        route = self.router(flat, record=record_routing)
        weights = self.cache.get(self.layer_idx)
        expert_out = None
        if self.fast_grouped_mm and grouped_mm_experts_forward is not None and flat.is_cuda:
            try:
                self.last_backend = "native-grouped" if native_grouped_mm_available(flat.device) else "hf-grouped-fallback"
                expert_out = grouped_mm_experts_forward(
                    _ExpertView(weights, self.num_experts), flat, route.indices, route.weights
                )
                self.last_backend_error = None
            except Exception as exc:
                expert_out = None
                self.fast_grouped_mm = False
                self.last_backend = f"eager-after-{type(exc).__name__}"
                self.last_backend_error = str(exc)
        if expert_out is None:
            if grouped_mm_experts_forward is None:
                self.last_backend = "eager-no-grouped-mm"
            expert_out = eager_experts_forward(flat, route.indices, route.weights, weights, self.num_experts)
        shared_out = self.shared(flat)
        return (expert_out + shared_out).view(shape)
