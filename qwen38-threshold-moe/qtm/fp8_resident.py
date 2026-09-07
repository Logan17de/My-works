from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .experts import ExpertStore, GPUExpertLayer


def resolve_fp8_dtype(name: str) -> torch.dtype:
    key = str(name).lower().replace("torch.", "")
    table = {
        "e4m3": torch.float8_e4m3fn,
        "e4m3fn": torch.float8_e4m3fn,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2,
        "float8_e5m2": torch.float8_e5m2,
    }
    if key not in table:
        raise ValueError(f"Unsupported FP8 dtype: {name}")
    return table[key]


@dataclass
class FP8ResidentLayer:
    gate_up: torch.Tensor
    down: torch.Tensor


class FP8ResidentExpertCache:
    """Keep every routed expert on GPU in FP8; materialize only one BF16 work layer.

    The local BF16 ExpertStore remains the optimizer master. Forward/backward never fetch
    weights from disk after preload: each current work layer is dequantized GPU->GPU from
    the resident FP8 copy. After the CPU Adafactor update, only that layer's new FP8 copy
    is refreshed on GPU.
    """

    def __init__(
        self,
        store: ExpertStore,
        *,
        device: torch.device,
        compute_dtype: torch.dtype = torch.bfloat16,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        reserve_gib: float = 18.0,
        expert_chunk: int = 16,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("FP8 resident experts require CUDA")
        self.store = store
        self.device = device
        self.compute_dtype = compute_dtype
        self.fp8_dtype = fp8_dtype
        self.capacity = store.num_layers
        self.expert_chunk = max(1, int(expert_chunk))
        self.progress_callback = progress_callback
        self.layers: list[FP8ResidentLayer] = []
        self.active: GPUExpertLayer | None = None
        self.last_load_seconds = 0.0
        self.last_loaded_layers = 0
        self.last_loaded_gib = 0.0
        self.last_refresh_seconds = 0.0

        fp8_bytes = store.bytes_per_layer() // 2 * store.num_layers
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        reserve_bytes = int(float(reserve_gib) * 1024**3)
        if fp8_bytes + reserve_bytes > free_bytes:
            raise RuntimeError(
                "Full FP8 expert bank does not fit in VRAM: "
                f"needs {fp8_bytes / 1024**3:.1f} GiB + {reserve_gib:.1f} GiB reserve, "
                f"but only {free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB is free"
            )
        print(
            f"FP8 resident expert bank: {store.num_layers} layers | "
            f"{fp8_bytes / 1024**3:.2f} GiB FP8 | reserve {reserve_gib:.1f} GiB | "
            f"free {free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB",
            flush=True,
        )
        self._preload_all()

    @property
    def resident_gib(self) -> float:
        return (self.store.bytes_per_layer() // 2 * self.store.num_layers) / 1024**3

    @torch.no_grad()
    def _quantize_cpu_to_gpu(self, src: torch.Tensor) -> torch.Tensor:
        # Cast on CPU first so PCIe carries one byte/parameter instead of BF16's two.
        q_cpu = src.to(dtype=self.fp8_dtype)
        return q_cpu.to(device=self.device, non_blocking=False)

    @torch.no_grad()
    def _preload_all(self) -> None:
        started_all = time.time()
        for layer_idx in range(self.store.num_layers):
            started = time.time()
            initialized_now = not self.store.layer_initialized(layer_idx)
            cpu = self.store.get_layer(layer_idx)
            gu = self._quantize_cpu_to_gpu(cpu.gate_up)
            dn = self._quantize_cpu_to_gpu(cpu.down)
            self.layers.append(FP8ResidentLayer(gu, dn))
            elapsed = time.time() - started
            if self.progress_callback:
                self.progress_callback(
                    {
                        "layer_idx": layer_idx,
                        "seconds": elapsed,
                        "initialized_now": initialized_now,
                        "cached_layers": len(self.layers),
                        "capacity": self.capacity,
                        "layer_gib": self.store.bytes_per_layer() / 2 / 1024**3,
                        "resident_fp8": True,
                    }
                )
        torch.cuda.synchronize(self.device)
        print(
            f"FP8 expert preload complete: {len(self.layers)}/{self.store.num_layers} layers "
            f"in {time.time() - started_all:.1f}s",
            flush=True,
        )

    def _materialize_work_layer(self, layer_idx: int) -> GPUExpertLayer:
        if self.active is not None and self.active.layer_idx == layer_idx:
            self.last_load_seconds = 0.0
            self.last_loaded_layers = 0
            self.last_loaded_gib = 0.0
            return self.active
        self.active = None
        started = time.time()
        resident = self.layers[layer_idx]
        gu = resident.gate_up.to(dtype=self.compute_dtype).detach().requires_grad_(True)
        dn = resident.down.to(dtype=self.compute_dtype).detach().requires_grad_(True)
        self.active = GPUExpertLayer(gate_up=gu, down=dn, layer_idx=layer_idx)
        self.last_load_seconds = time.time() - started
        self.last_loaded_layers = 0  # no host/disk load; GPU dequant only
        self.last_loaded_gib = 0.0
        return self.active

    def ensure_window(self, current: int, direction: int) -> GPUExpertLayer:
        del direction
        return self._materialize_work_layer(current)

    def get(self, layer_idx: int) -> GPUExpertLayer:
        return self._materialize_work_layer(layer_idx)

    @torch.no_grad()
    def refresh_from_master(self, layer_idx: int) -> None:
        """Quantize the updated BF16 master to FP8 and overwrite the resident GPU layer."""
        started = time.time()
        cpu = self.store.get_layer(layer_idx)
        dst = self.layers[layer_idx]
        # Chunk to keep the temporary CPU FP8 allocation bounded.
        for start in range(0, self.store.num_experts, self.expert_chunk):
            end = min(start + self.expert_chunk, self.store.num_experts)
            gu = cpu.gate_up[start:end].to(dtype=self.fp8_dtype).to(self.device)
            dn = cpu.down[start:end].to(dtype=self.fp8_dtype).to(self.device)
            dst.gate_up[start:end].copy_(gu)
            dst.down[start:end].copy_(dn)
            del gu, dn
        self.active = None
        self.last_refresh_seconds = time.time() - started

    def clear(self) -> None:
        # Resident FP8 weights remain allocated. Only the current BF16 work layer is released.
        self.active = None
        gc.collect()


def _route_for_grouped_mm(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
):
    num_tokens = hidden_states.shape[0]
    top_k = top_k_index.shape[-1]
    flat_ids = top_k_index.reshape(-1)
    order = torch.argsort(flat_ids)
    sorted_ids = flat_ids.index_select(0, order)
    token_idx = torch.div(order, top_k, rounding_mode="floor")
    x_sorted = hidden_states.index_select(0, token_idx)
    route_sorted = top_k_weights.reshape(-1).index_select(0, order)
    counts = torch.bincount(sorted_ids, minlength=num_experts)
    offs = counts.cumsum(0).to(dtype=torch.int32)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
    return x_sorted, route_sorted, offs, inverse, num_tokens, top_k


def mxfp8_grouped_mm_experts_forward(
    experts_view,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """TorchAO MXFP8 grouped-GEMM expert forward with differentiable FP8 FWD/BWD."""
    try:
        from torchao.prototype.moe_training import _to_mxfp8_then_scaled_grouped_mm
    except Exception as exc:  # pragma: no cover - depends on runtime package
        raise RuntimeError(
            "MXFP8 expert compute requested but torchao.prototype.moe_training is unavailable. "
            "Install requirements-fp8.txt."
        ) from exc

    x_sorted, route_sorted, offs, inverse, num_tokens, top_k = _route_for_grouped_mm(
        hidden_states, top_k_index, top_k_weights, experts_view.num_experts
    )
    gate_up = _to_mxfp8_then_scaled_grouped_mm(
        x_sorted,
        experts_view.gate_up_proj.transpose(-2, -1),
        offs,
    )
    gate, up = gate_up.chunk(2, dim=-1)
    middle = F.silu(gate) * up
    out_sorted = _to_mxfp8_then_scaled_grouped_mm(
        middle,
        experts_view.down_proj.transpose(-2, -1),
        offs,
    )
    out_sorted = out_sorted * route_sorted.to(out_sorted.dtype).unsqueeze(-1)
    pair_out = out_sorted.index_select(0, inverse)
    return pair_out.view(num_tokens, top_k, hidden_states.shape[-1]).sum(dim=1).to(hidden_states.dtype)


def install_mxfp8_backend() -> None:
    """Route the existing StreamedMoEBlock through TorchAO MXFP8 grouped GEMMs.

    The legacy StreamedMoEBlock catches grouped-kernel errors and otherwise falls back to
    eager expert loops. For the FP8 run we intentionally replace that fallback with a hard
    error so the notebook can never silently train in BF16 while claiming FP8.
    """
    from . import experts as experts_module

    experts_module.grouped_mm_experts_forward = mxfp8_grouped_mm_experts_forward

    def _no_silent_fallback(*args, **kwargs):
        raise RuntimeError(
            "MXFP8 grouped expert kernel failed. Refusing the eager BF16 fallback for an FP8 run; "
            "inspect the preceding kernel error/runtime compatibility."
        )

    experts_module.eager_experts_forward = _no_silent_fallback
