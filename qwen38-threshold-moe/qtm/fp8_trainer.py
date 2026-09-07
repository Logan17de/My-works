from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from .experts import pack_gradient
from .fp8_resident import FP8ResidentExpertCache, install_mxfp8_backend, resolve_fp8_dtype
from .trainer import LayerwiseTrainer, _clip


class FP8ResidentTrainer(LayerwiseTrainer):
    """Full-GPU FP8 routed-expert trainer.

    - all routed expert weights stay resident on GPU in FP8
    - one current BF16 work layer is dequantized GPU->GPU for autograd
    - expert GEMMs use TorchAO MXFP8 in forward and backward
    - BF16 expert masters + Adafactor state stay on local Colab disk, never Drive during steps
    - the layer superbatch is fused into one batch per layer instead of 8 Python-level model calls
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        install_mxfp8_backend()
        super().__init__(cfg)
        t = cfg["training"]
        fp8_dtype = resolve_fp8_dtype(t.get("fp8_weight_dtype", "e4m3fn"))

        # The legacy cache created by the base class is still empty here. Replace it before training.
        self.cache.clear()
        self.cache = FP8ResidentExpertCache(
            self.store,
            device=self.device,
            compute_dtype=self.compute_dtype,
            fp8_dtype=fp8_dtype,
            reserve_gib=float(t.get("fp8_vram_reserve_gib", 18.0)),
            expert_chunk=int(t.get("fp8_refresh_expert_chunk", 16)),
            progress_callback=self._cache_progress,
        )
        self.cache_capacity = self.store.num_layers
        self.model.attach_cache(self.cache)
        for layer in self.model.layers:
            layer.moe.fast_grouped_mm = True

        routed_params = (
            int(cfg["model"]["num_layers"])
            * int(cfg["model"]["num_experts"])
            * 3
            * int(cfg["model"]["hidden_size"])
            * int(cfg["model"]["expert_intermediate_size"])
        )
        resident_params = self.model.resident_parameter_count()
        total_params = routed_params + resident_params
        print(
            "FP8 mode active | "
            f"params={total_params / 1e9:.3f}B total ({routed_params / 1e9:.3f}B routed) | "
            f"routed FP8 VRAM={routed_params / 1024**3:.2f} GiB | "
            f"BF16 routed masters(local)={routed_params * 2 / 1024**3:.2f} GiB",
            flush=True,
        )

    @staticmethod
    def _split_sizes(batches: list[torch.Tensor]) -> list[int]:
        return [int(batch.shape[0]) for batch in batches]

    @torch.no_grad()
    def _forward_boundaries(self, batches: list[torch.Tensor], step: int, lr: float) -> list[list[torch.Tensor]]:
        boundaries: list[list[torch.Tensor]] = []
        for batch in batches:
            ids = batch[:, :-1].to(self.device, non_blocking=False)
            h = self.model.initial_hidden(ids)
            boundaries.append([h.to("cpu", dtype=self.boundary_dtype)])
            del h, ids

        split_sizes = self._split_sizes(batches)
        total_layers = len(self.model.layers)
        for layer_idx in range(total_layers):
            self.current_step = step
            self.cache.ensure_window(layer_idx, +1)
            compute_started = time.time()
            h = torch.cat([x[-1] for x in boundaries], dim=0).to(self.device, dtype=self.compute_dtype)
            out = self.model.forward_layer(layer_idx, h, record_routing=True)
            out_cpu = out.to("cpu", dtype=self.boundary_dtype)
            for mb_idx, chunk in enumerate(out_cpu.split(split_sizes, dim=0)):
                boundaries[mb_idx].append(chunk)
            compute_seconds = time.time() - compute_started
            del h, out, out_cpu
            self._report_backend_error_once(layer_idx)
            self.logger.live(
                step=step,
                layer=f"{layer_idx + 1}/{total_layers}",
                loss=self.last_loss,
                lr=lr,
                phase="forward",
                expert_stats=self.controller.live_layer_stats(layer_idx),
                cache_capacity=self.cache_capacity,
                loaded_layers=0,
                load_gib=0.0,
                load_seconds=self.cache.last_load_seconds,
                compute_seconds=compute_seconds,
                backend="mxfp8-resident",
            )
        return boundaries

    def _head_backward(self, batches, boundaries, step: int, lr: float):
        t = self.cfg["training"]
        self.head_optimizer.zero_grad(set_to_none=True)
        split_sizes = self._split_sizes(batches)
        h = torch.cat([x[-1] for x in boundaries], dim=0).to(self.device, dtype=self.compute_dtype)
        h = h.detach().requires_grad_(True)
        final = self.model.final_hidden(h)
        labels = torch.cat([batch[:, 1:] for batch in batches], dim=0).to(self.device, non_blocking=False)
        total_tokens = int(labels.numel())
        total_loss_sum = 0.0
        seqlen = labels.shape[1]
        chunk_tokens = int(t["lm_chunk_tokens"])
        for start in range(0, seqlen, chunk_tokens):
            end = min(start + chunk_tokens, seqlen)
            logits = self.model.lm_head(final[:, start:end]).float()
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels[:, start:end].reshape(-1), reduction="sum"
            )
            (loss_sum / total_tokens).backward(retain_graph=end < seqlen)
            total_loss_sum += float(loss_sum.detach().item())
            del logits, loss_sum

        if h.grad is None:
            raise RuntimeError("Missing final-boundary gradient")
        grad_cpu = h.grad.detach().to("cpu", dtype=self.boundary_dtype)
        grad_h = [chunk.contiguous() for chunk in grad_cpu.split(split_sizes, dim=0)]
        _clip(list(self.model.final_mixer.parameters()) + list(self.model.lm_head.parameters()), float(t["grad_clip"]))
        self.head_optimizer.step()
        mean_loss = total_loss_sum / total_tokens
        self.logger.live(
            step=step,
            layer="LM",
            loss=mean_loss,
            lr=lr,
            phase="backward",
            cache_capacity=self.cache_capacity,
            backend="BF16-head",
        )
        del labels, final, h, grad_cpu
        return grad_h, mean_loss

    def _layers_backward(self, batches, boundaries, grad_h, step: int, lr: float, expert_lr: float):
        t = self.cfg["training"]
        transfer = t["gradient_transfer_dtype"]
        expert_chunk = int(t["expert_update_chunk"])
        int8_block = int(t.get("gradient_int8_block_size", 65536))
        split_sizes = self._split_sizes(batches)
        total_layers = len(self.model.layers)

        for layer_idx in reversed(range(total_layers)):
            self.current_step = step
            gpu_experts = self.cache.ensure_window(layer_idx, -1)
            gpu_experts.zero_grad()
            optimizer = self.layer_optimizers[layer_idx]
            optimizer.zero_grad(set_to_none=True)

            compute_started = time.time()
            h_in = torch.cat([boundaries[i][layer_idx] for i in range(len(batches))], dim=0)
            h_in = h_in.to(self.device, dtype=self.compute_dtype).detach().requires_grad_(True)
            out = self.model.forward_layer(layer_idx, h_in, record_routing=False)
            upstream = torch.cat(grad_h, dim=0).to(self.device, dtype=out.dtype)
            torch.autograd.backward(out, upstream)
            compute_seconds = time.time() - compute_started

            if h_in.grad is None:
                raise RuntimeError(f"Missing dX at layer {layer_idx}")
            dx_cpu = h_in.grad.detach().to("cpu", dtype=self.boundary_dtype)
            grad_h = [chunk.contiguous() for chunk in dx_cpu.split(split_sizes, dim=0)]

            _clip(self.model.layers[layer_idx].parameters(), float(t["grad_clip"]))
            if gpu_experts.gate_up.grad is None or gpu_experts.down.grad is None:
                raise RuntimeError(f"Missing routed expert gradient at layer {layer_idx}")
            _clip([gpu_experts.gate_up, gpu_experts.down], float(t["grad_clip"]))
            gate_packet = pack_gradient(
                gpu_experts.gate_up.grad, transfer, expert_chunk=expert_chunk, block_size=int8_block
            )
            down_packet = pack_gradient(
                gpu_experts.down.grad, transfer, expert_chunk=expert_chunk, block_size=int8_block
            )
            gpu_experts.zero_grad()

            update_started = time.time()
            self.adafactor.step_layer(layer_idx, gate_packet, down_packet, lr=expert_lr)
            self.cache.refresh_from_master(layer_idx)
            update_seconds = time.time() - update_started
            optimizer.step()
            del gate_packet, down_packet, upstream, out, h_in, dx_cpu

            self._report_backend_error_once(layer_idx)
            self.logger.live(
                step=step,
                layer=f"{layer_idx + 1}/{total_layers}",
                loss=self.last_loss,
                lr=lr,
                phase="backward",
                expert_stats=self.controller.live_layer_stats(layer_idx),
                cache_capacity=self.cache_capacity,
                loaded_layers=0,
                load_gib=0.0,
                load_seconds=0.0,
                compute_seconds=compute_seconds + update_seconds,
                backend=f"mxfp8 | upd {update_seconds:.1f}s",
            )
        self.cache.clear()
        return grad_h

    def _embedding_backward(self, batches, grad_h) -> None:
        t = self.cfg["training"]
        self.embed_optimizer.zero_grad(set_to_none=True)
        ids = torch.cat([batch[:, :-1] for batch in batches], dim=0).to(self.device, non_blocking=False)
        hidden = self.model.initial_hidden(ids)
        incoming = torch.cat(grad_h, dim=0).to(self.device, dtype=hidden.dtype)
        torch.autograd.backward(hidden, incoming)
        _clip(self.model.embed_tokens.parameters(), float(t["grad_clip"]))
        self.embed_optimizer.step()
        del hidden, ids, incoming
