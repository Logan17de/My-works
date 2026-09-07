from __future__ import annotations

import gc
import json
import math
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import CheckpointManager
from .data import PackedTokenStream
from .experts import ExpertLayerCache, ExpertStore, pack_gradient, resolve_dtype
from .logging_utils import TrainingLogger
from .model import QwenThresholdMoE
from .optimizer import FactoredAdafactorCPU, cosine_lr
from .persistence import WorkBackup
from .router import ThresholdController


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clip(params, max_norm: float) -> float:
    params = [p for p in params if p.grad is not None]
    if not params or max_norm <= 0:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm).item())


class LayerwiseTrainer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        m, th, t, d = cfg["model"], cfg["threshold"], cfg["training"], cfg["data"]
        seed_everything(int(t["seed"]))
        if t["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        self.device = torch.device(t["device"])
        self.compute_dtype = resolve_dtype(t["expert_compute_dtype"])
        self.boundary_dtype = resolve_dtype(t["boundary_dtype"])
        self.run_dir = Path(t["output_dir"]) / t["run_name"]
        self.work_dir = Path(t["work_dir"]) / t["run_name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.parent.mkdir(parents=True, exist_ok=True)
        self.logger = TrainingLogger(self.run_dir)
        self.last_loss = float("nan")
        self.current_step = 0
        self.cache_capacity = 1
        self._reported_backend_error = False

        self.backup = WorkBackup(self.work_dir, t.get("backup_dir"), t["run_name"])
        resume_requested = str(t.get("resume", "none")).lower() not in {"", "none"}
        if resume_requested:
            self.backup.restore_if_needed()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        self.controller = ThresholdController(
            int(m["num_layers"]), int(m["num_experts"]),
            enabled=bool(th["enabled"]),
            warmup_tokens=int(th["warmup_tokens"]),
            imbalance_threshold=float(th["imbalance_threshold"]),
            block_fraction=float(th["block_fraction"]),
            check_every_steps=int(th["check_every_steps"]),
        )

        self.checkpoints = CheckpointManager(self.run_dir, keep_last=int(t["keep_last_checkpoints"]))
        self.resume_state = self._resolve_resume(t.get("resume", "none"))
        if self.resume_state and self.resume_state["metadata"].get("expert_snapshot", False):
            expert_root = self.resume_state["expert_store_root"]
        else:
            expert_root = self.work_dir / "expert_store"
            if self.resume_state:
                self.resume_state["expert_store_root"] = expert_root

        self.store = ExpertStore(
            expert_root,
            num_layers=int(m["num_layers"]), num_experts=int(m["num_experts"]),
            hidden_size=int(m["hidden_size"]), intermediate_size=int(m["expert_intermediate_size"]),
            storage=t["expert_store"], init_std=float(m["initializer_range"]), seed=int(t["seed"]),
        )

        self.model = QwenThresholdMoE(m, self.controller)
        self.model.to(self.device, dtype=self.compute_dtype)
        for module in self.model.modules():
            if hasattr(module, "A_log") and isinstance(module.A_log, torch.nn.Parameter):
                module.A_log.data = module.A_log.data.float()
            if hasattr(module, "dt_bias") and isinstance(module.dt_bias, torch.nn.Parameter):
                module.dt_bias.data = module.dt_bias.data.float()

        self.cache_capacity = self._resolve_cache_capacity(t)
        self.cache = ExpertLayerCache(
            self.store,
            device=self.device,
            capacity=self.cache_capacity,
            compute_dtype=self.compute_dtype,
            progress_callback=self._cache_progress,
        )
        self.model.attach_cache(self.cache)

        self.layer_optimizers = [self._make_adamw(layer.parameters()) for layer in self.model.layers]
        self.embed_optimizer = self._make_adamw(self.model.embed_tokens.parameters())
        self.head_optimizer = self._make_adamw(list(self.model.final_mixer.parameters()) + list(self.model.lm_head.parameters()))
        self.adafactor = FactoredAdafactorCPU(
            self.work_dir / "adafactor_state", self.store,
            beta2=float(t["adafactor_beta2"]), eps=float(t["adafactor_eps"]),
            clip_threshold=float(t["adafactor_clip_threshold"]), weight_decay=float(t["weight_decay"]),
            expert_chunk=int(t["expert_update_chunk"]),
        )

        if self.resume_state:
            self._restore_resume()
        data_cfg = dict(d)
        data_cfg["tokenizer"] = m["tokenizer"]
        self.data = PackedTokenStream(data_cfg, resume_state=self.resume_state["data_state"] if self.resume_state else None)
        self.stop_requested = False
        self._install_signal_handlers()
        self.start_step = int(self.resume_state["metadata"]["step"]) if self.resume_state else 0
        self.last_completed_step = self.start_step

        backup_text = str(self.backup.backup_dir) if self.backup.enabled else "disabled"
        print(
            f"Work: local={self.work_dir} | persistent_backup={backup_text} | GPU expert cache={self.cache_capacity}/{len(self.model.layers)} layers",
            flush=True,
        )

    def _resolve_cache_capacity(self, t: dict[str, Any]) -> int:
        requested = t["expert_cache_layers"]
        if not (isinstance(requested, str) and requested.lower() == "auto"):
            return min(self.store.num_layers, max(1, int(requested)))
        if self.device.type != "cuda":
            return 1
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        reserve_bytes = int(float(t.get("gpu_cache_reserve_gib", 20.0)) * 1024**3)
        dtype_bytes = torch.empty((), dtype=self.compute_dtype).element_size()
        per_layer_bytes = int(self.store.bytes_per_layer() * (dtype_bytes / 2.0))
        usable = max(per_layer_bytes, free_bytes - reserve_bytes)
        capacity = max(1, min(self.store.num_layers, usable // per_layer_bytes))
        print(
            f"Auto expert cache: {capacity} layers | {per_layer_bytes / 1024**3:.2f} GiB/layer | "
            f"free {free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB | reserve {reserve_bytes / 1024**3:.1f} GiB",
            flush=True,
        )
        return int(capacity)

    def _cache_progress(self, stats: dict[str, Any]) -> None:
        kind = "init+load" if stats.get("initialized_now") else "load"
        self.logger.live(
            step=self.current_step,
            layer=f"{int(stats['layer_idx']) + 1}/{len(self.model.layers)}",
            loss=self.last_loss,
            lr=0.0,
            phase="cache",
            cache_capacity=self.cache_capacity,
            loaded_layers=1,
            load_gib=float(stats.get("layer_gib", 0.0)),
            load_seconds=float(stats.get("seconds", 0.0)),
            backend=kind,
        )

    def _make_adamw(self, params):
        t = self.cfg["training"]
        return torch.optim.AdamW(
            params,
            lr=float(t["learning_rate"]),
            betas=(float(t["adam_beta1"]), float(t["adam_beta2"])),
            eps=float(t["adam_eps"]),
            weight_decay=float(t["weight_decay"]),
            foreach=False,
        )

    def _resolve_resume(self, resume: str | None):
        if not resume or str(resume).lower() == "none":
            return None
        if str(resume).lower() == "latest":
            path = self.checkpoints.latest()
            if path is None:
                raise FileNotFoundError("No checkpoint exists for --resume latest")
        else:
            path = Path(resume)
        if self.backup.enabled:
            synced = self.backup.last_synced_step()
            checkpoint_step = int(path.name.split("_")[-1])
            if synced is None:
                raise RuntimeError("Persistent expert backup has no completed sync_state.json; cannot resume safely")
            if synced != checkpoint_step:
                raise RuntimeError(
                    f"Checkpoint/expert backup mismatch: checkpoint step={checkpoint_step}, expert backup step={synced}. "
                    "Refusing an inconsistent resume."
                )
        return CheckpointManager.load_state(path, self.device)

    def _restore_resume(self) -> None:
        state = self.resume_state
        self.model.load_state_dict(state["model"])
        opt = state["optimizers"]
        for optimizer, payload in zip(self.layer_optimizers, opt["layer_optimizers"]):
            optimizer.load_state_dict(payload)
        self.embed_optimizer.load_state_dict(opt["embed_optimizer"])
        self.head_optimizer.load_state_dict(opt["head_optimizer"])
        self.adafactor.load_state_dict(opt.get("adafactor", {}))
        self.controller.load_state_dict(state["threshold"])
        CheckpointManager.restore_rng(state["rng"])

    def _install_signal_handlers(self) -> None:
        def request_stop(signum, frame):
            self.stop_requested = True
            print(f"\nSignal {signum} received; finishing the current optimizer step, then checkpointing + syncing.", flush=True)
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    def _set_lr(self, step: int) -> tuple[float, float]:
        t = self.cfg["training"]
        lr = cosine_lr(
            step, base_lr=float(t["learning_rate"]), min_lr=float(t["min_learning_rate"]),
            warmup_steps=int(t["warmup_steps"]), max_steps=int(t["max_steps"]),
        )
        expert_lr = cosine_lr(
            step, base_lr=float(t["expert_learning_rate"]), min_lr=float(t["min_learning_rate"]),
            warmup_steps=int(t["warmup_steps"]), max_steps=int(t["max_steps"]),
        )
        for optimizer in self.layer_optimizers + [self.embed_optimizer, self.head_optimizer]:
            for group in optimizer.param_groups:
                group["lr"] = lr
        return lr, expert_lr

    def _fetch_superbatch(self) -> list[torch.Tensor]:
        t = self.cfg["training"]
        return [self.data.next_microbatch(int(t["microbatch_size"])) for _ in range(int(t["layer_superbatch"]))]

    def _report_backend_error_once(self, layer_idx: int) -> None:
        moe = self.model.layers[layer_idx].moe
        if moe.last_backend_error and not self._reported_backend_error:
            self.logger.newline()
            print(
                f"WARNING: Hugging Face grouped MoE failed at layer {layer_idx + 1}; using eager fallback. "
                f"Error: {moe.last_backend_error}",
                flush=True,
            )
            self._reported_backend_error = True

    @torch.no_grad()
    def _forward_boundaries(self, batches: list[torch.Tensor], step: int, lr: float) -> list[list[torch.Tensor]]:
        boundaries: list[list[torch.Tensor]] = []
        for batch in batches:
            ids = batch[:, :-1].to(self.device, non_blocking=False)
            h = self.model.initial_hidden(ids)
            boundaries.append([h.to("cpu", dtype=self.boundary_dtype)])
            del h, ids
        total_layers = len(self.model.layers)
        for layer_idx in range(total_layers):
            self.current_step = step
            self.cache.ensure_window(layer_idx, +1)
            compute_started = time.time()
            for mb_idx in range(len(batches)):
                h = boundaries[mb_idx][-1].to(self.device, dtype=self.compute_dtype)
                out = self.model.forward_layer(layer_idx, h, record_routing=True)
                boundaries[mb_idx].append(out.to("cpu", dtype=self.boundary_dtype))
                del h, out
            compute_seconds = time.time() - compute_started
            self._report_backend_error_once(layer_idx)
            self.logger.live(
                step=step,
                layer=f"{layer_idx + 1}/{total_layers}",
                loss=self.last_loss,
                lr=lr,
                phase="forward",
                expert_stats=self.controller.live_layer_stats(layer_idx),
                cache_capacity=self.cache_capacity,
                loaded_layers=self.cache.last_loaded_layers,
                load_gib=self.cache.last_loaded_gib,
                load_seconds=self.cache.last_load_seconds,
                compute_seconds=compute_seconds,
                backend=self.model.layers[layer_idx].moe.last_backend,
            )
        # Keep the final full cache window resident. Reverse traversal starts from the same window,
        # so this avoids reloading tens of GiB before backward. Auto sizing reserves headroom for LM backward.
        return boundaries

    def _head_backward(self, batches, boundaries, step: int, lr: float) -> tuple[list[torch.Tensor], float]:
        t = self.cfg["training"]
        self.head_optimizer.zero_grad(set_to_none=True)
        grad_h: list[torch.Tensor] = []
        total_tokens = sum((batch.shape[0] * (batch.shape[1] - 1)) for batch in batches)
        total_loss_sum = 0.0
        chunk_tokens = int(t["lm_chunk_tokens"])
        for mb_idx, batch in enumerate(batches):
            h = boundaries[mb_idx][-1].to(self.device, dtype=self.compute_dtype).detach().requires_grad_(True)
            final = self.model.final_hidden(h)
            labels = batch[:, 1:].to(self.device, non_blocking=False)
            seqlen = labels.shape[1]
            for start in range(0, seqlen, chunk_tokens):
                end = min(start + chunk_tokens, seqlen)
                logits = self.model.lm_head(final[:, start:end]).float()
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels[:, start:end].reshape(-1), reduction="sum"
                )
                (loss_sum / total_tokens).backward(retain_graph=end < seqlen)
                total_loss_sum += float(loss_sum.detach().item())
                del logits, loss_sum
            grad_h.append(h.grad.detach().to("cpu", dtype=self.boundary_dtype))
            del labels, final, h
        _clip(list(self.model.final_mixer.parameters()) + list(self.model.lm_head.parameters()), float(t["grad_clip"]))
        self.head_optimizer.step()
        mean_loss = total_loss_sum / total_tokens
        self.logger.live(
            step=step, layer="LM", loss=mean_loss, lr=lr, phase="backward", cache_capacity=self.cache_capacity
        )
        return grad_h, mean_loss

    def _layers_backward(self, batches, boundaries, grad_h, step: int, lr: float, expert_lr: float) -> list[torch.Tensor]:
        t = self.cfg["training"]
        transfer = t["gradient_transfer_dtype"]
        expert_chunk = int(t["expert_update_chunk"])
        int8_block = int(t.get("gradient_int8_block_size", 65536))
        total_layers = len(self.model.layers)
        for layer_idx in reversed(range(total_layers)):
            self.current_step = step
            gpu_experts = self.cache.ensure_window(layer_idx, -1)
            gpu_experts.zero_grad()
            optimizer = self.layer_optimizers[layer_idx]
            optimizer.zero_grad(set_to_none=True)
            compute_started = time.time()
            for mb_idx in range(len(batches)):
                h_in = boundaries[mb_idx][layer_idx].to(self.device, dtype=self.compute_dtype).detach().requires_grad_(True)
                out = self.model.forward_layer(layer_idx, h_in, record_routing=False)
                upstream = grad_h[mb_idx].to(self.device, dtype=out.dtype)
                torch.autograd.backward(out, upstream)
                grad_h[mb_idx] = h_in.grad.detach().to("cpu", dtype=self.boundary_dtype)
                del upstream, out, h_in
            _clip(self.model.layers[layer_idx].parameters(), float(t["grad_clip"]))
            if gpu_experts.gate_up.grad is None or gpu_experts.down.grad is None:
                raise RuntimeError(f"Missing routed expert gradient at layer {layer_idx}")
            _clip([gpu_experts.gate_up, gpu_experts.down], float(t["grad_clip"]))
            gate_packet = pack_gradient(gpu_experts.gate_up.grad, transfer, expert_chunk=expert_chunk, block_size=int8_block)
            down_packet = pack_gradient(gpu_experts.down.grad, transfer, expert_chunk=expert_chunk, block_size=int8_block)
            gpu_experts.zero_grad()
            self.adafactor.step_layer(layer_idx, gate_packet, down_packet, lr=expert_lr)
            optimizer.step()
            del gate_packet, down_packet
            compute_seconds = time.time() - compute_started
            self._report_backend_error_once(layer_idx)
            self.logger.live(
                step=step,
                layer=f"{layer_idx + 1}/{total_layers}",
                loss=self.last_loss,
                lr=lr,
                phase="backward",
                expert_stats=self.controller.live_layer_stats(layer_idx),
                cache_capacity=self.cache_capacity,
                loaded_layers=self.cache.last_loaded_layers,
                load_gib=self.cache.last_loaded_gib,
                load_seconds=self.cache.last_load_seconds,
                compute_seconds=compute_seconds,
                backend=self.model.layers[layer_idx].moe.last_backend,
            )
        self.cache.clear()
        return grad_h

    def _embedding_backward(self, batches, grad_h) -> None:
        t = self.cfg["training"]
        self.embed_optimizer.zero_grad(set_to_none=True)
        for mb_idx, batch in enumerate(batches):
            ids = batch[:, :-1].to(self.device, non_blocking=False)
            hidden = self.model.initial_hidden(ids)
            torch.autograd.backward(hidden, grad_h[mb_idx].to(self.device, dtype=hidden.dtype))
            del hidden, ids
        _clip(self.model.embed_tokens.parameters(), float(t["grad_clip"]))
        self.embed_optimizer.step()

    def train_step(self, step: int) -> dict[str, float]:
        self.current_step = step
        started = time.time()
        lr, expert_lr = self._set_lr(step)
        batches = self._fetch_superbatch()
        total_tokens = sum(batch.shape[0] * (batch.shape[1] - 1) for batch in batches)
        boundaries = self._forward_boundaries(batches, step, lr)
        grad_h, loss = self._head_backward(batches, boundaries, step, lr)
        self.last_loss = loss
        grad_h = self._layers_backward(batches, boundaries, grad_h, step, lr, expert_lr)
        self._embedding_backward(batches, grad_h)
        self.controller.finish_step(step + 1, total_tokens)
        routing_summary = self.controller.completed_step_summary()
        elapsed = max(1e-9, time.time() - started)
        del batches, boundaries, grad_h
        gc.collect()
        return {
            "step": step + 1,
            "loss": loss,
            "ppl": math.exp(min(20.0, loss)),
            "lr": lr,
            "expert_lr": expert_lr,
            "tokens": total_tokens,
            "tokens_per_sec": total_tokens / elapsed,
            "step_seconds": elapsed,
            "expert_cache_layers": self.cache_capacity,
            "threshold_tokens_seen": self.controller.tokens_seen,
            **routing_summary,
            "blocked_experts_layer0": int(self.controller.blocked[0].sum().item()),
            "routing_gap_layer0": self.controller.layer_gap(0),
        }

    def save_checkpoint(self, step: int, *, reason: str) -> Path:
        t = self.cfg["training"]
        self.logger.newline()
        self.logger.plot()
        # The huge mutable expert/Adafactor work tree is local during training. Mirror it only here.
        if self.backup.enabled:
            self.backup.sync(step, reason)
        print(f"Saving checkpoint at step {step} ({reason})...", flush=True)
        path = self.checkpoints.save(
            step=step,
            cfg=self.cfg,
            model=self.model,
            layer_optimizers=self.layer_optimizers,
            embed_optimizer=self.embed_optimizer,
            head_optimizer=self.head_optimizer,
            adafactor_state=self.adafactor.state_dict(),
            threshold_state=self.controller.state_dict(),
            data_state=self.data.state_dict(),
            expert_store_root=self.store.root,
            snapshot_experts=bool(t.get("snapshot_experts", False)),
            extra={
                "reason": reason,
                "persistent_work_backup": str(self.backup.backup_dir) if self.backup.enabled else None,
                "expert_cache_layers": self.cache_capacity,
            },
        )
        print(f"Checkpoint: {path}", flush=True)
        return path

    def train(self) -> None:
        t = self.cfg["training"]
        max_steps = int(t["max_steps"])
        save_every = int(t["save_every_steps"])
        plot_every = int(t["plot_every_steps"])
        try:
            for step in range(self.start_step, max_steps):
                metrics = self.train_step(step)
                self.last_completed_step = step + 1
                self.logger.record(metrics)
                self.logger.live(
                    step=step + 1,
                    layer="-",
                    loss=metrics["loss"],
                    lr=metrics["lr"],
                    tokens_per_sec=metrics["tokens_per_sec"],
                    phase="train",
                    expert_stats=metrics,
                    cache_capacity=self.cache_capacity,
                )
                if plot_every > 0 and (step + 1) % plot_every == 0:
                    self.logger.plot()
                if save_every > 0 and (step + 1) % save_every == 0:
                    self.save_checkpoint(step + 1, reason="periodic")
                if self.stop_requested:
                    self.save_checkpoint(step + 1, reason="interrupted")
                    break
            else:
                self.save_checkpoint(max_steps, reason="completed")
        except KeyboardInterrupt:
            if self.last_completed_step > self.start_step:
                self.save_checkpoint(self.last_completed_step, reason="keyboard-interrupt")
            raise
        finally:
            self.logger.plot()
            self.logger.newline()
