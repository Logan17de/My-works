from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RoutingResult:
    logits: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    natural_indices: torch.Tensor


class ThresholdController:
    """Cumulative per-layer expert-use tracker with a step-stable execution mask."""

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        *,
        enabled: bool,
        warmup_tokens: int,
        imbalance_threshold: float,
        block_fraction: float,
        check_every_steps: int = 1,
    ) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.enabled = enabled
        self.warmup_tokens = int(warmup_tokens)
        self.imbalance_threshold = float(imbalance_threshold)
        self.block_fraction = float(block_fraction)
        self.check_every_steps = max(1, int(check_every_steps))
        self.executed_counts = torch.zeros(num_layers, num_experts, dtype=torch.int64)
        self.natural_counts = torch.zeros_like(self.executed_counts)
        self.blocked = torch.zeros(num_layers, num_experts, dtype=torch.bool)
        self.tokens_seen = 0
        self.last_refresh_step = -1

    @torch.no_grad()
    def record(self, layer_idx: int, natural: torch.Tensor, executed: torch.Tensor) -> None:
        nat = natural.detach().reshape(-1).to("cpu", non_blocking=False)
        exe = executed.detach().reshape(-1).to("cpu", non_blocking=False)
        self.natural_counts[layer_idx] += torch.bincount(nat, minlength=self.num_experts)
        self.executed_counts[layer_idx] += torch.bincount(exe, minlength=self.num_experts)

    def mask_for(self, layer_idx: int, device: torch.device) -> torch.Tensor:
        return self.blocked[layer_idx].to(device=device, non_blocking=True)

    @torch.no_grad()
    def finish_step(self, step: int, trained_tokens: int) -> None:
        self.tokens_seen += int(trained_tokens)
        if not self.enabled or self.warmup_tokens < 0 or self.tokens_seen < self.warmup_tokens:
            return
        if step % self.check_every_steps != 0:
            return
        self.refresh_masks(step)

    @torch.no_grad()
    def refresh_masks(self, step: int) -> None:
        block_n = int(round(self.num_experts * self.block_fraction))
        block_n = min(max(block_n, 0), max(self.num_experts - 1, 0))
        new_masks = torch.zeros_like(self.blocked)
        if block_n == 0:
            self.blocked = new_masks
            self.last_refresh_step = step
            return
        for layer in range(self.num_layers):
            counts = self.executed_counts[layer]
            most = int(counts.max().item())
            least = int(counts.min().item())
            if most <= 0:
                continue
            gap = (most - least) / most
            if gap > self.imbalance_threshold:
                hottest = torch.topk(counts, k=block_n, largest=True).indices
                new_masks[layer, hottest] = True
        self.blocked = new_masks
        self.last_refresh_step = step

    def layer_gap(self, layer_idx: int) -> float:
        counts = self.executed_counts[layer_idx]
        most = int(counts.max().item())
        least = int(counts.min().item())
        return 0.0 if most <= 0 else (most - least) / most

    def state_dict(self) -> dict:
        return {
            "executed_counts": self.executed_counts,
            "natural_counts": self.natural_counts,
            "blocked": self.blocked,
            "tokens_seen": self.tokens_seen,
            "last_refresh_step": self.last_refresh_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.executed_counts.copy_(state["executed_counts"])
        self.natural_counts.copy_(state["natural_counts"])
        self.blocked.copy_(state["blocked"])
        self.tokens_seen = int(state.get("tokens_seen", 0))
        self.last_refresh_step = int(state.get("last_refresh_step", -1))


class SoftmaxTopKRouter(nn.Module):
    """Qwen-style softmax router with optional pretraining execution masking."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        layer_idx: int,
        controller: ThresholdController | None,
        *,
        normalize_topk: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.controller = controller
        self.normalize_topk = normalize_topk
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, hidden_states: torch.Tensor, *, record: bool = False) -> RoutingResult:
        flat = hidden_states.reshape(-1, self.hidden_size)
        logits = F.linear(flat, self.weight)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        natural_indices = torch.topk(probs, self.top_k, dim=-1).indices

        if self.controller is not None:
            blocked = self.controller.mask_for(self.layer_idx, probs.device)
            if bool(blocked.any()):
                selectable = probs.masked_fill(blocked.unsqueeze(0), -1.0)
            else:
                selectable = probs
        else:
            selectable = probs

        top_values, indices = torch.topk(selectable, self.top_k, dim=-1)
        top_values = top_values.clamp_min_(0.0)
        if self.normalize_topk:
            top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = top_values.to(logits.dtype)

        if record and self.controller is not None:
            self.controller.record(self.layer_idx, natural_indices, indices)
        return RoutingResult(logits=logits, weights=weights, indices=indices, natural_indices=natural_indices)
