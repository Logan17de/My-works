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
    """Cumulative per-layer expert-use tracker with a step-stable execution mask.

    Besides the cumulative counters used by the threshold curriculum, this controller
    keeps a lightweight per-step snapshot for live experiment monitoring.  The live
    metrics intentionally focus on the questions this experiment cares about:
    expert coverage, load skew, threshold masking, and how often masking changes the
    router's natural Top-K choice.
    """

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

        # Long-horizon counters used by the curriculum.
        self.executed_counts = torch.zeros(num_layers, num_experts, dtype=torch.int64)
        self.natural_counts = torch.zeros_like(self.executed_counts)
        self.blocked = torch.zeros(num_layers, num_experts, dtype=torch.bool)

        # Current optimizer-step counters used only for observability.
        self.step_executed_counts = torch.zeros_like(self.executed_counts)
        self.step_natural_counts = torch.zeros_like(self.natural_counts)
        self.step_rerouted = torch.zeros(num_layers, dtype=torch.int64)
        self.step_assignments = torch.zeros(num_layers, dtype=torch.int64)

        # Frozen snapshot of the most recently completed step, so logging can happen
        # after finish_step() clears the current counters.
        self.last_step_executed_counts = torch.zeros_like(self.executed_counts)
        self.last_step_natural_counts = torch.zeros_like(self.natural_counts)
        self.last_step_rerouted = torch.zeros(num_layers, dtype=torch.int64)
        self.last_step_assignments = torch.zeros(num_layers, dtype=torch.int64)

        self.tokens_seen = 0
        self.last_refresh_step = -1

    @torch.no_grad()
    def record(self, layer_idx: int, natural: torch.Tensor, executed: torch.Tensor) -> None:
        nat = natural.detach().reshape(-1).to("cpu", non_blocking=False)
        exe = executed.detach().reshape(-1).to("cpu", non_blocking=False)
        nat_counts = torch.bincount(nat, minlength=self.num_experts)
        exe_counts = torch.bincount(exe, minlength=self.num_experts)

        self.natural_counts[layer_idx] += nat_counts
        self.executed_counts[layer_idx] += exe_counts
        self.step_natural_counts[layer_idx] += nat_counts
        self.step_executed_counts[layer_idx] += exe_counts

        # Count natural Top-K assignments that had to be replaced specifically because
        # the curriculum blocked that expert.  This is more meaningful than comparing
        # Top-K positions, because one removed expert can shift several positions.
        blocked = self.blocked[layer_idx]
        rerouted = int(blocked[nat].sum().item()) if bool(blocked.any()) else 0
        self.step_rerouted[layer_idx] += rerouted
        self.step_assignments[layer_idx] += nat.numel()

    def mask_for(self, layer_idx: int, device: torch.device) -> torch.Tensor:
        return self.blocked[layer_idx].to(device=device, non_blocking=True)

    @staticmethod
    def _stats_from_counts(counts: torch.Tensor) -> dict[str, float | int]:
        total = int(counts.sum().item())
        if total <= 0:
            return {
                "active": 0,
                "gap": 0.0,
                "cv": 0.0,
                "hot_id": -1,
                "hot_share": 0.0,
            }
        active = int((counts > 0).sum().item())
        most = int(counts.max().item())
        least = int(counts.min().item())
        gap = 0.0 if most <= 0 else (most - least) / most
        mean = float(counts.float().mean().item())
        cv = 0.0 if mean <= 0 else float(counts.float().std(unbiased=False).item() / mean)
        hot_id = int(torch.argmax(counts).item())
        return {
            "active": active,
            "gap": float(gap),
            "cv": cv,
            "hot_id": hot_id,
            "hot_share": most / total,
        }

    def live_layer_stats(self, layer_idx: int) -> dict[str, float | int]:
        """Stats for the layer as it is being routed in the current forward step."""
        stats = self._stats_from_counts(self.step_executed_counts[layer_idx])
        assignments = int(self.step_assignments[layer_idx].item())
        stats.update(
            {
                "total_experts": self.num_experts,
                "blocked": int(self.blocked[layer_idx].sum().item()),
                "reroute_pct": (
                    100.0 * int(self.step_rerouted[layer_idx].item()) / assignments if assignments > 0 else 0.0
                ),
            }
        )
        return stats

    def completed_step_summary(self) -> dict[str, float | int]:
        """Compact all-layer summary for CSV, plots, and the final one-line step log."""
        if self.num_layers == 0:
            return {}
        active, gaps, cvs, hot_shares = [], [], [], []
        for layer in range(self.num_layers):
            s = self._stats_from_counts(self.last_step_executed_counts[layer])
            active.append(float(s["active"]))
            gaps.append(float(s["gap"]))
            cvs.append(float(s["cv"]))
            hot_shares.append(float(s["hot_share"]))
        assignments = int(self.last_step_assignments.sum().item())
        rerouted = int(self.last_step_rerouted.sum().item())
        blocked_per_layer = self.blocked.sum(dim=1)
        return {
            "expert_active_mean": sum(active) / len(active),
            "expert_active_min": min(active),
            "expert_gap_mean": sum(gaps) / len(gaps),
            "expert_gap_max": max(gaps),
            "expert_cv_mean": sum(cvs) / len(cvs),
            "expert_hot_share_max": max(hot_shares),
            "blocked_layers": int((blocked_per_layer > 0).sum().item()),
            "blocked_experts_total": int(blocked_per_layer.sum().item()),
            "reroute_pct": 100.0 * rerouted / assignments if assignments > 0 else 0.0,
        }

    @torch.no_grad()
    def finish_step(self, step: int, trained_tokens: int) -> None:
        # Snapshot routing from the step before clearing it.  Mask refresh happens
        # afterwards, so this snapshot describes the routing that actually executed.
        self.last_step_executed_counts.copy_(self.step_executed_counts)
        self.last_step_natural_counts.copy_(self.step_natural_counts)
        self.last_step_rerouted.copy_(self.step_rerouted)
        self.last_step_assignments.copy_(self.step_assignments)
        self.step_executed_counts.zero_()
        self.step_natural_counts.zero_()
        self.step_rerouted.zero_()
        self.step_assignments.zero_()

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
