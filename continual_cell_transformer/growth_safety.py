from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

import torch

from model import ContinualCellTransformer


T = TypeVar("T")


@contextmanager
def deterministic_attention() -> Iterator[None]:
    """Force the deterministic math SDPA backend during safety comparisons.

    CUDA may otherwise choose FlashAttention or memory-efficient SDPA kernels.
    Two identical full-model forwards can then differ by around 1e-5 to 1e-4,
    which is large enough to create a false zero-impact failure.
    """
    if not torch.cuda.is_available():
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            yield
        return
    except (ImportError, AttributeError, TypeError):
        pass

    # Compatibility path for older PyTorch releases.
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    ):
        yield


def zero_impact(
    model: ContinualCellTransformer,
    inputs: torch.Tensor,
    mutation: Callable[[], T],
    label: str,
    tolerance: float = 1e-5,
) -> T:
    """Apply a structural mutation only after deterministic equivalence checks.

    A repeated no-op forward is checked first. This distinguishes genuine model
    drift from an execution backend that is nondeterministic even before the
    mutation occurs.
    """
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad(), deterministic_attention():
            baseline_a = model(
                inputs,
                adaptive_inference=False,
            )["logits"].detach().clone()
            baseline_b = model(
                inputs,
                adaptive_inference=False,
            )["logits"].detach().clone()

            baseline_drift = float(
                (baseline_b - baseline_a).abs().max()
            )
            if baseline_drift > tolerance:
                raise RuntimeError(
                    f"{label} cannot be verified: repeated no-op forwards "
                    f"drifted by {baseline_drift:.3e}"
                )

            result = mutation()
            after = model(
                inputs,
                adaptive_inference=False,
            )["logits"].detach()

        mutation_drift = float((after - baseline_b).abs().max())
    finally:
        model.train(was_training)

    print(
        f"{label}: {result}; "
        f"baseline_drift={baseline_drift:.3e}; "
        f"max_logit_drift={mutation_drift:.3e}"
    )
    if mutation_drift > tolerance:
        raise RuntimeError(
            f"{label} was not zero-impact: drift={mutation_drift:.3e}"
        )
    return result
