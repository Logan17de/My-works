from __future__ import annotations

"""Compatibility shim for zero-impact micro-neuron growth.

Zero-impact growth is now implemented directly in ``cells.py`` by keeping the
cell-output normalization denominator independent of the number of activated
micro-neurons. Activating a new zero-output slot therefore changes neither old
parameters nor logits.
"""


def apply_zero_impact_growth_patch() -> None:
    """Retained for entry-point compatibility; no runtime patch is required."""
    return
