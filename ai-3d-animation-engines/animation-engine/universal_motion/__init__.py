"""Universal motion retargeting core.

Motion sources produce CanonicalMotion. Rig adapters describe how canonical
semantics map onto a target armature. The retarget core contains no ARDY or
MIA-specific bone names.
"""
from .model import CanonicalMotion, CANONICAL_JOINTS, CANONICAL_PARENTS

__all__ = ["CanonicalMotion", "CANONICAL_JOINTS", "CANONICAL_PARENTS"]
