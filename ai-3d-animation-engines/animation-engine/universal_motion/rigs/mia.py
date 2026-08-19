from __future__ import annotations

from pathlib import Path
from .base import JSONRigAdapter, RigPreset

def create_mia_adapter(armature)->JSONRigAdapter:
    preset_path=Path(__file__).resolve().parent.parent/"rig_presets"/"mia_mixamo.json"
    return JSONRigAdapter(armature,RigPreset.from_json(preset_path))
