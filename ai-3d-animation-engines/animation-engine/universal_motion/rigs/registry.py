from __future__ import annotations

from .mia import create_mia_adapter

def load_rig_adapter(preset_id:str,armature):
    key=preset_id.strip().lower()
    if key in {"mia","mia_mixamo","mixamo"}: return create_mia_adapter(armature)
    raise ValueError(f"Unsupported active rig preset {preset_id!r}. Active today: mia_mixamo. Rigify/UE5/MetaHuman adapters can be added without changing the retarget core.")
