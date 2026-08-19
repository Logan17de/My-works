from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class RigPreset:
    preset_id: str
    provider: str
    semantic_bones: dict[str,str]
    motion_chains: dict[str,tuple[str,...]]
    status: str = "active"

    @classmethod
    def from_json(cls,path: str|Path)->"RigPreset":
        q=Path(path).expanduser().resolve(); data=json.loads(q.read_text(encoding="utf-8"))
        if data.get("status","active")!="active": raise RuntimeError(f"Rig preset {data.get('id',q.stem)} is not active")
        semantic_bones={str(k):str(v) for k,v in data["semantic_bones"].items()}
        chains={semantic:tuple(data.get("motion_chains",{}).get(semantic,[semantic])) for semantic in semantic_bones}
        return cls(str(data["id"]),str(data.get("provider",data["id"])),semantic_bones,chains,"active")

class JSONRigAdapter:
    """Data-driven rig adapter around a Blender armature."""
    def __init__(self,armature,preset:RigPreset):
        self.armature=armature; self.preset=preset
        bone_names={b.name for b in armature.data.bones}; missing=sorted(set(preset.semantic_bones.values())-bone_names)
        if missing: raise RuntimeError(f"Rig {armature.name} does not satisfy preset {preset.preset_id}; missing bones: {missing}")

    @property
    def semantic_bones(self): return self.preset.semantic_bones
    def bone_name(self,semantic): return self.preset.semantic_bones[semantic]
    def motion_chain(self,semantic): return self.preset.motion_chains.get(semantic,(semantic,))
    def mapped_semantics(self): return tuple(self.preset.semantic_bones)

    def nearest_mapped_parent_semantic(self,semantic):
        bone=self.armature.data.bones[self.bone_name(semantic)]; mapped_by_bone={v:k for k,v in self.preset.semantic_bones.items()}; parent=bone.parent
        while parent is not None:
            if parent.name in mapped_by_bone: return mapped_by_bone[parent.name]
            parent=parent.parent
        return None

    def topological_semantics(self):
        remaining=set(self.mapped_semantics()); result=[]
        while remaining:
            progressed=False
            for semantic in sorted(remaining):
                parent=self.nearest_mapped_parent_semantic(semantic)
                if parent is None or parent in result:
                    result.append(semantic); remaining.remove(semantic); progressed=True; break
            if not progressed: raise RuntimeError(f"Could not topologically order rig semantics: {sorted(remaining)}")
        return result
