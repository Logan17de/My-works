from __future__ import annotations

from pathlib import Path
import numpy as np

from ..model import CanonicalMotion, CANONICAL_JOINTS, CANONICAL_PARENTS

C = np.array([[1.0,0.0,0.0],[0.0,0.0,-1.0],[0.0,1.0,0.0]], dtype=np.float64)

ARDY_TO_CANONICAL = {
    "Hips":"hips","Spine":"spine","Spine1":"spine_1","Spine2":"chest","Spine3":"upper_chest","Neck":"neck","Head":"head",
    "LeftShoulder":"left_shoulder","LeftArm":"left_upper_arm","LeftForeArm":"left_forearm","LeftHand":"left_hand","LeftHandEnd":"left_hand_end","LeftHandThumb1":"left_thumb_1",
    "RightShoulder":"right_shoulder","RightArm":"right_upper_arm","RightForeArm":"right_forearm","RightHand":"right_hand","RightHandEnd":"right_hand_end","RightHandThumb1":"right_thumb_1",
    "LeftUpLeg":"left_upper_leg","LeftLeg":"left_lower_leg","LeftFoot":"left_foot","LeftToeBase":"left_toe",
    "RightUpLeg":"right_upper_leg","RightLeg":"right_lower_leg","RightFoot":"right_foot","RightToeBase":"right_toe",
}
CONTACT_NAMES = ("left_heel","left_toe","right_heel","right_toe")

def _cvt_pos(arr): return np.einsum("ij,...j->...i", C, np.asarray(arr,dtype=np.float64))
def _cvt_rot(arr):
    a=np.asarray(arr,dtype=np.float64)
    return np.einsum("ij,...jk,lk->...il", C, a, C)

def _parents_from_schema():
    idx={name:i for i,name in enumerate(CANONICAL_JOINTS)}
    return np.asarray([-1 if CANONICAL_PARENTS[n] is None else idx[CANONICAL_PARENTS[n]] for n in CANONICAL_JOINTS],dtype=np.int32)

class ARDYMotionSource:
    """Adapter from validated ARDY Core27 bridge NPZ to CanonicalMotion."""
    source_name="ardy_core27"

    def load(self, bridge_path: str|Path) -> CanonicalMotion:
        path=Path(bridge_path).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        with np.load(path,allow_pickle=True) as z:
            required={"joint_names","joint_parents","neutral_joints","local_rot_mats","global_rot_mats","root_positions","posed_joints","fps"}
            missing=required.difference(z.files)
            if missing: raise KeyError(f"ARDY bridge missing keys: {sorted(missing)}")
            src_names=[str(x) for x in z["joint_names"].tolist()]
            src_parents=np.asarray(z["joint_parents"],dtype=np.int32)
            neutral=np.asarray(z["neutral_joints"],dtype=np.float64)
            local=np.asarray(z["local_rot_mats"],dtype=np.float64)
            glob=np.asarray(z["global_rot_mats"],dtype=np.float64)
            roots=np.asarray(z["root_positions"],dtype=np.float64)
            posed=np.asarray(z["posed_joints"],dtype=np.float64)
            fps=float(np.asarray(z["fps"]).reshape(-1)[0])
            contacts=np.asarray(z["foot_contacts"],dtype=np.float64) if "foot_contacts" in z.files else None

        src_idx={name:i for i,name in enumerate(src_names)}
        missing_names=sorted(set(ARDY_TO_CANONICAL)-set(src_idx))
        if missing_names: raise RuntimeError("ARDY Core27 mapping is incomplete: "+", ".join(missing_names))
        canonical_to_src={canon:src for src,canon in ARDY_TO_CANONICAL.items()}
        order=[src_idx[canonical_to_src[name]] for name in CANONICAL_JOINTS]

        for canon_name in CANONICAL_JOINTS:
            src_name=canonical_to_src[canon_name]; src_i=src_idx[src_name]; parent_i=int(src_parents[src_i])
            expected=CANONICAL_PARENTS[canon_name]
            actual=None
            if parent_i>=0:
                actual=ARDY_TO_CANONICAL.get(src_names[parent_i])
            if actual!=expected:
                raise RuntimeError(f"ARDY hierarchy mismatch for {src_name}: expected {expected!r}, got {actual!r}")

        rest=_cvt_pos(neutral[order]); rest=rest-rest[CANONICAL_JOINTS.index("hips")]
        local_c=_cvt_rot(local[:,order]); global_c=_cvt_rot(glob[:,order]); root_c=_cvt_pos(roots); posed_c=_cvt_pos(posed[:,order])
        contact_names=()
        if contacts is not None:
            if contacts.ndim!=2 or contacts.shape[0]!=local_c.shape[0]: raise ValueError(f"Invalid ARDY foot_contacts shape: {contacts.shape}")
            if contacts.shape[1]>=4:
                contacts=contacts[:,:4]; contact_names=CONTACT_NAMES
            else:
                contact_names=tuple(f"contact_{i}" for i in range(contacts.shape[1]))

        return CanonicalMotion(
            joint_names=CANONICAL_JOINTS, parents=_parents_from_schema(), local_rotations=local_c, global_rotations=global_c,
            root_positions=root_c, rest_positions=rest, posed_positions=posed_c, fps=fps,
            contact_names=contact_names, contacts=contacts, source_name=self.source_name,
            metadata={"source_file":str(path),"coordinate_conversion":"ARDY Y-up -> Blender/Unreal Z-up","rotation_semantics":"ARDY local rotations are rest-relative joint rotations in its FK model"},
        ).validate()
