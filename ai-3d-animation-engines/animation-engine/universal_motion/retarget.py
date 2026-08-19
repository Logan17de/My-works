from __future__ import annotations

import math
import numpy as np
import mathutils

from .model import CanonicalMotion

def rotation_only(matrix_world)->np.ndarray:
    q=matrix_world.to_quaternion(); q.normalize(); return np.asarray(q.to_matrix(),dtype=np.float64)

def mat4(rotation3,translation3):
    m=mathutils.Matrix.Identity(4)
    for r in range(3):
        for c in range(3): m[r][c]=float(rotation3[r,c])
    m.translation=mathutils.Vector(tuple(float(x) for x in translation3)); return m

def target_skeleton_height_world(armature,adapter):
    zs=[]; mw=armature.matrix_world
    for semantic in adapter.mapped_semantics():
        b=armature.data.bones[adapter.bone_name(semantic)]; zs.extend((float((mw@b.head_local).z),float((mw@b.tail_local).z)))
    h=max(zs)-min(zs)
    if not math.isfinite(h) or h<=1e-8: raise RuntimeError(f"Invalid target skeleton height: {h}")
    return h

def compose_chain(local_rotations,index,chain):
    out=np.eye(3,dtype=np.float64)
    for semantic in chain:
        if semantic not in index: raise KeyError(f"Canonical motion missing semantic {semantic!r}")
        out=out@local_rotations[index[semantic]]
    return out

def bake_motion(scene,armature,adapter,motion:CanonicalMotion,progress=None):
    """Bake CanonicalMotion onto a target rig without source/target-specific code."""
    motion.validate(); idx=motion.index; order=adapter.topological_semantics()
    if "hips" not in order: raise RuntimeError("Target rig adapter must map canonical 'hips'")
    target_height=target_skeleton_height_world(armature,adapter); motion_scale=target_height/motion.reference_height
    if not math.isfinite(motion_scale) or not (0.05<=motion_scale<=20.0): raise RuntimeError(f"Implausible motion scale ratio: {motion_scale}")

    rest_rot={}; rest_head={}; mapped_parent={}; rest_offset={}
    for semantic in order:
        bone=armature.data.bones[adapter.bone_name(semantic)]
        rest_rot[semantic]=np.asarray(bone.matrix_local.to_3x3(),dtype=np.float64)
        rest_head[semantic]=np.asarray(bone.head_local,dtype=np.float64)
        parent=adapter.nearest_mapped_parent_semantic(semantic); mapped_parent[semantic]=parent
        if parent is not None: rest_offset[semantic]=rest_head[semantic]-rest_head[parent]

    obj_rot=rotation_only(armature.matrix_world); obj_rot_inv=obj_rot.T
    world_to_arm_vec=np.asarray(armature.matrix_world.inverted().to_3x3(),dtype=np.float64)
    if armature.animation_data: armature.animation_data_clear()
    for pb in armature.pose.bones: pb.rotation_mode="QUATERNION"; pb.matrix_basis.identity()
    scene.frame_start=1; scene.frame_end=motion.frames
    root0=np.asarray(motion.root_positions[0],dtype=np.float64); prev_quat={s:None for s in order}; report_every=max(1,motion.frames//10)

    for fi in range(motion.frames):
        scene.frame_set(fi+1)
        target_local={s:compose_chain(motion.local_rotations[fi],idx,adapter.motion_chain(s)) for s in order}
        global_motion={}
        for s in order:
            parent=mapped_parent[s]; global_motion[s]=target_local[s] if parent is None else global_motion[parent]@target_local[s]
        global_arm={s:obj_rot_inv@global_motion[s]@obj_rot for s in order}
        root_delta_world=(np.asarray(motion.root_positions[fi],dtype=np.float64)-root0)*motion_scale
        root_delta_arm=world_to_arm_vec@root_delta_world; pose_head={}
        for s in order:
            parent=mapped_parent[s]
            head=rest_head[s]+root_delta_arm if parent is None else pose_head[parent]+global_arm[parent]@rest_offset[s]
            pose_head[s]=head
            pb=armature.pose.bones[adapter.bone_name(s)]; pb.matrix=mat4(global_arm[s]@rest_rot[s],head); pb.scale=(1.0,1.0,1.0)
            q=pb.rotation_quaternion.copy(); prev=prev_quat[s]
            if prev is not None and q.dot(prev)<0: q.negate(); pb.rotation_quaternion=q
            prev_quat[s]=q.copy(); pb.keyframe_insert(data_path="rotation_quaternion",frame=fi+1); pb.keyframe_insert(data_path="location",frame=fi+1); pb.keyframe_insert(data_path="scale",frame=fi+1)
        if progress and (fi==0 or fi+1==motion.frames or (fi+1)%report_every==0): progress.info(f"Universal bake {fi+1}/{motion.frames} ({(fi+1)/motion.frames*100:.0f}%)")

    action=armature.animation_data.action if armature.animation_data else None
    if action is None: raise RuntimeError("Universal retarget did not create an animation action")
    for fc in action.fcurves:
        for kp in fc.keyframe_points: kp.interpolation="LINEAR"
    return {"action":action,"target_height":target_height,"motion_scale":motion_scale,"frame_start":1,"frame_end":motion.frames,"adapter":adapter.preset.preset_id}
