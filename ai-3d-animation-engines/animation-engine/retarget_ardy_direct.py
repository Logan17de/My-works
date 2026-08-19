#!/usr/bin/env python3
"""Deterministically bake ARDY Core motion onto a MIA/Mixamo rig.

This path intentionally does NOT use Auto-Rig-Pro retargeting.  ARDY global
joint rotations are treated as motion deltas in Blender world coordinates,
converted into the target armature's object space, then composed with each
target bone's own rest orientation.  This removes any dependence on matching
bone rolls/rest axes between the synthetic ARDY bridge and the MIA rig.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np

from progress_utils import Progress

# ARDY -> Blender coordinate conversion used by the validated bridge.
C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)

BODY_MAP = {
    "Hips": "mixamorig:Hips",
    "Spine": "mixamorig:Spine",
    "Spine1": "mixamorig:Spine1",
    "Spine2": "mixamorig:Spine2",
    "Neck": "mixamorig:Neck",
    "Head": "mixamorig:Head",
    "LeftShoulder": "mixamorig:LeftShoulder",
    "LeftArm": "mixamorig:LeftArm",
    "LeftForeArm": "mixamorig:LeftForeArm",
    "LeftHand": "mixamorig:LeftHand",
    "RightShoulder": "mixamorig:RightShoulder",
    "RightArm": "mixamorig:RightArm",
    "RightForeArm": "mixamorig:RightForeArm",
    "RightHand": "mixamorig:RightHand",
    "LeftUpLeg": "mixamorig:LeftUpLeg",
    "LeftLeg": "mixamorig:LeftLeg",
    "LeftFoot": "mixamorig:LeftFoot",
    "LeftToeBase": "mixamorig:LeftToeBase",
    "RightUpLeg": "mixamorig:RightUpLeg",
    "RightLeg": "mixamorig:RightLeg",
    "RightFoot": "mixamorig:RightFoot",
    "RightToeBase": "mixamorig:RightToeBase",
}
TARGET_TO_SOURCE = {v: k for k, v in BODY_MAP.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="MIA-rigged character FBX")
    p.add_argument("--motion-bridge", required=True, help="Validated motion_bridge.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--preview-glb", default=None)
    p.add_argument("--fps", type=float, required=True)
    return p.parse_args()


def require_file(path: str | Path, label: str) -> Path:
    q = Path(path).expanduser().resolve()
    if not q.is_file() or q.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing/empty: {q}")
    return q


def set_scene_fps(scene, fps: float):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")
    rounded = max(1, int(round(fps)))
    scene.render.fps = rounded
    scene.render.fps_base = rounded / fps


def cvt_pos(v) -> np.ndarray:
    return C @ np.asarray(v, dtype=np.float64)


def cvt_rot(r) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    return C @ r @ C.T


def rotation_only(matrix_world) -> np.ndarray:
    q = matrix_world.to_quaternion()
    q.normalize()
    return np.asarray(q.to_matrix(), dtype=np.float64)


def target_skeleton_height_world(armature, target_names) -> float:
    zs = []
    mw = armature.matrix_world
    for name in target_names:
        b = armature.data.bones[name]
        zs.extend((float((mw @ b.head_local).z), float((mw @ b.tail_local).z)))
    h = max(zs) - min(zs)
    if not math.isfinite(h) or h <= 1e-8:
        raise RuntimeError(f"Invalid target skeleton height: {h}")
    return h


def source_skeleton_height(neutral: np.ndarray, source_idx: dict[str, int]) -> float:
    pts = np.stack([cvt_pos(neutral[source_idx[src]]) for src in BODY_MAP])
    h = float(pts[:, 2].max() - pts[:, 2].min())
    if not math.isfinite(h) or h <= 1e-8:
        raise RuntimeError(f"Invalid ARDY skeleton height: {h}")
    return h


def nearest_mapped_parent(bone, mapped_target_names: set[str]):
    parent = bone.parent
    while parent is not None and parent.name not in mapped_target_names:
        parent = parent.parent
    return parent


def topological_target_order(armature, mapped_target_names: set[str]) -> list[str]:
    remaining = set(mapped_target_names)
    result = []
    while remaining:
        progressed = False
        for name in sorted(remaining):
            parent = nearest_mapped_parent(armature.data.bones[name], mapped_target_names)
            if parent is None or parent.name in result:
                result.append(name)
                remaining.remove(name)
                progressed = True
                break
        if not progressed:
            raise RuntimeError(f"Could not topologically order mapped target bones: {sorted(remaining)}")
    return result


def mat4(rotation3: np.ndarray, translation3: np.ndarray) -> mathutils.Matrix:
    m = mathutils.Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            m[r][c] = float(rotation3[r, c])
    m.translation = mathutils.Vector(tuple(float(x) for x in translation3))
    return m


def main():
    a = parse_args()
    p = Progress("DIRECT RETARGET", 7)

    p.step("Validating MIA rig, ARDY bridge and FPS")
    target = require_file(a.target, "rigged target")
    bridge = require_file(a.motion_bridge, "motion bridge")
    if target.suffix.lower() != ".fbx":
        raise ValueError("Direct retarget currently requires the MIA-rigged target FBX")
    if not math.isfinite(a.fps) or a.fps <= 0:
        raise ValueError(f"Invalid FPS: {a.fps}")

    with np.load(bridge, allow_pickle=True) as z:
        required = {
            "joint_names", "joint_parents", "neutral_joints",
            "global_rot_mats", "root_positions", "posed_joints", "fps",
        }
        missing = required.difference(z.files)
        if missing:
            raise KeyError(f"motion_bridge.npz missing: {sorted(missing)}")
        names = [str(x) for x in z["joint_names"].tolist()]
        neutral = np.asarray(z["neutral_joints"], dtype=np.float64)
        global_rot = np.asarray(z["global_rot_mats"], dtype=np.float64)
        roots = np.asarray(z["root_positions"], dtype=np.float64)
        bridge_fps = float(np.asarray(z["fps"]).reshape(-1)[0])

    if abs(bridge_fps - a.fps) > 1e-4:
        raise RuntimeError(f"FPS mismatch: bridge={bridge_fps}, requested={a.fps}")
    if global_rot.ndim != 4 or global_rot.shape[1:] != (len(names), 3, 3):
        raise ValueError(f"Invalid global_rot_mats shape: {global_rot.shape}")
    if roots.shape != (global_rot.shape[0], 3):
        raise ValueError(f"Invalid root_positions shape: {roots.shape}")
    if not np.isfinite(global_rot).all() or not np.isfinite(roots).all():
        raise ValueError("ARDY motion contains NaN/Inf")
    source_idx = {name: i for i, name in enumerate(names)}
    missing_source = sorted(set(BODY_MAP) - set(source_idx))
    if missing_source:
        raise RuntimeError("ARDY bridge missing required body joints: " + ", ".join(missing_source))
    frames = int(global_rot.shape[0])
    p.info(f"Frames={frames} | FPS={a.fps:g} | shared body joints={len(BODY_MAP)}")

    p.step("Importing the already-rigged MIA character")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(target))
    scene = bpy.context.scene
    set_scene_fps(scene, a.fps)
    armatures = [o for o in scene.objects if o.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected exactly one armature, found {len(armatures)}")
    arm = armatures[0]
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("Rigged target contains no mesh")
    target_names = set(TARGET_TO_SOURCE)
    missing_target = sorted(target_names - {b.name for b in arm.data.bones})
    if missing_target:
        raise RuntimeError("MIA rig missing required body bones: " + ", ".join(missing_target))
    if arm.animation_data:
        arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis.identity()
    bpy.context.view_layer.update()
    p.info(f"Target armature={arm.name} | object scale={tuple(round(float(x), 6) for x in arm.scale)}")
    p.ok(f"Loaded {len(meshes)} bound mesh object(s)")

    p.step("Building deterministic rest-space conversion")
    target_height = target_skeleton_height_world(arm, target_names)
    source_height = source_skeleton_height(neutral, source_idx)
    motion_scale = target_height / source_height
    if not math.isfinite(motion_scale) or not (0.05 <= motion_scale <= 20.0):
        raise RuntimeError(f"Implausible root-motion scale ratio: {motion_scale}")
    order = topological_target_order(arm, target_names)
    target_rest_rot = {
        name: np.asarray(arm.data.bones[name].matrix_local.to_3x3(), dtype=np.float64)
        for name in target_names
    }
    target_rest_head = {
        name: np.asarray(arm.data.bones[name].head_local, dtype=np.float64)
        for name in target_names
    }
    mapped_parent = {}
    rest_offset = {}
    for name in target_names:
        parent = nearest_mapped_parent(arm.data.bones[name], target_names)
        mapped_parent[name] = parent.name if parent else None
        if parent:
            rest_offset[name] = target_rest_head[name] - target_rest_head[parent.name]

    obj_rot = rotation_only(arm.matrix_world)
    obj_rot_inv = obj_rot.T
    world_to_arm_vec = np.asarray(arm.matrix_world.inverted().to_3x3(), dtype=np.float64)
    p.info(f"Source skeleton={source_height:.4f} m | target skeleton={target_height:.4f} m")
    p.info(f"Root-motion scale={motion_scale:.6f}x")
    p.info("No Auto-Rig-Pro, no source armature scaling, no copied source bone rolls")
    p.ok("Target-specific rest-space conversion ready")

    p.step("Baking ARDY global motion directly onto target bones")
    scene.frame_start = 1
    scene.frame_end = frames
    root0 = cvt_pos(roots[0])
    prev_quat = {name: None for name in target_names}
    report_every = max(1, frames // 10)

    for fi in range(frames):
        scene.frame_set(fi + 1)

        delta_arm = {}
        for target_name, source_name in TARGET_TO_SOURCE.items():
            d_world = cvt_rot(global_rot[fi, source_idx[source_name]])
            delta_arm[target_name] = obj_rot_inv @ d_world @ obj_rot

        pose_head = {}
        root_delta_world = (cvt_pos(roots[fi]) - root0) * motion_scale
        root_delta_arm = world_to_arm_vec @ root_delta_world

        for target_name in order:
            parent_name = mapped_parent[target_name]
            if parent_name is None:
                head = target_rest_head[target_name] + root_delta_arm
            else:
                head = pose_head[parent_name] + delta_arm[parent_name] @ rest_offset[target_name]
            pose_head[target_name] = head

            desired_rot = delta_arm[target_name] @ target_rest_rot[target_name]
            pb = arm.pose.bones[target_name]
            pb.matrix = mat4(desired_rot, head)
            pb.scale = (1.0, 1.0, 1.0)

            q = pb.rotation_quaternion.copy()
            if prev_quat[target_name] is not None and q.dot(prev_quat[target_name]) < 0:
                q.negate()
                pb.rotation_quaternion = q
            prev_quat[target_name] = q.copy()

            pb.keyframe_insert(data_path="rotation_quaternion", frame=fi + 1)
            pb.keyframe_insert(data_path="location", frame=fi + 1)
            pb.keyframe_insert(data_path="scale", frame=fi + 1)

        if fi == 0 or fi + 1 == frames or (fi + 1) % report_every == 0:
            p.info(f"Baked {fi + 1}/{frames} frames ({(fi + 1) / frames * 100:.0f}%)")

    action = arm.animation_data.action if arm.animation_data else None
    if action is None:
        raise RuntimeError("Direct retarget did not create an animation action")
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    p.ok(f"Direct target action: {action.name} | frame range {action.frame_range[0]:.0f}-{action.frame_range[1]:.0f}")

    p.step("Checking target scale and non-root translation sanity")
    rest_height = target_skeleton_height_world(arm, target_names)
    if abs(rest_height / target_height - 1.0) > 0.005:
        raise RuntimeError("Target armature rest scale changed during direct retarget")
    max_non_root = 0.0
    for name in target_names:
        if name == "mixamorig:Hips":
            continue
        pb = arm.pose.bones[name]
        max_non_root = max(max_non_root, float(pb.location.length))
    local_height = max(
        float(arm.data.bones[n].head_local.z) for n in target_names
    ) - min(float(arm.data.bones[n].head_local.z) for n in target_names)
    normalized_non_root = max_non_root / max(abs(local_height), 1e-8)
    p.info(f"Max non-root local translation={max_non_root:.6f} ({normalized_non_root:.4f} of local skeleton height)")
    if normalized_non_root > 0.10:
        raise RuntimeError(
            f"Direct retarget generated excessive non-root translation ({normalized_non_root:.3f} body heights)"
        )
    p.ok("Target scale is locked and pose translations are sane")

    p.step("Exporting final animated FBX")
    out = Path(a.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = arm
    with p.heartbeat("FBX export", every=20):
        bpy.ops.export_scene.fbx(
            filepath=str(out),
            check_existing=False,
            use_selection=True,
            add_leaf_bones=False,
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False,
            bake_anim_force_startend_keying=True,
            bake_anim_step=1.0,
            bake_anim_simplify_factor=0.0,
            path_mode="COPY",
            embed_textures=True,
        )
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"Final FBX export failed: {out}")
    p.ok(f"Final FBX: {out} ({out.stat().st_size / 1024**2:.1f} MiB)")

    p.step("Creating optional clean animated GLB preview")
    if a.preview_glb:
        preview = Path(a.preview_glb).expanduser().resolve()
        preview.parent.mkdir(parents=True, exist_ok=True)
        try:
            with p.heartbeat("animated GLB preview export", every=20):
                bpy.ops.export_scene.gltf(
                    filepath=str(preview),
                    check_existing=False,
                    use_selection=True,
                    export_format="GLB",
                    export_animations=True,
                )
            if preview.is_file() and preview.stat().st_size > 0:
                p.ok(f"Preview GLB: {preview}")
            else:
                p.warn("Preview GLB was not produced; final FBX remains available")
        except Exception as exc:
            p.warn(f"Preview GLB export failed; final FBX remains available: {exc}")
    else:
        p.info("No preview GLB requested")

    p.done("Deterministic ARDY→MIA retarget complete")
    p.info("Artifacts are written; exiting without Blender/Python teardown")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
