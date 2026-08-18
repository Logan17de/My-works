#!/usr/bin/env python3
"""Convert enriched ARDY Core motion into an animated source FBX.

Run this with the Make-It-Animatable Python environment because it includes
the Blender `bpy` package.

ARDY coordinates: Y-up, Z-forward.
Blender bridge:   Z-up, -Y-forward.
"""

import argparse
import math
from pathlib import Path

import bpy
import mathutils
import numpy as np


# ARDY (x, y-up, z-forward) -> Blender (x, y-back, z-up)
C = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Enriched ARDY NPZ")
    p.add_argument("--output", required=True, help="Animated source FBX")
    p.add_argument("--scale", type=float, default=1.0)
    return p.parse_args()


def cvt_pos(v):
    return (C @ np.asarray(v, dtype=np.float64)) * SCALE


def cvt_rot(r):
    r = np.asarray(r, dtype=np.float64)
    return C @ r @ C.T


def safe_normalize(v, fallback):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def children_of(parents, parent_idx):
    return [i for i, p in enumerate(parents) if int(p) == parent_idx]


def tail_position(points, parents, idx):
    children = children_of(parents, idx)
    head = points[idx]
    if children:
        return np.mean(points[children], axis=0)
    parent = int(parents[idx])
    if parent >= 0:
        direction = safe_normalize(head - points[parent], [0, 1, 0])
    else:
        direction = np.array([0.0, 1.0, 0.0])
    return head + direction * 0.10 * SCALE


def desired_bone_matrix(head, tail, global_rot):
    """Absolute armature-space matrix with Blender bone local Y along head->tail."""
    y_axis = safe_normalize(tail - head, [0, 1, 0])
    r = cvt_rot(global_rot)

    # Preserve as much twist as possible by borrowing ARDY's transformed X axis.
    x_guess = r[:, 0]
    x_axis = x_guess - y_axis * np.dot(x_guess, y_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_guess = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(x_guess, y_axis)) > 0.95:
            x_guess = np.array([0.0, 0.0, 1.0])
        x_axis = x_guess - y_axis * np.dot(x_guess, y_axis)
    x_axis = safe_normalize(x_axis, [1, 0, 0])
    z_axis = safe_normalize(np.cross(x_axis, y_axis), [0, 0, 1])
    x_axis = safe_normalize(np.cross(y_axis, z_axis), [1, 0, 0])

    m = mathutils.Matrix.Identity(4)
    # Blender Matrix rows/cols are explicit here; columns are local basis axes.
    for row in range(3):
        m[row][0] = float(x_axis[row])
        m[row][1] = float(y_axis[row])
        m[row][2] = float(z_axis[row])
        m[row][3] = float(head[row])
    return m


def main():
    args = parse_args()
    global SCALE
    SCALE = args.scale

    data = np.load(args.input, allow_pickle=True)
    names = [str(x) for x in data["joint_names"].tolist()]
    parents = np.asarray(data["joint_parents"], dtype=np.int32)
    neutral = np.asarray(data["neutral_joints"], dtype=np.float64)
    posed = np.asarray(data["posed_joints"], dtype=np.float64)
    global_rots = np.asarray(data["global_rot_mats"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    if posed.shape[:2] != global_rots.shape[:2]:
        raise ValueError("posed_joints and global_rot_mats frame/joint dimensions differ")
    if posed.shape[1] != len(names):
        raise ValueError("Joint metadata does not match motion")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = max(1, int(round(fps)))
    scene.frame_start = 0
    scene.frame_end = len(posed) - 1

    arm_data = bpy.data.armatures.new("ARDY_Core_Armature")
    arm_obj = bpy.data.objects.new("ARDY_Core", arm_data)
    scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = []
    neutral_b = np.stack([cvt_pos(v) for v in neutral], axis=0)

    for i, name in enumerate(names):
        bone = arm_data.edit_bones.new(name)
        bone.head = neutral_b[i]
        tail = tail_position(neutral_b, parents, i)
        if np.linalg.norm(tail - neutral_b[i]) < 1e-5:
            tail = neutral_b[i] + np.array([0.0, 0.1 * SCALE, 0.0])
        bone.tail = tail
        bone.use_connect = False
        edit_bones.append(bone)

    for i, parent in enumerate(parents):
        if int(parent) >= 0:
            edit_bones[i].parent = edit_bones[int(parent)]

    bpy.ops.object.mode_set(mode="POSE")

    for frame_idx in range(len(posed)):
        scene.frame_set(frame_idx)
        pts = np.stack([cvt_pos(v) for v in posed[frame_idx]], axis=0)

        # Parents first. Core skeleton metadata is parent-before-child.
        for j, name in enumerate(names):
            pb = arm_obj.pose.bones[name]
            tail = tail_position(pts, parents, j)
            matrix = desired_bone_matrix(pts[j], tail, global_rots[frame_idx, j])
            pb.matrix = matrix
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert(data_path="location", frame=frame_idx)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
            pb.keyframe_insert(data_path="scale", frame=frame_idx)

    bpy.ops.object.mode_set(mode="OBJECT")
    output = str(Path(args.output).resolve())
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.fbx(
        filepath=output,
        check_existing=False,
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_simplify_factor=0.0,
        path_mode="AUTO",
    )
    print(output)


if __name__ == "__main__":
    main()
