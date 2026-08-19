#!/usr/bin/env python3
"""Retarget an ARDY source FBX onto a Make-It-Animatable humanoid rig.

This wrapper uses Make-It-Animatable's bundled Auto-Rig-Pro fork, but removes
one fragile part of the upstream helper: when source and target share exact
``mixamorig:*`` bone names, those mappings are forced explicitly instead of
leaving them to fuzzy auto-detection.

It also makes frame rate explicit so ARDY's timing cannot silently become
Blender's default FPS during the final FBX export.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

BASE_BONES = {
    "mixamorig:Hips",
    "mixamorig:Spine",
    "mixamorig:Spine1",
    "mixamorig:Spine2",
    "mixamorig:Neck",
    "mixamorig:Head",
    "mixamorig:LeftShoulder",
    "mixamorig:LeftArm",
    "mixamorig:LeftForeArm",
    "mixamorig:LeftHand",
    "mixamorig:RightShoulder",
    "mixamorig:RightArm",
    "mixamorig:RightForeArm",
    "mixamorig:RightHand",
    "mixamorig:LeftUpLeg",
    "mixamorig:LeftLeg",
    "mixamorig:LeftFoot",
    "mixamorig:LeftToeBase",
    "mixamorig:RightUpLeg",
    "mixamorig:RightLeg",
    "mixamorig:RightFoot",
    "mixamorig:RightToeBase",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="Rigged character FBX")
    p.add_argument("--animation", required=True, help="ARDY source animation FBX")
    p.add_argument("--output", required=True, help="Animated output FBX")
    p.add_argument("--fps", type=float, required=True, help="Expected ARDY motion FPS")
    p.add_argument("--preview-glb", default=None)
    p.add_argument("--inplace", action="store_true")
    return p.parse_args()


def require_file(path: str, label: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file() or p.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found/empty: {p}")
    return p


def set_scene_fps(scene, fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")
    rounded = max(1, int(round(fps)))
    scene.render.fps = rounded
    scene.render.fps_base = rounded / fps
    actual = float(scene.render.fps) / float(scene.render.fps_base)
    if abs(actual - fps) > 1e-6:
        raise RuntimeError(f"Could not set Blender scene FPS to {fps}; got {actual}")


def main() -> None:
    args = parse_args()
    target_path = require_file(args.target, "Target FBX")
    animation_path = require_file(args.animation, "Animation FBX")

    mia_root = Path(os.environ.get("MIA_ROOT", "/content/Make-It-Animatable")).resolve()
    if not (mia_root / "util" / "blender_utils.py").is_file():
        raise FileNotFoundError(f"Make-It-Animatable not found at {mia_root}")
    os.chdir(mia_root)

    from util import blender_utils

    bpy = blender_utils.bpy
    blender_utils.reset()
    set_scene_fps(bpy.context.scene, args.fps)

    character = blender_utils.load_file(str(target_path))
    target_armature = blender_utils.get_armature_obj(character)
    if target_armature is None:
        raise RuntimeError("Target file does not contain an armature")

    source_objects = blender_utils.load_file(str(animation_path))
    source_armature = blender_utils.get_armature_obj(source_objects)
    if source_armature is None:
        raise RuntimeError("ARDY source FBX does not contain an armature")
    source_action = (
        source_armature.animation_data.action
        if source_armature.animation_data is not None
        else None
    )
    if source_action is None:
        raise RuntimeError("ARDY source armature has no animation action")

    target_bones = {b.name for b in target_armature.data.bones}
    source_bones = {b.name for b in source_armature.data.bones}
    missing_target = sorted(BASE_BONES - target_bones)
    missing_source = sorted(BASE_BONES - source_bones)
    if missing_target:
        raise RuntimeError("Target rig missing Mixamo bones: " + ", ".join(missing_target))
    if missing_source:
        raise RuntimeError(
            "ARDY source bridge missing Mixamo bones: " + ", ".join(missing_source)
        )

    blender_utils.set_action(target_armature, source_action)

    blender_utils.enable_arp(target_armature)
    scn = bpy.context.scene
    set_scene_fps(scn, args.fps)
    scn.source_rig = source_armature.name
    scn.target_rig = target_armature.name
    if args.inplace:
        scn.arp_retarget_in_place = True

    bpy.context.view_layer.objects.active = target_armature
    target_armature.select_set(True)
    bpy.ops.arp.auto_scale()
    bpy.ops.arp.build_bones_list()

    mapping_items = {item.name: item for item in scn.bones_map_v2}
    missing_map_entries = sorted(BASE_BONES - mapping_items.keys())
    if missing_map_entries:
        raise RuntimeError(
            "Auto-Rig-Pro did not expose required target mapping entries: "
            + ", ".join(missing_map_entries)
        )

    for bone_name in BASE_BONES:
        mapping_items[bone_name].source_bone = bone_name

    hips = mapping_items["mixamorig:Hips"]
    scn.bones_map_index = list(scn.bones_map_v2).index(hips)
    hips.set_as_root = True

    bad = [
        item.name
        for item in scn.bones_map_v2
        if item.name in BASE_BONES and item.source_bone not in source_bones
    ]
    if bad:
        raise RuntimeError("Invalid forced retarget mappings: " + ", ".join(sorted(bad)))

    bpy.ops.arp.retarget()
    blender_utils.update()
    set_scene_fps(scn, args.fps)

    final_action = (
        target_armature.animation_data.action
        if target_armature.animation_data is not None
        else None
    )
    if final_action is None:
        raise RuntimeError("Retarget completed without creating a target animation action")

    keyframes = blender_utils.get_keyframes([target_armature])
    if len(keyframes) < 2:
        raise RuntimeError("Retargeted target contains fewer than two animation keyframes")

    for obj in source_objects:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    scene.frame_start = min(keyframes)
    scene.frame_end = max(keyframes)
    set_scene_fps(scene, args.fps)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in character:
        if obj.name in bpy.data.objects:
            obj.select_set(True)
    bpy.context.view_layer.objects.active = target_armature

    bpy.ops.export_scene.fbx(
        filepath=str(output),
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
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Final FBX export failed: {output}")
    actual_fps = float(scene.render.fps) / float(scene.render.fps_base)
    print(
        f"Retargeted FBX: {output} | frames {scene.frame_start}-{scene.frame_end} "
        f"| fps {actual_fps:g}"
    )

    if args.preview_glb:
        preview = Path(args.preview_glb).expanduser().resolve()
        preview.parent.mkdir(parents=True, exist_ok=True)
        try:
            bpy.ops.export_scene.gltf(
                filepath=str(preview),
                check_existing=False,
                use_selection=True,
                export_format="GLB",
                export_animations=True,
            )
            if preview.is_file() and preview.stat().st_size > 0:
                print(f"GLB preview: {preview}")
            else:
                print("WARNING: GLB preview export returned without a valid file")
        except Exception as exc:
            print(f"WARNING: GLB preview export failed; FBX is still valid: {exc}")


if __name__ == "__main__":
    main()
