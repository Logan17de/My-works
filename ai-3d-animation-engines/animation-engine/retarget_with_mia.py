#!/usr/bin/env python3
"""Retarget an ARDY source FBX onto a Make-It-Animatable humanoid rig."""

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="Rigged character FBX")
    p.add_argument("--animation", required=True, help="ARDY source animation FBX")
    p.add_argument("--output", required=True, help="Animated output FBX")
    p.add_argument("--preview-glb", default=None)
    p.add_argument("--inplace", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    mia_root = os.environ.get("MIA_ROOT", "/content/Make-It-Animatable")
    os.chdir(mia_root)

    from util import blender_utils

    bpy = blender_utils.bpy
    blender_utils.reset()

    character = blender_utils.load_file(str(Path(args.target).resolve()))
    armature = blender_utils.get_armature_obj(character)
    if armature is None:
        raise RuntimeError("Target file does not contain an armature.")

    blender_utils.load_mixamo_anim(
        character,
        str(Path(args.animation).resolve()),
        do_retarget=True,
        inplace=args.inplace,
    )
    blender_utils.update()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        check_existing=False,
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_simplify_factor=0.0,
        path_mode="COPY",
        embed_textures=True,
    )
    print(output)

    if args.preview_glb:
        preview = Path(args.preview_glb).resolve()
        preview.parent.mkdir(parents=True, exist_ok=True)
        try:
            bpy.ops.export_scene.gltf(
                filepath=str(preview),
                check_existing=False,
                use_selection=False,
                export_format="GLB",
                export_animations=True,
            )
            print(preview)
        except Exception as exc:
            print(f"GLB preview export failed (FBX is still valid output): {exc}")


if __name__ == "__main__":
    main()
