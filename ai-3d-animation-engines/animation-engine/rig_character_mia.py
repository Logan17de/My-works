#!/usr/bin/env python3
"""Headless Make-It-Animatable wrapper with output validation."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REQUIRED_BASE_BONES = {
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
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-fingers", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() not in {".glb", ".fbx", ".obj", ".ply"}:
        raise ValueError(f"Unsupported character format: {input_path.suffix}")

    mia_root = Path(os.environ.get("MIA_ROOT", "/content/Make-It-Animatable")).resolve()
    if not (mia_root / "app.py").is_file():
        raise FileNotFoundError(f"Make-It-Animatable not found at {mia_root}")
    os.chdir(mia_root)

    import app as mia
    from util import blender_utils

    # Upstream functions return Gradio update dictionaries. The headless path
    # does not consume those dictionaries, so hashable placeholders are enough.
    for name in (
        "state",
        "output_joints_coarse",
        "output_normed_input",
        "output_sample",
        "output_joints",
        "output_bw",
        "output_rest_vis",
        "output_rest_lbs",
        "output_anim_vis",
        "output_anim",
    ):
        setattr(mia, name, name)

    print("Loading Make-It-Animatable models...")
    mia.init_models()
    db = mia.DB()

    mia.prepare_input(
        str(input_path),
        is_gs=False,
        opacity_threshold=0.0,
        db=db,
        export_temp=False,
    )
    if not db.is_mesh:
        raise ValueError(
            "Animation Engine requires a polygon mesh. "
            "The selected file was loaded as a point cloud."
        )

    mia.preprocess(db)
    mia.infer(input_normal=False, db=db)
    mia.vis(
        bw_fix=True,
        bw_vis_bone="LeftArm",
        no_fingers=args.no_fingers,
        db=db,
    )
    mia.vis_blender(
        reset_to_rest=True,
        remove_fingers=args.no_fingers,
        rest_pose_type="No",
        ignore_pose_parts=[],
        animation_file=None,
        retarget=False,
        inplace=False,
        db=db,
    )

    source = Path(db.anim_path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Make-It-Animatable did not produce a valid FBX: {source}")

    armatures = blender_utils.get_all_armature_obj()
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one generated armature, found {len(armatures)}")
    armature = armatures[0]
    bone_names = {bone.name for bone in armature.data.bones}
    missing = sorted(REQUIRED_BASE_BONES - bone_names)
    if missing:
        raise RuntimeError(
            "Generated rig is not Mixamo-compatible; missing base bones: "
            + ", ".join(missing)
        )

    mesh_objects = blender_utils.get_all_mesh_obj()
    if not mesh_objects:
        raise RuntimeError("Generated rig contains no mesh object")
    if not any(
        any(mod.type == "ARMATURE" and mod.object == armature for mod in mesh.modifiers)
        for mesh in mesh_objects
    ):
        raise RuntimeError("Generated mesh is not bound to the generated armature")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to copy rigged FBX to {output}")

    print(
        f"Rig validated: {len(bone_names)} bones, "
        f"{sum(len(m.data.vertices) for m in mesh_objects)} vertices"
    )
    print(output)


if __name__ == "__main__":
    main()
