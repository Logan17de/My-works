#!/usr/bin/env python3
"""Headless wrapper around Make-It-Animatable's current inference pipeline."""

import argparse
import os
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-fingers", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    mia_root = os.environ.get("MIA_ROOT", "/content/Make-It-Animatable")
    os.chdir(mia_root)

    import app as mia

    # The upstream functions return UI update dictionaries. For headless use,
    # dummy hashable keys are enough; all useful state lives in DB.
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

    mia.init_models()
    db = mia.DB()
    mia.clear(db)

    mia.prepare_input(args.input, is_gs=False, opacity_threshold=0.0, db=db, export_temp=False)
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

    source = Path(db.anim_path)
    if not source.exists():
        raise FileNotFoundError(f"Make-It-Animatable did not produce {source}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    print(output)


if __name__ == "__main__":
    main()
