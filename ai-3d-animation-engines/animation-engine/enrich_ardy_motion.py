#!/usr/bin/env python3
"""Add ARDY Core skeleton metadata to a generated motion NPZ."""

import argparse
from pathlib import Path

import numpy as np
from ardy.skeleton import CoreSkeleton27


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    src = np.load(args.input, allow_pickle=True)
    data = {k: src[k] for k in src.files}

    skeleton = CoreSkeleton27()
    data["joint_names"] = np.asarray(skeleton.bone_order_names, dtype=object)
    data["joint_parents"] = skeleton.joint_parents.detach().cpu().numpy().astype(np.int32)
    data["neutral_joints"] = skeleton.neutral_joints.detach().cpu().numpy().astype(np.float32)
    data["skeleton_name"] = np.asarray("cskel27")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(out)


if __name__ == "__main__":
    main()
