#!/usr/bin/env python3
"""Validate an ARDY Core NPZ and add stable skeleton metadata for export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ardy.skeleton import CoreSkeleton27

REQUIRED_KEYS = {
    "local_rot_mats",
    "global_rot_mats",
    "root_positions",
    "posed_joints",
    "fps",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf values")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with np.load(input_path, allow_pickle=True) as src:
        missing = REQUIRED_KEYS.difference(src.files)
        if missing:
            raise KeyError(f"ARDY output is missing required keys: {sorted(missing)}")
        data = {k: src[k] for k in src.files}

    skeleton = CoreSkeleton27()
    names = list(skeleton.bone_order_names)
    parents = skeleton.joint_parents.detach().cpu().numpy().astype(np.int32)
    neutral = skeleton.neutral_joints.detach().cpu().numpy().astype(np.float32)

    local_rots = np.asarray(data["local_rot_mats"])
    global_rots = np.asarray(data["global_rot_mats"])
    root_positions = np.asarray(data["root_positions"])
    posed = np.asarray(data["posed_joints"])
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    frames = local_rots.shape[0]
    joints = len(names)
    expected = {
        "local_rot_mats": (frames, joints, 3, 3),
        "global_rot_mats": (frames, joints, 3, 3),
        "root_positions": (frames, 3),
        "posed_joints": (frames, joints, 3),
    }
    actual = {
        "local_rot_mats": local_rots.shape,
        "global_rot_mats": global_rots.shape,
        "root_positions": root_positions.shape,
        "posed_joints": posed.shape,
    }
    for key, shape in expected.items():
        if actual[key] != shape:
            raise ValueError(f"{key} has shape {actual[key]}, expected {shape}")
    if not (fps > 0):
        raise ValueError(f"Invalid fps: {fps}")
    for key in ("local_rot_mats", "global_rot_mats", "root_positions", "posed_joints"):
        _require_finite(key, np.asarray(data[key]))

    data["joint_names"] = np.asarray(names, dtype=object)
    data["joint_parents"] = parents
    data["neutral_joints"] = neutral
    data["skeleton_name"] = np.asarray("cskel27")

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"Validated ARDY Core motion: {frames} frames, {joints} joints, {fps:g} fps")
    print(out)


if __name__ == "__main__":
    main()
