#!/usr/bin/env python3
"""Render a validated ARDY skeleton motion NPZ to MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dpi", type=int, default=110)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as data:
        for key in ("posed_joints", "joint_parents", "fps"):
            if key not in data.files:
                raise KeyError(f"Missing required key: {key}")
        joints = np.asarray(data["posed_joints"], dtype=np.float32)
        parents = np.asarray(data["joint_parents"], dtype=np.int32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"posed_joints must be [T,J,3], got {joints.shape}")
    if parents.shape != (joints.shape[1],):
        raise ValueError("joint_parents length does not match posed_joints")
    if not np.isfinite(joints).all():
        raise ValueError("posed_joints contains NaN/Inf")
    if not fps > 0:
        raise ValueError(f"Invalid fps: {fps}")

    mins = joints.min(axis=(0, 1))
    maxs = joints.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.60, 0.75)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))

    def setup_axes() -> None:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(max(0.0, center[1] - radius), center[1] + radius)
        ax.set_xlabel("X")
        ax.set_ylabel("Z / forward")
        ax.set_zlabel("Y / up")
        ax.view_init(elev=14, azim=-70)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2200)

    with writer.saving(fig, str(output), dpi=args.dpi):
        for frame in joints:
            ax.cla()
            setup_axes()
            for j, parent in enumerate(parents):
                if parent < 0:
                    continue
                p0 = frame[parent]
                p1 = frame[j]
                ax.plot([p0[0], p1[0]], [p0[2], p1[2]], [p0[1], p1[1]])
            ax.scatter(frame[:, 0], frame[:, 2], frame[:, 1], s=10)
            writer.grab_frame()

    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Preview was not written: {output}")
    print(output)


if __name__ == "__main__":
    main()
