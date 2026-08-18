#!/usr/bin/env python3
"""Render an ARDY skeleton motion NPZ to MP4 using matplotlib."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dpi", type=int, default=110)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    joints = np.asarray(data["posed_joints"], dtype=np.float32)
    parents = np.asarray(data["joint_parents"], dtype=np.int32)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    mins = joints.min(axis=(0, 1))
    maxs = joints.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.60, 0.75)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))

    # ARDY is Y-up / Z-forward. Plot with Z-forward on depth and Y vertical.
    def setup_axes():
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(max(0.0, center[1] - radius), center[1] + radius)
        ax.set_xlabel("X")
        ax.set_ylabel("Z / forward")
        ax.set_zlabel("Y / up")
        ax.view_init(elev=14, azim=-70)

    output = Path(args.output)
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
    print(output)


if __name__ == "__main__":
    main()
