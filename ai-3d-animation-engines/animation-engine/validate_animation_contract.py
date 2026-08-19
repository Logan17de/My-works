#!/usr/bin/env python3
"""Validate scale, timing and motion transfer across the Animation Engine."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np


REQUIRED_BONES = [
    "mixamorig:Hips",
    "mixamorig:Head",
    "mixamorig:LeftHand",
    "mixamorig:RightHand",
    "mixamorig:LeftFoot",
    "mixamorig:RightFoot",
]
EFFECTORS = REQUIRED_BONES[1:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--character-source", required=True)
    p.add_argument("--rigged-target", required=True)
    p.add_argument("--source-animation", required=True)
    p.add_argument("--animated-target", required=True)
    p.add_argument("--expected-fps", type=float, required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--scale-tolerance", type=float, default=0.08)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_path(path: Path) -> None:
    ext = path.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported 3D format: {path}")


def world_mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    reset()
    load_path(path)
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mw = obj.matrix_world
        for v in obj.data.vertices:
            p = mw @ v.co
            points.append((p.x, p.y, p.z))
    if not points:
        raise RuntimeError(f"No mesh vertices found in {path}")
    pts = np.asarray(points, dtype=np.float64)
    return pts.min(axis=0), pts.max(axis=0)


def mesh_height(path: Path) -> float:
    lo, hi = world_mesh_bounds(path)
    h = float(hi[2] - lo[2])
    if not math.isfinite(h) or h <= 1e-8:
        raise RuntimeError(f"Invalid character height for {path}: {h}")
    return h


def armature_and_action(path: Path):
    reset()
    load_path(path)
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"Expected exactly one armature in {path}, found {len(arms)}")
    arm = arms[0]
    action = arm.animation_data.action if arm.animation_data else None
    if action is None:
        raise RuntimeError(f"No animation action in {path}")
    names = {b.name for b in arm.data.bones}
    missing = [n for n in REQUIRED_BONES if n not in names]
    if missing:
        raise RuntimeError(f"{path.name} missing required motion bones: {missing}")
    return arm, action


def skeleton_height(arm) -> float:
    zs = []
    mw = arm.matrix_world
    for b in arm.data.bones:
        for p in (b.head_local, b.tail_local):
            w = mw @ p
            zs.append(float(w.z))
    h = max(zs) - min(zs)
    if not math.isfinite(h) or h <= 1e-8:
        raise RuntimeError("Invalid armature height")
    return h


def sample_motion(path: Path, samples: int) -> dict:
    arm, action = armature_and_action(path)
    start = float(action.frame_range[0])
    end = float(action.frame_range[1])
    if end <= start:
        raise RuntimeError(f"Animation in {path} has an invalid frame range {start}-{end}")

    frames = np.linspace(start, end, max(3, samples))
    height = skeleton_height(arm)
    trajectories = {name: [] for name in REQUIRED_BONES}

    for frame in frames:
        bpy.context.scene.frame_set(int(round(frame)))
        bpy.context.view_layer.update()
        for name in REQUIRED_BONES:
            pb = arm.pose.bones[name]
            p = arm.matrix_world @ pb.head
            trajectories[name].append([p.x, p.y, p.z])

    trajectories = {k: np.asarray(v, dtype=np.float64) for k, v in trajectories.items()}
    hips = trajectories["mixamorig:Hips"]
    root_delta = (hips - hips[0]) / height

    effector_delta = {}
    for name in EFFECTORS:
        rel = trajectories[name] - hips
        effector_delta[name] = (rel - rel[0]) / height

    scene_fps = float(bpy.context.scene.render.fps) / float(bpy.context.scene.render.fps_base)
    return {
        "frame_start": start,
        "frame_end": end,
        "frame_span": end - start,
        "scene_fps": scene_fps,
        "height": height,
        "root_delta": root_delta,
        "effector_delta": effector_delta,
    }


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def motion_energy(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(a**2, axis=-1))))


def ratio(target: float, source: float) -> float | None:
    if source < 1e-5:
        return None
    return float(target / source)


def main() -> None:
    a = parse_args()
    paths = {
        "character_source": Path(a.character_source).expanduser().resolve(),
        "rigged_target": Path(a.rigged_target).expanduser().resolve(),
        "source_animation": Path(a.source_animation).expanduser().resolve(),
        "animated_target": Path(a.animated_target).expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label}: {path}")

    source_mesh_h = mesh_height(paths["character_source"])
    rigged_mesh_h = mesh_height(paths["rigged_target"])
    final_mesh_h = mesh_height(paths["animated_target"])

    rigged_scale_error = abs(rigged_mesh_h / source_mesh_h - 1.0)
    final_scale_error = abs(final_mesh_h / source_mesh_h - 1.0)

    source_motion = sample_motion(paths["source_animation"], a.samples)
    final_motion = sample_motion(paths["animated_target"], a.samples)

    root_rms = rms(source_motion["root_delta"], final_motion["root_delta"])
    root_src_energy = motion_energy(source_motion["root_delta"])
    root_dst_energy = motion_energy(final_motion["root_delta"])
    root_energy_ratio = ratio(root_dst_energy, root_src_energy)

    effectors = {}
    for name in EFFECTORS:
        s = source_motion["effector_delta"][name]
        t = final_motion["effector_delta"][name]
        se = motion_energy(s)
        te = motion_energy(t)
        effectors[name] = {
            "rms_normalized_body_height": rms(s, t),
            "source_motion_energy": se,
            "target_motion_energy": te,
            "motion_energy_ratio": ratio(te, se),
        }

    expected_fps = float(a.expected_fps)
    final_fps_error = abs(final_motion["scene_fps"] - expected_fps)
    warnings = []
    failures = []

    if rigged_scale_error > a.scale_tolerance:
        failures.append(
            f"Rigging changed mesh height by {rigged_scale_error*100:.1f}% "
            f"(allowed {a.scale_tolerance*100:.1f}%)."
        )
    if final_scale_error > a.scale_tolerance:
        failures.append(
            f"Final FBX changed mesh height by {final_scale_error*100:.1f}% "
            f"(allowed {a.scale_tolerance*100:.1f}%)."
        )
    if final_fps_error > 0.25:
        failures.append(
            f"Final FBX reports {final_motion['scene_fps']:.3f} fps, expected {expected_fps:.3f} fps."
        )

    if root_rms > 0.15:
        warnings.append(f"Root trajectory RMS drift is {root_rms:.3f} body heights.")
    if root_rms > 0.40:
        failures.append(f"Root trajectory drift is severe: {root_rms:.3f} body heights.")

    if root_energy_ratio is not None and root_src_energy > 0.02:
        if not 0.10 <= root_energy_ratio <= 10.0:
            failures.append(f"Root motion energy ratio is implausible: {root_energy_ratio:.3f}.")

    for name, metrics in effectors.items():
        er = metrics["motion_energy_ratio"]
        val = metrics["rms_normalized_body_height"]
        if val > 0.25:
            warnings.append(f"{name} trajectory RMS drift is {val:.3f} body heights.")
        if val > 0.55:
            failures.append(f"{name} trajectory drift is severe: {val:.3f} body heights.")
        if er is not None and metrics["source_motion_energy"] > 0.02 and not 0.05 <= er <= 20.0:
            failures.append(f"{name} motion energy ratio is implausible: {er:.3f}.")

    report = {
        "schema_version": 1,
        "scale_contract": {
            "source_mesh_height": source_mesh_h,
            "rigged_mesh_height": rigged_mesh_h,
            "final_mesh_height": final_mesh_h,
            "rigged_relative_error": rigged_scale_error,
            "final_relative_error": final_scale_error,
            "tolerance": a.scale_tolerance,
        },
        "timing_contract": {
            "expected_fps": expected_fps,
            "source_fbx_scene_fps": source_motion["scene_fps"],
            "final_fbx_scene_fps": final_motion["scene_fps"],
            "source_frame_span": source_motion["frame_span"],
            "final_frame_span": final_motion["frame_span"],
        },
        "motion_contract": {
            "root_rms_normalized_body_height": root_rms,
            "root_motion_energy_ratio": root_energy_ratio,
            "effectors": effectors,
            "note": (
                "Trajectory comparison uses motion deltas relative to Hips and normalizes each rig "
                "by its own body height, so different character proportions do not dominate the metric."
            ),
        },
        "warnings": warnings,
        "failures": failures,
        "passed": not failures,
    }

    out = Path(a.report).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Contract report: {out}")

    if failures and a.strict:
        raise RuntimeError("Animation I/O contract validation failed: " + " | ".join(failures))


if __name__ == "__main__":
    main()
