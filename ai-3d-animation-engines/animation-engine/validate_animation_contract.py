#!/usr/bin/env python3
"""Validate Animation Engine scale, timing, motion and actual skinned deformation."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np

from progress_utils import Progress

C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)

REQUIRED_BONES = [
    "mixamorig:Hips",
    "mixamorig:Head",
    "mixamorig:LeftHand",
    "mixamorig:RightHand",
    "mixamorig:LeftFoot",
    "mixamorig:RightFoot",
]
SOURCE_FOR_TARGET = {
    "mixamorig:Hips": "Hips",
    "mixamorig:Head": "Head",
    "mixamorig:LeftHand": "LeftHand",
    "mixamorig:RightHand": "RightHand",
    "mixamorig:LeftFoot": "LeftFoot",
    "mixamorig:RightFoot": "RightFoot",
}
EFFECTORS = REQUIRED_BONES[1:]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--character-source", required=True)
    p.add_argument("--rigged-target", required=True)
    p.add_argument("--motion-bridge", required=True)
    p.add_argument("--animated-target", required=True)
    p.add_argument("--expected-fps", type=float, required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--scale-tolerance", type=float, default=0.08)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_path(path: Path):
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


def world_mesh_bounds(path: Path):
    reset()
    load_path(path)
    pts = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mw = obj.matrix_world
        for v in obj.data.vertices:
            q = mw @ v.co
            pts.append((q.x, q.y, q.z))
    if not pts:
        raise RuntimeError(f"No mesh vertices found in {path}")
    arr = np.asarray(pts, dtype=np.float64)
    return arr.min(axis=0), arr.max(axis=0)


def mesh_height(path: Path):
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


def skeleton_height(arm):
    zs = []
    mw = arm.matrix_world
    for name in REQUIRED_BONES:
        b = arm.data.bones[name]
        zs.extend([float((mw @ b.head_local).z), float((mw @ b.tail_local).z)])
    h = max(zs) - min(zs)
    if not math.isfinite(h) or h <= 1e-8:
        raise RuntimeError("Invalid armature height")
    return h


def sample_target_motion(path: Path, samples: int, progress: Progress):
    arm, action = armature_and_action(path)
    start = float(action.frame_range[0])
    end = float(action.frame_range[1])
    if end <= start:
        raise RuntimeError(f"Animation in {path} has invalid frame range {start}-{end}")
    frames = np.linspace(start, end, max(3, samples))
    height = skeleton_height(arm)
    traj = {n: [] for n in REQUIRED_BONES}
    for idx, frame in enumerate(frames, 1):
        bpy.context.scene.frame_set(int(round(frame)))
        bpy.context.view_layer.update()
        for name in REQUIRED_BONES:
            pb = arm.pose.bones[name]
            q = arm.matrix_world @ pb.head
            traj[name].append([q.x, q.y, q.z])
        progress.info(f"target skeleton: sampled {idx}/{len(frames)} frames")
    traj = {k: np.asarray(v, dtype=np.float64) for k, v in traj.items()}
    hips = traj["mixamorig:Hips"]
    root_delta = (hips - hips[0]) / height
    eff = {}
    for name in EFFECTORS:
        rel = traj[name] - hips
        eff[name] = (rel - rel[0]) / height
    fps = float(bpy.context.scene.render.fps) / float(bpy.context.scene.render.fps_base)
    return {
        "frame_start": start,
        "frame_end": end,
        "frame_span": end - start,
        "scene_fps": fps,
        "height": height,
        "root_delta": root_delta,
        "effector_delta": eff,
        "sample_frames": [int(round(x)) for x in frames],
    }


def cvt_pos(v):
    return C @ np.asarray(v, dtype=np.float64)


def sample_source_bridge(path: Path, samples: int, progress: Progress):
    with np.load(path, allow_pickle=True) as z:
        required = {"joint_names", "neutral_joints", "posed_joints", "fps"}
        missing = required.difference(z.files)
        if missing:
            raise KeyError(f"motion bridge missing keys: {sorted(missing)}")
        names = [str(x) for x in z["joint_names"].tolist()]
        neutral = np.asarray(z["neutral_joints"], dtype=np.float64)
        posed = np.asarray(z["posed_joints"], dtype=np.float64)
        fps = float(np.asarray(z["fps"]).reshape(-1)[0])
    if posed.ndim != 3 or posed.shape[1:] != (len(names), 3):
        raise ValueError(f"Invalid posed_joints shape: {posed.shape}")
    idx = {n: i for i, n in enumerate(names)}
    missing_names = sorted(set(SOURCE_FOR_TARGET.values()) - set(idx))
    if missing_names:
        raise RuntimeError("ARDY bridge missing validation joints: " + ", ".join(missing_names))

    shared_neutral = np.stack([cvt_pos(neutral[idx[src]]) for src in SOURCE_FOR_TARGET.values()])
    height = float(shared_neutral[:, 2].max() - shared_neutral[:, 2].min())
    if not math.isfinite(height) or height <= 1e-8:
        raise RuntimeError(f"Invalid ARDY validation height: {height}")

    frame_ids = np.rint(np.linspace(0, posed.shape[0] - 1, max(3, samples))).astype(int)
    traj = {n: [] for n in REQUIRED_BONES}
    for sample_i, frame in enumerate(frame_ids, 1):
        for target_name, source_name in SOURCE_FOR_TARGET.items():
            traj[target_name].append(cvt_pos(posed[frame, idx[source_name]]))
        progress.info(f"source bridge: sampled {sample_i}/{len(frame_ids)} frames")
    traj = {k: np.asarray(v, dtype=np.float64) for k, v in traj.items()}
    hips = traj["mixamorig:Hips"]
    root_delta = (hips - hips[0]) / height
    eff = {}
    for name in EFFECTORS:
        rel = traj[name] - hips
        eff[name] = (rel - rel[0]) / height
    return {
        "frame_start": 0.0,
        "frame_end": float(posed.shape[0] - 1),
        "frame_span": float(posed.shape[0] - 1),
        "scene_fps": fps,
        "height": height,
        "root_delta": root_delta,
        "effector_delta": eff,
    }


def evaluated_world_vertices(obj) -> np.ndarray:
    deps = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(deps)
    mesh = eval_obj.to_mesh()
    try:
        mw = eval_obj.matrix_world
        return np.asarray(
            [[(mw @ v.co).x, (mw @ v.co).y, (mw @ v.co).z] for v in mesh.vertices],
            dtype=np.float64,
        )
    finally:
        eval_obj.to_mesh_clear()


def deformation_contract(path: Path, samples: int, progress: Progress):
    reset()
    load_path(path)
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"Expected one armature in {path}")
    arm = arms[0]
    action = arm.animation_data.action if arm.animation_data else None
    if action is None:
        raise RuntimeError("Animated target has no action for deformation validation")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("Animated target has no mesh for deformation validation")

    start = int(round(float(action.frame_range[0])))
    end = int(round(float(action.frame_range[1])))
    frames = np.rint(np.linspace(start, end, max(3, samples))).astype(int)

    arm.animation_data.action = None
    bpy.context.scene.frame_set(start)
    bpy.context.view_layer.update()
    references = {}
    for mesh_obj in meshes:
        verts = evaluated_world_vertices(mesh_obj)
        edges = np.asarray([[e.vertices[0], e.vertices[1]] for e in mesh_obj.data.edges], dtype=np.int64)
        if len(edges) == 0:
            continue
        lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        keep = lengths > 1e-7
        references[mesh_obj.name] = (edges[keep], lengths[keep])
    arm.animation_data.action = action
    if not references:
        raise RuntimeError("Could not build reference mesh-edge set")

    worst_p99 = 1.0
    worst_p01 = 1.0
    worst_extent = 0.0
    body_height = skeleton_height(arm)

    for sample_i, frame in enumerate(frames, 1):
        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        frame_ratios = []
        frame_pts = []
        for mesh_obj in meshes:
            if mesh_obj.name not in references:
                continue
            verts = evaluated_world_vertices(mesh_obj)
            edges, ref_lengths = references[mesh_obj.name]
            cur_lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
            ratios = cur_lengths / ref_lengths
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size:
                frame_ratios.append(ratios)
            frame_pts.append(verts)
        if not frame_ratios or not frame_pts:
            raise RuntimeError(f"Could not evaluate skinned mesh at frame {frame}")
        ratios = np.concatenate(frame_ratios)
        pts = np.concatenate(frame_pts, axis=0)
        p01 = float(np.quantile(ratios, 0.01))
        p99 = float(np.quantile(ratios, 0.99))
        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) / body_height)
        worst_p01 = min(worst_p01, p01)
        worst_p99 = max(worst_p99, p99)
        worst_extent = max(worst_extent, extent)
        progress.info(
            f"deformation frame {frame}: edge p01={p01:.3f} | p99={p99:.3f} | bbox/body={extent:.3f}"
        )

    return {
        "worst_edge_ratio_p01": worst_p01,
        "worst_edge_ratio_p99": worst_p99,
        "worst_bbox_diagonal_body_heights": worst_extent,
        "sampled_frames": [int(x) for x in frames],
    }


def rms(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def energy(a):
    return float(np.sqrt(np.mean(np.sum(a**2, axis=-1))))


def ratio(t, s):
    return None if s < 1e-5 else float(t / s)


def hard_exit_after_success(progress):
    progress.info("Validation report is fully written; exiting without Blender/Python teardown")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    p = Progress("CONTRACT CHECK", 7)
    a = parse_args()
    paths = {
        "character_source": Path(a.character_source).expanduser().resolve(),
        "rigged_target": Path(a.rigged_target).expanduser().resolve(),
        "motion_bridge": Path(a.motion_bridge).expanduser().resolve(),
        "animated_target": Path(a.animated_target).expanduser().resolve(),
    }

    p.step("Checking all required files")
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label}: {path}")
        p.info(f"{label}: {path.name} ({path.stat().st_size / 1024**2:.1f} MiB)")

    p.step("Measuring source/rig/final static scale")
    src_h = mesh_height(paths["character_source"])
    rig_h = mesh_height(paths["rigged_target"])
    final_h = mesh_height(paths["animated_target"])
    rig_err = abs(rig_h / src_h - 1.0)
    final_err = abs(final_h / src_h - 1.0)
    p.info(f"Source height={src_h:.4f} | rigged={rig_h:.4f} | final={final_h:.4f}")
    p.info(f"Relative static scale drift: rigged={rig_err * 100:.2f}% | final={final_err * 100:.2f}%")

    p.step("Sampling authoritative ARDY bridge trajectories")
    src = sample_source_bridge(paths["motion_bridge"], a.samples, p)

    p.step("Sampling final target skeleton trajectories")
    dst = sample_target_motion(paths["animated_target"], a.samples, p)

    p.step("Evaluating the actual skinned mesh across animation frames")
    deform = deformation_contract(paths["animated_target"], a.samples, p)

    p.step("Comparing FPS, root motion and effectors")
    root_rms = rms(src["root_delta"], dst["root_delta"])
    src_e = energy(src["root_delta"])
    dst_e = energy(dst["root_delta"])
    root_ratio = ratio(dst_e, src_e)
    effectors = {}
    for name in EFFECTORS:
        s = src["effector_delta"][name]
        t = dst["effector_delta"][name]
        se = energy(s)
        te = energy(t)
        effectors[name] = {
            "rms_normalized_body_height": rms(s, t),
            "source_motion_energy": se,
            "target_motion_energy": te,
            "motion_energy_ratio": ratio(te, se),
        }

    fps_err = abs(dst["scene_fps"] - float(a.expected_fps))
    warnings = []
    failures = []
    if rig_err > a.scale_tolerance:
        failures.append(
            f"Rigging changed mesh height by {rig_err * 100:.1f}% (allowed {a.scale_tolerance * 100:.1f}%)."
        )
    if final_err > a.scale_tolerance:
        failures.append(
            f"Final FBX changed static mesh height by {final_err * 100:.1f}% (allowed {a.scale_tolerance * 100:.1f}%)."
        )
    if fps_err > 0.25:
        failures.append(
            f"Final FBX reports {dst['scene_fps']:.3f} fps, expected {a.expected_fps:.3f} fps."
        )

    if deform["worst_edge_ratio_p99"] > 2.0:
        warnings.append(
            f"Skinned mesh edge stretch p99 reached {deform['worst_edge_ratio_p99']:.3f}x."
        )
    if deform["worst_edge_ratio_p99"] > 3.0:
        failures.append(
            f"Skinned mesh deformation is severe: p99 edge stretch reached {deform['worst_edge_ratio_p99']:.3f}x."
        )
    if deform["worst_edge_ratio_p01"] < 0.40:
        warnings.append(
            f"Skinned mesh edge compression p01 reached {deform['worst_edge_ratio_p01']:.3f}x."
        )
    if deform["worst_edge_ratio_p01"] < 0.15:
        failures.append(
            f"Skinned mesh deformation is severe: p01 edge compression reached {deform['worst_edge_ratio_p01']:.3f}x."
        )
    if deform["worst_bbox_diagonal_body_heights"] > 2.5:
        warnings.append(
            f"Animated mesh bounding-box diagonal reached {deform['worst_bbox_diagonal_body_heights']:.3f} body heights."
        )
    if deform["worst_bbox_diagonal_body_heights"] > 3.5:
        failures.append(
            f"Animated mesh extent is implausible: {deform['worst_bbox_diagonal_body_heights']:.3f} body heights."
        )

    if root_rms > 0.15:
        warnings.append(f"Root trajectory RMS drift is {root_rms:.3f} body heights.")
    if root_rms > 0.40:
        failures.append(f"Root trajectory drift is severe: {root_rms:.3f} body heights.")
    if root_ratio is not None and src_e > 0.02 and not 0.10 <= root_ratio <= 10.0:
        failures.append(f"Root motion energy ratio is implausible: {root_ratio:.3f}.")

    for name, m in effectors.items():
        er = m["motion_energy_ratio"]
        val = m["rms_normalized_body_height"]
        if val > 0.25:
            warnings.append(f"{name} trajectory RMS drift is {val:.3f} body heights.")
        if val > 0.55:
            failures.append(f"{name} trajectory drift is severe: {val:.3f} body heights.")
        if er is not None and m["source_motion_energy"] > 0.02 and not 0.05 <= er <= 20.0:
            failures.append(f"{name} motion energy ratio is implausible: {er:.3f}.")

    p.info(f"FPS source={src['scene_fps']:.3f} | final={dst['scene_fps']:.3f} | expected={a.expected_fps:.3f}")
    p.info(f"Root trajectory RMS={root_rms:.4f} body heights")

    p.step("Writing validation report and enforcing strict gate")
    report = {
        "schema_version": 2,
        "scale_contract": {
            "source_mesh_height": src_h,
            "rigged_mesh_height": rig_h,
            "final_static_mesh_height": final_h,
            "rigged_relative_error": rig_err,
            "final_relative_error": final_err,
            "tolerance": a.scale_tolerance,
        },
        "deformation_contract": deform,
        "timing_contract": {
            "expected_fps": float(a.expected_fps),
            "source_bridge_fps": src["scene_fps"],
            "final_fbx_scene_fps": dst["scene_fps"],
            "source_frame_span": src["frame_span"],
            "final_frame_span": dst["frame_span"],
        },
        "motion_contract": {
            "root_rms_normalized_body_height": root_rms,
            "root_motion_energy_ratio": root_ratio,
            "effectors": effectors,
            "note": "Source trajectories come directly from motion_bridge.npz; target trajectories come from the final FBX pose bones.",
        },
        "warnings": warnings,
        "failures": failures,
        "passed": not failures,
    }
    out = Path(a.report).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for w in warnings:
        p.warn(w)
    if failures:
        for failure in failures:
            p.warn("FAIL: " + failure)
    else:
        p.ok("All strict scale, deformation, timing and motion checks passed")
    p.ok(f"Report: {out}")
    if failures and a.strict:
        raise RuntimeError("Animation contract validation failed: " + " | ".join(failures))
    p.done("Animation contract validation complete")
    hard_exit_after_success(p)


if __name__ == "__main__":
    main()
