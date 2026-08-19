#!/usr/bin/env python3
"""Validate an ARDY Core NPZ and add stable skeleton metadata for export."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ardy.skeleton import CoreSkeleton27
from progress_utils import Progress

REQUIRED_KEYS={"local_rot_mats","global_rot_mats","root_positions","posed_joints","fps"}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); return p.parse_args()

def _require_finite(name,array):
    if not np.isfinite(array).all(): raise ValueError(f"{name} contains NaN/Inf values")

def main():
    p=Progress("ARDY VALIDATE",4); a=parse_args(); ip=Path(a.input).expanduser().resolve()
    p.step("Loading ARDY motion NPZ")
    if not ip.is_file(): raise FileNotFoundError(ip)
    with np.load(ip,allow_pickle=True) as src:
        missing=REQUIRED_KEYS.difference(src.files)
        if missing: raise KeyError(f"ARDY output is missing required keys: {sorted(missing)}")
        data={k:src[k] for k in src.files}
    p.info(f"Input: {ip} | keys={len(data)}")

    p.step("Loading official ARDY Core 27-joint skeleton metadata")
    skeleton=CoreSkeleton27(); names=list(skeleton.bone_order_names)
    parents=skeleton.joint_parents.detach().cpu().numpy().astype(np.int32)
    neutral=skeleton.neutral_joints.detach().cpu().numpy().astype(np.float32)
    p.info(f"Skeleton: cskel27 | joints={len(names)}")

    p.step("Checking tensor shapes, FPS and finite values")
    local=np.asarray(data["local_rot_mats"]); glob=np.asarray(data["global_rot_mats"])
    roots=np.asarray(data["root_positions"]); posed=np.asarray(data["posed_joints"])
    fps=float(np.asarray(data["fps"]).reshape(-1)[0]); frames=local.shape[0]; joints=len(names)
    expected={"local_rot_mats":(frames,joints,3,3),"global_rot_mats":(frames,joints,3,3),"root_positions":(frames,3),"posed_joints":(frames,joints,3)}
    actual={"local_rot_mats":local.shape,"global_rot_mats":glob.shape,"root_positions":roots.shape,"posed_joints":posed.shape}
    for key,shape in expected.items():
        if actual[key]!=shape: raise ValueError(f"{key} has shape {actual[key]}, expected {shape}")
    if not fps>0: raise ValueError(f"Invalid fps: {fps}")
    for key in ("local_rot_mats","global_rot_mats","root_positions","posed_joints"): _require_finite(key,np.asarray(data[key]))
    p.ok(f"Motion contract valid: {frames} frames × {joints} joints @ {fps:g} fps")

    p.step("Writing bridge NPZ with stable skeleton metadata")
    data["joint_names"]=np.asarray(names,dtype=object); data["joint_parents"]=parents; data["neutral_joints"]=neutral; data["skeleton_name"]=np.asarray("cskel27")
    out=Path(a.output).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**data)
    if not out.is_file() or out.stat().st_size==0: raise RuntimeError(f"Failed to write {out}")
    p.ok(f"Output: {out} ({out.stat().st_size/1024**2:.1f} MiB)")
    p.done("ARDY motion validation complete")
if __name__=="__main__": main()
