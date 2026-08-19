#!/usr/bin/env python3
"""Validate scale, timing and motion transfer across the Animation Engine with live progress."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import bpy, numpy as np
from progress_utils import Progress

REQUIRED_BONES=["mixamorig:Hips","mixamorig:Head","mixamorig:LeftHand","mixamorig:RightHand","mixamorig:LeftFoot","mixamorig:RightFoot"]
EFFECTORS=REQUIRED_BONES[1:]

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--character-source",required=True); p.add_argument("--rigged-target",required=True); p.add_argument("--source-animation",required=True); p.add_argument("--animated-target",required=True)
    p.add_argument("--expected-fps",type=float,required=True); p.add_argument("--report",required=True); p.add_argument("--samples",type=int,default=15); p.add_argument("--scale-tolerance",type=float,default=.08); p.add_argument("--strict",action="store_true"); return p.parse_args()

def reset(): bpy.ops.wm.read_factory_settings(use_empty=True)
def load_path(path):
    ext=path.suffix.lower()
    if ext in {".glb",".gltf"}: bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext==".fbx": bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext==".obj": bpy.ops.wm.obj_import(filepath=str(path))
    elif ext==".ply": bpy.ops.wm.ply_import(filepath=str(path))
    else: raise ValueError(f"Unsupported 3D format: {path}")
def world_mesh_bounds(path):
    reset(); load_path(path); pts=[]
    for obj in bpy.context.scene.objects:
        if obj.type!="MESH": continue
        mw=obj.matrix_world
        for v in obj.data.vertices:
            q=mw@v.co; pts.append((q.x,q.y,q.z))
    if not pts: raise RuntimeError(f"No mesh vertices found in {path}")
    arr=np.asarray(pts,dtype=np.float64); return arr.min(axis=0),arr.max(axis=0)
def mesh_height(path):
    lo,hi=world_mesh_bounds(path); h=float(hi[2]-lo[2])
    if not math.isfinite(h) or h<=1e-8: raise RuntimeError(f"Invalid character height for {path}: {h}")
    return h
def armature_and_action(path):
    reset(); load_path(path); arms=[o for o in bpy.context.scene.objects if o.type=="ARMATURE"]
    if len(arms)!=1: raise RuntimeError(f"Expected exactly one armature in {path}, found {len(arms)}")
    arm=arms[0]; action=arm.animation_data.action if arm.animation_data else None
    if action is None: raise RuntimeError(f"No animation action in {path}")
    names={b.name for b in arm.data.bones}; missing=[n for n in REQUIRED_BONES if n not in names]
    if missing: raise RuntimeError(f"{path.name} missing required motion bones: {missing}")
    return arm,action
def skeleton_height(arm):
    zs=[]; mw=arm.matrix_world
    for b in arm.data.bones:
        for q in (b.head_local,b.tail_local): zs.append(float((mw@q).z))
    h=max(zs)-min(zs)
    if not math.isfinite(h) or h<=1e-8: raise RuntimeError("Invalid armature height")
    return h
def sample_motion(path,samples,progress,label):
    arm,action=armature_and_action(path); start=float(action.frame_range[0]); end=float(action.frame_range[1])
    if end<=start: raise RuntimeError(f"Animation in {path} has invalid frame range {start}-{end}")
    frames=np.linspace(start,end,max(3,samples)); height=skeleton_height(arm); traj={n:[] for n in REQUIRED_BONES}
    for idx,frame in enumerate(frames,1):
        bpy.context.scene.frame_set(int(round(frame))); bpy.context.view_layer.update()
        for name in REQUIRED_BONES:
            pb=arm.pose.bones[name]; q=arm.matrix_world@pb.head; traj[name].append([q.x,q.y,q.z])
        progress.info(f"{label}: sampled {idx}/{len(frames)} frames")
    traj={k:np.asarray(v,dtype=np.float64) for k,v in traj.items()}; hips=traj["mixamorig:Hips"]; root_delta=(hips-hips[0])/height
    eff={}
    for name in EFFECTORS:
        rel=traj[name]-hips; eff[name]=(rel-rel[0])/height
    fps=float(bpy.context.scene.render.fps)/float(bpy.context.scene.render.fps_base)
    return {"frame_start":start,"frame_end":end,"frame_span":end-start,"scene_fps":fps,"height":height,"root_delta":root_delta,"effector_delta":eff}
def rms(a,b): return float(np.sqrt(np.mean(np.sum((a-b)**2,axis=-1))))
def energy(a): return float(np.sqrt(np.mean(np.sum(a**2,axis=-1))))
def ratio(t,s): return None if s<1e-5 else float(t/s)

def main():
    p=Progress("CONTRACT CHECK",6); a=parse_args()
    paths={"character_source":Path(a.character_source).expanduser().resolve(),"rigged_target":Path(a.rigged_target).expanduser().resolve(),"source_animation":Path(a.source_animation).expanduser().resolve(),"animated_target":Path(a.animated_target).expanduser().resolve()}
    p.step("Checking all required files")
    for label,path in paths.items():
        if not path.is_file() or path.stat().st_size==0: raise FileNotFoundError(f"{label}: {path}")
        p.info(f"{label}: {path.name} ({path.stat().st_size/1024**2:.1f} MiB)")

    p.step("Measuring character scale before/after rigging and animation")
    src_h=mesh_height(paths["character_source"]); rig_h=mesh_height(paths["rigged_target"]); final_h=mesh_height(paths["animated_target"])
    rig_err=abs(rig_h/src_h-1.); final_err=abs(final_h/src_h-1.)
    p.info(f"Source height={src_h:.4f} | rigged={rig_h:.4f} | final={final_h:.4f}")
    p.info(f"Relative scale drift: rigged={rig_err*100:.2f}% | final={final_err*100:.2f}%")

    p.step("Sampling ARDY source animation trajectories")
    src=sample_motion(paths["source_animation"],a.samples,p,"source")

    p.step("Sampling final retargeted character trajectories")
    dst=sample_motion(paths["animated_target"],a.samples,p,"target")

    p.step("Comparing FPS, root motion and effectors")
    root_rms=rms(src["root_delta"],dst["root_delta"]); src_e=energy(src["root_delta"]); dst_e=energy(dst["root_delta"]); root_ratio=ratio(dst_e,src_e)
    effectors={}
    for name in EFFECTORS:
        s=src["effector_delta"][name]; t=dst["effector_delta"][name]; se=energy(s); te=energy(t)
        effectors[name]={"rms_normalized_body_height":rms(s,t),"source_motion_energy":se,"target_motion_energy":te,"motion_energy_ratio":ratio(te,se)}
    fps_err=abs(dst["scene_fps"]-float(a.expected_fps)); warnings=[]; failures=[]
    if rig_err>a.scale_tolerance: failures.append(f"Rigging changed mesh height by {rig_err*100:.1f}% (allowed {a.scale_tolerance*100:.1f}%).")
    if final_err>a.scale_tolerance: failures.append(f"Final FBX changed mesh height by {final_err*100:.1f}% (allowed {a.scale_tolerance*100:.1f}%).")
    if fps_err>.25: failures.append(f"Final FBX reports {dst['scene_fps']:.3f} fps, expected {a.expected_fps:.3f} fps.")
    if root_rms>.15: warnings.append(f"Root trajectory RMS drift is {root_rms:.3f} body heights.")
    if root_rms>.40: failures.append(f"Root trajectory drift is severe: {root_rms:.3f} body heights.")
    if root_ratio is not None and src_e>.02 and not .10<=root_ratio<=10.: failures.append(f"Root motion energy ratio is implausible: {root_ratio:.3f}.")
    for name,m in effectors.items():
        er=m["motion_energy_ratio"]; val=m["rms_normalized_body_height"]
        if val>.25: warnings.append(f"{name} trajectory RMS drift is {val:.3f} body heights.")
        if val>.55: failures.append(f"{name} trajectory drift is severe: {val:.3f} body heights.")
        if er is not None and m["source_motion_energy"]>.02 and not .05<=er<=20.: failures.append(f"{name} motion energy ratio is implausible: {er:.3f}.")
    p.info(f"FPS source={src['scene_fps']:.3f} | final={dst['scene_fps']:.3f} | expected={a.expected_fps:.3f}")
    p.info(f"Root trajectory RMS={root_rms:.4f} body heights")

    p.step("Writing validation report and enforcing strict gate")
    report={"schema_version":1,"scale_contract":{"source_mesh_height":src_h,"rigged_mesh_height":rig_h,"final_mesh_height":final_h,"rigged_relative_error":rig_err,"final_relative_error":final_err,"tolerance":a.scale_tolerance},"timing_contract":{"expected_fps":float(a.expected_fps),"source_fbx_scene_fps":src["scene_fps"],"final_fbx_scene_fps":dst["scene_fps"],"source_frame_span":src["frame_span"],"final_frame_span":dst["frame_span"]},"motion_contract":{"root_rms_normalized_body_height":root_rms,"root_motion_energy_ratio":root_ratio,"effectors":effectors,"note":"Trajectory comparison uses motion deltas relative to Hips and normalizes each rig by its own body height."},"warnings":warnings,"failures":failures,"passed":not failures}
    out=Path(a.report).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    for w in warnings: p.warn(w)
    if failures:
        for f in failures: p.warn("FAIL: "+f)
    else: p.ok("All strict I/O contract checks passed")
    p.ok(f"Report: {out}")
    if failures and a.strict: raise RuntimeError("Animation I/O contract validation failed: "+" | ".join(failures))
    p.done("Animation contract validation complete")
if __name__=="__main__": main()
