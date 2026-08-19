#!/usr/bin/env python3
"""Convert validated ARDY Core motion into a Mixamo-readable source FBX."""
from __future__ import annotations
import argparse
from pathlib import Path
import bpy, numpy as np
from ardy_fbx_bridge import build_children,choose_tail,cvt_pos,source_bone_name,set_frame,validate_fk,validate_npz

def args():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--scale",type=float,default=1.); p.add_argument("--validation-tolerance",type=float,default=.005)
    return p.parse_args()

def main():
    a=args()
    if a.scale<=0: raise ValueError("--scale must be positive")
    ip=Path(a.input).expanduser().resolve()
    if not ip.is_file(): raise FileNotFoundError(ip)
    with np.load(ip,allow_pickle=True) as z: data={k:z[k] for k in z.files}
    T,J,fps=validate_npz(data)
    names=[str(x) for x in data["joint_names"].tolist()]
    parents=np.asarray(data["joint_parents"],dtype=np.int32)
    neutral=np.asarray(data["neutral_joints"],dtype=np.float64)
    local=np.asarray(data["local_rot_mats"],dtype=np.float64)
    roots=np.asarray(data["root_positions"],dtype=np.float64)
    posed=np.asarray(data["posed_joints"],dtype=np.float64)
    root=int(np.where(parents<0)[0][0]); neutral-=neutral[root]
    pts=np.stack([cvt_pos(v,a.scale) for v in neutral]); src=[source_bone_name(n) for n in names]
    children=build_children(parents)

    bpy.ops.wm.read_factory_settings(use_empty=True); scene=bpy.context.scene
    scene.render.fps=max(1,int(round(fps))); scene.frame_start=1; scene.frame_end=T
    scene.unit_settings.system="METRIC"; scene.unit_settings.scale_length=1.
    ad=bpy.data.armatures.new("ARDY_Core_Armature"); arm=bpy.data.objects.new("ARDY_Core_Source",ad)
    scene.collection.objects.link(arm); bpy.context.view_layer.objects.active=arm; arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT"); eb=[]
    for i,n in enumerate(src):
        b=ad.edit_bones.new(n); b.head=pts[i]; tail=choose_tail(i,names,parents,children,pts,a.scale)
        if np.linalg.norm(tail-pts[i])<1e-5*a.scale: tail=pts[i]+np.array([0.,.10*a.scale,0.])
        b.tail=tail; b.use_connect=False; b.use_deform=False; eb.append(b)
    for i,p in enumerate(parents):
        if int(p)>=0: eb[i].parent=eb[int(p)]
    bpy.ops.object.mode_set(mode="POSE")
    rest=[np.asarray(ad.bones[n].matrix_local.to_3x3(),dtype=np.float64) for n in src]
    validate_fk(scene,arm,sorted({0,T//2,T-1}),local,roots,posed,rest,src,root,a.scale,a.validation_tolerance)
    for pb in arm.pose.bones: pb.matrix_basis.identity()
    prev=[None]*J
    for f in range(T):
        scene.frame_set(f+1); set_frame(arm,f,local,roots,rest,src,root,a.scale,prev,True)
    action=arm.animation_data.action if arm.animation_data else None
    if action is None: raise RuntimeError("No Blender action was created")
    for fc in action.fcurves:
        for kp in fc.keyframe_points: kp.interpolation="LINEAR"

    bpy.ops.object.mode_set(mode="OBJECT"); bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True); bpy.context.view_layer.objects.active=arm
    out=Path(a.output).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.export_scene.fbx(filepath=str(out),check_existing=False,use_selection=True,add_leaf_bones=False,
        bake_anim=True,bake_anim_use_all_bones=True,bake_anim_use_nla_strips=False,bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,bake_anim_simplify_factor=0.,path_mode="AUTO")
    if not out.is_file() or out.stat().st_size==0: raise RuntimeError(f"FBX export failed: {out}")
    print(f"ARDY source FBX: {out} ({T} frames, {J} joints, {fps:g} fps)")

if __name__=="__main__": main()
