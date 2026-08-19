#!/usr/bin/env python3
"""Retarget an ARDY source FBX onto a Make-It-Animatable humanoid rig with live progress."""
from __future__ import annotations
import argparse, math, os, sys, importlib
from pathlib import Path
from progress_utils import Progress

BASE_BONES={
"mixamorig:Hips","mixamorig:Spine","mixamorig:Spine1","mixamorig:Spine2","mixamorig:Neck","mixamorig:Head",
"mixamorig:LeftShoulder","mixamorig:LeftArm","mixamorig:LeftForeArm","mixamorig:LeftHand",
"mixamorig:RightShoulder","mixamorig:RightArm","mixamorig:RightForeArm","mixamorig:RightHand",
"mixamorig:LeftUpLeg","mixamorig:LeftLeg","mixamorig:LeftFoot","mixamorig:LeftToeBase",
"mixamorig:RightUpLeg","mixamorig:RightLeg","mixamorig:RightFoot","mixamorig:RightToeBase"}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--target",required=True); p.add_argument("--animation",required=True); p.add_argument("--output",required=True)
    p.add_argument("--fps",type=float,required=True); p.add_argument("--preview-glb",default=None); p.add_argument("--inplace",action="store_true"); return p.parse_args()

def require_file(path,label):
    p=Path(path).expanduser().resolve()
    if not p.is_file() or p.stat().st_size==0: raise FileNotFoundError(f"{label} not found/empty: {p}")
    return p

def set_scene_fps(scene,fps):
    if not math.isfinite(fps) or fps<=0: raise ValueError(f"Invalid FPS: {fps}")
    rounded=max(1,int(round(fps))); scene.render.fps=rounded; scene.render.fps_base=rounded/fps
    actual=float(scene.render.fps)/float(scene.render.fps_base)
    if abs(actual-fps)>1e-6: raise RuntimeError(f"Could not set Blender scene FPS to {fps}; got {actual}")

def main():
    p=Progress("RETARGET",7); a=parse_args()
    p.step("Validating rigged character, ARDY source FBX and FPS")
    target=require_file(a.target,"Target FBX"); anim=require_file(a.animation,"Animation FBX")
    p.info(f"Target: {target.name} | source motion: {anim.name} | fps={a.fps:g}")

    p.step("Loading Make-It-Animatable / Auto-Rig-Pro Blender tools")
    mia_root=Path(os.environ.get("MIA_ROOT","/content/Make-It-Animatable")).resolve()
    if not (mia_root/"util"/"blender_utils.py").is_file(): raise FileNotFoundError(f"Make-It-Animatable not found at {mia_root}")
    mia_root_str=str(mia_root)
    if mia_root_str not in sys.path:
        sys.path.insert(0,mia_root_str)
    existing_pythonpath=os.environ.get("PYTHONPATH","")
    os.environ["PYTHONPATH"]=mia_root_str+(os.pathsep+existing_pythonpath if existing_pythonpath else "")
    os.chdir(mia_root); importlib.invalidate_caches()
    from util import blender_utils; bpy=blender_utils.bpy
    p.info(f"MIA import root: {mia_root}")
    blender_utils.reset(); set_scene_fps(bpy.context.scene,a.fps)

    p.step("Importing target character and ARDY source skeleton")
    character=blender_utils.load_file(str(target)); target_armature=blender_utils.get_armature_obj(character)
    if target_armature is None: raise RuntimeError("Target file does not contain an armature")
    source_objects=blender_utils.load_file(str(anim)); source_armature=blender_utils.get_armature_obj(source_objects)
    if source_armature is None: raise RuntimeError("ARDY source FBX does not contain an armature")
    source_action=source_armature.animation_data.action if source_armature.animation_data is not None else None
    if source_action is None: raise RuntimeError("ARDY source armature has no animation action")
    target_bones={b.name for b in target_armature.data.bones}; source_bones={b.name for b in source_armature.data.bones}
    mt=sorted(BASE_BONES-target_bones); ms=sorted(BASE_BONES-source_bones)
    if mt: raise RuntimeError("Target rig missing Mixamo bones: "+", ".join(mt))
    if ms: raise RuntimeError("ARDY source bridge missing Mixamo bones: "+", ".join(ms))
    p.ok(f"Shared body contract: {len(BASE_BONES)} exact Mixamo bones")

    p.step("Building exact source→target bone mapping")
    blender_utils.set_action(target_armature,source_action); blender_utils.enable_arp(target_armature)
    scn=bpy.context.scene; set_scene_fps(scn,a.fps); scn.source_rig=source_armature.name; scn.target_rig=target_armature.name
    if a.inplace: scn.arp_retarget_in_place=True
    bpy.context.view_layer.objects.active=target_armature; target_armature.select_set(True)
    bpy.ops.arp.auto_scale(); bpy.ops.arp.build_bones_list()
    mapping={item.name:item for item in scn.bones_map_v2}; missing=sorted(BASE_BONES-mapping.keys())
    if missing: raise RuntimeError("Auto-Rig-Pro did not expose required target mapping entries: "+", ".join(missing))
    for bone in BASE_BONES: mapping[bone].source_bone=bone
    hips=mapping["mixamorig:Hips"]; scn.bones_map_index=list(scn.bones_map_v2).index(hips); hips.set_as_root=True
    bad=[item.name for item in scn.bones_map_v2 if item.name in BASE_BONES and item.source_bone not in source_bones]
    if bad: raise RuntimeError("Invalid forced retarget mappings: "+", ".join(sorted(bad)))
    p.ok("Exact shared-bone map built; Hips marked as root")

    p.step("Retargeting and baking ARDY motion onto the character")
    with p.heartbeat("Auto-Rig-Pro retarget",every=20): bpy.ops.arp.retarget()
    blender_utils.update(); set_scene_fps(scn,a.fps)
    final_action=target_armature.animation_data.action if target_armature.animation_data is not None else None
    if final_action is None: raise RuntimeError("Retarget completed without creating a target animation action")
    keyframes=blender_utils.get_keyframes([target_armature])
    if len(keyframes)<2: raise RuntimeError("Retargeted target contains fewer than two animation keyframes")
    p.ok(f"Retarget produced {len(keyframes)} keyed frames/positions; range {min(keyframes)}-{max(keyframes)}")

    p.step("Exporting final animated FBX")
    for obj in source_objects:
        if obj.name in bpy.data.objects: bpy.data.objects.remove(obj,do_unlink=True)
    scene=bpy.context.scene; scene.frame_start=min(keyframes); scene.frame_end=max(keyframes); set_scene_fps(scene,a.fps)
    out=Path(a.output).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in character:
        if obj.name in bpy.data.objects: obj.select_set(True)
    bpy.context.view_layer.objects.active=target_armature
    with p.heartbeat("FBX export",every=20):
        bpy.ops.export_scene.fbx(filepath=str(out),check_existing=False,use_selection=True,add_leaf_bones=False,bake_anim=True,
            bake_anim_use_all_bones=True,bake_anim_use_nla_strips=False,bake_anim_use_all_actions=False,bake_anim_force_startend_keying=True,
            bake_anim_step=1.0,bake_anim_simplify_factor=0.0,path_mode="COPY",embed_textures=True)
    if not out.is_file() or out.stat().st_size==0: raise RuntimeError(f"Final FBX export failed: {out}")
    p.ok(f"Final FBX: {out} ({out.stat().st_size/1024**2:.1f} MiB)")

    p.step("Creating optional animated GLB preview")
    if a.preview_glb:
        preview=Path(a.preview_glb).expanduser().resolve(); preview.parent.mkdir(parents=True,exist_ok=True)
        try:
            with p.heartbeat("animated GLB preview export",every=20):
                bpy.ops.export_scene.gltf(filepath=str(preview),check_existing=False,use_selection=True,export_format="GLB",export_animations=True)
            if preview.is_file() and preview.stat().st_size>0: p.ok(f"Preview GLB: {preview}")
            else: p.warn("GLB preview export returned without a valid file; FBX remains valid")
        except Exception as exc: p.warn(f"GLB preview export failed; FBX remains valid: {exc}")
    else: p.info("No preview GLB requested")
    p.done("Retarget and export complete")

if __name__=="__main__": main()
