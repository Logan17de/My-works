#!/usr/bin/env python3
from __future__ import annotations

import argparse, math, os, sys
from pathlib import Path
import bpy
from progress_utils import Progress
from universal_motion import CanonicalMotion
from universal_motion.rigs import load_rig_adapter
from universal_motion.retarget import bake_motion
from universal_motion.ik import ContactCorrection

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--motion",required=True); p.add_argument("--target",required=True); p.add_argument("--rig-preset",default="mia_mixamo"); p.add_argument("--output",required=True); p.add_argument("--preview-glb",default=None); return p.parse_args()

def set_scene_fps(scene,fps):
    if not math.isfinite(fps) or fps<=0: raise ValueError(f"Invalid FPS: {fps}")
    rounded=max(1,int(round(fps))); scene.render.fps=rounded; scene.render.fps_base=rounded/fps

def require(path,label):
    q=Path(path).expanduser().resolve()
    if not q.is_file() or q.stat().st_size==0: raise FileNotFoundError(f"{label} missing/empty: {q}")
    return q

def hard_exit(progress):
    progress.info("Universal retarget artifacts are fully written; bypassing bpy teardown"); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)

def main():
    a=parse_args(); p=Progress("UNIVERSAL RETARGET",7)
    p.step("Loading canonical motion and target rig")
    motion=CanonicalMotion.load(a.motion); target=require(a.target,"target rig")
    if target.suffix.lower()!=".fbx": raise ValueError("V1 universal retarget expects an already-rigged FBX target")
    p.info(f"Motion source={motion.source_name} | frames={motion.frames} | fps={motion.fps:g}")

    p.step("Importing target character into headless Blender")
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=str(target)); scene=bpy.context.scene; set_scene_fps(scene,motion.fps)
    armatures=[o for o in scene.objects if o.type=="ARMATURE"]; meshes=[o for o in scene.objects if o.type=="MESH"]
    if len(armatures)!=1: raise RuntimeError(f"Expected one target armature, found {len(armatures)}")
    if not meshes: raise RuntimeError("Target FBX has no mesh")
    armature=armatures[0]; p.info(f"Target armature={armature.name} | meshes={len(meshes)}")

    p.step(f"Loading target Rig Adapter: {a.rig_preset}")
    adapter=load_rig_adapter(a.rig_preset,armature); p.info(f"Rig provider={adapter.preset.provider} | mapped semantics={len(adapter.semantic_bones)}")
    for semantic,chain in adapter.preset.motion_chains.items():
        if tuple(chain)!=(semantic,): p.info(f"Collapse rule: {semantic} <= {' + '.join(chain)}")
    p.ok("Rig adapter contract satisfied")

    p.step("Running universal rest-space retarget core")
    result=bake_motion(scene,armature,adapter,motion,progress=p); p.info(f"Motion scale={result['motion_scale']:.6f} | target height={result['target_height']:.4f}"); p.ok(f"Universal action={result['action'].name} | frames={result['frame_start']}-{result['frame_end']}")

    p.step("Running separate IK/contact correction layer")
    correction=ContactCorrection().apply(scene=scene,armature=armature,adapter=adapter,motion=motion,progress=p); p.info(f"Correction solver={correction['solver']} | applied={correction['applied']}")

    p.step("Exporting universal animated FBX")
    out=Path(a.output).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True); bpy.ops.object.select_all(action="DESELECT"); armature.select_set(True)
    for mesh in meshes: mesh.select_set(True)
    bpy.context.view_layer.objects.active=armature
    with p.heartbeat("FBX export",every=20):
        bpy.ops.export_scene.fbx(filepath=str(out),check_existing=False,use_selection=True,add_leaf_bones=False,bake_anim=True,bake_anim_use_all_bones=True,bake_anim_use_nla_strips=False,bake_anim_use_all_actions=False,bake_anim_force_startend_keying=True,bake_anim_step=1.0,bake_anim_simplify_factor=0.0,path_mode="COPY",embed_textures=True)
    require(out,"animated FBX"); p.ok(f"Animated FBX: {out} ({out.stat().st_size/1024**2:.1f} MiB)")

    p.step("Creating optional GLB preview")
    if a.preview_glb:
        preview=Path(a.preview_glb).expanduser().resolve(); preview.parent.mkdir(parents=True,exist_ok=True)
        try:
            with p.heartbeat("GLB preview export",every=20): bpy.ops.export_scene.gltf(filepath=str(preview),check_existing=False,use_selection=True,export_format="GLB",export_animations=True)
            if preview.is_file() and preview.stat().st_size>0: p.ok(f"Preview GLB: {preview}")
            else: p.warn("Preview GLB was not produced; FBX remains authoritative")
        except Exception as exc: p.warn(f"Optional GLB preview failed: {exc}")
    else: p.info("No preview GLB requested")
    p.done("Universal motion retarget complete"); hard_exit(p)

if __name__=="__main__": main()
