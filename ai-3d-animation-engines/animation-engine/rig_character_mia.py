#!/usr/bin/env python3
"""Headless Make-It-Animatable wrapper with output validation and live progress."""
from __future__ import annotations
import argparse, os, shutil, sys, importlib
from pathlib import Path
from progress_utils import Progress

REQUIRED_BASE_BONES={
"mixamorig:Hips","mixamorig:Spine","mixamorig:Spine1","mixamorig:Spine2","mixamorig:Neck","mixamorig:Head",
"mixamorig:LeftShoulder","mixamorig:LeftArm","mixamorig:LeftForeArm","mixamorig:LeftHand",
"mixamorig:RightShoulder","mixamorig:RightArm","mixamorig:RightForeArm","mixamorig:RightHand",
"mixamorig:LeftUpLeg","mixamorig:LeftLeg","mixamorig:LeftFoot","mixamorig:LeftToeBase",
"mixamorig:RightUpLeg","mixamorig:RightLeg","mixamorig:RightFoot","mixamorig:RightToeBase"}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--no-fingers",action="store_true",default=False); return p.parse_args()

def _patch_gradio_for_headless(progress):
    """MIA's app replaces gradio.helpers.log_message with a UI-context logger.

    In our headless Colab subprocess there is no active Gradio Blocks context, so
    gr.Info/gr.Warning otherwise raise LookupError(ContextVar 'blocks'). Replace
    only notification logging with console output; model/inference APIs stay intact.
    """
    import gradio.helpers as gr_helpers

    def headless_log_message(message, level="info", duration=None, visible=True, *args, **kwargs):
        text=str(message)
        if str(level).lower() in {"warning","error"}:
            progress.warn(f"MIA: {text}")
        else:
            progress.info(f"MIA: {text}")
        return None

    gr_helpers.log_message=headless_log_message
    progress.info("Patched MIA Gradio notifications for headless Colab execution")

def main():
    p=Progress("AUTO-RIG",7); a=parse_args(); ip=Path(a.input).expanduser().resolve()
    p.step("Validating selected humanoid mesh")
    if not ip.is_file(): raise FileNotFoundError(ip)
    if ip.suffix.lower() not in {".glb",".fbx",".obj",".ply"}: raise ValueError(f"Unsupported character format: {ip.suffix}")
    p.info(f"Input: {ip} ({ip.stat().st_size/1024**2:.1f} MiB) | fingers={'removed' if a.no_fingers else 'kept'}")

    p.step("Loading Make-It-Animatable code and Blender integration")
    mia_root=Path(os.environ.get("MIA_ROOT","/content/Make-It-Animatable")).resolve()
    if not (mia_root/"app.py").is_file(): raise FileNotFoundError(f"Make-It-Animatable not found at {mia_root}")

    # When this helper is launched by absolute path, Python puts the helper's
    # directory (My-works/animation-engine) in sys.path, not the later cwd.
    # chdir() alone therefore does not make MIA's top-level `app.py` importable.
    mia_root_str=str(mia_root)
    if mia_root_str not in sys.path:
        sys.path.insert(0,mia_root_str)
    existing_pythonpath=os.environ.get("PYTHONPATH","")
    os.environ["PYTHONPATH"]=mia_root_str+(os.pathsep+existing_pythonpath if existing_pythonpath else "")
    os.chdir(mia_root)
    importlib.invalidate_caches()
    p.info(f"MIA import root: {mia_root}")

    import app as mia
    from util import blender_utils
    _patch_gradio_for_headless(p)
    p.ok(f"Loaded MIA app module: {Path(mia.__file__).resolve()}")
    for name in ("state","output_joints_coarse","output_normed_input","output_sample","output_joints","output_bw","output_rest_vis","output_rest_lbs","output_anim_vis","output_anim"):
        setattr(mia,name,name)

    p.step("Loading Make-It-Animatable neural-network models")
    with p.heartbeat("MIA model initialization",every=20): mia.init_models()
    db=mia.DB(); p.ok("MIA models loaded")

    p.step("Importing and preprocessing character geometry")
    mia.prepare_input(str(ip),is_gs=False,opacity_threshold=0.0,db=db,export_temp=False)
    if not db.is_mesh: raise ValueError("Animation Engine requires a polygon mesh; selected file was loaded as a point cloud.")
    with p.heartbeat("character preprocessing",every=20): mia.preprocess(db)
    p.ok("Geometry normalized and sampled")

    p.step("Predicting joints and skinning weights")
    with p.heartbeat("MIA inference",every=20): mia.infer(input_normal=False,db=db)
    mia.vis(bw_fix=True,bw_vis_bone="LeftArm",no_fingers=a.no_fingers,db=db)
    p.ok("Joint/weight prediction complete")

    p.step("Building Blender armature and exporting rigged FBX")
    with p.heartbeat("Blender rig construction/export",every=20):
        mia.vis_blender(reset_to_rest=True,remove_fingers=a.no_fingers,rest_pose_type="No",ignore_pose_parts=[],animation_file=None,retarget=False,inplace=False,db=db)
    source=Path(db.anim_path).resolve()
    if not source.is_file() or source.stat().st_size==0: raise RuntimeError(f"Make-It-Animatable did not produce a valid FBX: {source}")

    p.step("Validating skeleton names, mesh binding and output")
    armatures=blender_utils.get_all_armature_obj()
    if len(armatures)!=1: raise RuntimeError(f"Expected one generated armature, found {len(armatures)}")
    armature=armatures[0]; bone_names={b.name for b in armature.data.bones}; missing=sorted(REQUIRED_BASE_BONES-bone_names)
    if missing: raise RuntimeError("Generated rig is not Mixamo-compatible; missing base bones: "+", ".join(missing))
    meshes=blender_utils.get_all_mesh_obj()
    if not meshes: raise RuntimeError("Generated rig contains no mesh object")
    if not any(any(mod.type=="ARMATURE" and mod.object==armature for mod in mesh.modifiers) for mesh in meshes): raise RuntimeError("Generated mesh is not bound to the generated armature")
    out=Path(a.output).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,out)
    if not out.is_file() or out.stat().st_size==0: raise RuntimeError(f"Failed to copy rigged FBX to {out}")
    verts=sum(len(m.data.vertices) for m in meshes)
    p.ok(f"Rig validated: {len(bone_names)} bones, {verts:,} vertices")
    p.ok(f"Rigged FBX: {out} ({out.stat().st_size/1024**2:.1f} MiB)")
    p.done("Auto-rigging complete")
if __name__=="__main__": main()
