#!/usr/bin/env python3
"""Resume Animation Engine from an existing ARDY bridge and MIA rig."""
from __future__ import annotations
import argparse, json, os, shlex, shutil, subprocess
from pathlib import Path
import numpy as np
from progress_utils import Progress


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--character",required=True)
    p.add_argument("--output-dir",default="/content/animation_outputs")
    p.add_argument("--no-fingers",action="store_true")
    p.add_argument("--already-rigged",action="store_true")
    p.add_argument("--reuse-existing-rig",action="store_true",help="Reuse output-dir/character_rigged.fbx when it already exists and is non-empty")
    p.add_argument("--reuse-existing-retarget",action="store_true",help="Reuse output-dir/character_animated.fbx when it already exists and is non-empty")
    return p.parse_args()


def run(cmd,p,label,*,cwd=None,env=None):
    p.info("Command: "+" ".join(shlex.quote(str(x)) for x in cmd))
    with p.heartbeat(label,every=30):
        subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,check=True)


def require(path,label):
    q=Path(path).expanduser().resolve()
    if not q.is_file() or q.stat().st_size==0:
        raise FileNotFoundError(f"{label} missing/empty: {q}")
    return q


def gpu_preflight(conda_env,p,label):
    code=(
        "import torch; assert torch.cuda.is_available(), 'CUDA is not visible'; "
        "d=torch.cuda.get_device_properties(0); "
        f"print('[GPU CHECK][{label}] '+d.name+' | CUDA=True | VRAM=%.1f GiB'%(d.total_memory/1024**3), flush=True)"
    )
    run(["/opt/conda/bin/conda","run","--no-capture-output","-n",conda_env,"python","-c",code],p,f"{label} CUDA preflight")


def main():
    a=parse_args(); p=Progress("ANIMATION RESUME",5)
    tools=Path(__file__).resolve().parent
    out=Path(a.output_dir).expanduser().resolve()
    character=require(a.character,"character")
    bridge=require(out/"motion_bridge.npz","motion bridge")
    motion_preview=require(out/"motion_preview.mp4","motion preview")
    with np.load(bridge,allow_pickle=True) as z:
        fps=float(np.asarray(z["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps<=0:
        raise RuntimeError(f"Invalid FPS in {bridge}: {fps}")

    env=os.environ.copy(); env["MPLBACKEND"]="Agg"
    mia_root="/content/Make-It-Animatable"
    env["MIA_ROOT"]=mia_root
    env["PYTHONPATH"]=mia_root+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    p.step("Preparing/reusing MIA-rigged humanoid")
    rigged=out/"character_rigged.fbx"
    if a.already_rigged:
        if character.suffix.lower()!=".fbx": raise ValueError("--already-rigged requires FBX input")
        shutil.copy2(character,rigged)
        p.info("Character marked already-rigged; copied input FBX")
    elif a.reuse_existing_rig and rigged.is_file() and rigged.stat().st_size>0:
        p.ok(f"Reusing existing rig artifact: {rigged} ({rigged.stat().st_size/1024**2:.1f} MiB)")
    else:
        gpu_preflight("mia",p,"MIA")
        cmd=["/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",str(tools/"rig_character_mia.py"),"--input",str(character),"--output",str(rigged)]
        if a.no_fingers: cmd.append("--no-fingers")
        run(cmd,p,"MIA joint/skin prediction",env=env)
    require(rigged,"rigged character")

    p.step("Directly baking ARDY motion into the MIA target rest-space")
    final_fbx=out/"character_animated.fbx"; preview_glb=out/"character_animated_preview.glb"
    if a.reuse_existing_retarget and final_fbx.is_file() and final_fbx.stat().st_size>0:
        p.ok(f"Reusing existing animated FBX: {final_fbx} ({final_fbx.stat().st_size/1024**2:.1f} MiB)")
        p.info("Skipping direct retarget; strict deformation validation runs next.")
    else:
        run([
            "/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",
            str(tools/"retarget_ardy_direct.py"),
            "--target",str(rigged),
            "--motion-bridge",str(bridge),
            "--output",str(final_fbx),
            "--preview-glb",str(preview_glb),
            "--fps",str(fps),
        ],p,"deterministic ARDY→MIA bake/export",env=env)
    require(final_fbx,"final animated FBX")

    p.step("Validating scale, skinned deformation, FPS and motion")
    report=out/"animation_contract_report.json"
    run([
        "/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",
        str(tools/"validate_animation_contract.py"),
        "--character-source",str(character),
        "--rigged-target",str(rigged),
        "--motion-bridge",str(bridge),
        "--animated-target",str(final_fbx),
        "--expected-fps",str(fps),
        "--report",str(report),
        "--strict",
    ],p,"strict direct-retarget validation",env=env)
    data=json.loads(require(report,"contract report").read_text(encoding="utf-8"))
    if not data.get("passed"): raise RuntimeError("Animation contract did not pass")

    p.step("Packaging Unreal handoff")
    material_source=out/"character_material_source.glb"
    if character.suffix.lower()==".glb" and not material_source.is_file(): shutil.copy2(character,material_source)
    package=out/"unreal_package"
    if package.exists(): shutil.rmtree(package)
    package.mkdir(parents=True)
    for src in (final_fbx,report,motion_preview): shutil.copy2(src,package/src.name)
    if preview_glb.is_file() and preview_glb.stat().st_size: shutil.copy2(preview_glb,package/preview_glb.name)
    if material_source.is_file(): shutil.copy2(material_source,package/material_source.name)
    manifest={
        "animation_master":final_fbx.name,
        "material_master":material_source.name if material_source.is_file() else None,
        "animated_preview_glb":preview_glb.name if preview_glb.is_file() else None,
        "motion_preview":motion_preview.name,
        "contract_report":report.name,
        "fps":fps,
        "retarget_engine":"direct_ardy_rest_space_v2",
        "auto_rig_pro_used_for_motion_transfer":False,
    }
    (package/"package_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    archive=shutil.make_archive(str(out/"unreal_character_package"),"zip",package)
    p.ok(f"Package: {archive}")

    p.step("Resume complete")
    p.done("Animation Engine resumed successfully")
    print("FINAL_FBX="+str(final_fbx),flush=True)
    print("PACKAGE_ZIP="+archive,flush=True)


if __name__=="__main__": main()
