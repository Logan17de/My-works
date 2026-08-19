#!/usr/bin/env python3
"""Run the complete Animation Engine with deterministic ARDY→MIA retargeting."""
from __future__ import annotations
import argparse, json, os, shlex, shutil, subprocess
from pathlib import Path
import numpy as np
from progress_utils import Progress


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--character",required=True)
    p.add_argument("--prompt",required=True)
    p.add_argument("--duration",type=float,default=6.0)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--output-dir",default="/content/animation_outputs")
    p.add_argument("--source-manifest",default=None)
    p.add_argument("--already-rigged",action="store_true")
    p.add_argument("--no-fingers",action="store_true")
    return p.parse_args()


def run(cmd,progress,label,*,cwd=None,env=None):
    progress.info("Command: "+" ".join(shlex.quote(str(x)) for x in cmd))
    with progress.heartbeat(label,every=30):
        subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,check=True)


def require_file(path,label):
    path=path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size==0:
        raise FileNotFoundError(f"{label} missing/empty: {path}")
    return path


def gpu_preflight(conda_env, progress, label):
    code=(
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA is not visible in this environment'; "
        "p=torch.cuda.get_device_properties(0); "
        "print('[GPU CHECK][%s] '+p.name+' | CUDA=True | VRAM=%%.1f GiB'%%(p.total_memory/1024**3), flush=True)"
        % label
    )
    run(["/opt/conda/bin/conda","run","--no-capture-output","-n",conda_env,"python","-c",code],progress,f"{label} CUDA preflight")


def main():
    p=Progress("ANIMATION ENGINE",8)
    a=parse_args()

    p.step("Validating character, prompt, Hugging Face token and output folder")
    if a.duration<=0: raise ValueError("--duration must be positive")
    if not os.environ.get("HF_TOKEN","").strip():
        raise RuntimeError("HF_TOKEN is required for ARDY's gated Llama text encoder.")
    character=require_file(Path(a.character),"character")
    if character.suffix.lower() not in {".glb",".fbx",".obj",".ply"}:
        raise ValueError(f"Unsupported character format: {character.suffix}")
    if a.already_rigged and character.suffix.lower()!=".fbx":
        raise ValueError("--already-rigged requires an FBX character")
    tools=Path(__file__).resolve().parent
    out=Path(a.output_dir).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env["MPLBACKEND"]="Agg"
    mia_root="/content/Make-It-Animatable"
    mia_env={**env,"MIA_ROOT":mia_root}
    mia_env["PYTHONPATH"]=mia_root+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p.info(f"Character: {character.name} ({character.stat().st_size/1024**2:.1f} MiB)")
    p.info(f"Prompt: {a.prompt}")
    p.info(f"Duration: {a.duration:g}s | seed={a.seed} | fingers={'removed' if a.no_fingers else 'kept'}")
    material_source=None
    if character.suffix.lower()==".glb":
        material_source=out/"character_material_source.glb"
        shutil.copy2(character,material_source)
        p.info("Preserved source GLB as PBR material master")

    p.step("Generating body motion with NVIDIA ARDY")
    gpu_preflight("ardy",p,"ARDY")
    motion_stem=out/"motion"
    run([
        "/opt/conda/bin/conda","run","--no-capture-output","-n","ardy","python",
        "scripts/generate.py",a.prompt,
        "--model","core","--duration",str(a.duration),"--seed",str(a.seed),"--output",str(motion_stem),
    ],p,"ARDY diffusion/motion generation",cwd="/content/ardy",env=env)
    motion_npz=require_file(Path(str(motion_stem)+".npz"),"ARDY motion")
    p.ok(f"Raw ARDY motion: {motion_npz}")

    p.step("Validating ARDY output and attaching skeleton metadata")
    bridge=out/"motion_bridge.npz"
    run([
        "/opt/conda/bin/conda","run","--no-capture-output","-n","ardy","python",
        str(tools/"enrich_ardy_motion.py"),"--input",str(motion_npz),"--output",str(bridge),
    ],p,"ARDY NPZ validation",env=env)
    require_file(bridge,"motion bridge")
    with np.load(bridge,allow_pickle=True) as z:
        fps=float(np.asarray(z["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps<=0:
        raise RuntimeError(f"Invalid ARDY FPS: {fps}")
    p.info(f"Motion FPS contract: {fps:g}")

    p.step("Rendering ARDY skeleton preview")
    motion_preview=out/"motion_preview.mp4"
    run([
        "/opt/conda/bin/conda","run","--no-capture-output","-n","ardy","python",
        str(tools/"preview_ardy_motion.py"),"--input",str(bridge),"--output",str(motion_preview),
    ],p,"motion preview render",env=env)
    require_file(motion_preview,"motion preview")

    p.step("Rigging the selected humanoid with Make-It-Animatable")
    rigged=out/"character_rigged.fbx"
    if a.already_rigged:
        shutil.copy2(character,rigged)
        p.info("Character marked already-rigged; copied FBX without MIA inference")
    else:
        gpu_preflight("mia",p,"MIA")
        cmd=[
            "/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",
            str(tools/"rig_character_mia.py"),"--input",str(character),"--output",str(rigged),
        ]
        if a.no_fingers: cmd.append("--no-fingers")
        run(cmd,p,"MIA joint/skin prediction",env=mia_env)
    require_file(rigged,"rigged character")

    p.step("Directly baking ARDY motion into the MIA target rest-space")
    final_fbx=out/"character_animated.fbx"
    preview_glb=out/"character_animated_preview.glb"
    run([
        "/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",
        str(tools/"retarget_ardy_direct.py"),
        "--target",str(rigged),
        "--motion-bridge",str(bridge),
        "--output",str(final_fbx),
        "--preview-glb",str(preview_glb),
        "--fps",str(fps),
    ],p,"deterministic ARDY→MIA bake/export",env=mia_env)
    require_file(final_fbx,"final animated FBX")

    p.step("Running strict scale/deformation/FPS/motion validation")
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
    ],p,"final direct-retarget contract validation",env=mia_env)
    require_file(report,"contract report")
    report_data=json.loads(report.read_text(encoding="utf-8"))
    if not report_data.get("passed"):
        raise RuntimeError("Animation contract report did not pass.")

    p.step("Packaging Unreal handoff files")
    package=out/"unreal_package"
    if package.exists(): shutil.rmtree(package)
    package.mkdir(parents=True)
    for src in (final_fbx,report,motion_preview): shutil.copy2(src,package/src.name)
    if preview_glb.is_file() and preview_glb.stat().st_size:
        shutil.copy2(preview_glb,package/preview_glb.name)
    if material_source:
        shutil.copy2(material_source,package/material_source.name)
    source_manifest_name=None
    if a.source_manifest:
        src_manifest=require_file(Path(a.source_manifest),"source asset manifest")
        source_manifest_name="source_asset_manifest.json"
        shutil.copy2(src_manifest,package/source_manifest_name)
    package_manifest={
        "animation_master":final_fbx.name,
        "material_master":material_source.name if material_source else None,
        "animated_preview_glb":preview_glb.name if preview_glb.is_file() else None,
        "source_asset_manifest":source_manifest_name,
        "motion_preview":motion_preview.name,
        "contract_report":report.name,
        "fps":fps,
        "retarget_engine":"direct_ardy_rest_space_v2",
        "auto_rig_pro_used_for_motion_transfer":False,
        "unreal_note":"Import FBX as Skeletal Mesh/Animation. Use automatic/custom sample rate matching fps. Reapply PBR from the material-master GLB when present.",
    }
    (package/"package_manifest.json").write_text(json.dumps(package_manifest,indent=2),encoding="utf-8")
    archive=shutil.make_archive(str(out/"unreal_character_package"),"zip",package)
    p.ok(f"Package: {archive} ({Path(archive).stat().st_size/1024**2:.1f} MiB)")
    p.done("Animation Engine finished successfully")
    print("PACKAGE_ZIP="+archive,flush=True)
    print("FINAL_FBX="+str(final_fbx),flush=True)
    print("MOTION_PREVIEW="+str(motion_preview),flush=True)
    print("CONTRACT_REPORT="+str(report),flush=True)


if __name__=="__main__": main()
