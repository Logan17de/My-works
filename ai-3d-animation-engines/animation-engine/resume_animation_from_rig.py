#!/usr/bin/env python3
"""Resume Animation Engine from the auto-rig stage using existing ARDY outputs."""
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
    ardy_source=require(out/"ardy_source.fbx","ARDY source FBX")
    with np.load(bridge,allow_pickle=True) as z:
        fps=float(np.asarray(z["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps<=0:
        raise RuntimeError(f"Invalid FPS in {bridge}: {fps}")

    env=os.environ.copy(); env["MPLBACKEND"]="Agg"
    mia_root="/content/Make-It-Animatable"
    env["MIA_ROOT"]=mia_root
    env["PYTHONPATH"]=mia_root+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    p.step("Rigging selected humanoid with Make-It-Animatable")
    rigged=out/"character_rigged.fbx"
    if a.already_rigged:
        if character.suffix.lower()!=".fbx": raise ValueError("--already-rigged requires FBX input")
        shutil.copy2(character,rigged)
    else:
        gpu_preflight("mia",p,"MIA")
        cmd=["/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",str(tools/"rig_character_mia.py"),"--input",str(character),"--output",str(rigged)]
        if a.no_fingers: cmd.append("--no-fingers")
        run(cmd,p,"MIA joint/skin prediction",env=env)
    require(rigged,"rigged character")

    p.step("Retargeting existing ARDY motion")
    final_fbx=out/"character_animated.fbx"; preview_glb=out/"character_animated_preview.glb"
    run(["/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",str(tools/"retarget_with_mia.py"),"--target",str(rigged),"--animation",str(ardy_source),"--output",str(final_fbx),"--preview-glb",str(preview_glb),"--fps",str(fps)],p,"retarget/export",env=env)
    require(final_fbx,"final animated FBX")

    p.step("Validating final scale/FPS/motion contract")
    report=out/"animation_contract_report.json"
    run(["/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",str(tools/"validate_animation_contract.py"),"--character-source",str(character),"--rigged-target",str(rigged),"--source-animation",str(ardy_source),"--animated-target",str(final_fbx),"--expected-fps",str(fps),"--report",str(report),"--strict"],p,"contract validation",env=env)
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
    manifest={"animation_master":final_fbx.name,"material_master":material_source.name if material_source.is_file() else None,"animated_preview_glb":preview_glb.name if preview_glb.is_file() else None,"motion_preview":motion_preview.name,"contract_report":report.name,"fps":fps}
    (package/"package_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    archive=shutil.make_archive(str(out/"unreal_character_package"),"zip",package)
    p.ok(f"Package: {archive}")

    p.step("Resume complete")
    p.done("Animation Engine resumed successfully")
    print("FINAL_FBX="+str(final_fbx),flush=True)
    print("PACKAGE_ZIP="+archive,flush=True)

if __name__=="__main__": main()
