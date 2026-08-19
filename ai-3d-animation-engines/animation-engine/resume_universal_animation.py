#!/usr/bin/env python3
"""Resume Universal Animation Engine from existing ARDY motion and/or MIA rig."""
from __future__ import annotations

import argparse, json, os, shlex, shutil, subprocess
from pathlib import Path
import numpy as np
from progress_utils import Progress

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--character",required=True); p.add_argument("--output-dir",default="/content/animation_outputs"); p.add_argument("--rig-preset",default="mia_mixamo"); p.add_argument("--reuse-existing-rig",action="store_true"); p.add_argument("--no-fingers",action="store_true"); return p.parse_args()

def run(cmd,p,label,*,env=None):
    p.info("Command: "+" ".join(shlex.quote(str(x)) for x in cmd))
    with p.heartbeat(label,every=30): subprocess.run([str(x) for x in cmd],env=env,check=True)

def require(path,label):
    q=Path(path).expanduser().resolve()
    if not q.is_file() or q.stat().st_size==0: raise FileNotFoundError(f"{label} missing/empty: {q}")
    return q

def main():
    a=parse_args(); p=Progress("UNIVERSAL RESUME",5); tools=Path(__file__).resolve().parent; out=Path(a.output_dir).expanduser().resolve(); character=require(a.character,"character"); bridge=require(out/"motion_bridge.npz","motion bridge"); motion_preview=require(out/"motion_preview.mp4","motion preview")
    with np.load(bridge,allow_pickle=True) as z: fps=float(np.asarray(z["fps"]).reshape(-1)[0])
    env=os.environ.copy(); env["MPLBACKEND"]="Agg"; mia_root="/content/Make-It-Animatable"; env["MIA_ROOT"]=mia_root; env["PYTHONPATH"]=mia_root+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    p.step("Building/reusing Canonical Motion")
    canonical=out/"canonical_motion.npz"
    run(["/opt/conda/bin/conda","run","--no-capture-output","-n","ardy","python",str(tools/"build_canonical_motion.py"),"--source","ardy","--input",str(bridge),"--output",str(canonical)],p,"canonical motion build",env=env); require(canonical,"canonical motion")

    p.step("Preparing/reusing target rig")
    rigged=out/"character_rigged.fbx"
    if a.reuse_existing_rig and rigged.is_file() and rigged.stat().st_size>0: p.ok(f"Reusing target rig: {rigged}")
    else:
        cmd=["/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",str(tools/"rig_character_mia.py"),"--input",str(character),"--output",str(rigged)]
        if a.no_fingers: cmd.append("--no-fingers")
        run(cmd,p,"MIA rig provider",env=env)
    require(rigged,"target rig")

    p.step("Universal retarget")
    final_fbx=out/"character_animated.fbx"; preview_glb=out/"character_animated_preview.glb"
    run(["/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",str(tools/"run_universal_retarget.py"),"--motion",str(canonical),"--target",str(rigged),"--rig-preset",a.rig_preset,"--output",str(final_fbx),"--preview-glb",str(preview_glb)],p,"universal retarget",env=env); require(final_fbx,"animated FBX")

    p.step("Strict validation")
    report=out/"animation_contract_report.json"
    run(["/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",str(tools/"validate_animation_contract.py"),"--character-source",str(character),"--rigged-target",str(rigged),"--motion-bridge",str(bridge),"--animated-target",str(final_fbx),"--expected-fps",str(fps),"--report",str(report),"--strict"],p,"universal validation",env=env)
    data=json.loads(require(report,"report").read_text(encoding="utf-8"))
    if not data.get("passed"): raise RuntimeError("Universal validation did not pass")

    p.step("Packaging")
    package=out/"unreal_package"
    if package.exists(): shutil.rmtree(package)
    package.mkdir(parents=True)
    for src in (final_fbx,report,motion_preview,canonical): shutil.copy2(src,package/src.name)
    if preview_glb.is_file() and preview_glb.stat().st_size: shutil.copy2(preview_glb,package/preview_glb.name)
    archive=shutil.make_archive(str(out/"unreal_character_package"),"zip",package); p.done("Universal Animation Engine resumed successfully"); print("FINAL_FBX="+str(final_fbx),flush=True); print("PACKAGE_ZIP="+archive,flush=True)

if __name__=="__main__": main()
