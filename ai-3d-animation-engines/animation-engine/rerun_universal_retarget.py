#!/usr/bin/env python3
"""Re-run only universal retarget + validation + packaging from existing artifacts."""
from __future__ import annotations

import argparse, json, os, shlex, shutil, subprocess
from pathlib import Path
from universal_motion import CanonicalMotion
from progress_utils import Progress


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--character",required=True,help="Original character source used for scale validation")
    p.add_argument("--output-dir",default="/content/animation_outputs")
    p.add_argument("--rig-preset",default="mia_mixamo")
    return p.parse_args()


def run(cmd,p,label,*,env=None):
    p.info("Command: "+" ".join(shlex.quote(str(x)) for x in cmd))
    with p.heartbeat(label,every=30):
        subprocess.run([str(x) for x in cmd],env=env,check=True)


def require(path,label):
    q=Path(path).expanduser().resolve()
    if not q.is_file() or q.stat().st_size==0:
        raise FileNotFoundError(f"{label} missing/empty: {q}")
    return q


def main():
    a=parse_args(); p=Progress("UNIVERSAL RE-RETARGET",4)
    tools=Path(__file__).resolve().parent
    out=Path(a.output_dir).expanduser().resolve()
    character=require(a.character,"original character")
    canonical=require(out/"canonical_motion.npz","canonical motion")
    bridge=require(out/"motion_bridge.npz","motion bridge")
    rigged=require(out/"character_rigged.fbx","MIA rig")
    motion_preview=require(out/"motion_preview.mp4","motion preview")
    motion=CanonicalMotion.load(canonical)

    env=os.environ.copy(); env["MPLBACKEND"]="Agg"
    mia_root="/content/Make-It-Animatable"
    env["MIA_ROOT"]=mia_root
    env["PYTHONPATH"]=mia_root+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    p.step("Removing only old retarget/validation/package artifacts")
    final_fbx=out/"character_animated.fbx"
    preview_glb=out/"character_animated_preview.glb"
    report=out/"animation_contract_report.json"
    archive=out/"unreal_character_package.zip"
    package=out/"unreal_package"
    for path in (final_fbx,preview_glb,report,archive):
        if path.exists():
            path.unlink(); p.info(f"Removed old: {path.name}")
    if package.exists():
        shutil.rmtree(package); p.info("Removed old: unreal_package/")
    p.ok("ARDY motion, canonical motion and MIA rig preserved")

    p.step("Re-baking canonical motion with target-rest-local retarget v2")
    run([
        "/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",
        str(tools/"run_universal_retarget.py"),
        "--motion",str(canonical),
        "--target",str(rigged),
        "--rig-preset",a.rig_preset,
        "--output",str(final_fbx),
        "--preview-glb",str(preview_glb),
    ],p,"target-rest-local universal retarget",env=env)
    require(final_fbx,"new animated FBX")

    p.step("Running strict skinned-deformation and motion validation")
    run([
        "/opt/conda/bin/conda","run","--no-capture-output","-n","mia","python",
        str(tools/"validate_animation_contract.py"),
        "--character-source",str(character),
        "--rigged-target",str(rigged),
        "--motion-bridge",str(bridge),
        "--animated-target",str(final_fbx),
        "--expected-fps",str(motion.fps),
        "--report",str(report),
        "--strict",
    ],p,"strict animation validation",env=env)
    data=json.loads(require(report,"contract report").read_text(encoding="utf-8"))
    if not data.get("passed"):
        raise RuntimeError("Strict animation validation did not pass")

    p.step("Packaging corrected Unreal handoff")
    package.mkdir(parents=True,exist_ok=True)
    material_source=out/"character_material_source.glb"
    if character.suffix.lower()==".glb" and not material_source.is_file():
        shutil.copy2(character,material_source)
    for src in (final_fbx,report,motion_preview,canonical):
        shutil.copy2(src,package/src.name)
    if preview_glb.is_file() and preview_glb.stat().st_size:
        shutil.copy2(preview_glb,package/preview_glb.name)
    if material_source.is_file():
        shutil.copy2(material_source,package/material_source.name)
    manifest={
        "schema_version":1,
        "architecture":"universal_motion_retargeting_v1",
        "retarget_core":"universal_rest_local_v2",
        "root_translation_only":True,
        "rig_preset":a.rig_preset,
        "canonical_motion":canonical.name,
        "animation_master":final_fbx.name,
        "animated_preview_glb":preview_glb.name if preview_glb.is_file() else None,
        "material_master":material_source.name if material_source.is_file() else None,
        "motion_preview":motion_preview.name,
        "contract_report":report.name,
        "fps":motion.fps,
        "ik_contact_solver":"none_v1",
        "auto_rig_pro_used_for_motion_transfer":False,
    }
    (package/"universal_engine_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    archive_path=shutil.make_archive(str(out/"unreal_character_package"),"zip",package)
    p.ok(f"Corrected package: {archive_path}")
    p.done("Re-retarget complete")
    print("FINAL_FBX="+str(final_fbx),flush=True)
    print("PREVIEW_GLB="+str(preview_glb),flush=True)
    print("CONTRACT_REPORT="+str(report),flush=True)
    print("PACKAGE_ZIP="+archive_path,flush=True)


if __name__=="__main__": main()
