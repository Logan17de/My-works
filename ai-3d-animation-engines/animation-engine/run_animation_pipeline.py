#!/usr/bin/env python3
"""Run the complete Animation Engine from one command inside the combined Colab."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--character", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="/content/animation_outputs")
    p.add_argument("--source-manifest", default=None)
    p.add_argument("--already-rigged", action="store_true")
    p.add_argument("--no-fingers", action="store_true")
    return p.parse_args()


def run(cmd: list[str], *, cwd: str | None = None, env: dict | None = None) -> None:
    print("Running:", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=True)


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing/empty: {path}")
    return path


def main() -> None:
    a = parse_args()
    if a.duration <= 0:
        raise ValueError("--duration must be positive")
    if not os.environ.get("HF_TOKEN", "").strip():
        raise RuntimeError("HF_TOKEN is required for ARDY's gated Llama text encoder.")

    character = require_file(Path(a.character), "character")
    if character.suffix.lower() not in {".glb", ".fbx", ".obj", ".ply"}:
        raise ValueError(f"Unsupported character format: {character.suffix}")
    if a.already_rigged and character.suffix.lower() != ".fbx":
        raise ValueError("--already-rigged requires an FBX character")

    tools = Path(__file__).resolve().parent
    out = Path(a.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    mia_env = {**env, "MIA_ROOT": "/content/Make-It-Animatable"}

    material_source = None
    if character.suffix.lower() == ".glb":
        material_source = out / "character_material_source.glb"
        shutil.copy2(character, material_source)

    motion_stem = out / "motion"
    run([
        "/opt/conda/bin/conda", "run", "-n", "ardy", "python", "scripts/generate.py",
        a.prompt, "--model", "core", "--duration", str(a.duration),
        "--seed", str(a.seed), "--output", str(motion_stem),
    ], cwd="/content/ardy", env=env)

    motion_npz = require_file(Path(str(motion_stem) + ".npz"), "ARDY motion")
    motion_bridge = out / "motion_bridge.npz"
    motion_preview = out / "motion_preview.mp4"

    run([
        "/opt/conda/bin/conda", "run", "-n", "ardy", "python",
        str(tools / "enrich_ardy_motion.py"),
        "--input", str(motion_npz), "--output", str(motion_bridge),
    ])
    require_file(motion_bridge, "motion bridge")

    with np.load(motion_bridge, allow_pickle=True) as z:
        fps = float(np.asarray(z["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid ARDY FPS: {fps}")
    print("ARDY FPS contract:", fps)

    run([
        "/opt/conda/bin/conda", "run", "-n", "ardy", "python",
        str(tools / "preview_ardy_motion.py"),
        "--input", str(motion_bridge), "--output", str(motion_preview),
    ])
    require_file(motion_preview, "motion preview")

    ardy_source = out / "ardy_source.fbx"
    run([
        "/opt/conda/bin/conda", "run", "-n", "mia", "python",
        str(tools / "ardy_motion_to_fbx.py"),
        "--input", str(motion_bridge), "--output", str(ardy_source),
    ])
    require_file(ardy_source, "ARDY source FBX")

    rigged = out / "character_rigged.fbx"
    if a.already_rigged:
        shutil.copy2(character, rigged)
    else:
        rig_cmd = [
            "/opt/conda/bin/conda", "run", "-n", "mia", "python",
            str(tools / "rig_character_mia.py"),
            "--input", str(character), "--output", str(rigged),
        ]
        if a.no_fingers:
            rig_cmd.append("--no-fingers")
        run(rig_cmd, env=mia_env)
    require_file(rigged, "rigged character")

    final_fbx = out / "character_animated.fbx"
    preview_glb = out / "character_animated_preview.glb"
    run([
        "/opt/conda/bin/conda", "run", "-n", "mia", "python",
        str(tools / "retarget_with_mia.py"),
        "--target", str(rigged),
        "--animation", str(ardy_source),
        "--output", str(final_fbx),
        "--preview-glb", str(preview_glb),
        "--fps", str(fps),
    ], env=mia_env)
    require_file(final_fbx, "final animated FBX")

    report = out / "animation_contract_report.json"
    run([
        "/opt/conda/bin/conda", "run", "-n", "mia", "python",
        str(tools / "validate_animation_contract.py"),
        "--character-source", str(character),
        "--rigged-target", str(rigged),
        "--source-animation", str(ardy_source),
        "--animated-target", str(final_fbx),
        "--expected-fps", str(fps),
        "--report", str(report),
        "--strict",
    ])
    require_file(report, "contract report")
    report_data = json.loads(report.read_text(encoding="utf-8"))
    if not report_data.get("passed"):
        raise RuntimeError("Animation contract report did not pass.")

    package = out / "unreal_package"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)

    for src in (final_fbx, report, motion_preview):
        shutil.copy2(src, package / src.name)
    if preview_glb.is_file() and preview_glb.stat().st_size:
        shutil.copy2(preview_glb, package / preview_glb.name)
    if material_source:
        shutil.copy2(material_source, package / material_source.name)

    source_manifest_name = None
    if a.source_manifest:
        source_manifest = require_file(Path(a.source_manifest), "source asset manifest")
        source_manifest_name = "source_asset_manifest.json"
        shutil.copy2(source_manifest, package / source_manifest_name)

    package_manifest = {
        "animation_master": final_fbx.name,
        "material_master": material_source.name if material_source else None,
        "animated_preview_glb": preview_glb.name if preview_glb.is_file() else None,
        "source_asset_manifest": source_manifest_name,
        "motion_preview": motion_preview.name,
        "contract_report": report.name,
        "fps": fps,
        "unreal_note": (
            "Import FBX as Skeletal Mesh/Animation. Use automatic/custom sample rate matching fps. "
            "Reapply PBR from the material-master GLB when present."
        ),
    }
    (package / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2), encoding="utf-8"
    )

    archive = shutil.make_archive(str(out / "unreal_character_package"), "zip", package)
    print("Animation Engine complete.")
    print("PACKAGE_ZIP=" + archive)
    print("FINAL_FBX=" + str(final_fbx))
    print("MOTION_PREVIEW=" + str(motion_preview))
    print("CONTRACT_REPORT=" + str(report))


if __name__ == "__main__":
    main()
