#!/usr/bin/env python3
"""Batch TRELLIS.2 image-to-GLB runner for the Colab Drive workflow."""
from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/content/huggingface")

TRELLIS_ROOT = Path(os.environ.get("TRELLIS2_ROOT", "/content/TRELLIS.2")).resolve()
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

import torch
from PIL import Image
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

ROOT = Path("/content/drive/MyDrive/Shared/Trellis")
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
DONE_DIR = ROOT / "done"
FAILED_DIR = ROOT / "failed"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def move_replace(src: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    dst = destination_dir / src.name
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def main() -> None:
    for folder in (INPUT_DIR, OUTPUT_DIR, DONE_DIR, FAILED_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not images:
        print(f"✅ No images waiting in {INPUT_DIR}")
        return

    print(f"Found {len(images)} image(s). Loading TRELLIS.2 once...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = True
    pipeline.cuda()
    print("✅ TRELLIS.2 ready. Starting batch.\n")

    success = 0
    failed = 0

    for index, image_path in enumerate(images, 1):
        output_path = OUTPUT_DIR / f"{image_path.stem}.glb"
        print(f"[{index}/{len(images)}] {image_path.name}")

        try:
            if output_path.is_file() and output_path.stat().st_size > 0:
                print("  ↳ Existing GLB found; moving input to done.")
                move_replace(image_path, DONE_DIR)
                success += 1
                continue

            with Image.open(image_path) as source:
                image = source.copy()

            mesh = pipeline.run(image)[0]
            mesh.simplify(16_777_216)

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=500_000,
                texture_size=2048,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            glb.export(str(output_path), extension_webp=False)

            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("GLB export did not produce a valid file")

            move_replace(image_path, DONE_DIR)
            success += 1
            print(f"  ✅ Saved: {output_path}")

            del image, mesh, glb
            torch.cuda.empty_cache()

        except Exception as exc:
            failed += 1
            print(f"  ❌ Failed: {exc}")
            traceback.print_exc()
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()
            try:
                move_replace(image_path, FAILED_DIR)
            except Exception:
                pass
            torch.cuda.empty_cache()

    print("\n============================")
    print(f"✅ Completed: {success}")
    print(f"❌ Failed:    {failed}")
    print(f"📁 Output:    {OUTPUT_DIR}")
    print("============================")


if __name__ == "__main__":
    main()
