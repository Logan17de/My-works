#!/usr/bin/env python3
"""Visible batch TRELLIS.2 image-to-GLB runner for the Colab Drive workflow."""
from __future__ import annotations

import functools
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/content/huggingface")

HF_READY_MARKER = Path(os.environ["HF_HOME"]) / "trellis2_preload_ready.json"
if HF_READY_MARKER.is_file():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

TRELLIS_ROOT = Path(os.environ.get("TRELLIS2_ROOT", "/content/TRELLIS.2")).resolve()
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

import torch
from PIL import Image
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

ROOT = Path("/content/drive/MyDrive/Shared/Trellis")
INPUT_DIR = ROOT / "Input image"
OUTPUT_DIR = ROOT / "Output"
DONE_DIR = ROOT / "Done"
FAILED_DIR = ROOT / "Failed"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def move_replace(src: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    dst = destination_dir / src.name
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def attach_stage_trace(pipeline: Trellis2ImageTo3DPipeline) -> None:
    """Trace TRELLIS internals while still using the upstream pipeline.run()."""
    stages = {
        "preprocess_image": "background removal + image preprocessing",
        "get_cond": "DINOv3 image encoding / conditioning",
        "sample_sparse_structure": "sparse 3D structure sampling",
        "sample_shape_slat": "shape latent generation",
        "sample_shape_slat_cascade": "high-resolution shape refinement",
        "sample_tex_slat": "texture latent generation",
        "decode_latent": "mesh + PBR voxel decoding",
    }

    for method_name, label in stages.items():
        original = getattr(pipeline, method_name, None)
        if not callable(original):
            continue

        def make_wrapper(fn, stage_label, name):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                detail = ""
                if name == "get_cond":
                    resolution = kwargs.get("resolution")
                    if resolution is None and len(args) >= 2:
                        resolution = args[1]
                    if resolution is not None:
                        detail = f" ({resolution}px)"
                print(f"      ▶ {stage_label}{detail}", flush=True)
                started = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - started
                    print(f"      ✓ {stage_label}{detail} — {elapsed:.1f}s", flush=True)
            return wrapper

        setattr(pipeline, method_name, make_wrapper(original, label, method_name))


def main() -> None:
    for folder in (INPUT_DIR, OUTPUT_DIR, DONE_DIR, FAILED_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("TRELLIS.2 DRIVE BATCH", flush=True)
    print(f"Input : {INPUT_DIR}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print("=" * 72, flush=True)

    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not images:
        print("✅ Input image folder is empty. Nothing to process.", flush=True)
        return

    print(f"\n[BATCH] Found {len(images)} image(s).", flush=True)
    print("[MODEL LOAD] microsoft/TRELLIS.2-4B", flush=True)
    if HF_READY_MARKER.is_file():
        print(f"[MODEL LOAD] Using pre-downloaded Hugging Face cache: {HF_READY_MARKER}", flush=True)
    else:
        print("[MODEL LOAD] Cache marker not found; Hugging Face may download missing files now.", flush=True)

    started = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = True
    print("[MODEL LOAD] Checkpoints resolved. Initializing CUDA pipeline...", flush=True)
    pipeline.cuda()
    attach_stage_trace(pipeline)
    print(f"[MODEL LOAD] ✅ TRELLIS ready in {time.perf_counter() - started:.1f}s\n", flush=True)

    success = 0
    failed = 0

    for index, image_path in enumerate(images, 1):
        output_path = OUTPUT_DIR / f"{image_path.stem}.glb"
        image_started = time.perf_counter()
        print("\n" + "-" * 72, flush=True)
        print(f"[IMAGE {index}/{len(images)}] {image_path.name}", flush=True)
        print("-" * 72, flush=True)

        try:
            if output_path.is_file() and output_path.stat().st_size > 0:
                print(f"[SKIP] Existing GLB: {output_path.name}", flush=True)
                move_replace(image_path, DONE_DIR)
                success += 1
                continue

            print("[1/4] Reading source image", flush=True)
            with Image.open(image_path) as source:
                image = source.copy()
            print(f"      ✓ {image.width}×{image.height} | mode={image.mode}", flush=True)

            print("[2/4] TRELLIS image → 3D inference", flush=True)
            mesh = pipeline.run(image)[0]
            print(f"      ✓ Raw mesh: {mesh.vertices.shape[0]:,} vertices | {mesh.faces.shape[0]:,} faces", flush=True)

            print("[3/4] Remesh + UV unwrap + PBR texture bake", flush=True)
            bake_started = time.perf_counter()
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
                verbose=True,
            )
            print(f"      ✓ GLB geometry/textures prepared in {time.perf_counter() - bake_started:.1f}s", flush=True)

            print("[4/4] Exporting GLB to Google Drive", flush=True)
            export_started = time.perf_counter()
            glb.export(str(output_path), extension_webp=False)
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("GLB export did not produce a valid file")
            size_mib = output_path.stat().st_size / 1024**2
            print(f"      ✓ Saved {output_path} | {size_mib:.1f} MiB | {time.perf_counter() - export_started:.1f}s", flush=True)

            move_replace(image_path, DONE_DIR)
            success += 1
            print(f"[DONE] ✅ {image_path.name} → {output_path.name} | total {time.perf_counter() - image_started:.1f}s", flush=True)

            del image, mesh, glb
            torch.cuda.empty_cache()

        except Exception as exc:
            failed += 1
            print(f"[FAILED] ❌ {image_path.name}: {exc}", flush=True)
            traceback.print_exc()
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()
            try:
                move_replace(image_path, FAILED_DIR)
            except Exception:
                pass
            torch.cuda.empty_cache()

    print("\n" + "=" * 72, flush=True)
    print("BATCH FINISHED", flush=True)
    print(f"✅ Completed: {success}", flush=True)
    print(f"❌ Failed:    {failed}", flush=True)
    print(f"📁 Output:    {OUTPUT_DIR}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
