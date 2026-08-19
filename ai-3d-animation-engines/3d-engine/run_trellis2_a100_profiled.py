#!/usr/bin/env python3
"""A100-aware TRELLIS.2 image-to-3D runner with sub-stage timing and VRAM profiling.

The runner keeps TRELLIS' official low-VRAM model-by-model offload policy for
large flow/decoder modules, but in `smart` residency mode it keeps the DINOv3
image encoder resident across the consecutive 512/1024 conditioning passes.
That removes one needless CPU<->GPU round trip while avoiding the unsafe
`low_vram=False` all-model residency path on 40 GB A100s.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/content/huggingface")

HF_READY_MARKER = Path(os.environ["HF_HOME"]) / "trellis2_preload_ready.json"
HF_CACHE_ONLY = HF_READY_MARKER.is_file()
if HF_CACHE_ONLY:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

TRELLIS_ROOT = Path(os.environ.get("TRELLIS2_ROOT", "/content/TRELLIS.2")).expanduser().resolve()
TRELLIS_PACKAGE = TRELLIS_ROOT / "trellis2" / "__init__.py"
if not TRELLIS_PACKAGE.is_file():
    raise RuntimeError(
        f"TRELLIS.2 source package not found at {TRELLIS_PACKAGE}. "
        "Run the installer first or set TRELLIS2_ROOT."
    )
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils

AXIS_INDEX = {"width": 0, "height": 1, "depth": 2}


def gib(n: int | float) -> float:
    return float(n) / (1024 ** 3)


def allocated_gib() -> float:
    return gib(torch.cuda.memory_allocated())


def reserved_gib() -> float:
    return gib(torch.cuda.memory_reserved())


def parameter_gib(module: Any) -> float:
    if module is None or not hasattr(module, "parameters"):
        return 0.0
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in getattr(module, "buffers", lambda: [])():
        total += b.numel() * b.element_size()
    return gib(total)


@dataclass
class ProfileRecord:
    name: str
    seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    end_allocated_gib: float
    end_reserved_gib: float


class Profiler:
    def __init__(self) -> None:
        self.records: list[ProfileRecord] = []

    def run(self, name: str, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before_alloc = allocated_gib()
        before_reserved = reserved_gib()
        print(
            f"\n[A100 PROFILE] ▶ {name} | "
            f"start alloc={before_alloc:.2f} GiB reserved={before_reserved:.2f} GiB",
            flush=True,
        )
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        rec = ProfileRecord(
            name=name,
            seconds=seconds,
            peak_allocated_gib=gib(torch.cuda.max_memory_allocated()),
            peak_reserved_gib=gib(torch.cuda.max_memory_reserved()),
            end_allocated_gib=allocated_gib(),
            end_reserved_gib=reserved_gib(),
        )
        self.records.append(rec)
        print(
            f"[A100 PROFILE] ✓ {name} | {seconds:.1f}s | "
            f"peak alloc={rec.peak_allocated_gib:.2f} GiB | "
            f"peak reserved={rec.peak_reserved_gib:.2f} GiB | "
            f"end alloc={rec.end_allocated_gib:.2f} GiB",
            flush=True,
        )
        return result

    def to_dict(self) -> list[dict[str, Any]]:
        return [r.__dict__.copy() for r in self.records]

    def print_summary(self) -> None:
        print("\n" + "=" * 92, flush=True)
        print("A100 TRELLIS PROFILE SUMMARY", flush=True)
        print("=" * 92, flush=True)
        for r in self.records:
            print(
                f"{r.name:<38} {r.seconds:>8.1f}s | "
                f"peak alloc {r.peak_allocated_gib:>6.2f} GiB | "
                f"peak reserved {r.peak_reserved_gib:>6.2f} GiB",
                flush=True,
            )
        print("-" * 92, flush=True)
        print(f"Profiled total: {sum(r.seconds for r in self.records):.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="/content/trellis_a100_outputs")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--envmap", default="/content/TRELLIS.2/assets/hdri/forest.exr")
    parser.add_argument("--name", default="asset")
    parser.add_argument(
        "--asset-type",
        choices=["object", "character", "environment", "other"],
        default="object",
    )
    parser.add_argument(
        "--pipeline-type",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        default="1024_cascade",
    )
    parser.add_argument(
        "--residency",
        choices=["official", "smart"],
        default="smart",
        help=(
            "official = upstream low-VRAM transfers; smart = keep DINOv3 resident "
            "across both conditioning passes while retaining safe stage offload elsewhere"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--target-axis", choices=["width", "height", "depth", "longest"], required=True)
    parser.add_argument("--target-size-m", type=float, required=True)
    parser.add_argument("--decimation-target", type=int, default=500_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--skip-preview", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[Path, Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required.")
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.target_size_m <= 0:
        raise ValueError("--target-size-m must be positive")
    if args.decimation_target <= 0 or args.texture_size <= 0:
        raise ValueError("decimation/texture values must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_path, output_dir


def verify_dinov3_patch() -> None:
    path = TRELLIS_ROOT / "trellis2" / "modules" / "image_feature_extractor.py"
    text = path.read_text(encoding="utf-8")
    marker = 'encoder = getattr(self.model, "model", self.model)'
    if marker not in text:
        raise RuntimeError(
            "DINOv3 compatibility patch is missing. "
            "Rerun install_3d.sh or patch_trellis_dinov3.py before inference."
        )
    print("[PATCH] ✓ DINOv3/Transformers compatibility patch detected.", flush=True)


def get_condition(model, image: Image.Image, resolution: int) -> dict[str, torch.Tensor]:
    model.image_size = resolution
    cond = model([image])
    return {"cond": cond, "neg_cond": torch.zeros_like(cond)}


def run_profiled_inference(
    pipeline: Trellis2ImageTo3DPipeline,
    image: Image.Image,
    args: argparse.Namespace,
    profiler: Profiler,
):
    pipeline_type = args.pipeline_type
    device = torch.device("cuda")

    # Keep the official low-VRAM path for all large flow and decoder modules.
    pipeline.low_vram = True
    pipeline.cuda()

    image = profiler.run("1. image preprocessing / RMBG", lambda: pipeline.preprocess_image(image))
    torch.manual_seed(args.seed)

    if args.residency == "smart":
        def smart_conditioning():
            # DINO is used twice back-to-back for 1024-class pipelines. Keep only
            # this reusable encoder resident; all heavy 3D models stay offloaded.
            pipeline.image_cond_model.to(device)
            try:
                cond_512 = get_condition(pipeline.image_cond_model, image, 512)
                cond_1024 = (
                    get_condition(pipeline.image_cond_model, image, 1024)
                    if pipeline_type != "512"
                    else None
                )
            finally:
                pipeline.image_cond_model.cpu()
            return cond_512, cond_1024

        cond_512, cond_1024 = profiler.run(
            "2. DINO conditioning (smart resident)",
            smart_conditioning,
        )
    else:
        def official_conditioning():
            c512 = pipeline.get_cond([image], 512)
            c1024 = pipeline.get_cond([image], 1024) if pipeline_type != "512" else None
            return c512, c1024

        cond_512, cond_1024 = profiler.run(
            "2. DINO conditioning (official offload)",
            official_conditioning,
        )

    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
    coords = profiler.run(
        "3. sparse structure sampling",
        lambda: pipeline.sample_sparse_structure(cond_512, ss_res),
    )

    if pipeline_type == "512":
        shape_slat = profiler.run(
            "4. shape SLat 512",
            lambda: pipeline.sample_shape_slat(
                cond_512,
                pipeline.models["shape_slat_flow_model_512"],
                coords,
            ),
        )
        res = 512
        tex_slat = profiler.run(
            "5. texture SLat 512",
            lambda: pipeline.sample_tex_slat(
                cond_512,
                pipeline.models["tex_slat_flow_model_512"],
                shape_slat,
            ),
        )
    elif pipeline_type == "1024":
        assert cond_1024 is not None
        shape_slat = profiler.run(
            "4. shape SLat 1024",
            lambda: pipeline.sample_shape_slat(
                cond_1024,
                pipeline.models["shape_slat_flow_model_1024"],
                coords,
            ),
        )
        res = 1024
        tex_slat = profiler.run(
            "5. texture SLat 1024",
            lambda: pipeline.sample_tex_slat(
                cond_1024,
                pipeline.models["tex_slat_flow_model_1024"],
                shape_slat,
            ),
        )
    else:
        assert cond_1024 is not None
        target_res = 1024 if pipeline_type == "1024_cascade" else 1536
        shape_slat, res = profiler.run(
            f"4. shape SLat cascade → {target_res}",
            lambda: pipeline.sample_shape_slat_cascade(
                cond_512,
                cond_1024,
                pipeline.models["shape_slat_flow_model_512"],
                pipeline.models["shape_slat_flow_model_1024"],
                512,
                target_res,
                coords,
                max_num_tokens=args.max_num_tokens,
            ),
        )
        tex_slat = profiler.run(
            f"5. texture SLat {res}",
            lambda: pipeline.sample_tex_slat(
                cond_1024,
                pipeline.models["tex_slat_flow_model_1024"],
                shape_slat,
            ),
        )

    # Conditioning tensors are no longer needed. Release them before decode.
    del cond_512, cond_1024, coords
    torch.cuda.empty_cache()

    meshes = profiler.run(
        f"6. decode shape + PBR voxels ({res})",
        lambda: pipeline.decode_latent(shape_slat, tex_slat, res),
    )
    return meshes


def load_envmap(path: str):
    env_path = Path(path).expanduser().resolve()
    image = cv2.imread(str(env_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(env_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(image, dtype=torch.float32, device="cuda"))


def scale_to_contract(mesh, target_axis: str, target_size_m: float):
    before = np.asarray(mesh.extents, dtype=np.float64)
    current = float(before.max()) if target_axis == "longest" else float(before[AXIS_INDEX[target_axis]])
    if current <= 1e-9:
        raise RuntimeError(f"Invalid generated extent for {target_axis}: {before}")
    factor = float(target_size_m / current)
    mesh.apply_scale(factor)
    after = np.asarray(mesh.extents, dtype=np.float64)
    return before, after, factor


def model_size_table(pipeline: Trellis2ImageTo3DPipeline) -> dict[str, float]:
    sizes: dict[str, float] = {}
    sizes["image_cond_model"] = parameter_gib(pipeline.image_cond_model)
    sizes["rembg_model"] = parameter_gib(pipeline.rembg_model)
    for name, model in pipeline.models.items():
        sizes[name] = parameter_gib(model)
    print("\n[MODEL FOOTPRINT] CPU parameter/buffer sizes", flush=True)
    for name, size in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:<34} {size:6.2f} GiB", flush=True)
    print(f"  {'TOTAL (not simultaneous GPU use)':<34} {sum(sizes.values()):6.2f} GiB", flush=True)
    return sizes


def main() -> None:
    args = parse_args()
    input_path, output_dir = validate(args)
    verify_dinov3_patch()

    props = torch.cuda.get_device_properties(0)
    total_vram = gib(props.total_memory)
    print("=" * 92, flush=True)
    print("TRELLIS.2 A100 PROFILED RUNNER", flush=True)
    print(f"GPU: {props.name} | VRAM: {total_vram:.1f} GiB", flush=True)
    print(f"Pipeline: {args.pipeline_type} | residency: {args.residency}", flush=True)
    print(f"HF cache-only: {HF_CACHE_ONLY}", flush=True)
    print(
        "Safety policy: official low-VRAM offload remains enabled for 3D flow/decoder models. "
        "Smart mode only keeps DINOv3 resident across consecutive conditioning passes.",
        flush=True,
    )
    if total_vram < 35 and args.residency == "smart":
        print("[WARN] <35 GiB GPU detected; A100 40GB+ is recommended for smart mode.", flush=True)
    if args.pipeline_type == "1536_cascade":
        print("[WARN] 1536_cascade is expensive and may be unsuitable for a 40 GB A100.", flush=True)
    print("=" * 92, flush=True)

    profiler = Profiler()

    pipeline = profiler.run(
        "0. load TRELLIS pipeline to CPU",
        lambda: Trellis2ImageTo3DPipeline.from_pretrained(args.model),
    )
    pipeline.cuda()  # In low_vram mode this sets target device without loading all models.
    sizes = model_size_table(pipeline)

    image = Image.open(input_path)
    meshes = run_profiled_inference(pipeline, image, args, profiler)
    mesh = meshes[0]

    print(
        f"\n[INFERENCE DONE] vertices={mesh.vertices.shape[0]:,} "
        f"faces={mesh.faces.shape[0]:,}",
        flush=True,
    )

    def make_glb():
        mesh.simplify(16_777_216)
        return o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )

    glb = profiler.run(
        f"7. GLB remesh/UV/PBR bake ({args.decimation_target:,} faces, {args.texture_size}px)",
        make_glb,
    )
    before, after, factor = scale_to_contract(glb, args.target_axis, args.target_size_m)

    glb_path = output_dir / f"{args.name}.glb"
    profiler.run(
        "8. GLB serialization",
        lambda: glb.export(str(glb_path), extension_webp=False),
    )

    preview_path = output_dir / f"{args.name}_preview.mp4"
    if not args.skip_preview:
        def render_preview():
            envmap = load_envmap(args.envmap)
            frames = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
            imageio.mimsave(preview_path, frames, fps=args.preview_fps)
        profiler.run("9. optional turntable preview", render_preview)

    profile = {
        "schema_version": 1,
        "asset_name": args.name,
        "asset_type": args.asset_type,
        "gpu": {
            "name": props.name,
            "total_vram_gib": total_vram,
        },
        "trellis": {
            "model": args.model,
            "pipeline_type": args.pipeline_type,
            "residency": args.residency,
            "low_vram": True,
            "smart_residency_note": (
                "DINOv3 is kept on GPU across 512/1024 conditioning only; "
                "all large 3D flow/decoder modules retain official stage offload."
            ),
            "hf_cache_only": HF_CACHE_ONLY,
        },
        "model_footprint_gib": sizes,
        "profile": profiler.to_dict(),
        "geometry": {
            "coordinate_system": "glTF 2.0 right-handed, Y-up",
            "units": "meters",
            "target_axis": args.target_axis,
            "target_size_m": args.target_size_m,
            "normalized_extents_before_scale": before.tolist(),
            "extents_m": after.tolist(),
            "extents_cm_for_unreal_reference": (after * 100.0).tolist(),
            "scale_factor": factor,
        },
        "export": {
            "glb": str(glb_path),
            "unreal_safe_standard_textures": True,
            "decimation_target": args.decimation_target,
            "texture_size": args.texture_size,
            "preview": str(preview_path) if preview_path.is_file() else None,
        },
    }
    profile_path = output_dir / f"{args.name}_a100_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "asset_name": args.name,
        "asset_type": args.asset_type,
        "producer": {"engine": "TRELLIS.2", "model": args.model, "format": "GLB"},
        "geometry": profile["geometry"],
        "consumer_contracts": {
            "unreal_static_asset": {
                "ready": True,
                "preferred_input": glb_path.name,
                "note": "Real-world scale is baked into geometry; standard GLB textures are used."
            },
            "animation_engine": {
                "eligible": args.asset_type == "character",
                "note": "Only manually selected humanoids should enter the Animation Engine."
            },
        },
        "profiling": {
            "report": profile_path.name,
            "pipeline_type": args.pipeline_type,
            "residency": args.residency,
        },
    }
    manifest_path = output_dir / f"{args.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    profiler.print_summary()
    print(f"\n[OUTPUT] GLB: {glb_path}", flush=True)
    print(f"[OUTPUT] Manifest: {manifest_path}", flush=True)
    print(f"[OUTPUT] Profile: {profile_path}", flush=True)
    if preview_path.is_file():
        print(f"[OUTPUT] Preview: {preview_path}", flush=True)


if __name__ == "__main__":
    main()
