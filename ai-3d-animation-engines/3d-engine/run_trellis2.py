#!/usr/bin/env python3
"""Direct TRELLIS.2 image-to-3D runner with explicit Unreal I/O contracts and live progress."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils

from progress_utils import Progress

AXIS_INDEX = {"width": 0, "height": 1, "depth": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Reference image path")
    parser.add_argument("--output-dir", default="outputs/trellis2")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--envmap", default="assets/hdri/forest.exr")
    parser.add_argument("--name", default="asset")
    parser.add_argument("--asset-type", choices=["object", "character", "environment", "other"], default="object")
    parser.add_argument("--target-axis", choices=["width", "height", "depth", "longest"], required=True)
    parser.add_argument("--target-size-m", type=float, required=True)
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--skip-preview", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 requires an NVIDIA CUDA GPU.")
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if args.decimation_target <= 0:
        raise ValueError("--decimation-target must be positive")
    if args.texture_size <= 0:
        raise ValueError("--texture-size must be positive")
    if not np.isfinite(args.target_size_m) or args.target_size_m <= 0:
        raise ValueError("--target-size-m must be a finite positive number")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_path, output_dir


def load_envmap(path: str) -> EnvMap:
    env_path = Path(path).expanduser().resolve()
    image = cv2.imread(str(env_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"HDR environment map not found/readable: {env_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(image, dtype=torch.float32, device="cuda"))


def material_contract(mesh) -> dict:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    alpha_mode = getattr(material, "alphaMode", None) if material is not None else None
    has_base_color = bool(material is not None and getattr(material, "baseColorTexture", None) is not None)
    has_mr = bool(material is not None and getattr(material, "metallicRoughnessTexture", None) is not None)
    has_alpha_channel = False
    if has_base_color:
        try:
            arr = np.asarray(material.baseColorTexture)
            has_alpha_channel = arr.ndim == 3 and arr.shape[-1] == 4
        except Exception:
            pass
    return {
        "pbr": {
            "base_color_texture": has_base_color,
            "metallic_roughness_texture": has_mr,
            "alpha_channel_present": has_alpha_channel,
            "alpha_mode": alpha_mode or "OPAQUE",
        },
        "texture_compatibility": {
            "webp_required_extension": False,
            "note": "Standard GLB textures are used; EXT_texture_webp is intentionally not required.",
        },
    }


def scale_to_contract(mesh, target_axis: str, target_size_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    before = np.asarray(mesh.extents, dtype=np.float64)
    if before.shape != (3,) or not np.isfinite(before).all() or np.any(before <= 0):
        raise RuntimeError(f"Invalid generated mesh extents: {before}")
    current = float(before.max()) if target_axis == "longest" else float(before[AXIS_INDEX[target_axis]])
    if current <= 1e-9:
        raise RuntimeError(f"Generated mesh has zero extent along target axis {target_axis!r}")
    scale_factor = float(target_size_m / current)
    mesh.apply_scale(scale_factor)
    after = np.asarray(mesh.extents, dtype=np.float64)
    actual = float(after.max()) if target_axis == "longest" else float(after[AXIS_INDEX[target_axis]])
    if not np.isclose(actual, target_size_m, rtol=1e-5, atol=1e-7):
        raise RuntimeError(f"Scale contract failed: requested {target_size_m} m on {target_axis}, got {actual} m")
    return before, after, scale_factor


def main() -> None:
    p = Progress("3D GENERATION", 7)
    args = parse_args()

    p.step("Validating input, output path and GPU")
    input_path, output_dir = validate_args(args)
    props = torch.cuda.get_device_properties(0)
    p.info(f"Input: {input_path.name}")
    p.info(f"Asset type: {args.asset_type} | target {args.target_axis}={args.target_size_m:g} m")
    p.info(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GiB")

    p.step(f"Loading TRELLIS.2 model: {args.model}")
    with p.heartbeat("model download/load", every=30):
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
        pipeline.cuda()
    p.ok("Model loaded on GPU")

    p.step("Generating high-resolution 3D representation from the image")
    image = Image.open(input_path)
    p.info(f"Reference image: {image.width}×{image.height} px, mode={image.mode}")
    with p.heartbeat("TRELLIS inference", every=20):
        mesh = pipeline.run(image)[0]
    p.ok(f"Generation finished: {mesh.vertices.shape[0]:,} vertices, {mesh.faces.shape[0]:,} faces")

    p.step("Preparing mesh and baking PBR textures into GLB")
    mesh.simplify(16_777_216)
    p.info(f"Decimation target: {args.decimation_target:,} faces | texture: {args.texture_size}×{args.texture_size}")
    with p.heartbeat("O-Voxel remesh / UV unwrap / texture bake", every=20):
        glb = o_voxel.postprocess.to_glb(
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

    p.step("Applying explicit real-world scale")
    normalized_extents, extents_m, scale_factor = scale_to_contract(glb, args.target_axis, args.target_size_m)
    p.info(f"Scale factor from TRELLIS normalized space: {scale_factor:.6g}")
    p.info(f"Final size: W={extents_m[0]:.3f} m, H={extents_m[1]:.3f} m, D={extents_m[2]:.3f} m")

    p.step("Exporting Unreal-safe GLB and asset manifest")
    glb_path = output_dir / f"{args.name}.glb"
    with p.heartbeat("GLB serialization / texture embedding", every=20):
        glb.export(str(glb_path), extension_webp=False)
    if not glb_path.is_file() or glb_path.stat().st_size == 0:
        raise RuntimeError(f"GLB export did not create a valid file: {glb_path}")

    contract = {
        "schema_version": 1,
        "asset_name": args.name,
        "asset_type": args.asset_type,
        "source_image": input_path.name,
        "producer": {"engine": "TRELLIS.2", "model": args.model, "format": "GLB"},
        "geometry": {
            "coordinate_system": "glTF 2.0 right-handed, Y-up",
            "units": "meters",
            "normalized_extents_before_scale": normalized_extents.tolist(),
            "extents_m": extents_m.tolist(),
            "extents_cm_for_unreal_reference": (extents_m * 100.0).tolist(),
            "scale_factor_from_trellis_normalized_space": scale_factor,
            "target_axis": args.target_axis,
            "target_size_m": args.target_size_m,
            "scale_resolved": True,
        },
        "consumer_contracts": {
            "unreal_static_asset": {"ready": True, "preferred_input": glb_path.name, "note": "Real-world scale is baked into geometry; standard GLB textures are used."},
            "animation_engine": {"eligible": args.asset_type == "character", "note": "Only manually selected humanoids should enter the Animation Engine."},
        },
        **material_contract(glb),
    }
    manifest_path = output_dir / f"{args.name}_manifest.json"
    manifest_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    p.ok(f"GLB: {glb_path} ({glb_path.stat().st_size / 1024**2:.1f} MiB)")
    p.ok(f"Manifest: {manifest_path}")

    p.step("Rendering optional turntable preview")
    if args.skip_preview:
        p.info("Preview skipped by --skip-preview")
    else:
        preview_path = output_dir / f"{args.name}_preview.mp4"
        try:
            with p.heartbeat("preview render", every=20):
                envmap = load_envmap(args.envmap)
                frames = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
                imageio.mimsave(preview_path, frames, fps=args.preview_fps)
            p.ok(f"Preview: {preview_path}")
        except Exception as exc:
            p.warn(f"Preview failed, but the GLB is valid: {exc}")

    p.done("3D Engine finished successfully")


if __name__ == "__main__":
    main()
