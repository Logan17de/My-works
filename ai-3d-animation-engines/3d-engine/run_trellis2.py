#!/usr/bin/env python3
"""Direct TRELLIS.2 image-to-3D runner with explicit Unreal I/O contracts."""

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


AXIS_INDEX = {
    "width": 0,   # glTF X
    "height": 1,  # glTF Y (up)
    "depth": 2,   # glTF Z
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Reference image path")
    parser.add_argument("--output-dir", default="outputs/trellis2")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--envmap", default="assets/hdri/forest.exr")
    parser.add_argument("--name", default="asset")
    parser.add_argument("--asset-type", choices=["object", "character", "environment", "other"], default="object")
    parser.add_argument("--target-axis", choices=["width", "height", "depth", "longest"], required=True)
    parser.add_argument(
        "--target-size-m",
        type=float,
        required=True,
        help="Real-world size in meters along --target-axis. This makes scale explicit before Unreal import.",
    )
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

    if target_axis == "longest":
        current = float(before.max())
    else:
        current = float(before[AXIS_INDEX[target_axis]])
    if current <= 1e-9:
        raise RuntimeError(f"Generated mesh has zero extent along target axis {target_axis!r}")

    scale_factor = float(target_size_m / current)
    mesh.apply_scale(scale_factor)
    after = np.asarray(mesh.extents, dtype=np.float64)

    actual = float(after.max()) if target_axis == "longest" else float(after[AXIS_INDEX[target_axis]])
    if not np.isclose(actual, target_size_m, rtol=1e-5, atol=1e-7):
        raise RuntimeError(
            f"Scale contract failed: requested {target_size_m} m on {target_axis}, got {actual} m"
        )
    return before, after, scale_factor


def main() -> None:
    args = parse_args()
    input_path, output_dir = validate_args(args)

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GiB")

    print(f"Loading TRELLIS.2 model: {args.model}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    image = Image.open(input_path)
    mesh = pipeline.run(image)[0]
    mesh.simplify(16_777_216)  # nvdiffrast face-count limit from upstream example.

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

    normalized_extents, extents_m, scale_factor = scale_to_contract(
        glb, args.target_axis, args.target_size_m
    )

    # Do not require EXT_texture_webp: keep the GLB portable to Unreal and other core glTF consumers.
    glb_path = output_dir / f"{args.name}.glb"
    glb.export(str(glb_path), extension_webp=False)
    if not glb_path.is_file() or glb_path.stat().st_size == 0:
        raise RuntimeError(f"GLB export did not create a valid file: {glb_path}")

    contract = {
        "schema_version": 1,
        "asset_name": args.name,
        "asset_type": args.asset_type,
        "source_image": input_path.name,
        "producer": {
            "engine": "TRELLIS.2",
            "model": args.model,
            "format": "GLB",
        },
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
            "unreal_static_asset": {
                "ready": True,
                "preferred_input": glb_path.name,
                "note": "Real-world scale is baked into geometry; standard GLB textures are used.",
            },
            "animation_engine": {
                "eligible": args.asset_type == "character",
                "note": "Only manually selected humanoids should enter the Animation Engine.",
            },
        },
        **material_contract(glb),
    }

    manifest_path = output_dir / f"{args.name}_manifest.json"
    manifest_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    print(f"GLB: {glb_path}")
    print(f"Manifest: {manifest_path}")
    print(
        "Final extents (m): "
        f"width={extents_m[0]:.4f}, height={extents_m[1]:.4f}, depth={extents_m[2]:.4f}"
    )

    if args.skip_preview:
        return

    preview_path = output_dir / f"{args.name}_preview.mp4"
    try:
        envmap = load_envmap(args.envmap)
        frames = render_utils.make_pbr_vis_frames(
            render_utils.render_video(mesh, envmap=envmap)
        )
        imageio.mimsave(preview_path, frames, fps=args.preview_fps)
        print(f"Preview: {preview_path}")
    except Exception as exc:
        print(f"WARNING: preview rendering failed, but GLB export succeeded: {exc}")


if __name__ == "__main__":
    main()
