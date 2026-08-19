#!/usr/bin/env python3
"""Direct TRELLIS.2 image-to-3D runner for the Colab 3D Engine."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import imageio.v2 as imageio
import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Reference image path")
    parser.add_argument("--output-dir", default="outputs/trellis2")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--envmap", default="assets/hdri/forest.exr")
    parser.add_argument("--name", default="asset")
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


def main() -> None:
    args = parse_args()
    input_path, output_dir = validate_args(args)

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GiB")

    print(f"Loading TRELLIS.2 model: {args.model}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    # Follow the upstream example closely: let TRELLIS.2 handle the image mode.
    image = Image.open(input_path)
    mesh = pipeline.run(image)[0]
    mesh.simplify(16_777_216)  # nvdiffrast face-count limit from upstream example.

    # Export the valuable artifact first. Preview rendering is best-effort and
    # must never destroy an otherwise successful 3D generation job.
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
    glb_path = output_dir / f"{args.name}.glb"
    glb.export(str(glb_path), extension_webp=True)
    if not glb_path.is_file() or glb_path.stat().st_size == 0:
        raise RuntimeError(f"GLB export did not create a valid file: {glb_path}")
    print(f"GLB: {glb_path}")

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
