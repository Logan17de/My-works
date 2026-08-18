#!/usr/bin/env python3
"""Direct TRELLIS.2 image-to-3D runner for the Colab 3D Engine."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import imageio.v2 as imageio
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Reference image path")
    parser.add_argument("--output-dir", default="outputs/trellis2")
    parser.add_argument(
        "--model",
        default="microsoft/TRELLIS.2-4B",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--envmap",
        default="assets/hdri/forest.exr",
        help="HDRI used only for the preview render",
    )
    parser.add_argument("--name", default="asset")
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip turntable MP4 rendering and only export GLB",
    )
    return parser.parse_args()


def load_envmap(path: str) -> EnvMap:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"HDR environment map not found: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(image, dtype=torch.float32, device="cuda"))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 requires an NVIDIA CUDA GPU.")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    image = Image.open(input_path).convert("RGBA")
    mesh = pipeline.run(image)[0]

    # Matches the official example: keep below the nvdiffrast face-count limit
    # before rendering/export post-processing.
    mesh.simplify(16_777_216)

    if not args.skip_preview:
        envmap = load_envmap(args.envmap)
        frames = render_utils.make_pbr_vis_frames(
            render_utils.render_video(mesh, envmap=envmap)
        )
        preview_path = output_dir / f"{args.name}_preview.mp4"
        imageio.mimsave(preview_path, frames, fps=args.preview_fps)
        print(f"Preview: {preview_path}")

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
    print(f"GLB: {glb_path}")


if __name__ == "__main__":
    main()
