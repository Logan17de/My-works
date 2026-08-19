#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIA_ROOT="/content/Make-It-Animatable"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is missing. Load it from Colab Secrets or a hidden prompt first." >&2
  exit 2
fi
if [ ! -d "$MIA_ROOT" ]; then
  echo "ERROR: $MIA_ROOT does not exist. Run the Animation installer through stage 8 first." >&2
  exit 2
fi
if [ ! -x /opt/conda/bin/conda ] || [ ! -x /opt/conda/envs/mia/bin/python ]; then
  echo "ERROR: MIA conda environment is missing. Run the Animation installer through stage 8 first." >&2
  exit 2
fi

cd "$MIA_ROOT"
echo "[MIA RESUME][1/4] Installing/updating authenticated Hugging Face downloader"
conda run --no-capture-output -n mia python -m pip install --progress-bar on --upgrade huggingface_hub hf_xet

echo "[MIA RESUME][2/4] Downloading gated Mixamo templates + MIA model weights"
HF_HOME=/content/huggingface HF_XET_HIGH_PERFORMANCE=1 \
  conda run --no-capture-output -n mia python "$SCRIPT_DIR/download_mia_assets.py" \
    --mia-root "$MIA_ROOT" \
    --temp-root /tmp/mia-hf-data

echo "[MIA RESUME][3/4] Restoring FBX2glTF + validating assets"
if [ ! -s util/FBX2glTF ]; then
  wget --progress=bar:force:noscroll \
    https://github.com/facebookincubator/FBX2glTF/releases/download/v0.9.7/FBX2glTF-linux-x64 \
    -O util/FBX2glTF
fi
chmod +x util/FBX2glTF
BONE_FILE="$(find data/Mixamo -type f -name 'bones*.fbx' | head -n 1 || true)"
[ -n "$BONE_FILE" ] || { echo "ERROR: Mixamo template FBX missing" >&2; exit 1; }
[ -s "$BONE_FILE" ] || { echo "ERROR: Mixamo template FBX is empty" >&2; exit 1; }
WEIGHT_COUNT="$(find output/best/new -type f -name '*.pth' | wc -l)"
[ "$WEIGHT_COUNT" -gt 0 ] || { echo "ERROR: MIA weights missing" >&2; exit 1; }
echo "[MIA RESUME] Template: $BONE_FILE"
echo "[MIA RESUME] Weight files: $WEIGHT_COUNT"

echo "[MIA RESUME][4/4] Running MIA / Blender GPU smoke test"
conda run --no-capture-output -n mia python -c 'import torch,bpy,trimesh,pytorch3d; assert torch.cuda.is_available(), "MIA environment cannot see the NVIDIA GPU"; print("MIA smoke test: OK", flush=True); print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), flush=True); print("Blender:", bpy.app.version_string, flush=True)'

echo "[MIA RESUME] ✅ Animation Engine installation is complete from stage 9 onward."
