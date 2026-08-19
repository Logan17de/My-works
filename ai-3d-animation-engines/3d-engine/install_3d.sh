#!/usr/bin/env bash
set -Eeuo pipefail

TRELLIS_REF="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
NVDIFFRAST_REF="253ac4fcea7de5f396371124af597e6cc957bfae"
NVDIFFREC_REF="b296927cc7fd01c2ac1087c8065c4d7248f72da4"
CUMESH_REF="12289e1062f0603f2f0d0771b02e1395d247f26f"
FLEXGEMM_REF="6dd94a859c26ee8246888502eada3dd8ad85532e"
TOTAL=9
STEP=0
START_TS=$(date +%s)
CURRENT_STAGE="starting"
CURRENT_ACTION="starting"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ENGINE_ROOT/cache_common.sh"
cache_init

elapsed() {
  local now diff h m s
  now=$(date +%s); diff=$((now-START_TS)); h=$((diff/3600)); m=$(((diff%3600)/60)); s=$((diff%60))
  if (( h > 0 )); then printf '%02d:%02d:%02d' "$h" "$m" "$s"; else printf '%02d:%02d' "$m" "$s"; fi
}
stage() {
  STEP=$((STEP+1)); CURRENT_STAGE="$1"; CURRENT_ACTION="$1"
  printf '\n[TRELLIS INSTALL][%d/%d][%s] ▶ %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_STAGE"
}
action() {
  CURRENT_ACTION="$1"
  printf '[TRELLIS INSTALL][%d/%d][%s]   → %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_ACTION"
}
info() { printf '[TRELLIS INSTALL][%d/%d][%s]   %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$*"; }
trap 'code=$?; printf "\n[TRELLIS INSTALL][%d/%d][%s] ❌ FAILED\n  Stage: %s\n  Action: %s\n  Exit: %d\n" "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_STAGE" "$CURRENT_ACTION" "$code" >&2; exit "$code"' ERR

echo "============================================================"
echo " TRELLIS.2 Colab installer"
echo " Fresh runtime + optional Google Drive source/build cache"
echo " Model weights are NOT stored in Drive by this installer."
echo "============================================================"

stage "Installing Linux build tools"
action "Refreshing apt package index"
apt-get update
action "Installing compiler, Git, FFmpeg and graphics headers"
apt-get install -y git git-lfs build-essential cmake ninja-build wget ffmpeg sudo \
  libjpeg-dev libgl1-mesa-dev libegl1-mesa-dev

stage "Preparing Conda"
if [ ! -x /opt/conda/bin/conda ]; then
  action "Downloading Miniforge (conda-forge-first Conda distribution)"
  wget --progress=bar:force:noscroll \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    -O /tmp/miniforge.sh
  action "Installing Miniforge to /opt/conda"
  bash /tmp/miniforge.sh -b -p /opt/conda
else
  action "Reusing existing /opt/conda installation"
fi
source /opt/conda/etc/profile.d/conda.sh
info "Conda: $(conda --version)"
info "Environment creation uses conda-forge with --override-channels."

stage "Restoring/downloading pinned TRELLIS.2 source"
action "Checking Google Drive source cache before GitHub"
rm -rf /tmp/extensions
cache_git_repo \
  "TRELLIS2" \
  "https://github.com/microsoft/TRELLIS.2.git" \
  "$TRELLIS_REF" \
  "/content/TRELLIS.2" \
  "recursive"
info "Pinned TRELLIS commit: $TRELLIS_REF"

stage "Creating isolated trellis2 environment"
action "Removing stale trellis2 environment if present"
conda env remove -n trellis2 -y >/dev/null 2>&1 || true
action "Creating Python 3.10 environment from conda-forge only"
conda create -n trellis2 --override-channels -c conda-forge python=3.10 pip -y
action "Activating trellis2 environment"
conda activate trellis2
info "Python executable: $(command -v python)"
info "Python version: $(python --version 2>&1)"
action "Upgrading pip/build helpers"
python -m pip install --progress-bar on --upgrade pip setuptools wheel packaging ninja

stage "Installing PyTorch + CUDA 12.4 runtime"
action "Downloading PyTorch 2.6.0 / torchvision 0.21.0 cu124 wheels"
python -m pip install --progress-bar on torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
action "Checking PyTorch CUDA runtime metadata"
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda, flush=True)
PY

stage "Preparing CUDA 12.4 compiler toolkit"
if [ -x /usr/local/cuda-12.4/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda-12.4
  action "Using Colab system CUDA 12.4 toolkit"
else
  action "System CUDA 12.4 compiler not found; installing CUDA toolkit 12.4 into trellis2"
  conda install -y --override-channels \
    -c nvidia/label/cuda-12.4.1 -c conda-forge cuda-toolkit
  export CUDA_HOME="$CONDA_PREFIX"
fi
export PATH="$CUDA_HOME/bin:$PATH"
action "Verifying nvcc"
"$CUDA_HOME/bin/nvcc" --version | tail -n 1
info "CUDA_HOME=$CUDA_HOME"

stage "Installing TRELLIS Python dependencies"
cd /content/TRELLIS.2
action "Running upstream setup.sh --basic"
. ./setup.sh --basic

stage "Restoring/building TRELLIS CUDA/native extensions"
info "Drive cache stores compiled wheels keyed by Python/PyTorch/CUDA/GPU architecture."
action "Computing binary compatibility key"
GPU_CC="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}{b}")')"
PY_TAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
TORCH_TAG="$(python -c 'import torch; print(torch.__version__.split("+")[0].replace(".", ""))')"
WHEEL_KEY="trellis-${TRELLIS_REF:0:8}-${PY_TAG}-torch${TORCH_TAG}-cu124-sm${GPU_CC}"
export TORCH_CUDA_ARCH_LIST="${GPU_CC:0:1}.${GPU_CC:1:1}"
info "Compiled-wheel cache key: $WHEEL_KEY"
info "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

mkdir -p /tmp/extensions

action "flash-attn: use cached wheel or build once"
cache_install_or_build_wheel "flash-attn" "$WHEEL_KEY" "flash-attn==2.7.3" "no-deps"

action "nvdiffrast: restore source only on cache miss, then use/build wheel"
cache_git_repo "nvdiffrast" "https://github.com/NVlabs/nvdiffrast.git" "$NVDIFFRAST_REF" "/tmp/extensions/nvdiffrast" "plain"
cache_install_or_build_wheel "nvdiffrast" "$WHEEL_KEY" "/tmp/extensions/nvdiffrast" "no-deps"

action "nvdiffrec renderutils: restore source only on cache miss, then use/build wheel"
cache_git_repo "nvdiffrec" "https://github.com/JeffreyXiang/nvdiffrec.git" "$NVDIFFREC_REF" "/tmp/extensions/nvdiffrec" "plain"
cache_install_or_build_wheel "nvdiffrec" "$WHEEL_KEY" "/tmp/extensions/nvdiffrec" "no-deps"

action "CuMesh: restore source only on cache miss, then use/build wheel"
cache_git_repo "CuMesh" "https://github.com/JeffreyXiang/CuMesh.git" "$CUMESH_REF" "/tmp/extensions/CuMesh" "recursive"
cache_install_or_build_wheel "cumesh" "$WHEEL_KEY" "/tmp/extensions/CuMesh" "no-deps"

action "O-Voxel: use cached wheel or build from pinned TRELLIS source"
cache_install_or_build_wheel "o-voxel" "$WHEEL_KEY" "/content/TRELLIS.2/o-voxel" "no-deps"

action "FlexGEMM: restore source only on cache miss, then use/build wheel"
cache_git_repo "FlexGEMM" "https://github.com/JeffreyXiang/FlexGEMM.git" "$FLEXGEMM_REF" "/tmp/extensions/FlexGEMM" "recursive"
cache_install_or_build_wheel "flexgemm" "$WHEEL_KEY" "/tmp/extensions/FlexGEMM" "no-deps"

stage "Running GPU/import smoke test"
action "Importing TRELLIS/O-Voxel and checking CUDA visibility"
python - <<'PY'
import shutil
import torch, o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the NVIDIA GPU")
print("TRELLIS.2 import smoke test: OK", flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
print("Free disk (GiB):", round(shutil.disk_usage('/content').free/1024**3, 1), flush=True)
PY

printf '\n[TRELLIS INSTALL][%d/%d][%s] ✅ Installation complete.\n' "$TOTAL" "$TOTAL" "$(elapsed)"
if [ "$CACHE_ENABLED" -eq 1 ]; then
  printf '[CACHE] Persistent sources/wheels are in: %s\n' "$ENGINE_CACHE_ROOT"
fi
printf 'You can now run the 3D generation cell.\n'
