#!/usr/bin/env bash
set -Eeuo pipefail

TRELLIS_REF="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
TOTAL=9
STEP=0
START_TS=$(date +%s)
CURRENT_STAGE="starting"
CURRENT_ACTION="starting"

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
echo " Progress is streamed live; long CUDA builds may take minutes."
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
info "Environment creation below uses --override-channels -c conda-forge, so user/default Anaconda channels cannot interfere."

stage "Downloading pinned TRELLIS.2 source"
action "Removing stale TRELLIS checkout and extension build folder"
rm -rf /content/TRELLIS.2 /tmp/extensions
action "Cloning Microsoft TRELLIS.2"
git clone --progress --recursive https://github.com/microsoft/TRELLIS.2.git /content/TRELLIS.2
action "Checking out pinned TRELLIS revision"
git -C /content/TRELLIS.2 checkout "$TRELLIS_REF"
action "Synchronizing and downloading submodules"
git -C /content/TRELLIS.2 submodule sync --recursive
git -C /content/TRELLIS.2 submodule update --init --recursive --force --progress
info "Pinned commit: $TRELLIS_REF"

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

stage "Building TRELLIS CUDA/native extensions"
info "This is normally the slowest stage. Individual compiler output appears below."
action "Building flash-attn, nvdiffrast, nvdiffrec, CuMesh, O-Voxel and FlexGEMM"
. ./setup.sh --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

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

printf '\n[TRELLIS INSTALL][%d/%d][%s] ✅ Installation complete. You can now run the 3D generation cell.\n' "$TOTAL" "$TOTAL" "$(elapsed)"
