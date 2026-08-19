#!/usr/bin/env bash
set -Eeuo pipefail

TRELLIS_REF="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
TOTAL=9
STEP=0
START_TS=$(date +%s)
CURRENT_STAGE="starting"

elapsed() {
  local now diff h m s
  now=$(date +%s); diff=$((now-START_TS)); h=$((diff/3600)); m=$(((diff%3600)/60)); s=$((diff%60))
  if (( h > 0 )); then printf '%02d:%02d:%02d' "$h" "$m" "$s"; else printf '%02d:%02d' "$m" "$s"; fi
}
stage() {
  STEP=$((STEP+1)); CURRENT_STAGE="$1"
  printf '\n[TRELLIS INSTALL][%d/%d][%s] ▶ %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_STAGE"
}
info() { printf '[TRELLIS INSTALL][%d/%d][%s]   %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$*"; }
trap 'code=$?; printf "\n[TRELLIS INSTALL][%d/%d][%s] ❌ FAILED during: %s (exit %d)\n" "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_STAGE" "$code" >&2; exit "$code"' ERR

echo "============================================================"
echo " TRELLIS.2 Colab installer"
echo " Progress is streamed live; long CUDA builds may take minutes."
echo "============================================================"

stage "Installing Linux build tools"
apt-get update
apt-get install -y git git-lfs build-essential cmake ninja-build wget ffmpeg sudo \
  libjpeg-dev libgl1-mesa-dev libegl1-mesa-dev

stage "Preparing Miniconda"
if [ ! -x /opt/conda/bin/conda ]; then
  info "Miniconda not found; downloading installer..."
  wget --progress=bar:force:noscroll https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
else
  info "Miniconda already present; reusing it."
fi
source /opt/conda/etc/profile.d/conda.sh

stage "Downloading pinned TRELLIS.2 source"
rm -rf /content/TRELLIS.2 /tmp/extensions
git clone --progress --recursive https://github.com/microsoft/TRELLIS.2.git /content/TRELLIS.2
git -C /content/TRELLIS.2 checkout "$TRELLIS_REF"
git -C /content/TRELLIS.2 submodule sync --recursive
git -C /content/TRELLIS.2 submodule update --init --recursive --force --progress
info "Pinned commit: $TRELLIS_REF"

stage "Creating isolated trellis2 environment"
conda env remove -n trellis2 -y >/dev/null 2>&1 || true
conda create -n trellis2 python=3.10 -y
conda activate trellis2
python -m pip install --upgrade pip setuptools wheel packaging ninja

stage "Installing PyTorch + CUDA 12.4 runtime"
python -m pip install --progress-bar on torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda, flush=True)
PY

stage "Preparing CUDA 12.4 compiler toolkit"
if [ -x /usr/local/cuda-12.4/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda-12.4
  info "Using Colab system CUDA toolkit: $CUDA_HOME"
else
  info "System CUDA 12.4 compiler not found; installing toolkit into Conda environment."
  conda install -y -c nvidia/label/cuda-12.4.1 cuda-toolkit
  export CUDA_HOME="$CONDA_PREFIX"
fi
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version | tail -n 1

stage "Installing TRELLIS Python dependencies"
cd /content/TRELLIS.2
info "Running upstream setup: --basic"
. ./setup.sh --basic

stage "Building TRELLIS CUDA/native extensions"
info "This is normally the slowest stage. Individual compilers will print below."
info "Building: flash-attn, nvdiffrast, nvdiffrec, cumesh, o-voxel, flexgemm"
. ./setup.sh --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

stage "Running GPU/import smoke test"
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
