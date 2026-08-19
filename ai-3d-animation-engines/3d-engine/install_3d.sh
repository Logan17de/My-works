#!/usr/bin/env bash
set -euo pipefail

TRELLIS_REF="75fbf0183001ed9876c8dbb35de6b68552ee08bd"

apt-get update -qq
apt-get install -y -qq git git-lfs build-essential cmake ninja-build wget ffmpeg sudo \
  libjpeg-dev libgl1-mesa-dev libegl1-mesa-dev

if [ ! -x /opt/conda/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
fi
source /opt/conda/etc/profile.d/conda.sh

rm -rf /content/TRELLIS.2 /tmp/extensions
git clone -q --recursive https://github.com/microsoft/TRELLIS.2.git /content/TRELLIS.2
git -C /content/TRELLIS.2 checkout -q "$TRELLIS_REF"
git -C /content/TRELLIS.2 submodule sync --recursive
git -C /content/TRELLIS.2 submodule update --init --recursive --force

conda env remove -n trellis2 -y >/dev/null 2>&1 || true
conda create -n trellis2 python=3.10 -y -q
conda activate trellis2
python -m pip install -q --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

if [ -x /usr/local/cuda-12.4/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda-12.4
else
  conda install -y -q -c nvidia/label/cuda-12.4.1 cuda-toolkit
  export CUDA_HOME="$CONDA_PREFIX"
fi
export PATH="$CUDA_HOME/bin:$PATH"

cd /content/TRELLIS.2
. ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

python - <<'PY'
import torch, o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
assert torch.cuda.is_available()
print("TRELLIS.2 smoke test: OK", torch.__version__, torch.version.cuda)
PY
