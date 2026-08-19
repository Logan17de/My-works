#!/usr/bin/env bash
set -euo pipefail
ARDY_REF="693f74d13b3d04a0a22ce127ee79c929dd89756b"
MIA_REF="d60cc7e01ff8da46448e458dbf450e8967b34e77"

apt-get update -qq
apt-get install -y -qq git git-lfs build-essential cmake ninja-build wget ffmpeg
if [ ! -x /opt/conda/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
fi
source /opt/conda/etc/profile.d/conda.sh

rm -rf /content/ardy
git clone -q https://github.com/nv-tlabs/ardy.git /content/ardy
git -C /content/ardy checkout -q "$ARDY_REF"
conda env remove -n ardy -y >/dev/null 2>&1 || true
conda create -n ardy python=3.11 -y -q
conda run -n ardy python -m pip install -q --upgrade pip setuptools wheel
conda run -n ardy python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
cd /content/ardy
conda run -n ardy python -m pip install -e .
conda run -n ardy python -m pip install matplotlib
conda run -n ardy python -c 'import torch; from ardy.skeleton import CoreSkeleton27; assert len(CoreSkeleton27().bone_order_names)==27; print("ARDY smoke test: OK",torch.__version__)'

rm -rf /content/Make-It-Animatable /tmp/mia-hf-data
git clone -q --recursive https://github.com/jasongzy/Make-It-Animatable /content/Make-It-Animatable
git -C /content/Make-It-Animatable checkout -q "$MIA_REF"
git -C /content/Make-It-Animatable submodule sync --recursive
git -C /content/Make-It-Animatable submodule update --init --recursive --force
conda env remove -n mia -y >/dev/null 2>&1 || true
conda create -n mia python=3.11 -y -q
cd /content/Make-It-Animatable
conda run -n mia python -m pip install -q --upgrade pip
printf "gradio>=5.25,<6\n" >/tmp/mia-constraints.txt
PIP_CONSTRAINT=/tmp/mia-constraints.txt conda run -n mia python -m pip install -r requirements.txt

git lfs install --skip-repo >/dev/null
mkdir -p data
GIT_LFS_SKIP_SMUDGE=1 git -C data clone -q https://huggingface.co/datasets/jasongzy/Mixamo
GIT_LFS_SKIP_SMUDGE=1 git clone -q https://huggingface.co/jasongzy/Make-It-Animatable /tmp/mia-hf-data
git -C data/Mixamo lfs pull -I 'bones*.fbx'
git -C /tmp/mia-hf-data lfs pull -I 'output/best/new'
mkdir -p output/best
cp -r /tmp/mia-hf-data/output/best/new output/best/
wget -q https://github.com/facebookincubator/FBX2glTF/releases/download/v0.9.7/FBX2glTF-linux-x64 -O util/FBX2glTF
chmod +x util/FBX2glTF

BONE_FILE="$(find data/Mixamo -type f -name 'bones*.fbx' | head -n 1 || true)"
[ -n "$BONE_FILE" ] || { echo "ERROR: Mixamo template FBX missing" >&2; exit 1; }
! head -c 200 "$BONE_FILE" | grep -q 'git-lfs.github.com/spec' || { echo "ERROR: Mixamo FBX is still an LFS pointer" >&2; exit 1; }
find output/best/new -type f -name '*.pth' | grep -q . || { echo "ERROR: MIA weights missing" >&2; exit 1; }
! grep -RIl '^version https://git-lfs.github.com/spec/v1' output/best/new | grep -q . || { echo "ERROR: MIA weights still LFS pointers" >&2; exit 1; }
conda run -n mia python -c 'import torch,bpy,trimesh,pytorch3d; print("MIA smoke test: OK",torch.__version__,bpy.app.version_string)'
