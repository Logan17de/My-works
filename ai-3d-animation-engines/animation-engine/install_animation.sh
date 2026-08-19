#!/usr/bin/env bash
set -Eeuo pipefail
ARDY_REF="693f74d13b3d04a0a22ce127ee79c929dd89756b"
MIA_REF="d60cc7e01ff8da46448e458dbf450e8967b34e77"
TOTAL=11
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
  printf '\n[ANIMATION INSTALL][%d/%d][%s] ▶ %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_STAGE"
}
info() { printf '[ANIMATION INSTALL][%d/%d][%s]   %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$*"; }
trap 'code=$?; printf "\n[ANIMATION INSTALL][%d/%d][%s] ❌ FAILED during: %s (exit %d)\n" "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_STAGE" "$code" >&2; exit "$code"' ERR

echo "============================================================"
echo " ARDY + Make-It-Animatable Colab installer"
echo " Progress is streamed live. Model/weight downloads may be large."
echo "============================================================"

stage "Installing Linux build tools"
apt-get update
apt-get install -y git git-lfs build-essential cmake ninja-build wget ffmpeg

stage "Preparing Miniconda"
if [ ! -x /opt/conda/bin/conda ]; then
  info "Miniconda not found; downloading installer..."
  wget --progress=bar:force:noscroll https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
else
  info "Miniconda already present; reusing it."
fi
source /opt/conda/etc/profile.d/conda.sh

stage "Downloading pinned ARDY source"
rm -rf /content/ardy
git clone --progress https://github.com/nv-tlabs/ardy.git /content/ardy
git -C /content/ardy checkout "$ARDY_REF"
info "Pinned ARDY commit: $ARDY_REF"

stage "Creating ARDY environment"
conda env remove -n ardy -y >/dev/null 2>&1 || true
conda create -n ardy python=3.11 -y
conda run --no-capture-output -n ardy python -m pip install --upgrade pip setuptools wheel

stage "Installing ARDY + PyTorch dependencies"
conda run --no-capture-output -n ardy python -m pip install --progress-bar on torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
cd /content/ardy
conda run --no-capture-output -n ardy python -m pip install -e .
conda run --no-capture-output -n ardy python -m pip install matplotlib

stage "Running ARDY smoke test"
conda run --no-capture-output -n ardy python - <<'PY'
import torch
from ardy.skeleton import CoreSkeleton27
assert len(CoreSkeleton27().bone_order_names) == 27
if not torch.cuda.is_available():
    raise RuntimeError("ARDY environment cannot see the NVIDIA GPU")
print("ARDY smoke test: OK", torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY

stage "Downloading pinned Make-It-Animatable source"
rm -rf /content/Make-It-Animatable /tmp/mia-hf-data
git clone --progress --recursive https://github.com/jasongzy/Make-It-Animatable /content/Make-It-Animatable
git -C /content/Make-It-Animatable checkout "$MIA_REF"
git -C /content/Make-It-Animatable submodule sync --recursive
git -C /content/Make-It-Animatable submodule update --init --recursive --force --progress
info "Pinned MIA commit: $MIA_REF"

stage "Creating Make-It-Animatable environment"
conda env remove -n mia -y >/dev/null 2>&1 || true
conda create -n mia python=3.11 -y
cd /content/Make-It-Animatable
conda run --no-capture-output -n mia python -m pip install --upgrade pip
printf "gradio>=5.25,<6\n" >/tmp/mia-constraints.txt
PIP_CONSTRAINT=/tmp/mia-constraints.txt conda run --no-capture-output -n mia python -m pip install -r requirements.txt

stage "Downloading MIA templates and model weights"
git lfs install --skip-repo >/dev/null
mkdir -p data
GIT_LFS_SKIP_SMUDGE=1 git -C data clone --progress https://huggingface.co/datasets/jasongzy/Mixamo
GIT_LFS_SKIP_SMUDGE=1 git clone --progress https://huggingface.co/jasongzy/Make-It-Animatable /tmp/mia-hf-data
info "Downloading Mixamo skeleton templates through Git LFS..."
git -C data/Mixamo lfs pull -I 'bones*.fbx'
info "Downloading Make-It-Animatable neural-network weights through Git LFS..."
git -C /tmp/mia-hf-data lfs pull -I 'output/best/new'
mkdir -p output/best
cp -r /tmp/mia-hf-data/output/best/new output/best/
info "Downloading FBX2glTF helper..."
wget --progress=bar:force:noscroll https://github.com/facebookincubator/FBX2glTF/releases/download/v0.9.7/FBX2glTF-linux-x64 -O util/FBX2glTF
chmod +x util/FBX2glTF

stage "Validating downloaded MIA assets"
BONE_FILE="$(find data/Mixamo -type f -name 'bones*.fbx' | head -n 1 || true)"
[ -n "$BONE_FILE" ] || { echo "ERROR: Mixamo template FBX missing" >&2; exit 1; }
! head -c 200 "$BONE_FILE" | grep -q 'git-lfs.github.com/spec' || { echo "ERROR: Mixamo FBX is still an LFS pointer" >&2; exit 1; }
find output/best/new -type f -name '*.pth' | grep -q . || { echo "ERROR: MIA weights missing" >&2; exit 1; }
! grep -RIl '^version https://git-lfs.github.com/spec/v1' output/best/new | grep -q . || { echo "ERROR: MIA weights still LFS pointers" >&2; exit 1; }
info "Template: $BONE_FILE"
info "Weight files: $(find output/best/new -type f -name '*.pth' | wc -l)"

stage "Running Make-It-Animatable / Blender smoke test"
conda run --no-capture-output -n mia python - <<'PY'
import torch, bpy, trimesh, pytorch3d
if not torch.cuda.is_available():
    raise RuntimeError("MIA environment cannot see the NVIDIA GPU")
print("MIA smoke test: OK", flush=True)
print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), flush=True)
print("Blender:", bpy.app.version_string, flush=True)
PY

printf '\n[ANIMATION INSTALL][%d/%d][%s] ✅ Installation complete. You can now run the Animation Engine.\n' "$TOTAL" "$TOTAL" "$(elapsed)"
