#!/usr/bin/env bash
set -Eeuo pipefail
ARDY_REF="693f74d13b3d04a0a22ce127ee79c929dd89756b"
MIA_REF="d60cc7e01ff8da46448e458dbf450e8967b34e77"
TOTAL=11
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
  printf '\n[ANIMATION INSTALL][%d/%d][%s] ▶ %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_STAGE"
}
action() {
  CURRENT_ACTION="$1"
  printf '[ANIMATION INSTALL][%d/%d][%s]   → %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_ACTION"
}
info() { printf '[ANIMATION INSTALL][%d/%d][%s]   %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$*"; }
trap 'code=$?; printf "\n[ANIMATION INSTALL][%d/%d][%s] ❌ FAILED\n  Stage: %s\n  Action: %s\n  Exit: %d\n" "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_STAGE" "$CURRENT_ACTION" "$code" >&2; exit "$code"' ERR

echo "============================================================"
echo " ARDY + Make-It-Animatable Colab installer"
echo " Progress is streamed live. Model/weight downloads may be large."
echo "============================================================"

stage "Installing Linux build tools"
action "Refreshing apt package index"
apt-get update
action "Installing compiler, Git-LFS, CMake, Ninja and FFmpeg"
apt-get install -y git git-lfs build-essential cmake ninja-build wget ffmpeg

stage "Preparing Conda"
if [ ! -x /opt/conda/bin/conda ]; then
  action "Downloading Miniforge"
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

stage "Downloading pinned ARDY source"
action "Removing stale ARDY checkout"
rm -rf /content/ardy
action "Cloning NVIDIA ARDY"
git clone --progress https://github.com/nv-tlabs/ardy.git /content/ardy
action "Checking out pinned ARDY revision"
git -C /content/ardy checkout "$ARDY_REF"
info "Pinned ARDY commit: $ARDY_REF"

stage "Creating ARDY environment"
action "Removing stale ardy environment if present"
conda env remove -n ardy -y >/dev/null 2>&1 || true
action "Creating Python 3.11 ARDY environment from conda-forge only"
conda create -n ardy --override-channels -c conda-forge python=3.11 pip -y
action "Upgrading ARDY pip/build helpers"
conda run --no-capture-output -n ardy python -m pip install --progress-bar on --upgrade pip setuptools wheel

stage "Installing ARDY + PyTorch dependencies"
action "Installing PyTorch 2.6.0 cu124"
conda run --no-capture-output -n ardy python -m pip install --progress-bar on torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
action "Installing ARDY package"
cd /content/ardy
conda run --no-capture-output -n ardy python -m pip install -e .
action "Installing ARDY preview dependency"
conda run --no-capture-output -n ardy python -m pip install matplotlib

stage "Running ARDY smoke test"
action "Checking ARDY skeleton import and GPU visibility"
conda run --no-capture-output -n ardy python -c 'import torch; from ardy.skeleton import CoreSkeleton27; assert len(CoreSkeleton27().bone_order_names)==27; assert torch.cuda.is_available(), "ARDY environment cannot see the NVIDIA GPU"; print("ARDY smoke test: OK", torch.__version__, torch.cuda.get_device_name(0), flush=True)'

stage "Downloading pinned Make-It-Animatable source"
action "Removing stale MIA checkout and temporary weights"
rm -rf /content/Make-It-Animatable /tmp/mia-hf-data
action "Cloning Make-It-Animatable with submodules"
git clone --progress --recursive https://github.com/jasongzy/Make-It-Animatable /content/Make-It-Animatable
action "Checking out pinned MIA revision"
git -C /content/Make-It-Animatable checkout "$MIA_REF"
action "Synchronizing MIA submodules"
git -C /content/Make-It-Animatable submodule sync --recursive
git -C /content/Make-It-Animatable submodule update --init --recursive --force --progress
info "Pinned MIA commit: $MIA_REF"

stage "Creating Make-It-Animatable environment"
action "Removing stale mia environment if present"
conda env remove -n mia -y >/dev/null 2>&1 || true
action "Creating Python 3.11 MIA environment from conda-forge only"
conda create -n mia --override-channels -c conda-forge python=3.11 pip -y
cd /content/Make-It-Animatable
action "Upgrading pip"
conda run --no-capture-output -n mia python -m pip install --progress-bar on --upgrade pip
action "Installing Make-It-Animatable requirements"
printf "gradio>=5.25,<6\n" >/tmp/mia-constraints.txt
PIP_CONSTRAINT=/tmp/mia-constraints.txt conda run --no-capture-output -n mia python -m pip install -r requirements.txt

stage "Downloading MIA templates and model weights"
action "Initializing Git-LFS"
git lfs install --skip-repo >/dev/null
mkdir -p data
action "Cloning Mixamo template dataset metadata"
GIT_LFS_SKIP_SMUDGE=1 git -C data clone --progress https://huggingface.co/datasets/jasongzy/Mixamo
action "Cloning MIA model repository metadata"
GIT_LFS_SKIP_SMUDGE=1 git clone --progress https://huggingface.co/jasongzy/Make-It-Animatable /tmp/mia-hf-data
action "Downloading Mixamo skeleton templates through Git-LFS"
git -C data/Mixamo lfs pull -I 'bones*.fbx'
action "Downloading Make-It-Animatable neural-network weights through Git-LFS"
git -C /tmp/mia-hf-data lfs pull -I 'output/best/new'
mkdir -p output/best
cp -r /tmp/mia-hf-data/output/best/new output/best/
action "Downloading FBX2glTF helper"
wget --progress=bar:force:noscroll https://github.com/facebookincubator/FBX2glTF/releases/download/v0.9.7/FBX2glTF-linux-x64 -O util/FBX2glTF
chmod +x util/FBX2glTF

stage "Validating downloaded MIA assets"
action "Checking Mixamo template and weight files"
BONE_FILE="$(find data/Mixamo -type f -name 'bones*.fbx' | head -n 1 || true)"
[ -n "$BONE_FILE" ] || { echo "ERROR: Mixamo template FBX missing" >&2; exit 1; }
! head -c 200 "$BONE_FILE" | grep -q 'git-lfs.github.com/spec' || { echo "ERROR: Mixamo FBX is still an LFS pointer" >&2; exit 1; }
find output/best/new -type f -name '*.pth' | grep -q . || { echo "ERROR: MIA weights missing" >&2; exit 1; }
! grep -RIl '^version https://git-lfs.github.com/spec/v1' output/best/new | grep -q . || { echo "ERROR: MIA weights still LFS pointers" >&2; exit 1; }
info "Template: $BONE_FILE"
info "Weight files: $(find output/best/new -type f -name '*.pth' | wc -l)"

stage "Running Make-It-Animatable / Blender smoke test"
action "Importing MIA/Blender stack and checking GPU visibility"
conda run --no-capture-output -n mia python -c 'import torch,bpy,trimesh,pytorch3d; assert torch.cuda.is_available(), "MIA environment cannot see the NVIDIA GPU"; print("MIA smoke test: OK", flush=True); print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), flush=True); print("Blender:", bpy.app.version_string, flush=True)'

printf '\n[ANIMATION INSTALL][%d/%d][%s] ✅ Installation complete. You can now run the Animation Engine.\n' "$TOTAL" "$TOTAL" "$(elapsed)"
