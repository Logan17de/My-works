#!/usr/bin/env bash
set -Eeuo pipefail
ARDY_REF="693f74d13b3d04a0a22ce127ee79c929dd89756b"
MIA_REF="d60cc7e01ff8da46448e458dbf450e8967b34e77"
TOTAL=11
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
echo " Fresh runtime + optional Google Drive source/build cache"
echo " ARDY/Llama/MIA runtime model weights are NOT stored in Drive."
echo "============================================================"

# MIA requires the gated jasongzy/Mixamo Hugging Face dataset. Fail before
# spending GPU time if the notebook did not provide a token. The token remains
# in the process environment only; it is never printed or added to a Git URL.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is required before Animation Engine installation." >&2
  echo "Run the Animation Hugging Face sign-in cell (or add HF_TOKEN to Colab Secrets) first." >&2
  exit 2
fi

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

stage "Restoring/downloading pinned ARDY source"
action "Checking Google Drive source cache before GitHub"
cache_git_repo \
  "ARDY" \
  "https://github.com/nv-tlabs/ardy.git" \
  "$ARDY_REF" \
  "/content/ardy" \
  "plain"
info "Pinned ARDY commit: $ARDY_REF"

stage "Creating ARDY environment"
action "Removing stale ardy environment if present"
conda env remove -n ardy -y >/dev/null 2>&1 || true
action "Creating Python 3.11 ARDY environment from conda-forge only"
conda create -n ardy --override-channels -c conda-forge python=3.11 pip -y
action "Upgrading ARDY pip/build helpers"
conda run --no-capture-output -n ardy python -m pip install --progress-bar on --upgrade pip setuptools wheel cmake packaging

stage "Installing ARDY + PyTorch dependencies"
action "Installing PyTorch 2.6.0 cu124"
conda run --no-capture-output -n ardy python -m pip install --progress-bar on torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
action "Restoring/building cached ARDY wheel (includes native MotionCorrection extension)"
cd /content/ardy
ARDY_WHEEL_KEY="ardy-${ARDY_REF:0:8}-cp311-torch260-linux-x86_64"
CACHE_PYTHON="/opt/conda/envs/ardy/bin/python"
cache_install_or_build_wheel "ardy" "$ARDY_WHEEL_KEY" "/content/ardy" "with-deps"
CACHE_PYTHON="python"
action "Installing ARDY preview dependency"
conda run --no-capture-output -n ardy python -m pip install matplotlib

stage "Running ARDY smoke test"
action "Checking ARDY skeleton import and GPU visibility"
conda run --no-capture-output -n ardy python -c 'import torch; from ardy.skeleton import CoreSkeleton27; assert len(CoreSkeleton27().bone_order_names)==27; assert torch.cuda.is_available(), "ARDY environment cannot see the NVIDIA GPU"; print("ARDY smoke test: OK", torch.__version__, torch.cuda.get_device_name(0), flush=True)'

stage "Restoring/downloading pinned Make-It-Animatable source"
action "Removing stale temporary MIA model-weight checkout"
rm -rf /tmp/mia-hf-data
action "Checking Google Drive source cache before GitHub"
cache_git_repo \
  "Make-It-Animatable" \
  "https://github.com/jasongzy/Make-It-Animatable" \
  "$MIA_REF" \
  "/content/Make-It-Animatable" \
  "recursive"
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
info "The gated Mixamo templates and MIA weights are downloaded fresh to local /content."
info "HF_TOKEN is read from the environment and is never printed or embedded in Git URLs."
action "Installing Hugging Face Hub/Xet downloader in the MIA environment"
conda run --no-capture-output -n mia python -m pip install --progress-bar on --upgrade huggingface_hub hf_xet
action "Authenticating and downloading only required MIA assets with visible HF progress"
HF_HOME=/content/huggingface HF_XET_HIGH_PERFORMANCE=1 \
  conda run --no-capture-output -n mia python "$SCRIPT_DIR/download_mia_assets.py" \
    --mia-root /content/Make-It-Animatable \
    --temp-root /tmp/mia-hf-data
action "Restoring/downloading FBX2glTF helper binary"
cache_download_file \
  "FBX2glTF-linux-x64-v0.9.7" \
  "https://github.com/facebookincubator/FBX2glTF/releases/download/v0.9.7/FBX2glTF-linux-x64" \
  "util/FBX2glTF"
chmod +x util/FBX2glTF

stage "Validating downloaded MIA assets"
action "Checking Mixamo template and weight files"
BONE_FILE="$(find data/Mixamo -type f -name 'bones*.fbx' | head -n 1 || true)"
[ -n "$BONE_FILE" ] || { echo "ERROR: Mixamo template FBX missing" >&2; exit 1; }
[ -s "$BONE_FILE" ] || { echo "ERROR: Mixamo template FBX is empty" >&2; exit 1; }
find output/best/new -type f -name '*.pth' | grep -q . || { echo "ERROR: MIA weights missing" >&2; exit 1; }
info "Template: $BONE_FILE"
info "Weight files: $(find output/best/new -type f -name '*.pth' | wc -l)"

stage "Running Make-It-Animatable / Blender smoke test"
action "Importing MIA/Blender stack and checking GPU visibility"
conda run --no-capture-output -n mia python -c 'import torch,bpy,trimesh,pytorch3d; assert torch.cuda.is_available(), "MIA environment cannot see the NVIDIA GPU"; print("MIA smoke test: OK", flush=True); print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), flush=True); print("Blender:", bpy.app.version_string, flush=True)'

printf '\n[ANIMATION INSTALL][%d/%d][%s] ✅ Installation complete.\n' "$TOTAL" "$TOTAL" "$(elapsed)"
if [ "$CACHE_ENABLED" -eq 1 ]; then
  printf '[CACHE] Persistent sources/builds are in: %s\n' "$ENGINE_CACHE_ROOT"
fi
printf 'You can now run the Animation Engine.\n'
