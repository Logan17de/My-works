#!/usr/bin/env bash
set -Eeuo pipefail

NVDIFFRAST_REF="253ac4fcea7de5f396371124af597e6cc957bfae"
NVDIFFREC_REF="b296927cc7fd01c2ac1087c8065c4d7248f72da4"
CUMESH_REF="12289e1062f0603f2f0d0771b02e1395d247f26f"
FLEXGEMM_REF="6dd94a859c26ee8246888502eada3dd8ad85532e"
TOTAL=7
STEP=0
START_TS=$(date +%s)
CURRENT_ACTION="starting"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ENGINE_ROOT/cache_common.sh"
cache_init

elapsed() {
  local now diff m s
  now=$(date +%s); diff=$((now-START_TS)); m=$((diff/60)); s=$((diff%60))
  printf '%02d:%02d' "$m" "$s"
}
step() {
  STEP=$((STEP+1)); CURRENT_ACTION="$1"
  printf '\n[TRELLIS EXT][%d/%d][%s] ▶ %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_ACTION"
}
info() { printf '[TRELLIS EXT][%d/%d][%s]   %s\n' "$STEP" "$TOTAL" "$(elapsed)" "$*"; }
trap 'code=$?; printf "\n[TRELLIS EXT][%d/%d][%s] ❌ FAILED\n  Action: %s\n  Exit: %d\n" "$STEP" "$TOTAL" "$(elapsed)" "$CURRENT_ACTION" "$code" >&2; exit "$code"' ERR

if [ ! -x /opt/conda/bin/conda ]; then
  echo "ERROR: Conda is not installed. Run the main TRELLIS installer first." >&2
  exit 1
fi
if [ ! -x /opt/conda/envs/trellis2/bin/python ]; then
  echo "ERROR: trellis2 environment is missing. Run the main TRELLIS installer first." >&2
  exit 1
fi
if [ ! -d /content/TRELLIS.2 ]; then
  echo "ERROR: /content/TRELLIS.2 is missing. Run the main TRELLIS installer first." >&2
  exit 1
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate trellis2
cd /content/TRELLIS.2

if [ -x /usr/local/cuda-12.4/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda-12.4
else
  export CUDA_HOME="$CONDA_PREFIX"
fi
export PATH="$CUDA_HOME/bin:$PATH"

TRELLIS_REF="$(git -C /content/TRELLIS.2 rev-parse HEAD)"
GPU_CC="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}{b}")')"
PY_TAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
TORCH_TAG="$(python -c 'import torch; print(torch.__version__.split("+")[0].replace(".", ""))')"
BASE_KEY="${PY_TAG}-torch${TORCH_TAG}-cu124-sm${GPU_CC}"
export TORCH_CUDA_ARCH_LIST="${GPU_CC:0:1}.${GPU_CC:1:1}"
mkdir -p /tmp/extensions /content/.ai3d_prebuilt
info "Binary compatibility: $BASE_KEY"
info "TRELLIS revision: ${TRELLIS_REF:0:12}"

step "Installing official FlashAttention 2.7.3 wheel"
ABI_FLAG="$(python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
FLASH_WHEEL_NAME="flash_attn-2.7.3+cu12torch2.6cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl"
FLASH_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/${FLASH_WHEEL_NAME}"
FLASH_WHEEL_LOCAL="/content/.ai3d_prebuilt/${FLASH_WHEEL_NAME}"
FLASH_CACHE_LABEL="flash-attn-v2.7.3-cu12-torch2.6-cp310-abi${ABI_FLAG}.whl"
info "PyTorch C++ ABI: $ABI_FLAG"
info "Using official prebuilt wheel: $FLASH_WHEEL_NAME"
cache_download_file "$FLASH_CACHE_LABEL" "$FLASH_WHEEL_URL" "$FLASH_WHEEL_LOCAL"
python -m pip install --no-deps "$FLASH_WHEEL_LOCAL"
python -c 'import flash_attn; print("FlashAttention import: OK", flash_attn.__version__, flush=True)'

step "Installing/restoring nvdiffrast"
cache_git_repo "nvdiffrast" "https://github.com/NVlabs/nvdiffrast.git" "$NVDIFFRAST_REF" "/tmp/extensions/nvdiffrast" "plain"
cache_install_or_build_wheel "nvdiffrast" "${BASE_KEY}-nvdiffrast-${NVDIFFRAST_REF:0:8}" "/tmp/extensions/nvdiffrast" "no-deps"

step "Installing/restoring nvdiffrec renderutils"
cache_git_repo "nvdiffrec" "https://github.com/JeffreyXiang/nvdiffrec.git" "$NVDIFFREC_REF" "/tmp/extensions/nvdiffrec" "plain"
cache_install_or_build_wheel "nvdiffrec" "${BASE_KEY}-nvdiffrec-${NVDIFFREC_REF:0:8}" "/tmp/extensions/nvdiffrec" "no-deps"

step "Installing/restoring CuMesh"
cache_git_repo "CuMesh" "https://github.com/JeffreyXiang/CuMesh.git" "$CUMESH_REF" "/tmp/extensions/CuMesh" "recursive"
cache_install_or_build_wheel "cumesh" "${BASE_KEY}-cumesh-${CUMESH_REF:0:8}" "/tmp/extensions/CuMesh" "no-deps"

step "Installing/restoring O-Voxel"
cache_install_or_build_wheel "o-voxel" "${BASE_KEY}-ovoxel-${TRELLIS_REF:0:8}" "/content/TRELLIS.2/o-voxel" "no-deps"

step "Installing/restoring FlexGEMM"
cache_git_repo "FlexGEMM" "https://github.com/JeffreyXiang/FlexGEMM.git" "$FLEXGEMM_REF" "/tmp/extensions/FlexGEMM" "recursive"
cache_install_or_build_wheel "flexgemm" "${BASE_KEY}-flexgemm-${FLEXGEMM_REF:0:8}" "/tmp/extensions/FlexGEMM" "no-deps"

step "Validating native extension imports"
python - <<'PY'
import flash_attn
import o_voxel
print("flash_attn:", flash_attn.__version__, flush=True)
print("o_voxel: OK", flush=True)
PY

printf '\n[TRELLIS EXT][%d/%d][%s] ✅ Native extension stage complete.\n' "$TOTAL" "$TOTAL" "$(elapsed)"
