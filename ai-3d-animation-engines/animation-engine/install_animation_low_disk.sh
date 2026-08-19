#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colab's local disk is small. Prevent pip from keeping a second copy of every
# wheel/archive while the two isolated ARDY/MIA environments are being built.
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1

usage() {
  df -h /content | tail -n 1 || true
}

echo "============================================================"
echo " Animation Engine low-disk install"
echo " pip cache: disabled"
echo " temporary MIA weight staging: removed immediately after copy"
echo " Conda/build/package caches: cleaned after install"
echo "============================================================"
echo "[DISK] Before installation:"
usage

bash "$SCRIPT_DIR/install_animation.sh"

echo "\n[DISK CLEANUP][1/5] Removing temporary model/build staging"
rm -rf \
  /tmp/mia-hf-data \
  /tmp/pip-* \
  /tmp/extensions \
  /content/.ai3d_wheel_build \
  /content/.ai3d_cached_wheels \
  /root/.cache/pip \
  /root/.cache/torch_extensions || true

echo "[DISK CLEANUP][2/5] Cleaning Conda package caches"
if [ -x /opt/conda/bin/conda ]; then
  /opt/conda/bin/conda clean -a -y || true
fi

echo "[DISK CLEANUP][3/5] Cleaning apt package cache"
apt-get clean || true
rm -rf /var/lib/apt/lists/* || true

echo "[DISK CLEANUP][4/5] Removing Python bytecode/cache directories from source trees"
find /content/ardy /content/Make-It-Animatable \
  -type d \( -name '__pycache__' -o -name '.pytest_cache' \) \
  -prune -exec rm -rf {} + 2>/dev/null || true

echo "[DISK CLEANUP][5/5] Disk status after cleanup"
usage

echo "[DISK] Largest retained Animation Engine directories:"
du -sh \
  /opt/conda/envs/ardy \
  /opt/conda/envs/mia \
  /content/ardy \
  /content/Make-It-Animatable \
  /content/huggingface \
  2>/dev/null | sort -hr || true

echo "[DISK] ✅ Low-disk Animation Engine installation complete."
