#!/usr/bin/env bash
# Shared persistent-cache helpers for fresh Colab runtimes.
# The notebook sets ENGINE_CACHE_ROOT to a Google Drive folder when caching is enabled.
set -Eeuo pipefail

CACHE_ENABLED=0
CACHE_PYTHON="${CACHE_PYTHON:-python}"

cache_init() {
  if [ -n "${ENGINE_CACHE_ROOT:-}" ]; then
    mkdir -p "$ENGINE_CACHE_ROOT/sources" "$ENGINE_CACHE_ROOT/wheels" "$ENGINE_CACHE_ROOT/downloads"
    CACHE_ENABLED=1
    printf '[CACHE] Enabled: %s\n' "$ENGINE_CACHE_ROOT"
  else
    CACHE_ENABLED=0
    printf '[CACHE] Disabled. ENGINE_CACHE_ROOT is not set; this run will use only /content.\n'
  fi
}

cache_git_repo() {
  # usage: cache_git_repo label url ref dest recursive|plain
  local label="$1" url="$2" ref="$3" dest="$4" mode="${5:-plain}"
  local parent base archive tmp_archive actual
  parent="$(dirname "$dest")"
  base="$(basename "$dest")"
  archive="${ENGINE_CACHE_ROOT:-/tmp}/sources/${label}-${ref}.tar.gz"

  rm -rf "$dest"
  mkdir -p "$parent"

  if [ "$CACHE_ENABLED" -eq 1 ] && [ -s "$archive" ]; then
    printf '[CACHE HIT] Source %s -> restoring %s\n' "$label" "$archive"
    if tar -xzf "$archive" -C "$parent"; then
      if [ -d "$dest/.git" ]; then
        actual="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"
        if [ "$actual" = "$ref" ]; then
          printf '[CACHE HIT] Source %s restored at %s\n' "$label" "$ref"
          return 0
        fi
        printf '[CACHE] Source %s cache revision mismatch (%s); rebuilding cache.\n' "$label" "$actual"
      else
        printf '[CACHE] Source %s archive is invalid; rebuilding cache.\n' "$label"
      fi
    else
      printf '[CACHE] Source %s archive could not be extracted; rebuilding cache.\n' "$label"
    fi
    rm -rf "$dest"
  fi

  printf '[CACHE MISS] Source %s -> cloning from %s\n' "$label" "$url"
  if [ "$mode" = "recursive" ]; then
    git clone --progress --recursive "$url" "$dest"
  else
    git clone --progress "$url" "$dest"
  fi
  git -C "$dest" checkout "$ref"
  if [ "$mode" = "recursive" ]; then
    git -C "$dest" submodule sync --recursive
    git -C "$dest" submodule update --init --recursive --force --progress
  fi

  if [ "$CACHE_ENABLED" -eq 1 ]; then
    printf '[CACHE SAVE] Source %s -> %s\n' "$label" "$archive"
    tmp_archive="/content/.${label}-${ref}.tar.gz.tmp.$$"
    tar -czf "$tmp_archive" -C "$parent" "$base"
    cp "$tmp_archive" "$archive.tmp"
    mv "$archive.tmp" "$archive"
    rm -f "$tmp_archive"
    printf '[CACHE SAVE] Source %s saved (%s)\n' "$label" "$(du -h "$archive" | awk '{print $1}')"
  fi
}

cache_has_wheel() {
  # usage: cache_has_wheel label cache_key
  local label="$1" cache_key="$2" drive_dir
  drive_dir="${ENGINE_CACHE_ROOT:-/tmp}/wheels/$cache_key/$label"
  [ "$CACHE_ENABLED" -eq 1 ] && [ -d "$drive_dir" ] && \
    [ -n "$(find "$drive_dir" -maxdepth 1 -type f -name '*.whl' -print -quit 2>/dev/null || true)" ]
}

cache_install_or_build_wheel() {
  # usage: cache_install_or_build_wheel label cache_key target [with-deps|no-deps]
  # target can be a package spec (flash-attn==...) or a local source directory.
  local label="$1" cache_key="$2" target="$3" dep_mode="${4:-no-deps}"
  local drive_dir local_dir build_dir cached wheel
  local -a install_args
  drive_dir="${ENGINE_CACHE_ROOT:-/tmp}/wheels/$cache_key/$label"
  local_dir="/content/.ai3d_cached_wheels/$cache_key/$label"
  build_dir="/content/.ai3d_wheel_build/$cache_key/$label"
  install_args=(--no-deps)
  if [ "$dep_mode" = "with-deps" ]; then
    install_args=()
  fi

  rm -rf "$local_dir" "$build_dir"
  mkdir -p "$local_dir" "$build_dir"

  cached=""
  if [ "$CACHE_ENABLED" -eq 1 ] && [ -d "$drive_dir" ]; then
    cached="$(find "$drive_dir" -maxdepth 1 -type f -name '*.whl' -print -quit || true)"
  fi

  if [ -n "$cached" ]; then
    printf '[CACHE HIT] Wheel %s -> %s\n' "$label" "$(basename "$cached")"
    cp "$cached" "$local_dir/"
    wheel="$local_dir/$(basename "$cached")"
    if "$CACHE_PYTHON" -m pip install "${install_args[@]}" "$wheel"; then
      printf '[CACHE HIT] Wheel %s installed without rebuilding.\n' "$label"
      return 0
    fi
    printf '[CACHE] Cached wheel %s failed to install; deleting it and rebuilding.\n' "$label"
    rm -f "$cached" "$wheel"
  fi

  printf '[CACHE MISS] Wheel %s -> building now. First compatible run pays this cost.\n' "$label"
  "$CACHE_PYTHON" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$build_dir" "$target"
  wheel="$(find "$build_dir" -maxdepth 1 -type f -name '*.whl' -print -quit || true)"
  if [ -z "$wheel" ]; then
    echo "ERROR: wheel build for $label completed without producing a .whl" >&2
    return 1
  fi
  "$CACHE_PYTHON" -m pip install "${install_args[@]}" "$wheel"

  if [ "$CACHE_ENABLED" -eq 1 ]; then
    mkdir -p "$drive_dir"
    rm -f "$drive_dir"/*.whl
    printf '[CACHE SAVE] Wheel %s -> %s\n' "$label" "$drive_dir/$(basename "$wheel")"
    cp "$wheel" "$drive_dir/$(basename "$wheel").tmp"
    mv "$drive_dir/$(basename "$wheel").tmp" "$drive_dir/$(basename "$wheel")"
  fi
}

cache_download_file() {
  # usage: cache_download_file label url destination
  local label="$1" url="$2" dest="$3" cache_file
  cache_file="${ENGINE_CACHE_ROOT:-/tmp}/downloads/$label"
  mkdir -p "$(dirname "$dest")"
  if [ "$CACHE_ENABLED" -eq 1 ] && [ -s "$cache_file" ]; then
    printf '[CACHE HIT] Download %s -> restoring cached file\n' "$label"
    cp "$cache_file" "$dest"
    return 0
  fi
  printf '[CACHE MISS] Download %s -> %s\n' "$label" "$url"
  wget --progress=bar:force:noscroll "$url" -O "$dest"
  if [ "$CACHE_ENABLED" -eq 1 ]; then
    cp "$dest" "$cache_file.tmp"
    mv "$cache_file.tmp" "$cache_file"
    printf '[CACHE SAVE] Download %s saved to Drive cache.\n' "$label"
  fi
}
