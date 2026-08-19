#!/usr/bin/env python3
"""Authenticate to Hugging Face and pre-download TRELLIS.2 runtime dependencies.

The token is read only from HF_TOKEN / the local HF token store. This script:
1) verifies the token,
2) verifies gated access before the expensive TRELLIS load,
3) pre-downloads the exact runtime repos with visible Hugging Face progress bars,
4) writes a cache-ready marker consumed by run_trellis2.py.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_HOME", "/content/huggingface")
os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)


def _ram_gib() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1024**3
    except (ValueError, OSError, AttributeError):
        return 0.0


# Hugging Face reads many environment variables at import time. Enable Xet high
# performance only on large-memory runtimes (HF recommends >=64 GB RAM).
RAM_GIB = _ram_gib()
if RAM_GIB >= 64 and "HF_XET_HIGH_PERFORMANCE" not in os.environ:
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from huggingface_hub import HfApi, get_token, hf_hub_download, login, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, enable_progress_bars

DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TRELLIS_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS_IMAGE_REPO = "microsoft/TRELLIS-image-large"
RMBG_REPO = "briaai/RMBG-2.0"
READY_MARKER = Path(os.environ["HF_HOME"]) / "trellis2_preload_ready.json"

GATED_CHECKS = (
    (DINO_REPO, "config.json", "Meta DINOv3 image encoder"),
    (RMBG_REPO, "config.json", "BRIA RMBG-2.0 background-removal model"),
)


@dataclass(frozen=True)
class RepoPlan:
    repo_id: str
    patterns: tuple[str, ...]
    label: str


PLANS = (
    RepoPlan(
        TRELLIS_REPO,
        ("pipeline.json", "ckpts/*.json", "ckpts/*.safetensors"),
        "TRELLIS.2 4B checkpoints",
    ),
    RepoPlan(
        TRELLIS_IMAGE_REPO,
        (
            "ckpts/ss_dec_conv3d_16l8_fp16.json",
            "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        ),
        "TRELLIS sparse-structure decoder",
    ),
    RepoPlan(
        DINO_REPO,
        ("config.json", "preprocessor_config.json", "model.safetensors"),
        "DINOv3 image encoder (gated)",
    ),
    RepoPlan(
        RMBG_REPO,
        (
            "config.json",
            "preprocessor_config.json",
            "BiRefNet_config.py",
            "birefnet.py",
            "model.safetensors",
        ),
        "RMBG-2.0 background-removal model (gated)",
    ),
)


def gib(n: int | None) -> str:
    if not n:
        return "unknown size"
    return f"{n / 1024**3:.2f} GiB"


def select_files(siblings: Iterable[object], patterns: tuple[str, ...]) -> list[object]:
    selected = []
    for item in siblings:
        name = getattr(item, "rfilename", "")
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            selected.append(item)
    selected.sort(key=lambda x: getattr(x, "rfilename", ""))
    return selected


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Run the notebook HF sign-in cell first. "
            "Use a personal READ token; never hard-code it into the notebook."
        )
    return token.strip()


def verify_auth_and_gates(api: HfApi, token: str) -> str:
    # Keep the token out of command arguments/logs. login() stores it only in the
    # runtime's HF_HOME; the Colab itself receives it through an environment var.
    login(token=token, add_to_git_credential=False)
    who = api.whoami(token=token)
    username = who.get("name") or who.get("fullname") or "authenticated user"
    print(f"[HF AUTH] ✅ Logged in as: {username}", flush=True)

    for repo_id, probe_file, label in GATED_CHECKS:
        print(f"[HF ACCESS] Checking {label}: {repo_id}", flush=True)
        try:
            hf_hub_download(repo_id, probe_file, token=token)
        except GatedRepoError as exc:
            raise RuntimeError(
                f"Token is valid, but this account does not have access to {repo_id}. "
                "Open that model page while signed into the SAME Hugging Face account, "
                "accept/request access, then rerun this cell."
            ) from exc
        except HfHubHTTPError as exc:
            if getattr(exc.response, "status_code", None) in (401, 403):
                raise RuntimeError(
                    f"Hugging Face authentication/access failed for {repo_id}. "
                    "Confirm the token belongs to the account that accepted the model terms."
                ) from exc
            raise
        print(f"[HF ACCESS] ✅ Access confirmed: {repo_id}", flush=True)
    return str(username)


def download_plan(api: HfApi, token: str, plan: RepoPlan, index: int, total_plans: int) -> dict:
    print("\n" + "=" * 78, flush=True)
    print(f"[HF DOWNLOAD][{index}/{total_plans}] {plan.label}", flush=True)
    print(f"[HF DOWNLOAD] Repo: {plan.repo_id}", flush=True)

    info = api.model_info(plan.repo_id, token=token, files_metadata=True)
    files = select_files(info.siblings or [], plan.patterns)
    if not files:
        raise RuntimeError(f"No files matched the expected patterns in {plan.repo_id}: {plan.patterns}")

    known_total = sum(int(getattr(f, "size", 0) or 0) for f in files)
    print(f"[HF DOWNLOAD] Files: {len(files)} | selected size: {gib(known_total)}", flush=True)
    for i, item in enumerate(files, start=1):
        name = str(getattr(item, "rfilename", ""))
        size = int(getattr(item, "size", 0) or 0)
        pct = i / len(files) * 100.0
        print(f"[HF PLAN] {i:02d}/{len(files):02d} ({pct:5.1f}%) {name} | {gib(size)}", flush=True)

    started = time.time()
    # snapshot_download performs concurrent downloads and keeps every file in the
    # standard HF cache. HF/Xet emits live byte/file progress bars in Colab.
    snapshot_path = snapshot_download(
        repo_id=plan.repo_id,
        allow_patterns=list(plan.patterns),
        token=token,
        max_workers=8,
    )
    elapsed = time.time() - started
    print(f"[HF DOWNLOAD] ✅ {plan.label} ready | 100.0% | {elapsed / 60:.1f} min", flush=True)
    print(f"[HF CACHE] Snapshot: {snapshot_path}", flush=True)
    return {
        "repo_id": plan.repo_id,
        "snapshot_path": snapshot_path,
        "files": [str(getattr(f, "rfilename", "")) for f in files],
        "selected_bytes": known_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate token + gated access without downloading every runtime weight",
    )
    args = parser.parse_args()

    enable_progress_bars()
    READY_MARKER.unlink(missing_ok=True)
    token = get_hf_token()

    try:
        hub_ver = version("huggingface_hub")
    except PackageNotFoundError:
        hub_ver = "unknown"
    try:
        xet_ver = version("hf_xet")
    except PackageNotFoundError:
        xet_ver = None

    print(
        f"[HF SETUP] huggingface_hub={hub_ver} | hf_xet={xet_ver or 'not installed'} | RAM={RAM_GIB:.1f} GiB",
        flush=True,
    )
    print(f"[HF SETUP] HF_HOME={os.environ['HF_HOME']}", flush=True)
    print(
        f"[HF SETUP] HF_XET_HIGH_PERFORMANCE={os.environ.get('HF_XET_HIGH_PERFORMANCE', '0')}",
        flush=True,
    )

    api = HfApi(token=token)
    username = verify_auth_and_gates(api, token)
    if args.check_only:
        print("[HF READY] ✅ Authentication and gated-model access checks passed.", flush=True)
        return

    print("\n[HF DOWNLOAD] Starting explicit TRELLIS dependency pre-download.", flush=True)
    print("[HF DOWNLOAD] Hugging Face/Xet progress bars will show file and byte progress below.", flush=True)

    downloaded = []
    for i, plan in enumerate(PLANS, start=1):
        downloaded.append(download_plan(api, token, plan, i, len(PLANS)))

    READY_MARKER.parent.mkdir(parents=True, exist_ok=True)
    READY_MARKER.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_unix": time.time(),
                "hf_home": os.environ["HF_HOME"],
                "authenticated_user": username,
                "repos": downloaded,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78, flush=True)
    print("[HF READY] ✅ All TRELLIS.2 runtime files are present in the local Hugging Face cache.", flush=True)
    print(f"[HF READY] Cache marker: {READY_MARKER}", flush=True)
    print("[HF READY] Generation will now switch to cache-only/offline HF loading automatically.", flush=True)


if __name__ == "__main__":
    main()
