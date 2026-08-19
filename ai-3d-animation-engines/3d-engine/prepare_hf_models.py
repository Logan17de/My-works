#!/usr/bin/env python3
"""Authenticate to Hugging Face and pre-download TRELLIS.2 dependencies with visible progress.

A valid personal HF_TOKEN is required. TRELLIS.2 depends on gated models from
Meta (DINOv3) and BRIA (RMBG-2.0), so both access agreements must be accepted
on the Hugging Face website before this script can download them.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable

os.environ.setdefault("HF_HOME", "/content/huggingface")
os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

from huggingface_hub import HfApi, get_token, hf_hub_download, login
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, enable_progress_bars

DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TRELLIS_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS_IMAGE_REPO = "microsoft/TRELLIS-image-large"
RMBG_REPO = "briaai/RMBG-2.0"

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


def total_ram_gib() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1024**3
    except (ValueError, OSError, AttributeError):
        return 0.0


def select_files(siblings: Iterable[object], patterns: tuple[str, ...]) -> list[object]:
    selected = []
    for item in siblings:
        name = getattr(item, "rfilename", "")
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            selected.append(item)
    selected.sort(
        key=lambda x: (
            str(getattr(x, "rfilename", "")).endswith((".safetensors", ".bin")),
            getattr(x, "rfilename", ""),
        )
    )
    return selected


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Run the notebook HF sign-in cell first. "
            "Use a personal READ token; do not hard-code it into the notebook."
        )
    return token.strip()


def verify_auth_and_gates(api: HfApi, token: str) -> str:
    login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
    who = api.whoami(token=token)
    username = who.get("name") or who.get("fullname") or "authenticated user"
    print(f"[HF AUTH] ✅ Logged in as: {username}", flush=True)

    for repo_id, probe_file, label in GATED_CHECKS:
        print(f"[HF ACCESS] Checking {label}: {repo_id}", flush=True)
        try:
            hf_hub_download(repo_id, probe_file, token=token)
        except GatedRepoError as exc:
            raise RuntimeError(
                f"Your Hugging Face token is valid, but this account does not yet have access to {repo_id}. "
                "Open that model page in your browser while signed into the SAME account, accept/request the "
                "repository terms, then rerun the HF sign-in/download cell."
            ) from exc
        except HfHubHTTPError as exc:
            if getattr(exc.response, "status_code", None) in (401, 403):
                raise RuntimeError(
                    f"Hugging Face authentication/access failed for {repo_id}. Confirm the token belongs to "
                    "the same account that accepted the repository terms."
                ) from exc
            raise
        print(f"[HF ACCESS] ✅ Access confirmed: {repo_id}", flush=True)
    return str(username)


def download_plan(api: HfApi, token: str, plan: RepoPlan, index: int, total_plans: int) -> None:
    print("\n" + "=" * 78, flush=True)
    print(f"[HF DOWNLOAD][{index}/{total_plans}] {plan.label}", flush=True)
    print(f"[HF DOWNLOAD] Repo: {plan.repo_id}", flush=True)
    info = api.model_info(plan.repo_id, token=token, files_metadata=True)
    files = select_files(info.siblings or [], plan.patterns)
    if not files:
        raise RuntimeError(f"No files matched the expected patterns in {plan.repo_id}: {plan.patterns}")

    known_total = sum(int(getattr(f, "size", 0) or 0) for f in files)
    print(f"[HF DOWNLOAD] Files: {len(files)} | selected size: {gib(known_total)}", flush=True)

    completed_known = 0
    started_repo = time.time()
    for file_index, item in enumerate(files, start=1):
        filename = str(getattr(item, "rfilename"))
        size = int(getattr(item, "size", 0) or 0)
        before_pct = (completed_known / known_total * 100.0) if known_total else 0.0
        print(
            f"\n[HF FILE][{file_index}/{len(files)}] {filename} | {gib(size)} | overall {before_pct:5.1f}%",
            flush=True,
        )
        path = hf_hub_download(plan.repo_id, filename, token=token)
        completed_known += size
        after_pct = (
            completed_known / known_total * 100.0
            if known_total
            else file_index / len(files) * 100.0
        )
        print(f"[HF FILE] ✅ ready | overall {after_pct:5.1f}% | cache: {path}", flush=True)

    elapsed = time.time() - started_repo
    print(f"[HF DOWNLOAD] ✅ {plan.label} ready in {elapsed / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate token + all gated TRELLIS dependencies without downloading every weight",
    )
    args = parser.parse_args()

    enable_progress_bars()
    token = get_hf_token()

    try:
        hub_ver = version("huggingface_hub")
    except PackageNotFoundError:
        hub_ver = "unknown"
    try:
        xet_ver = version("hf_xet")
    except PackageNotFoundError:
        xet_ver = None

    ram_gib = total_ram_gib()
    if ram_gib >= 64 and "HF_XET_HIGH_PERFORMANCE" not in os.environ:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    print(
        f"[HF SETUP] huggingface_hub={hub_ver} | hf_xet={xet_ver or 'not installed'} | RAM={ram_gib:.1f} GiB",
        flush=True,
    )
    print(f"[HF SETUP] HF_HOME={os.environ['HF_HOME']}", flush=True)
    print(
        f"[HF SETUP] HF_XET_HIGH_PERFORMANCE={os.environ.get('HF_XET_HIGH_PERFORMANCE', '0')}",
        flush=True,
    )

    api = HfApi(token=token)
    verify_auth_and_gates(api, token)
    if args.check_only:
        print("[HF READY] ✅ Authentication and all gated-model access checks passed.", flush=True)
        return

    print("\n[HF DOWNLOAD] Starting explicit TRELLIS dependency pre-download.", flush=True)
    print(
        "[HF DOWNLOAD] Large files show Hugging Face byte progress bars; each completed file updates the overall percentage.",
        flush=True,
    )
    for i, plan in enumerate(PLANS, start=1):
        download_plan(api, token, plan, i, len(PLANS))

    print("\n" + "=" * 78, flush=True)
    print(
        "[HF READY] ✅ All TRELLIS.2 runtime model files are present in the local Hugging Face cache.",
        flush=True,
    )
    print(
        "[HF READY] TRELLIS generation can now load from local cache instead of downloading silently.",
        flush=True,
    )


if __name__ == "__main__":
    main()
