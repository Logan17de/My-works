#!/usr/bin/env python3
"""Download only the Hugging Face assets required by Make-It-Animatable.

Uses HF_TOKEN from the environment so gated Mixamo access works in Colab without
embedding credentials in Git URLs or command-line arguments.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

MIXAMO_REPO = "jasongzy/Mixamo"
MIA_REPO = "jasongzy/Make-It-Animatable"


def token_from_env() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. In Colab add HF_TOKEN under Secrets (key icon) "
            "or run the Animation HF sign-in cell before installation."
        )
    return token


def verify_access(token: str) -> None:
    api = HfApi(token=token)
    who = api.whoami(token=token)
    username = who.get("name") or who.get("fullname") or "authenticated user"
    print(f"[MIA HF] ✅ Authenticated as: {username}", flush=True)

    print(f"[MIA HF] Checking gated dataset access: {MIXAMO_REPO}", flush=True)
    try:
        # model_info/dataset_info alone can list a gated repo; the actual file
        # download below is the definitive access check.
        api.dataset_info(MIXAMO_REPO, token=token)
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Cannot access Hugging Face dataset metadata for {MIXAMO_REPO}. "
            "Confirm the token is valid."
        ) from exc


def download_mixamo(token: str, destination: Path) -> None:
    print("\n[MIA HF][1/2] Downloading Mixamo skeleton templates", flush=True)
    print(f"[MIA HF] Repo: datasets/{MIXAMO_REPO}", flush=True)
    print("[MIA HF] Files: bones*.fbx only", flush=True)
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=MIXAMO_REPO,
            repo_type="dataset",
            token=token,
            allow_patterns=["bones*.fbx"],
            local_dir=str(destination),
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "Your Hugging Face token is valid, but your account has not accepted "
            "the jasongzy/Mixamo gated dataset conditions. Open the Mixamo dataset "
            "page while signed into the same HF account, accept the access terms, "
            "then rerun this cell."
        ) from exc
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) in (401, 403):
            raise RuntimeError(
                "Hugging Face authentication/access failed for jasongzy/Mixamo. "
                "Confirm HF_TOKEN belongs to the account that accepted the dataset terms."
            ) from exc
        raise

    bone_files = sorted(destination.glob("bones*.fbx"))
    if not bone_files:
        raise RuntimeError(f"No bones*.fbx files were downloaded to {destination}")
    for path in bone_files:
        print(f"[MIA HF] ✅ Template: {path.name} ({path.stat().st_size / 1024:.1f} KiB)", flush=True)


def download_weights(token: str, temp_root: Path, mia_root: Path) -> None:
    print("\n[MIA HF][2/2] Downloading Make-It-Animatable pretrained weights", flush=True)
    print(f"[MIA HF] Repo: {MIA_REPO}", flush=True)
    print("[MIA HF] Files: output/best/new/**", flush=True)
    if temp_root.exists():
        shutil.rmtree(temp_root)
    snapshot_download(
        repo_id=MIA_REPO,
        repo_type="model",
        token=token,
        allow_patterns=["output/best/new/**"],
        local_dir=str(temp_root),
    )

    src = temp_root / "output" / "best" / "new"
    if not src.is_dir():
        raise RuntimeError(f"Expected MIA weight directory missing after download: {src}")
    weights = sorted(src.rglob("*.pth"))
    if not weights:
        raise RuntimeError(f"No .pth MIA weights found in {src}")

    dst = mia_root / "output" / "best" / "new"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[MIA HF] ✅ Installed {len(weights)} weight files into {dst}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mia-root", default="/content/Make-It-Animatable")
    parser.add_argument("--temp-root", default="/tmp/mia-hf-data")
    args = parser.parse_args()

    token = token_from_env()
    mia_root = Path(args.mia_root).expanduser().resolve()
    temp_root = Path(args.temp_root).expanduser().resolve()
    if not mia_root.is_dir():
        raise RuntimeError(f"Make-It-Animatable checkout not found: {mia_root}")

    verify_access(token)
    download_mixamo(token, mia_root / "data" / "Mixamo")
    download_weights(token, temp_root, mia_root)
    print("\n[MIA HF] ✅ Required MIA templates and model weights are ready.", flush=True)


if __name__ == "__main__":
    main()
