from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


class WorkBackup:
    """Keep the hot expert/optimizer work tree on local disk and mirror it to persistent storage only at checkpoints."""

    def __init__(self, local_work_dir: str | Path, backup_root: str | Path | None, run_name: str) -> None:
        self.local_work_dir = Path(local_work_dir)
        self.backup_root = Path(backup_root) if backup_root else None
        self.run_name = run_name
        self.backup_dir = self.backup_root / run_name if self.backup_root else None

    @property
    def enabled(self) -> bool:
        return self.backup_dir is not None

    @property
    def state_path(self) -> Path | None:
        return self.backup_dir / "sync_state.json" if self.backup_dir else None

    def last_synced_step(self) -> int | None:
        path = self.state_path
        if path is None or not path.exists():
            return None
        try:
            return int(json.loads(path.read_text(encoding="utf-8")).get("step"))
        except Exception:
            return None

    def restore_if_needed(self) -> bool:
        """Restore the Drive/cloud copy only when the local runtime has no usable work tree."""
        if not self.enabled or self.backup_dir is None or not self.backup_dir.exists():
            return False
        local_manifest = self.local_work_dir / "expert_store" / "manifest.json"
        if local_manifest.exists():
            return False
        self.local_work_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Restoring local expert work from {self.backup_dir} ...", flush=True)
        self._copy_tree(self.backup_dir, self.local_work_dir, exclude_state=True)
        print(f"Local expert work restored to {self.local_work_dir}", flush=True)
        return True

    def sync(self, step: int, reason: str) -> float:
        """Overwrite the persistent mirror with the current local work tree."""
        if not self.enabled or self.backup_dir is None:
            return 0.0
        started = time.time()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        # Force dirty mmap pages toward storage before copying the authoritative files.
        if hasattr(os, "sync"):
            os.sync()
        print(f"Syncing local expert work -> persistent backup (step {step}, {reason}) ...", flush=True)
        self._copy_tree(self.local_work_dir, self.backup_dir, exclude_state=False)
        state = {
            "step": int(step),
            "reason": reason,
            "local_work_dir": str(self.local_work_dir),
            "synced_at_unix": time.time(),
        }
        (self.backup_dir / "sync_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        elapsed = time.time() - started
        print(f"Persistent expert backup complete in {elapsed:.1f}s: {self.backup_dir}", flush=True)
        return elapsed

    @staticmethod
    def _copy_tree(source: Path, target: Path, *, exclude_state: bool) -> None:
        source.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        rsync = shutil.which("rsync")
        if rsync:
            cmd = [rsync, "-a", "--inplace", "--delete"]
            if exclude_state:
                cmd += ["--exclude", "sync_state.json"]
            cmd += [str(source) + "/", str(target) + "/"]
            subprocess.run(cmd, check=True)
            return

        # Portable fallback. This copies only files whose size or mtime changed and removes stale paths.
        source_paths: set[Path] = set()
        for src in source.rglob("*"):
            rel = src.relative_to(source)
            if exclude_state and rel.as_posix() == "sync_state.json":
                continue
            source_paths.add(rel)
            dst = target / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            needs_copy = not dst.exists() or src.stat().st_size != dst.stat().st_size or src.stat().st_mtime_ns != dst.stat().st_mtime_ns
            if needs_copy:
                shutil.copy2(src, dst)

        for dst in sorted(target.rglob("*"), reverse=True):
            rel = dst.relative_to(target)
            if rel.as_posix() == "sync_state.json":
                continue
            if rel not in source_paths:
                if dst.is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                else:
                    dst.unlink(missing_ok=True)
