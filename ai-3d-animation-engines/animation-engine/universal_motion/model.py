from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

CANONICAL_JOINTS = (
    "hips", "spine", "spine_1", "chest", "upper_chest", "neck", "head",
    "left_shoulder", "left_upper_arm", "left_forearm", "left_hand", "left_hand_end", "left_thumb_1",
    "right_shoulder", "right_upper_arm", "right_forearm", "right_hand", "right_hand_end", "right_thumb_1",
    "left_upper_leg", "left_lower_leg", "left_foot", "left_toe",
    "right_upper_leg", "right_lower_leg", "right_foot", "right_toe",
)

CANONICAL_PARENTS = {
    "hips": None,
    "spine": "hips", "spine_1": "spine", "chest": "spine_1", "upper_chest": "chest", "neck": "upper_chest", "head": "neck",
    "left_shoulder": "upper_chest", "left_upper_arm": "left_shoulder", "left_forearm": "left_upper_arm", "left_hand": "left_forearm", "left_hand_end": "left_hand", "left_thumb_1": "left_hand",
    "right_shoulder": "upper_chest", "right_upper_arm": "right_shoulder", "right_forearm": "right_upper_arm", "right_hand": "right_forearm", "right_hand_end": "right_hand", "right_thumb_1": "right_hand",
    "left_upper_leg": "hips", "left_lower_leg": "left_upper_leg", "left_foot": "left_lower_leg", "left_toe": "left_foot",
    "right_upper_leg": "hips", "right_lower_leg": "right_upper_leg", "right_foot": "right_lower_leg", "right_toe": "right_foot",
}

@dataclass
class CanonicalMotion:
    joint_names: tuple[str, ...]
    parents: np.ndarray
    local_rotations: np.ndarray
    global_rotations: np.ndarray
    root_positions: np.ndarray
    rest_positions: np.ndarray
    posed_positions: np.ndarray
    fps: float
    contact_names: tuple[str, ...] = ()
    contacts: np.ndarray | None = None
    source_name: str = "unknown"
    metadata: dict | None = None

    def validate(self) -> "CanonicalMotion":
        j = len(self.joint_names)
        t = int(self.local_rotations.shape[0])
        if tuple(self.joint_names) != CANONICAL_JOINTS:
            raise ValueError("Canonical joint order does not match schema v1")
        expected = {
            "parents": (j,), "local_rotations": (t, j, 3, 3), "global_rotations": (t, j, 3, 3),
            "root_positions": (t, 3), "rest_positions": (j, 3), "posed_positions": (t, j, 3),
        }
        for name, shape in expected.items():
            arr = np.asarray(getattr(self, name))
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape}, expected {shape}")
            if name != "parents" and not np.isfinite(arr).all():
                raise ValueError(f"{name} contains NaN/Inf")
        if not np.isfinite(float(self.fps)) or float(self.fps) <= 0:
            raise ValueError(f"Invalid fps: {self.fps}")
        if int(np.sum(np.asarray(self.parents) < 0)) != 1:
            raise ValueError("Canonical motion must have exactly one root")
        if self.contacts is not None:
            c = np.asarray(self.contacts)
            if c.ndim != 2 or c.shape[0] != t or c.shape[1] != len(self.contact_names):
                raise ValueError(f"contacts shape {c.shape} incompatible with {t} frames and {len(self.contact_names)} contact names")
            if not np.isfinite(c).all():
                raise ValueError("contacts contains NaN/Inf")
        return self

    @property
    def frames(self) -> int:
        return int(self.local_rotations.shape[0])

    @property
    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.joint_names)}

    @property
    def reference_height(self) -> float:
        pts = np.asarray(self.rest_positions, dtype=np.float64)
        height = float(pts[:, 2].max() - pts[:, 2].min())
        if not np.isfinite(height) or height <= 1e-8:
            raise ValueError(f"Invalid canonical reference height: {height}")
        return height

    def save(self, path: str | Path) -> Path:
        self.validate()
        out = Path(path).expanduser().resolve(); out.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata or {}); metadata.setdefault("schema_version", 1)
        payload = {
            "schema_version": np.asarray(1, dtype=np.int32),
            "joint_names": np.asarray(self.joint_names, dtype=object),
            "parents": np.asarray(self.parents, dtype=np.int32),
            "local_rotations": np.asarray(self.local_rotations, dtype=np.float32),
            "global_rotations": np.asarray(self.global_rotations, dtype=np.float32),
            "root_positions": np.asarray(self.root_positions, dtype=np.float32),
            "rest_positions": np.asarray(self.rest_positions, dtype=np.float32),
            "posed_positions": np.asarray(self.posed_positions, dtype=np.float32),
            "fps": np.asarray(float(self.fps), dtype=np.float32),
            "contact_names": np.asarray(self.contact_names, dtype=object),
            "source_name": np.asarray(self.source_name),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        if self.contacts is not None: payload["contacts"] = np.asarray(self.contacts, dtype=np.float32)
        np.savez_compressed(out, **payload)
        if not out.is_file() or out.stat().st_size == 0: raise RuntimeError(f"Failed to write canonical motion: {out}")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalMotion":
        q = Path(path).expanduser().resolve()
        if not q.is_file(): raise FileNotFoundError(q)
        with np.load(q, allow_pickle=True) as z:
            names = tuple(str(x) for x in z["joint_names"].tolist())
            contact_names = tuple(str(x) for x in z.get("contact_names", np.asarray([], dtype=object)).tolist())
            metadata_raw = str(np.asarray(z.get("metadata_json", np.asarray("{}"))).reshape(-1)[0])
            obj = cls(
                joint_names=names,
                parents=np.asarray(z["parents"], dtype=np.int32),
                local_rotations=np.asarray(z["local_rotations"], dtype=np.float64),
                global_rotations=np.asarray(z["global_rotations"], dtype=np.float64),
                root_positions=np.asarray(z["root_positions"], dtype=np.float64),
                rest_positions=np.asarray(z["rest_positions"], dtype=np.float64),
                posed_positions=np.asarray(z["posed_positions"], dtype=np.float64),
                fps=float(np.asarray(z["fps"]).reshape(-1)[0]),
                contact_names=contact_names,
                contacts=np.asarray(z["contacts"], dtype=np.float64) if "contacts" in z.files else None,
                source_name=str(np.asarray(z.get("source_name", np.asarray("unknown"))).reshape(-1)[0]),
                metadata=json.loads(metadata_raw),
            )
        return obj.validate()
