#!/usr/bin/env python3
"""Apply the TRELLIS.2 DINOv3 layer-path compatibility fix in-place.

Current Transformers versions wrap the DINOv3 encoder under `.model`, while the
pinned TRELLIS.2 source expects `.layer` directly on the top-level model. This
implements the compatibility approach proposed upstream in microsoft/TRELLIS.2#156.
"""
from pathlib import Path
import os

root = Path(os.environ.get("TRELLIS2_ROOT", "/content/TRELLIS.2")).expanduser().resolve()
path = root / "trellis2" / "modules" / "image_feature_extractor.py"
if not path.is_file():
    raise FileNotFoundError(f"TRELLIS DINOv3 extractor not found: {path}")

text = path.read_text(encoding="utf-8")
old = "        for i, layer_module in enumerate(self.model.layer):\n"
marker = 'encoder = getattr(self.model, "model", self.model)'
new = (
    "        # Transformers >=5 wraps DINOv3ViTEncoder under self.model.model.\n"
    "        # Keep compatibility with both old and new Transformers layouts.\n"
    '        encoder = getattr(self.model, "model", self.model)\n'
    "        for i, layer_module in enumerate(encoder.layer):\n"
)

if old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[DINOv3 FIX] ✅ Patched: {path}", flush=True)
elif marker in text:
    print(f"[DINOv3 FIX] ✅ Already patched: {path}", flush=True)
else:
    raise RuntimeError(
        "TRELLIS DINOv3 extractor no longer matches the expected pinned source; "
        "refusing to modify it automatically."
    )
