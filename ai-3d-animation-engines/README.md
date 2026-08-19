# AI 3D + Animation Engines (Colab)

Two independent Colab-first engines for AI-assisted Unreal asset production.

## Recommended — one Colab, two engines

Open [`AI_3D_Animation_Engine_Colab.ipynb`](AI_3D_Animation_Engine_Colab.ipynb).

The **3D Engine and Animation Engine live in the same Colab notebook**, but remain independent:

- run only the 3D section for objects, environments or unanimated characters;
- run only the Animation section when you already have a humanoid;
- run 3D first and then manually select its generated GLB in the Animation section without downloading/re-uploading it.

The heavy environments remain isolated:

```text
trellis2 Conda env  -> TRELLIS.2 3D generation
ardy Conda env      -> ARDY motion generation
mia Conda env       -> Make-It-Animatable + Blender + retarget
```

## Fresh-A100 + Google Drive build cache

The notebook assumes disposable Colab runtimes. An optional Drive cache cell mounts:

```text
/content/drive/MyDrive/AI3D_Engine_Cache
```

The cache automatically stores and reuses:

```text
AI3D_Engine_Cache/
├── sources/     pinned source snapshots
├── wheels/      compiled/native CUDA wheels
└── downloads/   small reusable helper binaries
```

The first compatible run prints `CACHE MISS`, downloads/builds locally, then saves the reusable result to Drive. Later fresh runtimes print `CACHE HIT` and restore the compatible source/build artifact instead of rebuilding it.

TRELLIS compiled-wheel cache keys include the pinned TRELLIS revision, Python version, PyTorch version, CUDA version and GPU compute capability. This is intended to prevent, for example, an A100 `sm80` build from being treated as the same artifact as a different GPU architecture.

### Deliberately not cached in Drive

TRELLIS, ARDY and Llama runtime model weights are intentionally **not** placed in the Drive build cache. They download fresh into local Colab storage and still have to be loaded into CPU/GPU memory each fresh runtime.

This keeps Drive focused on avoiding repeat source downloads and expensive native/CUDA compilation rather than turning it into the runtime model filesystem.

## Architecture

```text
REFERENCE IMAGE
      |
      v
+-----------------------+
| 3D ENGINE             |
| TRELLIS.2             |
+-----------------------+
      |
      +--> object/environment.glb --> Unreal
      |
      +--> character.glb
                 |
          manual selection only
                 v
       +--------------------------+
       | ANIMATION ENGINE         |
       | Make-It-Animatable       |
       | + ARDY Core              |
       | + Auto-Rig-Pro retarget  |
       | + contract validator     |
       +--------------------------+
                 |
                 +--> character_animated.fbx
                 +--> character_material_source.glb
                 +--> animation_contract_report.json
                 |
                 v
               Unreal
```

Generating a 3D asset never automatically sends it into animation.

## Engine 1 — 3D Engine

The combined notebook calls:

- `3d-engine/install_3d.sh`
- `3d-engine/run_trellis2.py`

**Input**
- one reference image;
- asset type;
- one explicit real-world dimension.

**Outputs**
- `asset.glb` — PBR 3D asset;
- `asset_manifest.json` — scale/material/downstream I/O contract;
- optional turntable MP4.

The installer uses `cache_common.sh` to restore/download the pinned TRELLIS source and to reuse/build cached wheels for the expensive native stack, including FlashAttention, nvdiffrast, nvdiffrec, CuMesh, O-Voxel and FlexGEMM.

### 3D I/O guarantees

1. **Real-world scale is mandatory.** The notebook requires one target dimension in meters and uniformly scales the generated mesh before export.
2. **No required WebP GLB extension.** The export uses standard GLB textures for broader Unreal/glTF compatibility.
3. The manifest records dimensions, scale, PBR texture presence, alpha status and downstream eligibility.

## Engine 2 — Animation Engine

The combined notebook calls:

- `animation-engine/install_animation.sh`
- `animation-engine/run_animation_pipeline.py`

**Input**
- one manually selected humanoid `.glb`, `.fbx`, `.obj`, or polygon `.ply`;
- an ARDY text motion prompt.

**Pipeline**

1. ARDY Core generates `.npz` motion.
2. `enrich_ardy_motion.py` validates the exact motion tensors and attaches Core skeleton metadata.
3. `preview_ardy_motion.py` renders a skeleton MP4.
4. `ardy_motion_to_fbx.py` creates a validated ARDY source FBX.
5. `rig_character_mia.py` auto-rigs the selected mesh with Make-It-Animatable.
6. `retarget_with_mia.py` retargets ARDY motion while preserving FPS.
7. `validate_animation_contract.py` checks scale, timing and motion transfer.
8. `run_animation_pipeline.py` creates the Unreal handoff ZIP.

The Animation installer also caches the pinned ARDY and Make-It-Animatable source snapshots. ARDY's package wheel includes its native MotionCorrection extension and is persisted in the Drive wheel cache. MIA neural-network weights remain fresh/local by design.

## Character material contract

FBX is treated as the **skeletal/animation master**, not the canonical PBR material master.

If the selected character is a GLB, the pipeline preserves it as:

```text
character_material_source.glb
```

Use that as the PBR reference if Unreal's FBX material conversion does not reproduce the original metallic/roughness/alpha appearance.

## ARDY ↔ Mixamo skeleton contract

ARDY Core has 27 joints. The no-finger Make-It-Animatable body rig has 22 shared Mixamo body bones.

Shared body bones are mapped by exact `mixamorig:*` names. ARDY-only joints remain source helpers. Because this is semantic retargeting rather than skeleton identity, the final validator compares normalized motion trajectories instead of assuming success because an FBX file exists.

## Unreal handoff

See [`UNREAL_IMPORT.md`](UNREAL_IMPORT.md).

In short:

- static props/environments → import the 3D Engine GLB;
- animated characters → import `character_animated.fbx` as Skeletal Mesh/Animation;
- keep `character_material_source.glb` as the PBR reference when present;
- preserve the ARDY sample rate from the package/contract report;
- scale is resolved by the 3D Engine manifest.

## Recommended runtime sequence

```text
Fresh A100
   ↓
Shared Setup
   ↓
Mount Drive build cache
   ↓
Install only the engine needed
   ↓
CACHE HIT: restore compatible build artifacts
CACHE MISS: build/download once and save to Drive
   ↓
Generate multiple assets while the runtime is alive
   ↓
Download outputs
   ↓
Disconnect A100
```

## Pinned upstream revisions

- Microsoft TRELLIS.2: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- NVIDIA ARDY: `693f74d13b3d04a0a22ce127ee79c929dd89756b`
- Make-It-Animatable: `d60cc7e01ff8da46448e458dbf450e8967b34e77`
- nvdiffrast v0.4.0: `253ac4fcea7de5f396371124af597e6cc957bfae`
- nvdiffrec renderutils: `b296927cc7fd01c2ac1087c8065c4d7248f72da4`
- CuMesh: `12289e1062f0603f2f0d0771b02e1395d247f26f`
- FlexGEMM: `6dd94a859c26ee8246888502eada3dd8ad85532e`

## External projects / licenses

- Microsoft TRELLIS.2 — MIT
- NVIDIA ARDY code — Apache-2.0; model weights use NVIDIA's model license
- Make-It-Animatable — MIT
- Auto-Rig-Pro fork bundled by Make-It-Animatable
- Meta Llama 3 8B Instruct — used by ARDY's text encoder and gated on Hugging Face

Review upstream licenses before redistributing checkpoints, bundled code or generated assets.
