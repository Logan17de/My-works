# AI 3D + Animation Engines (Colab)

Two independent Colab-first engines for AI-assisted Unreal asset production.

## Recommended notebooks

- `TRELLIS2_A100_Optimized_Colab.ipynb` — 3D generation on A100.
- `Animation_Engine_L4_Low_Disk_Colab.ipynb` — animation on L4 24 GB.
- `AI_3D_Animation_Engine_Colab.ipynb` — combined/reference workflow with both engines available.

The 3D Engine and Animation Engine remain independent. Generate a GLB with TRELLIS when needed, then start a fresh animation runtime and upload/select that humanoid explicitly.

The heavy environments remain isolated:

```text
trellis2 Conda env  -> TRELLIS.2 3D generation
ardy Conda env      -> ARDY motion generation
mia Conda env       -> Make-It-Animatable rigging + Blender direct bake/export
```

## Disposable Colab runtimes + Google Drive build cache

The notebooks assume disposable Colab runtimes. An optional Drive cache mounts:

```text
/content/drive/MyDrive/AI3D_Engine_Cache
```

The cache stores reusable source/build artifacts:

```text
AI3D_Engine_Cache/
├── sources/     pinned source snapshots
├── wheels/      compiled/native CUDA wheels
└── downloads/   small reusable helper binaries
```

TRELLIS, ARDY and Llama runtime model weights are intentionally not kept in the Drive build cache. They download to local Colab storage for the current runtime.

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
       +-------------------------------+
       | ANIMATION ENGINE              |
       | Make-It-Animatable auto-rig   |
       | + NVIDIA ARDY Core motion     |
       | + direct rest-space baker     |
       | + skinned-mesh validator      |
       +-------------------------------+
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

The 3D notebook calls:

- `3d-engine/install_3d.sh`
- `3d-engine/run_trellis2.py` or the A100-profiled runner

**Input**
- one reference image;
- asset type;
- one explicit real-world dimension.

**Outputs**
- `asset.glb` — PBR 3D asset;
- `asset_manifest.json` — scale/material/downstream I/O contract;
- optional turntable MP4.

### 3D I/O guarantees

1. Real-world scale is mandatory.
2. Standard GLB texture encoding is used for broader Unreal/glTF compatibility.
3. The manifest records dimensions, scale, PBR texture presence, alpha status and downstream eligibility.

## Engine 2 — Animation Engine

The animation notebook calls:

- `animation-engine/install_animation_low_disk.sh`
- `animation-engine/run_animation_pipeline.py`

**Input**
- one manually selected humanoid `.glb`, `.fbx`, `.obj`, or polygon `.ply`;
- an ARDY text motion prompt.

**Current pipeline**

1. ARDY Core generates `.npz` motion.
2. `enrich_ardy_motion.py` validates the tensors and attaches Core skeleton metadata in `motion_bridge.npz`.
3. `preview_ardy_motion.py` renders a skeleton MP4.
4. `rig_character_mia.py` auto-rigs the selected mesh with Make-It-Animatable.
5. `retarget_ardy_direct.py` reads ARDY global joint rotations directly from `motion_bridge.npz`, converts them into the target MIA rig's own rest-space, and bakes the target action without Auto-Rig-Pro motion retargeting.
6. `validate_animation_contract.py` checks static scale, FPS, normalized motion trajectories, and the actual evaluated/skinned mesh deformation over sampled animation frames.
7. `run_animation_pipeline.py` creates the Unreal handoff ZIP only after the strict validator passes.

### Why the direct retargeter replaced the ARP bridge

ARDY Core and the MIA/Mixamo rig share semantic body-joint names, but their rest axes, bone rolls, proportions and helper-joint hierarchy are not identical. A synthetic ARDY FBX with Mixamo-like names can therefore look compatible while still interpreting rotations in the wrong local coordinate frames.

The current direct baker does not copy ARDY bone rolls and does not resize either armature. Instead it:

```text
ARDY global joint rotation
        ↓ coordinate conversion
motion delta in Blender space
        ↓ target armature object-space conversion
compose with target bone's own rest orientation
        ↓
bake directly onto the MIA target bone
```

Only root motion is scaled by the target/source skeleton-height ratio.

`ardy_motion_to_fbx.py` and `retarget_with_mia.py` remain legacy/debug helpers and are not used by the current production pipeline.

## Character material contract

FBX is treated as the skeletal/animation master, not the canonical PBR material master.

If the selected character is a GLB, the pipeline preserves it as:

```text
character_material_source.glb
```

Use that as the PBR reference if Unreal's FBX material conversion does not reproduce the original metallic/roughness/alpha appearance.

## ARDY ↔ MIA body contract

ARDY Core has 27 joints. The no-finger MIA/Mixamo body rig has 22 shared body bones.

The current direct baker uses the 22 shared body joints. ARDY-only helpers such as `Spine3` and hand-end helper joints are not exported as animation targets on the MIA rig. Missing finger bones are expected when `--no-fingers` is selected.

The validator does not assume success merely because an FBX file exists. It compares ARDY bridge trajectories to final target trajectories and also inspects the evaluated skinned mesh for severe edge stretching/compression or implausible animated extents.

## Unreal handoff

See [`UNREAL_IMPORT.md`](UNREAL_IMPORT.md).

In short:

- static props/environments → import the 3D Engine GLB;
- animated characters → import `character_animated.fbx` as Skeletal Mesh/Animation;
- keep `character_material_source.glb` as the PBR reference when present;
- preserve the ARDY sample rate from the package/contract report.

## Recommended runtime sequence

```text
A100 runtime
  -> TRELLIS 3D generation
  -> download GLB
  -> disconnect

Fresh L4 24 GB runtime
  -> Animation low-disk install
  -> upload humanoid GLB
  -> ARDY + MIA rig + direct bake
  -> strict deformation validation
  -> download Unreal package
  -> disconnect
```

This separation also avoids keeping the TRELLIS environment/model cache on the same ~113 GB Colab disk as ARDY/Llama/MIA.

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
- Auto-Rig-Pro fork bundled by Make-It-Animatable (legacy/debug retarget path only in this repo)
- Meta Llama 3 8B Instruct — used by ARDY's text encoder and gated on Hugging Face

Review upstream licenses before redistributing checkpoints, bundled code or generated assets.
