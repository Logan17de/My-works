# AI 3D + Animation Engines (Colab)

Two independent Colab-first engines for AI-assisted Unreal asset production.

## Recommended — one Colab, two engines

Open [`AI_3D_Animation_Engine_Colab.ipynb`](AI_3D_Animation_Engine_Colab.ipynb).

This is now the primary entry point. The **3D Engine and Animation Engine live in the same Colab notebook**, but remain independent:

- run only the 3D section when you want objects, environments or unanimated characters;
- run only the Animation section when you already have a humanoid;
- run 3D first and then manually choose its generated GLB in the Animation section without downloading/re-uploading it.

The heavy environments are still isolated so their dependencies do not collide:

```text
trellis2 Conda env  -> TRELLIS.2 3D generation
ardy Conda env      -> ARDY motion generation
mia Conda env       -> Make-It-Animatable + Blender + retarget
```

The older per-engine notebooks are retained as focused/debug notebooks:

- [`3d-engine/TRELLIS2_Colab.ipynb`](3d-engine/TRELLIS2_Colab.ipynb)
- [`animation-engine/ARDY_Animation_Colab.ipynb`](animation-engine/ARDY_Animation_Colab.ipynb)

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

The engines are deliberately separate. Generating a 3D asset never implies that it should be animated.

## Engine 1 — 3D Engine

The combined notebook calls `3d-engine/install_3d.sh` and `3d-engine/run_trellis2.py`.

**Input**
- one reference image;
- asset type;
- one explicit real-world dimension.

**Outputs**
- `asset.glb` — PBR 3D asset;
- `asset_manifest.json` — scale/material/downstream I/O contract;
- optional turntable MP4.

### I/O guarantees

The 3D Engine follows TRELLIS.2/O-Voxel's official GLB path, but adds two production constraints:

1. **Real-world scale is mandatory.** TRELLIS generates in normalized space, so the notebook requires a target axis and target size in meters and uniformly scales the mesh before export.
2. **The exported GLB does not require `EXT_texture_webp`.** `extension_webp=False` is used so Unreal and other standard glTF consumers do not depend on WebP extension support.

The manifest records:
- final width/height/depth in meters and centimeters;
- requested target dimension;
- scale factor from TRELLIS normalized space;
- PBR texture presence;
- alpha-mode status;
- whether the asset is eligible to enter the Animation Engine.

TRELLIS/O-Voxel currently exports material alpha in an OPAQUE material by default. The manifest records that fact; transparent assets may still need an Unreal material adjustment.

## Engine 2 — Animation Engine

The combined notebook calls `animation-engine/install_animation.sh` and then `animation-engine/run_animation_pipeline.py`.

**Input**
- one humanoid `.glb`, `.fbx`, `.obj`, or polygon `.ply` selected manually;
- an ARDY text motion prompt.

**Pipeline**

1. ARDY Core generates `.npz` motion.
2. `enrich_ardy_motion.py` validates the exact ARDY output tensors and attaches Core skeleton metadata.
3. `preview_ardy_motion.py` renders a quick skeleton MP4.
4. `ardy_motion_to_fbx.py` builds a validated ARDY source FBX:
   - root translation only;
   - rotation-only child bones;
   - exact `mixamorig:*` names for the 22 shared body bones;
   - ARDY-only helper bones retained where necessary;
   - FK validation against ARDY `posed_joints`.
5. `rig_character_mia.py` auto-rigs the chosen mesh with Make-It-Animatable and validates the resulting bound Mixamo-style body skeleton.
6. `retarget_with_mia.py` forces exact shared Mixamo mappings and explicitly preserves the ARDY FPS during retarget/export.
7. `validate_animation_contract.py` reimports the source, rigged and final files and verifies:
   - character scale did not drift beyond tolerance;
   - final FBX FPS matches ARDY;
   - root motion was not lost;
   - head/hand/foot motion was not frozen or catastrophically distorted.
8. `run_animation_pipeline.py` packages the validated Unreal handoff ZIP.

### Character material contract

FBX is treated as the **skeletal/animation master**, not the canonical PBR material master.

If the selected character is a GLB, the pipeline preserves it as:

`character_material_source.glb`

Use that file as the source of TRELLIS PBR appearance if Unreal's FBX material conversion does not reproduce metallic/roughness/alpha correctly.

## ARDY ↔ Mixamo skeleton contract

ARDY Core has 27 joints. The no-finger Make-It-Animatable body rig has 22 shared Mixamo body bones.

The shared bones are mapped by exact `mixamorig:*` names. ARDY's extra joints (notably `Spine3` and hand helper/end joints) stay in the source animation skeleton instead of being falsely renamed into target bones. Auto-Rig-Pro retargeting handles the hierarchy difference.

Because this is a semantic retarget rather than a one-to-one skeleton identity, the pipeline does not assume success merely because an FBX exists. The final contract validator compares normalized motion trajectories and fails only on severe transfer problems.

## Unreal handoff

See [`UNREAL_IMPORT.md`](UNREAL_IMPORT.md).

In short:

- static props/environments: import the 3D Engine GLB;
- animated character: import `character_animated.fbx` as Skeletal Mesh/Animation;
- keep/import `character_material_source.glb` as the PBR reference when present;
- preserve the ARDY sample rate (use the package/contract report rather than assuming 30 FPS);
- asset scale is already resolved by the 3D Engine manifest.

## Colab strategy

The unified notebook uses one disposable Colab runtime while keeping the three heavy software environments separate.

Keep permanently:
- the unified notebook/scripts in GitHub;
- source images/models;
- GLB/FBX outputs;
- manifests and validation reports.

Redownload per fresh Colab runtime:
- upstream repositories;
- checkpoints;
- Conda environments.

## Pinned upstream revisions

- Microsoft TRELLIS.2: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- NVIDIA ARDY: `693f74d13b3d04a0a22ce127ee79c929dd89756b`
- Make-It-Animatable: `d60cc7e01ff8da46448e458dbf450e8967b34e77`

Pinning matters because the notebooks call upstream Python APIs directly.

## External projects / licenses

- Microsoft TRELLIS.2 — MIT
- NVIDIA ARDY code — Apache-2.0; model weights use NVIDIA's model license
- Make-It-Animatable — MIT
- Auto-Rig-Pro fork bundled by Make-It-Animatable
- Meta Llama 3 8B Instruct — used by ARDY's text encoder and gated on Hugging Face

Review upstream licenses before redistributing checkpoints, bundled code, or generated assets.
