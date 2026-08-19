# AI 3D + Animation Engines (Colab)

Two independent Colab-first engines for AI-assisted Unreal asset production.

## Architecture

```text
REFERENCE IMAGE
      |
      v
+-------------------+
| 3D ENGINE         |
| TRELLIS.2         |
+-------------------+
      |
      +----> prop.glb ----------------------> Unreal
      +----> chair.glb ---------------------> Unreal
      +----> building.glb ------------------> Unreal
      |
      +----> character.glb
                    |
             manual selection
                    v
          +----------------------+
          | ANIMATION ENGINE     |
          | Make-It-Animatable   |
          | + ARDY Core          |
          | + validated bridge   |
          +----------------------+
                    |
                    v
             animated_character.fbx
                    |
                    v
                  Unreal
```

The engines are deliberately separate. The 3D Engine can generate any asset; only a humanoid you explicitly choose enters the Animation Engine.

## Engine 1 — 3D Engine

Open [`3d-engine/TRELLIS2_Colab.ipynb`](3d-engine/TRELLIS2_Colab.ipynb).

**Input**
- one reference image

**Outputs**
- `asset.glb` — PBR 3D asset
- `asset_preview.mp4` — optional turntable preview

The notebook calls TRELLIS.2 directly; ComfyUI is not required.

### Reliability changes

- refuses to run the official path below 24 GB VRAM;
- recreates the `trellis2` environment cleanly when the install cell is rerun;
- pins the upstream TRELLIS.2 commit;
- uses PyTorch/CUDA 12.4 as expected by upstream;
- installs a matching CUDA 12.4 toolkit inside Conda when Colab does not expose `/usr/local/cuda-12.4`;
- smoke-tests imports before model download/generation;
- exports the GLB **before** optional preview rendering, so an HDRI/render failure cannot discard a successful asset.

## Engine 2 — Animation Engine

Open [`animation-engine/ARDY_Animation_Colab.ipynb`](animation-engine/ARDY_Animation_Colab.ipynb).

**Input**
- a humanoid `.glb`, `.fbx`, `.obj`, or polygon `.ply` that you manually select;
- a text motion prompt.

**Pipeline**

1. ARDY Core generates motion as `.npz`.
2. `enrich_ardy_motion.py` validates the ARDY shapes/keys and attaches Core skeleton metadata.
3. `preview_ardy_motion.py` renders a quick skeleton MP4.
4. `ardy_motion_to_fbx.py` creates a clean source animation:
   - root translation only;
   - rotation-only child bones;
   - exact `mixamorig:*` names for the 22 shared base humanoid bones;
   - ARDY-only helper bones kept separate;
   - runtime FK validation against ARDY `posed_joints` before FBX export.
5. `rig_character_mia.py` auto-rigs the selected mesh with Make-It-Animatable and validates that the result is a bound Mixamo-style humanoid.
6. `retarget_with_mia.py` uses Make-It-Animatable's bundled Auto-Rig-Pro fork, forcing exact shared Mixamo bone mappings before baking.
7. Final animated FBX is exported for Unreal.

**Outputs**
- `motion.npz`
- `motion_bridge.npz`
- `motion_preview.mp4`
- `ardy_source.fbx`
- `character_rigged.fbx`
- `character_animated.fbx`
- `character_animated.glb` when the optional preview export succeeds

## Important scope

TRELLIS.2 and ARDY are upstream model stages. The **ARDY → arbitrary generated character** path is integration code in this repository, not an official NVIDIA workflow.

The bridge now validates its own kinematics, but final visual quality still depends on:
- the generated character having clear humanoid geometry;
- auto-rig blend weights;
- unusual proportions, clothing, hair, accessories, tails, wings, etc.;
- retarget quality.

The default Animation Engine intentionally removes finger bones from the target rig because ARDY Core is a body-motion model, not a full finger-animation system.

## Colab strategy

The notebooks use disposable runtimes and separate environments to avoid dependency conflicts.

Keep permanently:
- these notebooks/scripts in GitHub;
- source images/models;
- final GLB/FBX/NPZ/preview outputs in Drive.

Redownload in a fresh runtime:
- model repositories;
- model weights;
- Conda environments.

## Pinned upstream revisions

The notebooks currently target:

- Microsoft TRELLIS.2: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- NVIDIA ARDY: `693f74d13b3d04a0a22ce127ee79c929dd89756b`
- Make-It-Animatable: `d60cc7e01ff8da46448e458dbf450e8967b34e77`

Pinning matters because these notebooks call upstream Python APIs directly.

## External projects / licenses

- Microsoft TRELLIS.2 — MIT
- NVIDIA ARDY code — Apache-2.0; released model weights use NVIDIA's model license
- Make-It-Animatable — MIT
- Auto-Rig-Pro fork bundled by Make-It-Animatable
- Meta Llama 3 8B Instruct — used by ARDY's text encoder and gated on Hugging Face

Review upstream licenses before redistributing checkpoints, bundled code, or generated assets.
