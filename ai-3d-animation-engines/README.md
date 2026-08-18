# AI 3D + Animation Engines (Colab)

Two independent Colab-first engines for AI-assisted Unreal asset production.

## Why two engines?

The 3D engine creates **all kinds of assets**: props, furniture, buildings, environment pieces, and characters. Most generated assets never need a skeleton or motion.

The animation engine is therefore deliberately separate. You manually choose only a humanoid character that should move, then send that file into the animation engine.

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
          | + ARDY               |
          | + retarget bridge    |
          +----------------------+
                    |
                    v
             animated_character.fbx
                    |
                    v
                  Unreal
```

## Engine 1 — 3D Engine

Open [`3d-engine/TRELLIS2_Colab.ipynb`](3d-engine/TRELLIS2_Colab.ipynb) in Google Colab.

Input:
- one reference image

Output:
- `asset.glb` — PBR-ready 3D asset
- `asset_preview.mp4` — turntable preview

The notebook uses Microsoft's official TRELLIS.2 repository directly. No ComfyUI is required.

> TRELLIS.2's official implementation currently targets Linux and an NVIDIA GPU with at least 24 GB VRAM. An A100/H100-class Colab runtime is therefore the clean path. Lower-VRAM community hacks are intentionally not part of this engine.

## Engine 2 — Animation Engine

Open [`animation-engine/ARDY_Animation_Colab.ipynb`](animation-engine/ARDY_Animation_Colab.ipynb).

Input:
- a humanoid `.glb` / `.fbx` that **you manually select**
- a text motion prompt, for example `A person walks forward and waves with the right hand.`

Pipeline:

1. Make-It-Animatable rigs the selected character to a Mixamo-style humanoid skeleton.
2. ARDY generates Core-skeleton motion from the prompt.
3. `enrich_ardy_motion.py` adds Core skeleton metadata to ARDY's `.npz`.
4. `ardy_motion_to_fbx.py` builds an animated ARDY source skeleton in FBX.
5. Make-It-Animatable's bundled Auto-Rig-Pro retargeter transfers the source motion to the generated character rig.
6. The engine exports an animated FBX for Unreal.

Outputs:
- `motion.npz` — original ARDY motion
- `motion_bridge.npz` — motion + skeleton metadata
- `motion_preview.mp4` — quick skeleton preview
- `ardy_source.fbx` — ARDY motion as an animation skeleton
- `character_rigged.fbx` — selected character after auto-rigging
- `character_animated.fbx` — final retargeted character for Unreal
- `character_animated.glb` — quick browser/GLB preview when export succeeds

## Important boundary

TRELLIS.2 and ARDY are official model stages. The **ARDY → arbitrary generated character** bridge is integration code in this repository, not an official NVIDIA feature. ARDY officially emits motion `.npz` data; the FBX creation and retargeting steps here are glue code.

That means:
- motion generation itself is the stable stage;
- auto-rigging quality depends on the generated character geometry;
- retargeting may need adjustment for unusual body proportions, clothing, accessories, tails, wings, etc.

## Colab strategy

The notebooks intentionally install their large dependencies into disposable Colab runtimes.

Keep permanently:
- these notebooks/scripts in GitHub;
- your input images/models;
- final `.glb`, `.fbx`, `.npz`, and preview files in Drive.

Re-download on a fresh runtime:
- TRELLIS.2 code + weights;
- ARDY code + weights;
- Make-It-Animatable code + weights.

This avoids maintaining conflicting CUDA/PyTorch stacks on the local PC.

## External projects

This integration relies on:
- Microsoft TRELLIS.2 — MIT
- NVIDIA ARDY — Apache-2.0 code; model license is listed by NVIDIA
- Make-It-Animatable — MIT
- Auto-Rig-Pro fork bundled by Make-It-Animatable
- Meta Llama 3 8B Instruct for ARDY text encoding; access must be granted on Hugging Face

Review each upstream license before redistributing generated assets, checkpoints, or bundled dependencies.
