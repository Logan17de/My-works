# Unreal Engine import contract

This file describes the handoff from the two Colab engines. It is intentionally about **data contracts**, not Unreal gameplay setup.

## Static objects / environments

Use `asset.glb` from the 3D Engine.

The GLB is exported with:
- glTF 2.0 coordinates;
- real-world scale baked in meters;
- standard GLB texture encoding;
- TRELLIS/O-Voxel PBR base-color and metallic/roughness textures.

Also keep `asset_manifest.json`. The manifest records final dimensions in both meters and centimeters.

### Transparency

TRELLIS/O-Voxel preserves an alpha channel but its current exporter sets the material to `OPAQUE`.

If the asset is meant to contain transparent/translucent surfaces, inspect the alpha texture and configure the Unreal material blend mode appropriately.

## Animated characters

The Animation Engine ZIP contains:

- `character_animated.fbx` — skeletal mesh / skeleton / baked animation master;
- `character_material_source.glb` — PBR material master when the selected source was GLB;
- `animation_contract_report.json` — static scale, evaluated deformation, FPS and motion-transfer verification;
- `package_manifest.json`;
- `motion_preview.mp4`;
- optional `character_animated_preview.glb`.

### Recommended import responsibilities

```text
character_animated.fbx
    authoritative for:
    - skeleton
    - skin weights
    - baked animation
    - animation frame range / timing

character_material_source.glb
    authoritative for:
    - base color
    - metallic
    - roughness
    - alpha/PBR appearance
```

Do not rely on FBX to reproduce the full TRELLIS metallic/roughness material graph perfectly.

### Animation sample rate

Read `fps` from `package_manifest.json` / `animation_contract_report.json`.

Do not blindly force 30 FPS. The pipeline writes the ARDY rate into Blender before final FBX export and validates it by reimporting the result.

### Scale and deformation

The original character should already have a real-world height/size from the 3D Engine.

The Animation Engine checks source → rigged → animated static mesh height. It also evaluates the **actual skinned mesh** at sampled animation frames and measures mesh-edge stretch/compression plus animated extents.

This distinction is important: an FBX can preserve the armature/object scale while still producing a visibly broken skin deformation. The strict validator is designed to reject that case before packaging.

## Current ARDY → MIA transfer contract

ARDY Core and Make-It-Animatable are not identical skeletons:

- ARDY Core: 27 joints;
- no-finger MIA/Mixamo body: 22 shared body bones.

The production path no longer uses Auto-Rig-Pro to infer the ARDY→MIA motion-space conversion.

Instead `retarget_ardy_direct.py` reads `motion_bridge.npz` and, for each of the 22 shared body joints:

1. reads the ARDY global rotation;
2. converts it into Blender coordinates;
3. converts that motion delta into the target armature object space;
4. composes it with the **target MIA bone's own rest orientation**;
5. bakes the resulting pose directly on the target rig.

ARDY-only helper joints such as `Spine3` and hand-end helpers are not animation targets on the final MIA rig. Finger helpers are also absent when the no-finger rig option is used.

Only Hips/root motion is proportionally scaled using the source/target skeleton-height ratio. The target character and target armature are never resized during motion transfer.

## Why the validator exists

The final validator compares:

- normalized root trajectory;
- Head motion relative to Hips;
- Left/Right Hand motion relative to Hips;
- Left/Right Foot motion relative to Hips;
- static source/rig/final mesh height;
- evaluated skinned-mesh edge stretch/compression across sampled frames;
- animated mesh extent;
- final FBX FPS.

Source motion is read directly from `motion_bridge.npz`, so validation does not depend on a synthetic ARDY FBX or on matching source/target bone rolls.

The report may contain warnings for moderate drift. `passed=false` is reserved for hard contract failures. The Unreal ZIP is created only after the strict contract passes.
