# Unreal Engine import contract

This file describes the handoff from the two Colab engines. It is intentionally about **data contracts**, not Unreal gameplay setup.

## Static objects / environments

Use:

`asset.glb`

from the 3D Engine.

The GLB is exported with:
- glTF 2.0 coordinates;
- real-world scale baked in meters;
- standard GLB texture encoding (no required `EXT_texture_webp`);
- TRELLIS/O-Voxel PBR base-color and metallic/roughness textures.

Also keep:

`asset_manifest.json`

The manifest records final dimensions in both meters and centimeters.

### Transparency

TRELLIS/O-Voxel preserves an alpha channel but its current exporter sets the material to `OPAQUE`.

If the asset is meant to contain transparent/translucent surfaces (glass, leaves, hair cards, etc.), inspect the alpha texture and configure the Unreal material blend mode appropriately.

## Animated characters

The Animation Engine ZIP contains:

- `character_animated.fbx` — skeletal mesh / skeleton / baked animation master;
- `character_material_source.glb` — PBR material master when the selected source was GLB;
- `animation_contract_report.json` — scale, FPS and motion-transfer verification;
- `package_manifest.json`;
- `motion_preview.mp4`.

### Recommended import responsibilities

Treat each file as authoritative for a different thing:

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

Do not blindly force 30 FPS. The pipeline explicitly writes the ARDY rate into Blender before final FBX export and validates it by reimporting the result.

### Scale

The original character should already have a real-world height/size from the 3D Engine.

The Animation Engine checks source → rigged → animated mesh height and fails in strict mode if the rigging/retarget path changes it beyond the configured tolerance.

## Why the validator exists

ARDY Core and Make-It-Animatable are not identical skeletons:

- ARDY Core: 27 joints
- no-finger MIA/Mixamo body: 22 shared body bones

The bridge retains ARDY helper joints instead of inventing false one-to-one mappings. After retargeting, the validator compares:

- normalized root trajectory;
- Head motion relative to Hips;
- Left/Right Hand motion relative to Hips;
- Left/Right Foot motion relative to Hips.

It subtracts the first-frame pose and normalizes each rig by its own body height. This makes the check much less sensitive to different character proportions while still detecting:
- lost root motion;
- frozen limbs;
- severe trajectory distortion;
- major retarget failures.

The report may contain warnings for moderate drift. `passed=false` is reserved for hard contract failures.
