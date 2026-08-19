# Universal Motion Retargeting Engine

The Animation Engine no longer needs to be architected as `ARDY -> MIA`.

## Runtime architecture

```text
Motion Sources
  ARDY (active)
  BVH / mocap / SOMA / video-motion (future)
        |
        v
Motion Adapter
        |
        v
Canonical Motion
  - semantic joint names
  - local rest-relative rotations
  - global rotations
  - root trajectory
  - rest / posed joint positions
  - contacts
  - FPS
        |
        v
Universal Retarget Core
        ^
        |
Target Rig Adapter
  MIA/Mixamo (active)
  Rigify / UE5 / MetaHuman (future)
        |
        v
IK / Contact Correction
  V1: isolated extension point; contact data preserved
  Future: foot lock, pelvis correction, hand IK, ground alignment
        |
        v
Strict deformation + motion validator
        |
        v
FBX / GLB preview / Unreal package
```

## Design rules

1. The retarget core contains no ARDY or MIA bone names.
2. Motion sources convert to `CanonicalMotion` before retargeting.
3. Target rigs are described by adapters/presets.
4. Source bone rolls and source armature transforms are never copied to the target rig.
5. Missing target joints are handled by explicit collapse rules in the rig preset.
6. Retargeting and IK/contact correction remain separate stages.
7. The final gate evaluates the actual skinned/deformed mesh, not only static object dimensions.

## Active V1 components

- Motion source: NVIDIA ARDY Core27.
- Rig provider: Make-It-Animatable.
- Rig preset: `mia_mixamo`.
- Retarget core: `universal_motion/retarget.py`.
- IK/contact correction: `none_v1` (architectural hook only; ARDY contact channels are preserved).
- Validator: `validate_animation_contract.py`.

The MIA preset maps target semantics to `mixamorig:*` bones. ARDY Core27 has an extra `Spine3` joint that the no-finger MIA/Mixamo body rig does not have, so V1 explicitly collapses canonical `chest + upper_chest` motion into target `mixamorig:Spine2`. This preserves the downstream Neck/Shoulder global motion instead of dropping the extra spine rotation.

## Files

```text
animation-engine/
  run_universal_animation_pipeline.py   full top-level pipeline
  resume_universal_animation.py         resume from existing source motion / rig
  build_canonical_motion.py             Motion Adapter + Skeleton Mapper CLI
  run_universal_retarget.py             Blender/headless target retarget CLI
  universal_motion/
    model.py                             CanonicalMotion schema
    sources/ardy.py                      ARDY MotionSource adapter
    retarget.py                          source/rig-neutral retarget math
    ik.py                                separate correction-layer interface
    rigs/
      base.py                            generic data-driven rig adapter
      mia.py                             active MIA adapter
      registry.py                        rig-adapter registry
    rig_presets/
      mia_mixamo.json                    active target skeleton preset
```

## Add a new motion source

Implement an adapter that emits `CanonicalMotion`. It must not know anything about MIA, Mixamo, Rigify, UE5, or MetaHuman.

## Add a new target rig

Add a rig preset/adapter that resolves canonical semantics to target bone names/rest transforms. It must not know anything about ARDY.

This separation lets the engine evolve to `SOMA -> UE5`, `BVH -> Rigify`, etc. without rewriting the retarget mathematics.
