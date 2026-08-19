from __future__ import annotations

import math
import numpy as np
import mathutils

from .model import CanonicalMotion


def rotation_only(matrix_world) -> np.ndarray:
    q = matrix_world.to_quaternion()
    q.normalize()
    return np.asarray(q.to_matrix(), dtype=np.float64)


def target_skeleton_height_world(armature, adapter):
    zs = []
    mw = armature.matrix_world
    for semantic in adapter.mapped_semantics():
        b = armature.data.bones[adapter.bone_name(semantic)]
        zs.extend((float((mw @ b.head_local).z), float((mw @ b.tail_local).z)))
    h = max(zs) - min(zs)
    if not math.isfinite(h) or h <= 1e-8:
        raise RuntimeError(f"Invalid target skeleton height: {h}")
    return h


def compose_chain(local_rotations, index, chain):
    """Collapse one or more canonical local rotations into one semantic joint.

    Canonical rotations are rest-relative local joint rotations.  If the target
    rig has fewer spine joints than the source, multiplying the local deltas in
    hierarchy order preserves the total orientation change without inventing a
    source bone roll on the target.
    """
    out = np.eye(3, dtype=np.float64)
    for semantic in chain:
        if semantic not in index:
            raise KeyError(f"Canonical motion missing semantic {semantic!r}")
        out = out @ np.asarray(local_rotations[index[semantic]], dtype=np.float64)
    return out


def bone_rest_local_rotation(bone) -> np.ndarray:
    """Return this Blender bone's rest orientation relative to its real parent.

    PoseBone.matrix_basis is defined relative to the parent and the bone's own
    rest transform.  Therefore the retarget core must work in this rest-local
    basis instead of assigning a target pose matrix in armature/global space.
    """
    if bone.parent is None:
        rel = bone.matrix_local.copy()
    else:
        rel = bone.parent.matrix_local.inverted() @ bone.matrix_local
    q = rel.to_quaternion()
    q.normalize()
    return np.asarray(q.to_matrix(), dtype=np.float64)


def orthonormal_rotation(matrix: np.ndarray, label: str) -> np.ndarray:
    """Project tiny numerical drift back onto SO(3) and reject bad matrices."""
    a = np.asarray(matrix, dtype=np.float64)
    if a.shape != (3, 3) or not np.isfinite(a).all():
        raise ValueError(f"{label} is not a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(a)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    if abs(np.linalg.det(r) - 1.0) > 1e-5:
        raise RuntimeError(f"{label} could not be normalized to a rotation")
    return r


def bake_motion(scene, armature, adapter, motion: CanonicalMotion, progress=None):
    """Bake CanonicalMotion onto a target rig using target rest-local deltas.

    Contract:
      * canonical local rotations are the authoritative motion deltas;
      * each delta is converted into the target bone's own rest-local basis;
      * only the canonical root/hips receives translation;
      * non-root PoseBone.location remains exactly zero;
      * no target bone length, source bone roll, or target rest offset is copied
        or recomputed by the retargeter.

    This is deliberately different from the old global-pose-matrix baker.  A
    full PoseBone.matrix assignment can decompose into non-zero location
    channels on ordinary FK bones, which stretches a skinned character even
    when the static armature height is unchanged.
    """
    motion.validate()
    idx = motion.index
    order = adapter.topological_semantics()
    if "hips" not in order:
        raise RuntimeError("Target rig adapter must map canonical 'hips'")

    target_height = target_skeleton_height_world(armature, adapter)
    motion_scale = target_height / motion.reference_height
    if not math.isfinite(motion_scale) or not (0.05 <= motion_scale <= 20.0):
        raise RuntimeError(f"Implausible motion scale ratio: {motion_scale}")

    # Canonical motion is stored in Blender/Unreal Z-up world axes.  Convert
    # rotation axes once into the target armature object's coordinate system.
    object_rotation = rotation_only(armature.matrix_world)
    world_to_armature_rotation = object_rotation.T
    world_to_armature_vector = np.asarray(
        armature.matrix_world.inverted().to_3x3(), dtype=np.float64
    )

    rest_local_rotation = {}
    for semantic in order:
        bone = armature.data.bones[adapter.bone_name(semantic)]
        rest_local_rotation[semantic] = bone_rest_local_rotation(bone)

    # Remove imported/default actions and guarantee a clean FK channel state.
    if armature.animation_data:
        armature.animation_data_clear()
    for pb in armature.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)

    scene.frame_start = 1
    scene.frame_end = motion.frames
    root0 = np.asarray(motion.root_positions[0], dtype=np.float64)
    root_rest_basis = rest_local_rotation["hips"]
    prev_quat = {semantic: None for semantic in order}
    report_every = max(1, motion.frames // 10)

    for fi in range(motion.frames):
        scene.frame_set(fi + 1)

        for semantic in order:
            # Source-local delta.  Collapsed target joints (currently the MIA
            # upper spine) compose consecutive canonical local rotations here.
            delta_canonical = compose_chain(
                motion.local_rotations[fi], idx, adapter.motion_chain(semantic)
            )
            delta_armature = (
                world_to_armature_rotation
                @ delta_canonical
                @ object_rotation
            )
            delta_armature = orthonormal_rotation(
                delta_armature, f"{semantic} frame {fi + 1} armature delta"
            )

            # Blender PoseBone.matrix_basis applies after the target bone's
            # rest-local transform B.  We need:
            #
            #       B @ Q = R_delta @ B
            #       Q     = inv(B) @ R_delta @ B
            #
            # so Q is the same semantic motion expressed in this target bone's
            # own rest-local axes, independent of source bone roll.
            basis = rest_local_rotation[semantic]
            basis_delta = basis.T @ delta_armature @ basis
            basis_delta = orthonormal_rotation(
                basis_delta, f"{semantic} frame {fi + 1} basis delta"
            )

            pb = armature.pose.bones[adapter.bone_name(semantic)]
            quat = mathutils.Matrix(basis_delta.tolist()).to_quaternion()
            quat.normalize()
            prev = prev_quat[semantic]
            if prev is not None and quat.dot(prev) < 0:
                quat.negate()
            pb.rotation_quaternion = quat
            prev_quat[semantic] = quat.copy()

            # Absolutely no FK-bone translation.  Bone offsets/lengths stay
            # authoritative from the target rig.
            if semantic != "hips":
                pb.location = (0.0, 0.0, 0.0)
                pb.scale = (1.0, 1.0, 1.0)
                pb.keyframe_insert(data_path="rotation_quaternion", frame=fi + 1)
                continue

            # Root motion is the only translation channel.  Convert the world
            # displacement to armature units (including FBX object scale), then
            # into the root bone's rest-local basis used by matrix_basis.
            root_delta_world = (
                np.asarray(motion.root_positions[fi], dtype=np.float64) - root0
            ) * motion_scale
            root_delta_armature = world_to_armature_vector @ root_delta_world
            root_delta_basis = root_rest_basis.T @ root_delta_armature
            if not np.isfinite(root_delta_basis).all():
                raise RuntimeError(f"Invalid root translation at frame {fi + 1}")
            pb.location = tuple(float(x) for x in root_delta_basis)
            pb.scale = (1.0, 1.0, 1.0)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=fi + 1)
            pb.keyframe_insert(data_path="location", frame=fi + 1)

        if progress and (
            fi == 0
            or fi + 1 == motion.frames
            or (fi + 1) % report_every == 0
        ):
            progress.info(
                f"Universal local-space bake {fi + 1}/{motion.frames} "
                f"({(fi + 1) / motion.frames * 100:.0f}%)"
            )

    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RuntimeError("Universal retarget did not create an animation action")

    # Hard contract: location curves may only exist on the target hips/root.
    root_bone_name = adapter.bone_name("hips")
    illegal_location_curves = []
    for fc in action.fcurves:
        if fc.data_path.endswith(".location"):
            expected_prefix = f'pose.bones["{root_bone_name}"]'
            if not fc.data_path.startswith(expected_prefix):
                illegal_location_curves.append(fc.data_path)
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    if illegal_location_curves:
        raise RuntimeError(
            "Universal retarget created non-root translation curves: "
            + ", ".join(sorted(set(illegal_location_curves)))
        )

    # Also verify the live pose channels themselves are zero for every non-root
    # mapped bone after the final frame.
    worst_non_root_location = 0.0
    worst_non_root_bone = None
    for semantic in order:
        if semantic == "hips":
            continue
        pb = armature.pose.bones[adapter.bone_name(semantic)]
        length = float(pb.location.length)
        if length > worst_non_root_location:
            worst_non_root_location = length
            worst_non_root_bone = pb.name
    if worst_non_root_location > 1e-6:
        raise RuntimeError(
            f"Non-root translation leaked into {worst_non_root_bone}: "
            f"{worst_non_root_location:.8f}"
        )

    if progress:
        progress.info("Retarget translation contract: hips/root only")
        progress.info(
            f"Max non-root pose translation: {worst_non_root_location:.8f}"
        )
        progress.info("Target bone rest offsets and bone lengths were never modified")

    return {
        "action": action,
        "target_height": target_height,
        "motion_scale": motion_scale,
        "frame_start": 1,
        "frame_end": motion.frames,
        "adapter": adapter.preset.preset_id,
        "retarget_version": "universal_rest_local_v2",
        "root_translation_only": True,
        "max_non_root_location": worst_non_root_location,
    }
