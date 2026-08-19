from __future__ import annotations

class ContactCorrection:
    """Extension point for IK/contact correction.

    V1 deliberately keeps retargeting and correction separate. ARDY contact
    channels are preserved in CanonicalMotion so future foot/pelvis/hand solvers
    can be added without changing the retarget core.
    """
    name="none_v1"

    def apply(self,*,scene,armature,adapter,motion,progress=None):
        if progress:
            if motion.contacts is None:
                progress.info("IK/contact correction: no contact channels available; skipped")
            else:
                progress.info(f"IK/contact correction: {len(motion.contact_names)} contact channels preserved; V1 performs no post-retarget edits")
        return {"solver":self.name,"applied":False}
