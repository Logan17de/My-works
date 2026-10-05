"""Optional explicit Google reviewer setup; defaults and pending staging unchanged."""
import os
import stat
from .google_reviewer_auth import AgentOnlyVerifier,GoogleReviewerAuth,GoogleReviewerProfile
from .google_public_keys import GooglePublicKeys


def load_google_review_profile(settings):
    path=settings.google_reviewer_profile_file
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as handle:
            meta=os.fstat(handle.fileno())
            if not stat.S_ISREG(meta.st_mode) or not 1<=meta.st_size<=16384:raise ValueError
            profile=GoogleReviewerProfile.model_validate_json(handle.read(16385))
        if not profile.enabled or settings.review_origin!=profile.origin:raise ValueError
        if settings.issuer in ('https://accounts.google.com','accounts.google.com') or settings.audience==profile.client_id:
            raise ValueError('agent access credentials need a separate issuer/resource audience')
        expected={(sub,profile.client_id) for sub in profile.reviewer_subjects}
        actual={(g.subject,g.client_id) for g in settings.principals if g.role=='reviewer'}
        if actual!=expected or any(g.client_id==profile.client_id for g in settings.principals if g.role=='agent'):raise ValueError
        return profile
    except Exception:
        raise RuntimeError('Google reviewer profile is disabled, pending or invalid') from None


def configure_google_review(service, *, key_source=None):
    if service.settings.reviewer_auth!='google_oidc':return service
    profile=load_google_review_profile(service.settings)
    service.google_reviewer_auth=GoogleReviewerAuth(profile,service.settings,
        key_source if key_source is not None else GooglePublicKeys(),clock=service.clock)
    service.verifier=AgentOnlyVerifier(service.verifier)
    return service
