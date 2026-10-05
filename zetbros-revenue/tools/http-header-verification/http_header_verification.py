"""Strict security-header verification using HTTP's case-insensitive names."""
from http.client import HTTPMessage


REQUIRED_SECURITY_HEADERS = (
    ('Cache-Control', 'no-store'),
    ('Referrer-Policy', 'no-referrer'),
    ('X-Frame-Options', 'DENY'),
    ('X-Content-Type-Options', 'nosniff'),
)


def verify_security_headers(headers: HTTPMessage) -> None:
    """Require one exact value for each field, regardless of field-name casing.

    Pass the original ``HTTPResponse.headers`` object. Converting headers to a
    plain dictionary loses case-insensitive lookup and duplicate-field evidence.
    Missing, duplicated, or unexpected required values raise ``RuntimeError``.
    """
    for name, expected in REQUIRED_SECURITY_HEADERS:
        if headers.get_all(name) != [expected]:
            raise RuntimeError('Required security header mismatch: ' + name)
