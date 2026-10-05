# HTTP header verification

`verify_security_headers(response.headers)` checks a standard-library
`http.client.HTTPResponse` without collapsing its headers into a dictionary.
Field names are case-insensitive. Each required field must appear exactly once
with its exact expected value: `Cache-Control: no-store`,
`Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, and
`X-Content-Type-Options: nosniff`. Missing, duplicate, conflicting, or unexpected
values fail verification.

Run from the repository root:

```sh
python -m unittest discover -s zetbros-revenue/tools/http-header-verification -p test_http_header_verification.py -v
```

Offline evidence: all 11 regression tests passed on Python 3.13.2. They use the
real HTTP wire-format parser with in-memory responses. Coverage includes lower,
upper, mixed, and original field-name casing; missing and incorrect values;
cache-control near matches; duplicate fields in both orders; unrelated fields;
and prefixed impostor fields. The helper and tests use only the Python standard
library, with no network, service, credential, or host-configuration operations.

This directory preserves the generic verification fix as standalone source.
Application runtime files are unchanged by this source-only addition.
