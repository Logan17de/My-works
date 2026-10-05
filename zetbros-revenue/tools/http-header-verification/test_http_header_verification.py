"""Offline regressions using HTTPResponse's real wire-format header parser."""
import http.client
import io
import unittest

from http_header_verification import verify_security_headers


REQUIRED = [
    ('Cache-Control', 'no-store'),
    ('Referrer-Policy', 'no-referrer'),
    ('X-Frame-Options', 'DENY'),
    ('X-Content-Type-Options', 'nosniff'),
]


class InMemorySocket:
    def __init__(self, wire):
        self.wire = wire

    def makefile(self, mode):
        if mode != 'rb':
            raise AssertionError('Unexpected HTTP parser mode')
        return io.BytesIO(self.wire)


class HeaderVerificationTests(unittest.TestCase):
    def parse_and_validate(self, pairs):
        body = b'Offline response body'
        fields = ''.join(name + ': ' + value + '\r\n'
                         for name, value in pairs)
        wire = (b'HTTP/1.1 200 OK\r\n' + fields.encode('ascii')
                + ('Content-Length: %d\r\n\r\n' % len(body)).encode('ascii')
                + body)
        response = http.client.HTTPResponse(InMemorySocket(wire))
        response.begin()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), body)
        verify_security_headers(response.headers)
        return response.headers

    def test_original_field_names(self):
        self.parse_and_validate(REQUIRED)

    def test_lowercase_field_names(self):
        self.parse_and_validate([(name.lower(), value)
                                 for name, value in REQUIRED])

    def test_uppercase_field_names(self):
        self.parse_and_validate([(name.upper(), value)
                                 for name, value in REQUIRED])

    def test_mixed_case_field_names(self):
        self.parse_and_validate([(name.swapcase(), value)
                                 for name, value in REQUIRED])

    def test_missing_required_headers_fail(self):
        for absent, _ in REQUIRED:
            with self.subTest(absent=absent):
                with self.assertRaisesRegex(RuntimeError, absent):
                    self.parse_and_validate([(name.lower(), value)
                                             for name, value in REQUIRED
                                             if name != absent])

    def test_wrong_required_values_fail(self):
        for wrong, _ in REQUIRED:
            with self.subTest(wrong=wrong):
                with self.assertRaisesRegex(RuntimeError, wrong):
                    self.parse_and_validate([
                        (name.lower(), 'wrong' if name == wrong else value)
                        for name, value in REQUIRED])

    def test_cache_control_near_matches_do_not_pass(self):
        for value in ('no-cache', 'public', 'xno-store', 'no-storex',
                      'no-store, public', 'no-store, max-age=0', 'NO-STORE'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, 'Cache-Control'):
                    self.parse_and_validate([
                        (name.lower(), value if name == 'Cache-Control' else expected)
                        for name, expected in REQUIRED])

    def test_conflicting_duplicate_headers_fail_in_both_orders(self):
        for duplicate, correct in REQUIRED:
            for values in ((correct, 'wrong'), ('wrong', correct)):
                with self.subTest(duplicate=duplicate, values=values):
                    pairs = [(name, value) for name, value in REQUIRED
                             if name != duplicate]
                    pairs += [(duplicate.lower(), values[0]),
                              (duplicate.upper(), values[1])]
                    with self.assertRaisesRegex(RuntimeError, duplicate):
                        self.parse_and_validate(pairs)

    def test_duplicate_required_values_fail(self):
        with self.assertRaisesRegex(RuntimeError, 'Cache-Control'):
            self.parse_and_validate(REQUIRED + [('cache-control', 'no-store')])

    def test_unrelated_headers_do_not_change_required_checks(self):
        self.parse_and_validate(REQUIRED + [('X-Offline-Test', 'ok')])

    def test_prefixed_header_cannot_replace_required_header(self):
        pairs = [('X-' + name if name == 'Cache-Control' else name, value)
                 for name, value in REQUIRED]
        with self.assertRaisesRegex(RuntimeError, 'Cache-Control'):
            self.parse_and_validate(pairs)


if __name__ == '__main__':
    unittest.main()
