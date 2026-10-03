"""Non-adversarial process guard for the OFFLINE SYNTHETIC demonstration.

This is a fail-closed test guard, not an OS sandbox or production security layer.
It does not inspect environment variables or store network call arguments.
"""
import sys


class NetworkAccessDenied(RuntimeError):
    pass


_installed = False
blocked_events = []


def install_offline_guard():
    """Install before constructing the fixture backend; no removal/fallback path."""
    global _installed
    if _installed:
        return

    def deny_transport(event, _arguments):
        if (event.startswith("socket.") or event in {
            "smtplib.connect", "imaplib.open", "urllib.Request",
            "subprocess.Popen", "os.system", "os.posix_spawn",
        }):
            blocked_events.append(event)  # Never retain hostnames, URLs, or data.
            raise NetworkAccessDenied("offline_synthetic_transport_blocked")

    sys.addaudithook(deny_transport)
    _installed = True


def guard_is_installed():
    return _installed
