"""Monitor file server for Sight runs. Robust replacement for `python -m http.server`.

Why this exists: three prior monitor deaths came from (a) detached-child reaping,
(b) reboot, (c) a wedge where the process stayed in Listen but stopped answering,
consistent with per-request logging blocking on a dead stderr pipe. This script:
  - uses ThreadingHTTPServer (one stuck client cannot wedge the accept loop)
  - suppresses per-request logging entirely (nothing to block on)
  - is meant to run as Task Scheduler job Sight-Monitor (ONLOGON), so it
    survives reboots and never belongs to a DC session process tree.
Serves C:\\Projects\\Sight\\runs\\vzd on 0.0.0.0:8791.
"""
import functools
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = r"C:\Projects\Sight\runs\vzd"
PORT = 8791


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass

    def end_headers(self):
        # Monitor pages poll the same log path; never let a cache serve stale bytes.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    handler = functools.partial(QuietHandler, directory=ROOT)
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), handler)
    httpd.daemon_threads = True
    httpd.serve_forever()


if __name__ == "__main__":
    main()
