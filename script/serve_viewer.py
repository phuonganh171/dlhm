#!/usr/bin/env python3
"""
HTTP server for hierarchy video HTML that follows symlinks.

Standard `python -m http.server` blocks paths that resolve outside the
document root. viewer_links/ symlinks point at the NAS mount in /tmp without
using repo disk space — this server resolves and serves them.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
from pathlib import Path


class SymlinkHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files under document_root, following symlinks when opening."""

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.exists(path):
            return self.send_error(404, "File not found")
        resolved = os.path.realpath(path)
        if os.path.isdir(resolved):
            parts = self.path.rstrip("/").split("/")
            if not self.path.endswith("/"):
                self.send_response(301)
                self.send_header("Location", self.path + "/")
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index_path = os.path.join(resolved, index)
                if os.path.isfile(index_path):
                    resolved = index_path
                    break
            else:
                return self.list_directory(resolved)
        if not os.path.isfile(resolved):
            return self.send_error(404, "File not found")
        try:
            f = open(resolved, "rb")
        except OSError:
            return self.send_error(404, "File not found")
        ctype = self.guess_type(resolved)
        self.send_response(200)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(os.fstat(f.fileno()).st_size))
        self.end_headers()
        return f


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve hierarchy viewer with symlink support")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--directory", type=Path, default=Path("."),
        help="Document root (repo root)",
    )
    args = parser.parse_args()
    root = args.directory.resolve()
    os.chdir(root)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), SymlinkHTTPRequestHandler) as httpd:
        print(f"Serving {root} on http://localhost:{args.port}/")
        print("Symlinks in viewer_links/ -> NAS mount are followed.")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
