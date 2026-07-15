#!/usr/bin/env python3
"""
HTTP server for the hierarchy annotation editor.

Serves the editor HTML and provides JSON API endpoints to load, save,
and track edit history for hierarchy JSON files in hierarchy_output/.

Endpoints:
    GET  /api/hierarchies           — list available hierarchy JSON files
    GET  /api/hierarchy/<filename>  — load a hierarchy JSON file
    POST /api/hierarchy/<filename>  — save (overwrite) a hierarchy JSON file
    GET  /api/history/<filename>    — get the edit history log for a file
    POST /api/history/<filename>    — append a single edit entry to history
    POST /api/regenerate/<filename> — regenerate the HTML viewer files from the JSON

Static files (HTML, JS, CSS, images) are served from the document root
with symlink support (same as serve_viewer.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socketserver
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote


HIERARCHY_DIR = "hierarchy_output"
HISTORY_SUFFIX = "_edit_history.json"
VIEWER_LINKS = "viewer_links"

# Only serve files matching this pattern as hierarchy files
HIERARCHY_RE = re.compile(r"^[\w\-]+_hierarchy[\w\-]*\.json$")


def is_safe_filename(name: str) -> bool:
    return bool(HIERARCHY_RE.match(name)) and ".." not in name


# e.g. "001_PKA_hierarchy_qwen27b.json" -> "001_PKA"
_TAKE_RE = re.compile(r"^(\d{3}_[A-Z]+)_hierarchy")


def extract_take_name(filename: str) -> str | None:
    m = _TAKE_RE.match(filename)
    return m.group(1) if m else None


def history_path_for(filename: str) -> Path:
    stem = filename.replace(".json", "")
    return Path(HIERARCHY_DIR) / f"{stem}{HISTORY_SUFFIX}"


class EditorHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve static files (with symlink support) + JSON API."""

    def do_GET(self):
        path = unquote(self.path).split("?")[0]

        if path == "/api/hierarchies":
            return self._list_hierarchies()
        if path.startswith("/api/hierarchy/"):
            return self._load_hierarchy(path[len("/api/hierarchy/"):])
        if path.startswith("/api/history/"):
            return self._load_history(path[len("/api/history/"):])

        return self._serve_static()

    def do_POST(self):
        path = unquote(self.path).split("?")[0]

        if path.startswith("/api/hierarchy/"):
            return self._save_hierarchy(path[len("/api/hierarchy/"):])
        if path.startswith("/api/history/"):
            return self._append_history(path[len("/api/history/"):])
        if path.startswith("/api/regenerate/"):
            return self._regenerate_html(path[len("/api/regenerate/"):])

        self.send_error(404)

    # -- API handlers --

    def _list_hierarchies(self):
        hdir = Path(HIERARCHY_DIR)
        if not hdir.is_dir():
            return self._json_response([])
        files = sorted(
            f.name for f in hdir.iterdir()
            if f.is_file()
            and HIERARCHY_RE.match(f.name)
            and HISTORY_SUFFIX not in f.name
        )
        self._json_response(files)

    def _load_hierarchy(self, filename: str):
        if not is_safe_filename(filename):
            return self.send_error(400, "Invalid filename")
        fpath = Path(HIERARCHY_DIR) / filename
        if not fpath.is_file():
            return self.send_error(404, "File not found")
        data = fpath.read_text(encoding="utf-8")
        self._raw_json_response(data)

    def _save_hierarchy(self, filename: str):
        if not is_safe_filename(filename):
            return self.send_error(400, "Invalid filename")
        body = self._read_body()
        if body is None:
            return
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as exc:
            return self.send_error(400, f"Invalid JSON: {exc}")

        fpath = Path(HIERARCHY_DIR) / filename
        # Write atomically via temp file
        tmp = fpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(fpath)
        self._json_response({"status": "ok", "file": filename})

    def _load_history(self, filename: str):
        if not is_safe_filename(filename):
            return self.send_error(400, "Invalid filename")
        hpath = history_path_for(filename)
        if not hpath.is_file():
            return self._json_response([])
        data = hpath.read_text(encoding="utf-8")
        self._raw_json_response(data)

    def _append_history(self, filename: str):
        if not is_safe_filename(filename):
            return self.send_error(400, "Invalid filename")
        body = self._read_body()
        if body is None:
            return
        try:
            entry = json.loads(body)
        except json.JSONDecodeError as exc:
            return self.send_error(400, f"Invalid JSON: {exc}")

        entry.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))

        hpath = history_path_for(filename)
        history = []
        if hpath.is_file():
            try:
                history = json.loads(hpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                history = []
        history.append(entry)
        hpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        self._json_response({"status": "ok", "entries": len(history)})

    # -- Regenerate HTML viewer --

    def _regenerate_html(self, filename: str):
        if not is_safe_filename(filename):
            return self.send_error(400, "Invalid filename")

        hier_json = Path(HIERARCHY_DIR) / filename
        if not hier_json.is_file():
            return self.send_error(404, "Hierarchy file not found")

        # Derive take name: e.g. "001_PKA_hierarchy_qwen27b.json" -> "001_PKA"
        take = extract_take_name(filename)
        if not take:
            return self._json_response(
                {"status": "error", "message": f"Cannot derive take name from {filename}"},
                code=400,
            )

        errors = []
        generated = []

        # 1) Static hierarchy HTML (visualize_hierarchy.py)
        static_html = Path(HIERARCHY_DIR) / filename.replace(".json", ".html")
        viz_script = Path("visualize_hierarchy.py")
        if viz_script.is_file():
            cmd = [sys.executable, str(viz_script),
                   "--input", str(hier_json),
                   "--output", str(static_html)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                generated.append(str(static_html))
            else:
                errors.append(f"visualize_hierarchy.py failed: {result.stderr[:500]}")

        # 2) Video sync HTML (render_hierarchy_video.py)
        render_script = Path("render_hierarchy_video.py")
        if render_script.is_file():
            video_html = Path(f"{take}_hierarchy_video_sync_qwen27b.html")
            viewer_take = Path(VIEWER_LINKS) / take
            colorimage_dir = viewer_take / "colorimage"

            if not colorimage_dir.is_dir():
                errors.append(
                    f"No colorimage dir at {colorimage_dir} — "
                    "mount NAS and create viewer_links first"
                )
            else:
                cmd = [sys.executable, str(render_script),
                       "--hierarchy", str(hier_json),
                       "--colorimage_dir", str(colorimage_dir),
                       "--web_root", ".",
                       "--output", str(video_html)]
                sim_dir = viewer_take / "simstation"
                if sim_dir.is_dir():
                    cmd.extend(["--simstation_dir", str(sim_dir)])
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    generated.append(str(video_html))
                else:
                    errors.append(f"render_hierarchy_video.py failed: {result.stderr[:500]}")

        status = "ok" if not errors else "partial" if generated else "error"
        self._json_response({
            "status": status,
            "generated": generated,
            "errors": errors,
        })

    # -- Helpers --

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self.send_error(400, "Empty body")
            return None
        return self.rfile.read(length).decode("utf-8")

    def _json_response(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _raw_json_response(self, text: str, code=200):
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    # -- Static file serving with symlink support --

    def _serve_static(self):
        path = self.translate_path(self.path)
        if not os.path.exists(path):
            return self.send_error(404, "File not found")
        resolved = os.path.realpath(path)
        if os.path.isdir(resolved):
            if not self.path.endswith("/"):
                self.send_response(301)
                self.send_header("Location", self.path + "/")
                self.end_headers()
                return
            for index in ("index.html", "index.htm"):
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
        self.copyfile(f, self.wfile)
        f.close()

    def log_message(self, fmt, *args):
        if self.path and self.path.startswith("/api/"):
            super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchy annotation editor server")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--directory", type=Path, default=Path("."),
        help="Document root (repo root)",
    )
    args = parser.parse_args()
    root = args.directory.resolve()
    os.chdir(root)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), EditorHTTPRequestHandler) as httpd:
        print(f"Editor server at http://localhost:{args.port}/hierarchy_editor.html")
        print(f"Document root: {root}")
        print(f"Hierarchy dir: {root / HIERARCHY_DIR}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
