#!/bin/bash
# Serve the hierarchy annotation editor.
#
# Usage:
#   bash script/serve_editor.sh [PORT]
#
# Then open http://localhost:PORT/hierarchy_editor.html

cd "$(dirname "$0")/.."
PORT="${1:-8081}"

echo "Starting Hierarchy Annotation Editor..."
echo "Open:  http://localhost:${PORT}/hierarchy_editor.html"
echo "(Ctrl+C to stop)"
exec python3 script/serve_editor.py --port "$PORT" --directory .
