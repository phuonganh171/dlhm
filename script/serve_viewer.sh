#!/bin/bash
# Serve summary_video_sync.html + mm-or frames from project root (dlhm).
cd "$(dirname "$0")/.."
PORT="${1:-8080}"
echo "Serving: $(pwd)"
echo "Open:    http://localhost:${PORT}/summary_video_sync.html"
echo "(Ctrl+C to stop)"
exec python3 -m http.server "$PORT" --directory .
