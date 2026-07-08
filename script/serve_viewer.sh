#!/bin/bash
# Serve hierarchy video HTML + frame images from project root.
# Uses serve_viewer.py so viewer_links/ symlinks to NAS (/tmp) work.
#
# Usage:
#   bash script/serve_viewer.sh [PORT] [HTML_FILE]
#
# Prerequisites: NAS mounted (bash script/mount_nas.sh) and viewer_links/ created
# by run.sh when rendering the HTML.

cd "$(dirname "$0")/.."
PORT="${1:-8080}"
HTML="${2:-}"

if [ -z "$HTML" ]; then
    HTML=$(ls -t *_hierarchy_video_sync_qwen27b.html 2>/dev/null | head -1)
fi

echo "Serving: $(pwd)"
if [ -n "$HTML" ]; then
    echo "Open:    http://localhost:${PORT}/${HTML}"
else
    echo "Open:    http://localhost:${PORT}/<your_hierarchy_video.html>"
fi
echo "Images load via viewer_links/ symlinks -> NAS mount in /tmp (no repo disk use)."
echo "(Ctrl+C to stop)"
exec python3 script/serve_viewer.py --port "$PORT" --directory .
