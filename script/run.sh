#!/bin/bash
set -e

VENV=/tmp/nhatvu/.venv312
HF_CACHE=/tmp/nhatvu/hf_cache
PYTHON=$VENV/bin/python3
# Locate uv: prefer one on PATH, fall back to the user-local pip install.
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
WORKDIR=/mnt/home/nhatvu/dlhm
HIER_DIR=$WORKDIR/hierarchy_output
LOG=/tmp/nhatvu/run.log

TAKE=001_PKA
TAKE_DIR=mm-or/MM-OR_data/MM-OR_processed/$TAKE
SRT=mm-or/MM-OR_data/MM-OR_processed/take_transcripts/${TAKE}.srt
HIER_JSON=$HIER_DIR/${TAKE}_hierarchy_qwen27b.json
HIER_HTML=$HIER_DIR/${TAKE}_hierarchy_qwen27b.html
VIDEO_HTML=hierarchy_video_sync_qwen27b.html

# ---------------------------------------------------------------------------
# 1. Setup virtual environment (skip if already exists)
#    Qwen3.5 requires transformers>=5.2.0, which needs Python>=3.10, so we use
#    uv to provision a standalone Python 3.12 and build the venv with it.
# ---------------------------------------------------------------------------
if [ ! -f "$PYTHON" ]; then
    echo "[setup] Creating Python 3.12 venv at $VENV via uv ..."
    if [ ! -f "$UV" ]; then
        echo "[setup] uv not found at $UV; install it first (pip install uv)." >&2
        exit 1
    fi
    $UV python install 3.12
    $UV venv --python 3.12 "$VENV"
    $UV pip install --python "$PYTHON" "transformers>=5.2.0" torch accelerate tqdm bitsandbytes
    echo "[setup] Done."
else
    echo "[setup] Venv already exists at $VENV, skipping install."
fi

mkdir -p "$HIER_DIR" /tmp/nhatvu

# ---------------------------------------------------------------------------
# 2. Build hierarchy (Qwen3.5-27B, 4-bit quantised)
# ---------------------------------------------------------------------------
echo "[hierarchy] Building level-0 + level-1 + level-2 hierarchy ..."
cd "$WORKDIR"
HF_HOME=$HF_CACHE $PYTHON build_hierarchy_qwen.py \
    --take_dir "$TAKE_DIR" \
    --start_tp 001114 \
    --max_frames 600 \
    --level2 \
    --model Qwen/Qwen3.5-27B \
    --output "$HIER_JSON"

# ---------------------------------------------------------------------------
# 3. Generate HTML hierarchy viewer
# ---------------------------------------------------------------------------
echo "[visualize] Generating hierarchy HTML viewer ..."
$PYTHON visualize_hierarchy.py \
    --input  "$HIER_JSON" \
    --output "$HIER_HTML"

# ---------------------------------------------------------------------------
# 4. Generate synchronized video player
# ---------------------------------------------------------------------------
echo "[render] Generating hierarchy video player ..."
$PYTHON render_hierarchy_video.py \
    --hierarchy "$HIER_JSON" \
    --srt "$SRT" \
    --colorimage_dir "${TAKE_DIR}/colorimage" \
    --output "$VIDEO_HTML"

echo ""
echo "Done. Outputs:"
echo "  Hierarchy JSON : $HIER_JSON"
echo "  Hierarchy HTML : $HIER_HTML"
echo "  Video player   : $VIDEO_HTML"
echo "  Model log      : $LOG"
echo ""
echo "To view in browser:"
echo "  python3 -m http.server 8080 --directory $WORKDIR"
