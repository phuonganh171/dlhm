#!/bin/bash
set -e

VENV=/tmp/nhatvu/.venv
HF_CACHE=/tmp/nhatvu/hf_cache
PYTHON=$VENV/bin/python3
PIP=$VENV/bin/pip
WORKDIR=/mnt/home/nhatvu/dlhm
OUT_DIR=$WORKDIR/annotation_output
LOG=/tmp/nhatvu/run.log

# ---------------------------------------------------------------------------
# 1. Setup virtual environment (skip if already exists)
# ---------------------------------------------------------------------------
if [ ! -f "$PYTHON" ]; then
    echo "[setup] Creating venv at $VENV ..."
    python3 -m venv "$VENV"
    $PIP install --upgrade pip --quiet
    $PIP install torch transformers accelerate tqdm bitsandbytes
    echo "[setup] Done."
else
    echo "[setup] Venv already exists at $VENV, skipping install."
fi

mkdir -p "$OUT_DIR" /tmp/nhatvu

# ---------------------------------------------------------------------------
# 2. Run annotation model (Qwen3-32B, 4-bit quantised)
# ---------------------------------------------------------------------------
echo "[annotate] Starting annotation model ..."
cd "$WORKDIR"
HF_HOME=$HF_CACHE $PYTHON annotation_model.py \
    --model Qwen/Qwen3-32B \
    --start_tp 001114 \
    --max_frames 600 \
    --output "$OUT_DIR/annotations_001_PKA_600_qwen3_32b.jsonl" \
    --log_file "$LOG" \
    --batch_size 2

# ---------------------------------------------------------------------------
# 3. Generate HTML visualisation
# ---------------------------------------------------------------------------
echo "[visualize] Generating HTML report ..."
$PYTHON visualize_annotations.py \
    --input  "$OUT_DIR/annotations_001_PKA_600_qwen3_32b.jsonl" \
    --output "$OUT_DIR/annotations_001_PKA_600_qwen3_32b.html"

# ---------------------------------------------------------------------------
# 4. Generate sync video viewer (summary + multi-camera frames)
# ---------------------------------------------------------------------------
echo "[render] Generating summary video viewer ..."
$PYTHON render_summary_video.py \
    --annotations "$OUT_DIR/annotations_001_PKA_600_qwen3_32b.jsonl" \
    --output "$WORKDIR/summary_video_sync_qwen3_32b.html" \
    --start_timestamp 1443

echo ""
echo "Done. Outputs:"
echo "  Annotations : $OUT_DIR/annotations_001_PKA_600_qwen3_32b.jsonl"
echo "  HTML report : $OUT_DIR/annotations_001_PKA_600_qwen3_32b.html"
echo "  Video viewer: $WORKDIR/summary_video_sync_qwen3_32b.html"
echo "  Model log   : $LOG"
echo ""
echo "To view sync player in browser, run:"
echo "  bash script/serve_viewer.sh"
echo "  # or: python3 -m http.server 8080 --directory $WORKDIR"
