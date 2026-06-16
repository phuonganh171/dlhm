#!/bin/sh
#SBATCH --job-name=dlhm
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/nhatvu/dlhm/logs/annotate_%j.out
#SBATCH --error=/mnt/home/nhatvu/dlhm/logs/annotate_%j.err

set -e

WORKDIR=/mnt/home/nhatvu/dlhm
NODE=$(hostname -s)
SCRATCH="/tmp/nhatvu-${SLURM_JOB_ID}-${NODE}"
VENV="$SCRATCH/.venv"
HF_CACHE="$SCRATCH/hf_cache"
LOG_FILE="$SCRATCH/annotate.log"
OUT_DIR=$WORKDIR/annotation_output

echo "[slurm] job=$SLURM_JOB_ID node=$NODE scratch=$SCRATCH"
mkdir -p "$WORKDIR/logs" "$OUT_DIR" "$SCRATCH"

# ---------------------------------------------------------------------------
# 1. Setup virtual environment (once per first run)
# ---------------------------------------------------------------------------
if [ ! -f "$VENV/bin/python3" ]; then
    echo "[setup] Creating venv at $VENV ..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip -q
    "$VENV/bin/pip" install torch transformers accelerate tqdm bitsandbytes
    echo "[setup] Done."
else
    echo "[setup] Venv already exists at $VENV, skipping install."
fi

# ---------------------------------------------------------------------------
# 2. Run annotation (Qwen3-32B, 4-bit NF4)
# ---------------------------------------------------------------------------
echo "[annotate] Starting Qwen3-32B annotation (job $SLURM_JOB_ID) ..."
cd "$WORKDIR"
HF_HOME="$HF_CACHE" "$VENV/bin/python3" annotation_model.py \
    --model Qwen/Qwen3-32B \
    --start_tp 001114 \
    --max_frames 600 \
    --output "$OUT_DIR/annotations_001_PKA_600_qwen3_32b.jsonl" \
    --log_file "$LOG_FILE" \
    --batch_size 2

# ---------------------------------------------------------------------------
# 3. Post-processing (HTML + video viewer)
# ---------------------------------------------------------------------------
echo "[visualize] Generating HTML report ..."
"$VENV/bin/python3" visualize_annotations.py \
    --input  "$OUT_DIR/annotations_001_PKA_600_qwen3_32b.jsonl" \
    --output "$OUT_DIR/annotations_001_PKA_600_qwen3_32b.html"

echo "[render] Generating summary video viewer ..."
"$VENV/bin/python3" render_summary_video.py \
    --annotations "$OUT_DIR/annotations_001_PKA_600_qwen3_32b.jsonl" \
    --output "$WORKDIR/summary_video_sync_qwen3_32b.html" \
    --start_timestamp 1443

echo ""
echo "Done (job $SLURM_JOB_ID). Outputs:"
echo "  Annotations : $OUT_DIR/annotations_001_PKA_600_qwen3_32b.jsonl"
echo "  HTML report : $OUT_DIR/annotations_001_PKA_600_qwen3_32b.html"
echo "  Video viewer: $WORKDIR/summary_video_sync_qwen3_32b.html"
echo "  Model log   : $LOG_FILE"
echo "  Scratch     : $SCRATCH (removed when job ends / node reboots)"
