#!/bin/bash
#SBATCH --job-name=trivial
#SBATCH --partition=NORMAL
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/trivial_baselines_%j.out
#SBATCH --error=logs/trivial_baselines_%j.err

# No GPU needed — trivial baselines are CPU-only.
# BERTScore (optional) would need a GPU, but we skip it here for speed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKDIR="/storage/user/vun/vun/dlhm"
RCLONE="$HOME/.local/bin/rclone"
NAS_REMOTE="nas:ge42faj"
NAS_MOUNT="/tmp/${USER}/nas_mount_$$"
SAMPLES_DIR="$WORKDIR/data_pipeline/samples"
RESULTS_DIR="$WORKDIR/data_pipeline/trivial_results"

cd "$WORKDIR"
mkdir -p logs "$SAMPLES_DIR" "$RESULTS_DIR"

echo "======================================"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Started: $(date)"
echo "======================================"

# ---------------------------------------------------------------------------
# 1. Mount NAS via rclone
# ---------------------------------------------------------------------------
echo "[1/5] Mounting NAS..."
mkdir -p "$NAS_MOUNT"

$RCLONE mount "$NAS_REMOTE" "$NAS_MOUNT" \
    --vfs-cache-mode full \
    --dir-cache-time 72h \
    --poll-interval 1m \
    --daemon

export MM_OR_PROCESSED_ROOT="$NAS_MOUNT/MM-OR_data/MM-OR_processed"

# Wait for mount to be ready
for i in $(seq 1 30); do
    if [ -d "$MM_OR_PROCESSED_ROOT/001_PKA" ]; then
        echo "[nas] Ready: $MM_OR_PROCESSED_ROOT"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[nas] ERROR: mount timed out" >&2
        exit 1
    fi
    sleep 1
done

# Cleanup rclone on exit
cleanup() {
    echo "[cleanup] Unmounting NAS..."
    fusermount -uz "$NAS_MOUNT" 2>/dev/null || true
    rmdir "$NAS_MOUNT" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Install dependencies (if needed)
# ---------------------------------------------------------------------------
echo "[2/5] Checking dependencies..."
pip install --quiet nltk rouge-score wandb 2>/dev/null || true
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# ---------------------------------------------------------------------------
# 3. Build samples (all splits, no augmentation for deterministic eval)
# ---------------------------------------------------------------------------
echo "[3/5] Building training samples from hierarchy JSONs + MM-OR images..."
python -m data_pipeline.build_samples \
    --split all \
    --no-augment \
    --output-dir "$SAMPLES_DIR"

echo "Sample counts:"
for f in "$SAMPLES_DIR"/*.jsonl; do
    echo "  $(basename "$f"): $(wc -l < "$f") samples"
done

# ---------------------------------------------------------------------------
# 4. Run trivial baselines
# ---------------------------------------------------------------------------
echo "[4/5] Running trivial baselines..."
python -m data_pipeline.trivial_baselines \
    --train-samples "$SAMPLES_DIR/train.jsonl" \
    --test-samples "$SAMPLES_DIR/test.jsonl" \
    --output-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# 5. Evaluate each baseline and log to wandb
# ---------------------------------------------------------------------------
echo "[5/5] Evaluating baselines and logging to wandb..."
python "$WORKDIR/script/eval_and_log_wandb.py" \
    --results-dir "$RESULTS_DIR" \
    --gt "$SAMPLES_DIR/test.jsonl" \
    --project "dlhm-hierarchy-baselines" \
    --no-bertscore

echo "======================================"
echo "Finished: $(date)"
echo "Results in: $RESULTS_DIR"
echo "======================================"
