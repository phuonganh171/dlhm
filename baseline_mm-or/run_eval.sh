#!/bin/bash
#SBATCH --job-name=b1_eval
#SBATCH --partition=NORMAL
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1,VRAM:48G
# Torch 2.0.1+cu118 has no sm_120 kernels — exclude Blackwell (node22).
#SBATCH --exclude=node22
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/b1_eval_%j.out
#SBATCH --error=logs/b1_eval_%j.err

# Evaluate trained Baseline 1 model on the test split.
# Runs inference (autoregressive and GT memory), evaluates, and logs to wandb.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths + conda env
# ---------------------------------------------------------------------------
WORKDIR="/storage/user/vun/vun/dlhm"
BASELINE_DIR="$WORKDIR/baseline_mm-or"
ORACLE_DIR="$BASELINE_DIR/ORacle"
LLAVA_DIR="$ORACLE_DIR/LLaVA"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="dlhm-b1"

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
# ORacle pins torch+cu118; bitsandbytes needs the toolkit libs
if command -v module >/dev/null 2>&1; then
    module load cuda/11.8.0
fi
export LD_LIBRARY_PATH="${CUDA_HOME:+$CUDA_HOME/lib64:}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LLAVA_DIR:${PYTHONPATH:-}"

RCLONE="$HOME/.local/bin/rclone"
NAS_REMOTE="nas:ge42faj"
NAS_MOUNT="/tmp/${USER}/nas_mount_$$"

SAMPLES_DIR="$WORKDIR/data_pipeline/samples"
TEST_SAMPLES="$SAMPLES_DIR/test.jsonl"
PRED_DIR="$BASELINE_DIR/predictions"

# Which checkpoint to evaluate (default: Phase 2 with memory)
MODEL_PATH="${MODEL_PATH:-$BASELINE_DIR/checkpoints/phase2_with_memory}"

cd "$WORKDIR"
mkdir -p logs "$PRED_DIR"

echo "======================================"
echo "Baseline 1 — Evaluation"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Python: $(which python)"
echo "Model: $MODEL_PATH"
echo "Started: $(date)"
echo "======================================"

# ---------------------------------------------------------------------------
# 1. Mount NAS
# ---------------------------------------------------------------------------
echo "[1/5] Mounting NAS..."
mkdir -p "$NAS_MOUNT"

$RCLONE mount "$NAS_REMOTE" "$NAS_MOUNT" \
    --vfs-cache-mode full \
    --dir-cache-time 72h \
    --poll-interval 1m \
    --daemon

export MM_OR_PROCESSED_ROOT="$NAS_MOUNT/MM-OR_data/MM-OR_processed"

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

cleanup() {
    echo "[cleanup] Unmounting NAS..."
    fusermount -uz "$NAS_MOUNT" 2>/dev/null || true
    rmdir "$NAS_MOUNT" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Build test samples if needed
# ---------------------------------------------------------------------------
echo "[2/5] Preparing test samples..."

if [ ! -f "$TEST_SAMPLES" ]; then
    echo "  Building test samples..."
    python -m data_pipeline.build_samples \
        --split test \
        --no-augment \
        --output-dir "$SAMPLES_DIR"
fi

echo "  Test samples: $(wc -l < "$TEST_SAMPLES") frames"

python -c "import transformers, peft, llava; print('  deps OK')" || {
    echo "ERROR: env '$ENV_NAME' incomplete. Run: bash baseline_mm-or/setup.sh" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Run inference — autoregressive memory
# ---------------------------------------------------------------------------
PRED_AUTO="$PRED_DIR/pred_autoregressive.jsonl"

echo "[3/5] Running inference (autoregressive memory)..."
python "$BASELINE_DIR/inference.py" \
    --model-path "$MODEL_PATH" \
    --test-samples "$TEST_SAMPLES" \
    --processed-root "$MM_OR_PROCESSED_ROOT" \
    --output "$PRED_AUTO" \
    --memory-mode predicted

# ---------------------------------------------------------------------------
# 4. Run inference — GT memory (oracle, for ablation)
# ---------------------------------------------------------------------------
PRED_GT="$PRED_DIR/pred_gt_memory.jsonl"

echo "[4/5] Running inference (GT memory)..."
python "$BASELINE_DIR/inference.py" \
    --model-path "$MODEL_PATH" \
    --test-samples "$TEST_SAMPLES" \
    --processed-root "$MM_OR_PROCESSED_ROOT" \
    --output "$PRED_GT" \
    --memory-mode gt

# ---------------------------------------------------------------------------
# 5. Evaluate and log to wandb
# ---------------------------------------------------------------------------
echo "[5/5] Evaluating and logging to wandb..."

python "$BASELINE_DIR/eval_predictions.py" \
    --gt "$TEST_SAMPLES" \
    --predictions "$PRED_AUTO" "$PRED_GT" \
    --names "b1_autoregressive" "b1_gt_memory" \
    --model-info "$MODEL_PATH" \
    --project "dlhm-hierarchy-baselines"

echo "======================================"
echo "Evaluation complete."
echo "Predictions: $PRED_DIR/"
echo "Finished: $(date)"
echo "======================================"
