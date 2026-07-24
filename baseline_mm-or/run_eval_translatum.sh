#!/bin/bash
#SBATCH --job-name=b1_eval
#SBATCH --partition=part-1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --gres=gpu:A40:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/b1_eval_%j.out
#SBATCH --error=logs/b1_eval_%j.err

# Evaluate trained Baseline 1 model on (a subset of) the test split.
#
# target ~1 day at ~10 s/frame:
#   - predicted memory only (skip GT ablation unless RUN_GT_MEMORY=1)
#   - takes 014_PKA,033_PKA,038_TKA (~6.8k frames ≈ 19 h for one pass)
#
# Overrides (env):
#   MODEL_PATH       checkpoint dir (default: phase2_with_memory)
#   EVAL_TAKES       comma-separated takes (default: 014_PKA,033_PKA,038_TKA)
#                    set empty EVAL_TAKES= to use all test takes
#   EVAL_MAX_GROUPS  optional cap on (take, role, L2) groups
#   RUN_GT_MEMORY=1  also run GT-memory ablation (≈2× time)

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths + conda env
# ---------------------------------------------------------------------------
WORKDIR="/home/guests/nhat_vu/dlhm"
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

# ---------------------------------------------------------------------------
# MM-OR dataset (local cluster path — no NAS mount needed)
# ---------------------------------------------------------------------------
export MM_OR_PROCESSED_ROOT="/home/guests/shared/ORDatasets/MM-OR"

SAMPLES_DIR="$WORKDIR/data_pipeline/samples"
TEST_SAMPLES="$SAMPLES_DIR/test.jsonl"
PRED_DIR="$BASELINE_DIR/predictions"

# Which checkpoint to evaluate (default: Phase 2 with memory)
MODEL_PATH="${MODEL_PATH:-$BASELINE_DIR/checkpoints/phase2_with_memory}"
# 1-day subset (override with EVAL_TAKES= for full test)
EVAL_TAKES="${EVAL_TAKES-014_PKA,033_PKA,038_TKA}"
EVAL_MAX_GROUPS="${EVAL_MAX_GROUPS:-}"
RUN_GT_MEMORY="${RUN_GT_MEMORY:-0}"

cd "$WORKDIR"
mkdir -p logs "$PRED_DIR"

echo "======================================"
echo "Baseline 1 — Evaluation"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Python: $(which python)"
echo "Model: $MODEL_PATH"
echo "MM_OR_PROCESSED_ROOT: $MM_OR_PROCESSED_ROOT"
echo "EVAL_TAKES: ${EVAL_TAKES:-<all>}"
echo "EVAL_MAX_GROUPS: ${EVAL_MAX_GROUPS:-<none>}"
echo "RUN_GT_MEMORY: $RUN_GT_MEMORY"
echo "Started: $(date)"
echo "======================================"

# ---------------------------------------------------------------------------
# 1. Verify dataset is accessible
# ---------------------------------------------------------------------------
echo "[1/5] Verifying dataset access..."
if [ ! -d "$MM_OR_PROCESSED_ROOT/001_PKA" ]; then
    echo "ERROR: MM-OR dataset not found at $MM_OR_PROCESSED_ROOT" >&2
    exit 1
fi
echo "  Dataset OK: $MM_OR_PROCESSED_ROOT"

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

INFER_EXTRA=()
if [ -n "${EVAL_TAKES}" ]; then
    INFER_EXTRA+=(--takes "$EVAL_TAKES")
fi
if [ -n "${EVAL_MAX_GROUPS}" ]; then
    INFER_EXTRA+=(--max-groups "$EVAL_MAX_GROUPS")
fi

# ---------------------------------------------------------------------------
# 3. Run inference — autoregressive memory (primary, 1-day default)
# ---------------------------------------------------------------------------
PRED_AUTO="$PRED_DIR/pred_autoregressive.jsonl"

echo "[3/5] Running inference (autoregressive memory)..."
python "$BASELINE_DIR/inference.py" \
    --model-path "$MODEL_PATH" \
    --test-samples "$TEST_SAMPLES" \
    --processed-root "$MM_OR_PROCESSED_ROOT" \
    --output "$PRED_AUTO" \
    --memory-mode predicted \
    "${INFER_EXTRA[@]}"

PRED_ARGS=("$PRED_AUTO")
NAME_ARGS=("b1_autoregressive")

# ---------------------------------------------------------------------------
# 4. Optional: GT memory ablation (≈2× wall time — off by default)
# ---------------------------------------------------------------------------
if [ "$RUN_GT_MEMORY" = "1" ]; then
    PRED_GT="$PRED_DIR/pred_gt_memory.jsonl"
    echo "[4/5] Running inference (GT memory)..."
    python "$BASELINE_DIR/inference.py" \
        --model-path "$MODEL_PATH" \
        --test-samples "$TEST_SAMPLES" \
        --processed-root "$MM_OR_PROCESSED_ROOT" \
        --output "$PRED_GT" \
        --memory-mode gt \
        "${INFER_EXTRA[@]}"
    PRED_ARGS+=("$PRED_GT")
    NAME_ARGS+=("b1_gt_memory")
else
    echo "[4/5] Skipping GT-memory ablation (set RUN_GT_MEMORY=1 to enable)."
fi

# ---------------------------------------------------------------------------
# 5. Evaluate and log to wandb
# ---------------------------------------------------------------------------
echo "[5/5] Evaluating and logging to wandb..."

python "$BASELINE_DIR/eval_predictions.py" \
    --gt "$TEST_SAMPLES" \
    --predictions "${PRED_ARGS[@]}" \
    --names "${NAME_ARGS[@]}" \
    --model-info "$MODEL_PATH" \
    --project "dlhm-hierarchy-baselines"

echo "======================================"
echo "Evaluation complete."
echo "Predictions: $PRED_DIR/"
echo "Finished: $(date)"
echo "======================================"
