#!/bin/bash
#SBATCH --job-name=b2_eval
#SBATCH --partition=NORMAL
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1,VRAM:48G
#SBATCH --exclude=node22
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/b2_eval_%j.out
#SBATCH --error=logs/b2_eval_%j.err

# Evaluate trained Baseline 2 (ORQA) model on (a subset of) the test split.
#
# Overrides (env):
#   MODEL_PATH       checkpoint dir (default: checkpoints/phase2_with_memory)
#   EVAL_TAKES       comma-separated takes (default: 014_PKA,033_PKA,038_TKA)
#   EVAL_MAX_GROUPS  optional cap on (take, role, L2) groups
#   RUN_GT_MEMORY=1  also run GT-memory ablation

set -euo pipefail

WORKDIR="/storage/user/vun/vun/dlhm"
BASELINE_DIR="$WORKDIR/baseline_orqa"
ORQA_DIR="$BASELINE_DIR/ORQA"
LLAMA_FACTORY_DIR="$ORQA_DIR/Qwen2-VL/LLaMA-Factory"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="dlhm-b2"

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
# shellcheck disable=SC1091
source "$BASELINE_DIR/lib_cuda_env.sh"
export PYTHONPATH="$LLAMA_FACTORY_DIR/src:${ORQA_DIR}:${PYTHONPATH:-}"

RCLONE="$HOME/.local/bin/rclone"
NAS_REMOTE="nas:ge42faj"
NAS_MOUNT="/tmp/${USER}/nas_mount_orqa"

SAMPLES_DIR="$WORKDIR/data_pipeline/samples"
TEST_SAMPLES="$SAMPLES_DIR/test.jsonl"
PRED_DIR="$BASELINE_DIR/predictions"

MODEL_PATH="${MODEL_PATH:-$BASELINE_DIR/checkpoints/phase2_with_memory}"
EVAL_TAKES="${EVAL_TAKES-014_PKA,033_PKA,038_TKA}"
EVAL_MAX_GROUPS="${EVAL_MAX_GROUPS:-}"
RUN_GT_MEMORY="${RUN_GT_MEMORY:-0}"

cd "$WORKDIR"
mkdir -p logs "$PRED_DIR"

echo "======================================"
echo "Baseline 2 — Evaluation"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Python: $(which python)"
echo "Model: $MODEL_PATH"
echo "EVAL_TAKES: ${EVAL_TAKES:-<all>}"
echo "Started: $(date)"
echo "======================================"

# ---------------------------------------------------------------------------
# 1. Mount NAS
# ---------------------------------------------------------------------------
echo "[1/5] Mounting NAS..."
# shellcheck disable=SC1091
source "$BASELINE_DIR/lib_nas_mount.sh"
b2_mount_nas 3 90 || exit 1
trap b2_unmount_nas EXIT

# ---------------------------------------------------------------------------
# 2. Build test samples if needed
# ---------------------------------------------------------------------------
echo "[2/5] Preparing test samples..."

if [ ! -f "$TEST_SAMPLES" ]; then
    python -m data_pipeline.build_samples \
        --split test \
        --no-augment \
        --output-dir "$SAMPLES_DIR"
fi

echo "  Test samples: $(wc -l < "$TEST_SAMPLES") frames"

python -c "import transformers, peft; from llamafactory.model.qwen2_vl.modeling_qwen2_vl import ImageEmbeddingPooler; print('  deps OK')" || {
    echo "ERROR: env '$ENV_NAME' incomplete. Run: bash baseline_orqa/setup.sh" >&2
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
# 3. Inference — autoregressive memory
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
NAME_ARGS=("b2_autoregressive")

# ---------------------------------------------------------------------------
# 4. Optional GT memory ablation
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
    NAME_ARGS+=("b2_gt_memory")
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
