#!/bin/bash
# Submit the full Baseline 2 pipeline as chained SLURM jobs (TranslaTUM cluster).
# Phase 1 (no memory) → Phase 2 (with memory, curriculum) → Evaluation.
# Matches ORQA paper: base then temporal variant from that checkpoint.
#
# Prerequisites (once on login node):
#   bash baseline_orqa/setup.sh
#
# Usage:
#   bash baseline_orqa/run_all_translatum.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$SCRIPT_DIR")"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="dlhm-b2"

cd "$WORKDIR"
mkdir -p logs

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "ERROR: conda env '$ENV_NAME' not found."
    echo "Run first: bash baseline_orqa/setup.sh"
    exit 1
fi
conda activate "$ENV_NAME"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_cuda_env.sh"
export PYTHONPATH="$SCRIPT_DIR/ORQA/Qwen2-VL/LLaMA-Factory/src:${SCRIPT_DIR}/ORQA:${PYTHONPATH:-}"
python -c "import torch, transformers, peft, bitsandbytes; from llamafactory.model.qwen2_vl.modeling_qwen2_vl import ImageEmbeddingPooler" || {
    echo "ERROR: env '$ENV_NAME' incomplete (torch/CUDA libs). Re-run: bash baseline_orqa/setup.sh"
    exit 1
}
echo "Preflight OK — using env $ENV_NAME ($(which python))"

echo "Submitting Baseline 2 pipeline (ORQA curriculum) — TranslaTUM cluster..."

JOB1=$(sbatch --parsable "$SCRIPT_DIR/train_phase1_translatum.sh")
echo "Phase 1 submitted: job $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 "$SCRIPT_DIR/train_phase2_translatum.sh")
echo "Phase 2 submitted: job $JOB2 (depends on $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 "$SCRIPT_DIR/run_eval_translatum.sh")
echo "Eval submitted:    job $JOB3 (depends on $JOB2)"

echo ""
echo "Pipeline: $JOB1 → $JOB2 → $JOB3"
echo "Monitor:  squeue -u \$USER"
echo "Wandb:    https://wandb.ai (project: dlhm-hierarchy-baselines)"
