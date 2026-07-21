#!/bin/bash
# Submit the full Baseline 1 pipeline as chained SLURM jobs.
# Phase 1 → Phase 2 → Evaluation, each starting after the previous succeeds.
#
# Prerequisites (once on login node):
#   bash baseline_mm-or/setup.sh   # creates conda env dlhm-b1 + installs deps
#
# Usage:
#   bash baseline_mm-or/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$SCRIPT_DIR")"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="dlhm-b1"

cd "$WORKDIR"
mkdir -p logs

# Preflight: env must exist before submitting GPU jobs
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "ERROR: conda env '$ENV_NAME' not found."
    echo "Run first: bash baseline_mm-or/setup.sh"
    exit 1
fi
conda activate "$ENV_NAME"
if command -v module >/dev/null 2>&1; then
    module load cuda/11.8.0
fi
export LD_LIBRARY_PATH="${CUDA_HOME:+$CUDA_HOME/lib64:}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SCRIPT_DIR/ORacle/LLaVA:${PYTHONPATH:-}"
python -c "import transformers, peft, bitsandbytes, deepspeed, llava" || {
    echo "ERROR: env '$ENV_NAME' incomplete. Re-run: bash baseline_mm-or/setup.sh"
    exit 1
}
echo "Preflight OK — using env $ENV_NAME ($(which python))"

echo "Submitting Baseline 1 pipeline..."

# Phase 1: visual grounding (no memory)
JOB1=$(sbatch --parsable "$SCRIPT_DIR/train_phase1.sh")
echo "Phase 1 submitted: job $JOB1"

# Phase 2: temporal curriculum (with memory) — starts after Phase 1 succeeds
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 "$SCRIPT_DIR/train_phase2.sh")
echo "Phase 2 submitted: job $JOB2 (depends on $JOB1)"

# Evaluation — starts after Phase 2 succeeds
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 "$SCRIPT_DIR/run_eval.sh")
echo "Eval submitted:    job $JOB3 (depends on $JOB2)"

echo ""
echo "Pipeline: $JOB1 → $JOB2 → $JOB3"
echo "Monitor:  squeue -u \$USER"
echo "Wandb:    https://wandb.ai (project: dlhm-hierarchy-baselines)"
