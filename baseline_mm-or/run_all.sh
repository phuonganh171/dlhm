#!/bin/bash
# Submit the full Baseline 1 pipeline as chained SLURM jobs.
# Phase 1 → Phase 2 → Evaluation, each starting after the previous succeeds.
#
# Usage:
#   bash baseline_mm-or/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$SCRIPT_DIR")"

cd "$WORKDIR"
mkdir -p logs

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
