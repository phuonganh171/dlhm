#!/bin/bash
#SBATCH --job-name=dlhm-hier
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/nhatvu/dlhm/logs/annotate_%j.out
#SBATCH --error=/mnt/home/nhatvu/dlhm/logs/annotate_%j.err

# Full pipeline: mount NAS -> build hierarchy (all takes) -> HTML viewers.
# All heavy data stays on NAS (/tmp mount). Repo only gets JSON/HTML + symlinks.

set -euo pipefail

WORKDIR=/mnt/home/nhatvu/dlhm
JOB_TAG="${SLURM_JOB_ID:-local}"
TMP_ROOT="/tmp/nhatvu"

# ---------------------------------------------------------------------------
# Environment (run.sh reads these; job-specific NAS mount avoids conflicts)
# ---------------------------------------------------------------------------
export USE_NAS=1
export NAS_REMOTE="${NAS_REMOTE:-nas:ge42faj}"
export NAS_MOUNT="${NAS_MOUNT:-${TMP_ROOT}/nas_mount_${JOB_TAG}}"
export HF_HOME="${HF_HOME:-${TMP_ROOT}/hf_cache}"
export TMPDIR="${TMPDIR:-${TMP_ROOT}}"
# HF_TOKEN default lives in script/run.sh; export here to override if needed.

# ---------------------------------------------------------------------------
# Job header
# ---------------------------------------------------------------------------
echo "============================================"
echo "[slurm] job=$SLURM_JOB_ID node=$(hostname -s)"
echo "[slurm] workdir=$WORKDIR"
echo "[slurm] nas_mount=$NAS_MOUNT"
echo "[slurm] hf_cache=$HF_HOME"
echo "============================================"

cd "$WORKDIR"

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
mkdir -p \
    "$WORKDIR/logs" \
    "$WORKDIR/hierarchy_output" \
    "$WORKDIR/viewer_links" \
    "$TMP_ROOT" \
    "$HF_HOME"

# ---------------------------------------------------------------------------
# Preflight: tools + GPU
# ---------------------------------------------------------------------------
echo "[setup] Checking required tools ..."
for cmd in rclone uv nvidia-smi; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[error] Required command not found: $cmd" >&2
        exit 1
    fi
    echo "  ok: $(command -v "$cmd")"
done

echo "[setup] GPU:"
nvidia-smi -L || { echo "[error] nvidia-smi failed" >&2; exit 1; }

# Drop stale HuggingFace login tokens that caused 401 on compute nodes
rm -f "$HF_HOME/token" "${HOME}/.cache/huggingface/token" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Cleanup NAS mount on job exit
# ---------------------------------------------------------------------------
cleanup() {
    local rc=$?
    if mountpoint -q "$NAS_MOUNT" 2>/dev/null; then
        echo "[cleanup] Unmounting $NAS_MOUNT"
        fusermount -u "$NAS_MOUNT" 2>/dev/null \
            || umount "$NAS_MOUNT" 2>/dev/null \
            || true
    fi
    exit "$rc"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Run pipeline (venv, NAS mount, all takes — handled by run.sh)
# ---------------------------------------------------------------------------
echo ""
echo "[pipeline] Starting run.sh --all ..."
bash "$WORKDIR/script/run.sh" --all

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Done (job $SLURM_JOB_ID)"
echo "  Hierarchy JSON/HTML : $WORKDIR/hierarchy_output/"
echo "  Video players       : $WORKDIR/*_hierarchy_video_sync_qwen27b.html"
echo "  Slurm logs          : $WORKDIR/logs/annotate_${SLURM_JOB_ID}.{out,err}"
echo "============================================"
