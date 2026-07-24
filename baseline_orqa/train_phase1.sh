#!/bin/bash
#SBATCH --job-name=b2_phase1
#SBATCH --partition=NORMAL
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1,VRAM:48G
#SBATCH --exclude=node22
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/b2_phase1_%j.out
#SBATCH --error=logs/b2_phase1_%j.err

# Phase 1: ORQA base — visual grounding only (no temporal memory).
# Matches paper base model + official qwen2vl_lora_sft_QA.yaml (Falsetemp).

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
# Stable path so tokenized caches / relative image_dir stay valid across jobs on a node
NAS_MOUNT="/tmp/${USER}/nas_mount_orqa"
SAMPLES_DIR="$WORKDIR/data_pipeline/samples"

TRAIN_DATA="$BASELINE_DIR/data/train_no_memory.json"
VAL_DATA="$BASELINE_DIR/data/val_no_memory.json"
CKPT_DIR="$BASELINE_DIR/checkpoints/phase1_no_memory"
# HF datasets format_cache_file_name requires a '.' (extension) when num_proc>1
CACHE_ROOT="$BASELINE_DIR/data/cache"
CACHE_FILE="$CACHE_ROOT/train_no_memory.arrow"
EVAL_CACHE_FILE="$CACHE_ROOT/val_no_memory.arrow"
CFG_TEMPLATE="$BASELINE_DIR/configs/hierarchy_lora_sft_phase1.yaml"
CFG_RUNTIME="$BASELINE_DIR/configs/hierarchy_lora_sft_phase1.runtime.yaml"

cd "$WORKDIR"
mkdir -p logs "$CKPT_DIR" "$CACHE_ROOT"

# shellcheck disable=SC1091
source "$BASELINE_DIR/lib_nas_mount.sh"

echo "======================================"
echo "Baseline 2 — Phase 1 (No Memory / ORQA base)"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Python: $(which python)"
echo "Started: $(date)"
echo "======================================"

# ---------------------------------------------------------------------------
# 1. Mount NAS
# ---------------------------------------------------------------------------
echo "[1/4] Mounting NAS..."
b2_mount_nas 3 90 || exit 1
trap b2_unmount_nas EXIT

# ---------------------------------------------------------------------------
# 2. Build samples + convert (no temporal aug; strip memory in JSON)
# ---------------------------------------------------------------------------
echo "[2/4] Preparing Phase 1 training + validation data..."

echo "  Building JSONL samples (train+val, no temporal augmentation)..."
python -m data_pipeline.build_samples \
    --split train \
    --no-augment \
    --output-dir "$SAMPLES_DIR"
python -m data_pipeline.build_samples \
    --split val \
    --no-augment \
    --output-dir "$SAMPLES_DIR"

echo "  Converting to Qwen2-VL QA JSON (relative image paths)..."
python "$BASELINE_DIR/convert_to_qwen_json.py" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$BASELINE_DIR/data" \
    --processed-root "$MM_OR_PROCESSED_ROOT" \
    --splits train val \
    --augment-views \
    --relative-images

# Drop stale tokenized caches (old absolute /tmp/nas_mount_PID paths)
rm -rf "$CACHE_ROOT"
mkdir -p "$CACHE_ROOT"

# Phase 1 must not leave *_with_memory.json, otherwise Phase 2 would
# skip rebuilding and miss temporal augmentation.
rm -f "$BASELINE_DIR/data/train_with_memory.json" \
      "$BASELINE_DIR/data/val_with_memory.json"

echo "  Train: $TRAIN_DATA ($(python -c "import json; print(len(json.load(open('$TRAIN_DATA'))))") samples)"
echo "  Val:   $VAL_DATA ($(python -c "import json; print(len(json.load(open('$VAL_DATA'))))") samples)"
echo "  image_dir: $MM_OR_PROCESSED_ROOT"

# Sanity: relative paths + openable under mount
python - <<PY
import json, random
from pathlib import Path
root = Path("$MM_OR_PROCESSED_ROOT")
data = json.load(open("$TRAIN_DATA"))
img0 = data[0]["images"][0]
assert not img0.startswith("/"), f"expected relative image path, got {img0}"
sample = random.sample(data, min(32, len(data)))
missing = []
for s in sample:
    for p in s["images"]:
        if not (root / p).is_file():
            missing.append(p)
if missing:
    raise SystemExit(f"ERROR: {len(missing)} sampled image(s) missing under {root}, e.g. {missing[0]}")
print(f"  relative images OK (spot-checked {len(sample)} samples)")
PY

python - <<PY
from pathlib import Path
tpl = Path("$CFG_TEMPLATE").read_text()
out = (
    tpl.replace("__DATA_JSON__", "$TRAIN_DATA")
       .replace("__EVAL_DATA_JSON__", "$VAL_DATA")
       .replace("__CACHE_DIR__", "$CACHE_FILE")
       .replace("__EVAL_CACHE_DIR__", "$EVAL_CACHE_FILE")
       .replace("__IMAGE_DIR__", "$MM_OR_PROCESSED_ROOT")
       .replace("__OUTPUT_DIR__", "$CKPT_DIR")
)
Path("$CFG_RUNTIME").write_text(out)
print("  wrote $CFG_RUNTIME")
PY

# ---------------------------------------------------------------------------
# 3. Verify env
# ---------------------------------------------------------------------------
echo "[3/4] Checking ORQA + conda env..."

if [ ! -d "$ORQA_DIR/.git" ] || [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "ERROR: ORQA/LLaMA-Factory missing. Run: bash baseline_orqa/setup.sh" >&2
    exit 1
fi

python -c "import transformers, peft, bitsandbytes; from llamafactory.train.tuner import run_exp; print('  deps OK')" || {
    echo "ERROR: env '$ENV_NAME' incomplete. Run: bash baseline_orqa/setup.sh" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 4. Launch training
# ---------------------------------------------------------------------------
echo "[4/4] Starting Phase 1 training..."

export WANDB_PROJECT="dlhm-hierarchy-baselines"
export WANDB_DIR="$WORKDIR/wandb"

cd "$LLAMA_FACTORY_DIR"
python -m src.train "$CFG_RUNTIME"

echo "======================================"
echo "Phase 1 complete."
echo "Checkpoint: $CKPT_DIR"
echo "Finished: $(date)"
echo "======================================"
