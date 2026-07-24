#!/bin/bash
#SBATCH --job-name=b2_phase2
#SBATCH --partition=part-1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --gres=gpu:A40:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/b2_phase2_%j.out
#SBATCH --error=logs/b2_phase2_%j.err

# Phase 2: ORQA-Temp — temporal memory + curriculum from Phase 1.
# Matches paper temporal variant + official qwen2vl_lora_sft_QA_temporality.yaml
# (previous_model_weights = Phase 1 checkpoint, Truetemp data).

set -euo pipefail

WORKDIR="/home/guests/nhat_vu/dlhm"
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

# ---------------------------------------------------------------------------
# MM-OR dataset (local cluster path — no NAS mount needed)
# ---------------------------------------------------------------------------
export MM_OR_PROCESSED_ROOT="/home/guests/shared/ORDatasets/MM-OR"

SAMPLES_DIR="$WORKDIR/data_pipeline/samples"

TRAIN_DATA="$BASELINE_DIR/data/train_with_memory.json"
VAL_DATA="$BASELINE_DIR/data/val_with_memory.json"
PHASE1_CKPT="$BASELINE_DIR/checkpoints/phase1_no_memory"
CKPT_DIR="$BASELINE_DIR/checkpoints/phase2_with_memory"
CACHE_ROOT="$BASELINE_DIR/data/cache"
CACHE_FILE="$CACHE_ROOT/train_with_memory.arrow"
EVAL_CACHE_FILE="$CACHE_ROOT/val_with_memory.arrow"
CFG_TEMPLATE="$BASELINE_DIR/configs/hierarchy_lora_sft_phase2.yaml"
CFG_RUNTIME="$BASELINE_DIR/configs/hierarchy_lora_sft_phase2.runtime.yaml"

cd "$WORKDIR"
mkdir -p logs "$CKPT_DIR" "$CACHE_ROOT"

echo "======================================"
echo "Baseline 2 — Phase 2 (With Memory / ORQA-Temp)"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Python: $(which python)"
echo "MM_OR_PROCESSED_ROOT: $MM_OR_PROCESSED_ROOT"
echo "Started: $(date)"
echo "======================================"

# ---------------------------------------------------------------------------
# Resolve Phase 1 checkpoint (prefer latest checkpoint-* under phase1 dir)
# ---------------------------------------------------------------------------
resolve_ckpt() {
    local root="$1"
    if [ ! -d "$root" ]; then
        echo ""
        return
    fi
    local latest
    latest=$(ls -d "$root"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -n "${latest:-}" ]; then
        echo "$latest"
    elif [ -f "$root/adapter_config.json" ] || [ -f "$root/visual_block.pt" ]; then
        echo "$root"
    else
        echo ""
    fi
}

PHASE1_WEIGHTS="$(resolve_ckpt "$PHASE1_CKPT")"
if [ -z "$PHASE1_WEIGHTS" ]; then
    echo "ERROR: Phase 1 checkpoint not found under $PHASE1_CKPT"
    echo "Run train_phase1.sh first."
    exit 1
fi
echo "Phase 1 weights (curriculum): $PHASE1_WEIGHTS"

# ---------------------------------------------------------------------------
# 1. Verify dataset is accessible
# ---------------------------------------------------------------------------
echo "[1/4] Verifying dataset access..."
if [ ! -d "$MM_OR_PROCESSED_ROOT/001_PKA" ]; then
    echo "ERROR: MM-OR dataset not found at $MM_OR_PROCESSED_ROOT" >&2
    exit 1
fi
echo "  Dataset OK: $MM_OR_PROCESSED_ROOT"

# ---------------------------------------------------------------------------
# 2. Build samples with temporal aug + convert with_memory
# ---------------------------------------------------------------------------
echo "[2/4] Preparing Phase 2 training + validation data..."

echo "  Building JSONL samples (train+val, with temporal augmentation)..."
python -m data_pipeline.build_samples \
    --split train \
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

rm -rf "$CACHE_ROOT"
mkdir -p "$CACHE_ROOT"

echo "  Train: $TRAIN_DATA ($(python -c "import json; print(len(json.load(open('$TRAIN_DATA'))))") samples)"
echo "  Val:   $VAL_DATA ($(python -c "import json; print(len(json.load(open('$VAL_DATA'))))") samples)"
echo "  image_dir: $MM_OR_PROCESSED_ROOT"

python - <<PY
import json, random
from pathlib import Path
root = Path("$MM_OR_PROCESSED_ROOT")
data = json.load(open("$TRAIN_DATA"))
assert not data[0]["images"][0].startswith("/"), data[0]["images"][0]
sample = random.sample(data, min(32, len(data)))
missing = [p for s in sample for p in s["images"] if not (root / p).is_file()]
if missing:
    raise SystemExit(f"ERROR: missing images under {root}, e.g. {missing[0]}")
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
       .replace("__PREVIOUS_WEIGHTS__", "$PHASE1_WEIGHTS")
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

python "$BASELINE_DIR/patches/optional_pc_audio.py" "$ORQA_DIR"
python "$BASELINE_DIR/patches/collator_skip_hierarchy_ids.py" "$ORQA_DIR"
python "$BASELINE_DIR/patches/skip_image_load_tokenize.py" "$ORQA_DIR"

python -c "import transformers, peft, bitsandbytes; from llamafactory.train.tuner import run_exp; print('  deps OK')" || {
    echo "ERROR: env '$ENV_NAME' incomplete. Run: bash baseline_orqa/setup.sh" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 4. Launch training
# ---------------------------------------------------------------------------
echo "[4/4] Starting Phase 2 (curriculum) training..."

export WANDB_PROJECT="dlhm-hierarchy-baselines"
export WANDB_DIR="$WORKDIR/wandb"

cd "$LLAMA_FACTORY_DIR"
python -m src.train "$CFG_RUNTIME"

echo "======================================"
echo "Phase 2 complete."
echo "Checkpoint: $CKPT_DIR"
echo "Finished: $(date)"
echo "======================================"
