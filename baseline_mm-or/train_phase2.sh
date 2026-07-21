#!/bin/bash
#SBATCH --job-name=b1_phase2
#SBATCH --partition=NORMAL
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1,VRAM:48G
# Torch 2.0.1+cu118 has no sm_120 kernels — exclude Blackwell (node22).
#SBATCH --exclude=node22
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/b1_phase2_%j.out
#SBATCH --error=logs/b1_phase2_%j.err

# Phase 2: Temporal curriculum learning (with memory).
# Fine-tunes from Phase 1 checkpoint, now with <memory_start>...<memory_end>
# blocks in the prompt. Temporal augmentation was applied during sample
# building (style mixing + 50% history dropout).

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

# Training data (with temporal memory)
TRAIN_DATA="$BASELINE_DIR/data/train_with_memory.json"

# Phase 1 checkpoint (curriculum resume)
PHASE1_CKPT="$BASELINE_DIR/checkpoints/phase1_no_memory"

# Output checkpoint directory
CKPT_DIR="$BASELINE_DIR/checkpoints/phase2_with_memory"

# DeepSpeed config
DS_CONFIG="$BASELINE_DIR/configs/deepspeed_zero2.json"

cd "$WORKDIR"
mkdir -p logs "$CKPT_DIR"

echo "======================================"
echo "Baseline 1 — Phase 2 Training (With Memory)"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Python: $(which python)"
echo "Started: $(date)"
echo "======================================"

# ---------------------------------------------------------------------------
# Verify Phase 1 checkpoint exists
# ---------------------------------------------------------------------------
if [ ! -d "$PHASE1_CKPT" ] || [ -z "$(ls -A "$PHASE1_CKPT" 2>/dev/null)" ]; then
    echo "ERROR: Phase 1 checkpoint not found at $PHASE1_CKPT"
    echo "Run train_phase1.sh first."
    exit 1
fi
echo "Phase 1 checkpoint: $PHASE1_CKPT"

# ---------------------------------------------------------------------------
# 1. Mount NAS
# ---------------------------------------------------------------------------
echo "[1/3] Mounting NAS..."
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
# 2. Build samples with augmentation + convert if needed
# ---------------------------------------------------------------------------
echo "[2/3] Preparing training data..."

# Always rebuild with-memory JSON against the live NAS mount (and temporal aug).
echo "  Building JSONL samples (with augmentation)..."
python -m data_pipeline.build_samples \
    --split train \
    --output-dir "$SAMPLES_DIR"

echo "  Converting to LLaVA JSON (skip missing images)..."
python "$BASELINE_DIR/convert_to_llava_json.py" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$BASELINE_DIR/data" \
    --processed-root "$MM_OR_PROCESSED_ROOT" \
    --splits train val

echo "  Training data: $TRAIN_DATA"
echo "  Samples: $(python -c "import json; print(len(json.load(open('$TRAIN_DATA'))))")"
echo "  image_folder: $MM_OR_PROCESSED_ROOT"

# ---------------------------------------------------------------------------
# 3. Verify env, then launch training (curriculum from Phase 1)
# ---------------------------------------------------------------------------
echo "[3/3] Checking deps and starting Phase 2 training..."

python -c "import transformers, peft, bitsandbytes, deepspeed, llava; print('  deps OK')" || {
    echo "ERROR: env '$ENV_NAME' incomplete. Run: bash baseline_mm-or/setup.sh" >&2
    exit 1
}

export WANDB_PROJECT="dlhm-hierarchy-baselines"
export GPUS_PER_NODE=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((28500 + RANDOM % 100))

cd "$LLAVA_DIR"

python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
    --lora_enable True \
    --bits 4 \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed "$DS_CONFIG" \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path "$TRAIN_DATA" \
    --image_folder "$MM_OR_PROCESSED_ROOT" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$CKPT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --max_grad_norm 0.1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "b1_phase2_hierarchy_with_memory" \
    --curriculum_learning_weights "$PHASE1_CKPT" \
    --mv_type "learned" \
    --unfreeze_n_vision_tower_layers 12 \
    --do_img_order_augment

echo "======================================"
echo "Phase 2 training complete."
echo "Checkpoint: $CKPT_DIR"
echo "Finished: $(date)"
echo "======================================"
