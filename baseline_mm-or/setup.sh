#!/bin/bash
set -euo pipefail

# Setup Baseline 1 conda env (dlhm-b1), clone ORacle, install deps, cache weights.
#
# Usage (on login node, once):
#   bash baseline_mm-or/setup.sh
#
# Then submit jobs — sbatch scripts activate this env automatically.

ENV_NAME="dlhm-b1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORACLE_DIR="$SCRIPT_DIR/ORacle"
LLAVA_DIR="$ORACLE_DIR/LLaVA"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"

echo "======================================"
echo "Baseline 1 (MM-OR) Setup — env: $ENV_NAME"
echo "======================================"

# -----------------------------------------------------------------------
# 0. Conda
# -----------------------------------------------------------------------
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# -----------------------------------------------------------------------
# 1. Clone ORacle repository
# -----------------------------------------------------------------------
if [ -d "$ORACLE_DIR/.git" ]; then
    echo "[1/4] ORacle repo already cloned at $ORACLE_DIR"
    (cd "$ORACLE_DIR" && git pull --ff-only 2>/dev/null || true)
else
    echo "[1/4] Cloning ORacle repository..."
    git clone https://github.com/egeozsoy/ORacle.git "$ORACLE_DIR"
fi

# -----------------------------------------------------------------------
# 2. Create conda env if missing
# -----------------------------------------------------------------------
echo "[2/4] Ensuring conda env '$ENV_NAME'..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  Env exists — activating"
else
    echo "  Creating env (Python 3.10)..."
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
echo "  Using: $(which python) ($(python --version))"

# -----------------------------------------------------------------------
# 3. Install Python dependencies (ORacle-compatible pins)
# -----------------------------------------------------------------------
echo "[3/4] Installing Python dependencies into $ENV_NAME..."

# PyTorch 2.0.1 + CUDA 11.8 wheels (matches ORacle pins; works on modern drivers)
pip install --upgrade pip setuptools wheel
pip install \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Core training stack (ORacle / LLaVA pins)
pip install \
    transformers==4.31.0 \
    "tokenizers>=0.12.1,<0.14" \
    sentencepiece==0.1.99 \
    accelerate==0.21.0 \
    peft==0.4.0 \
    bitsandbytes==0.41.0 \
    deepspeed==0.9.5 \
    einops==0.6.1 \
    einops-exts==0.0.4 \
    timm==0.6.13 \
    shortuuid \
    "pydantic>=1,<2" \
    "markdown2[all]" \
    "numpy==1.24.4" \
    "setuptools<70" \
    scikit-learn==1.2.2 \
    httpx==0.24.0 \
    uvicorn fastapi \
    ninja \
    wandb==0.16.0 \
    torchinfo==1.8.0 \
    pillow \
    "protobuf<4"


# Eval / data pipeline
pip install nltk rouge-score bert-score

# Install LLaVA as editable package from ORacle's fork
if [ -d "$LLAVA_DIR" ]; then
    echo "  Installing ORacle LLaVA editable..."
    pip install -e "$LLAVA_DIR" --no-deps

    # Patch ORacle LLaVA so training works without flash-attn / author paths.
    echo "  Applying ORacle LLaVA compatibility patches..."
    python - <<PY
from pathlib import Path

llava = Path(r"$LLAVA_DIR")

# 1) Token-frequency path is machine-specific — use standard CE by default.
trainer = llava / "llava/train/llava_trainer.py"
text = trainer.read_text()
old = """with open('/home/guests/ege_oezsoy/Oracle/data/llava_samples/train_token_freqs_7b_100perm.json') as f:
    token_frequencies = json.load(f)  # TODO switch to this for normal training
# token_frequencies = None # TODO switch to this for symbolic training"""
new = """# Token-frequency weighted loss is ORacle-author specific; use standard CE here.
_TOKEN_FREQ_PATH = os.environ.get("LLAVA_TOKEN_FREQ_PATH", "")
if _TOKEN_FREQ_PATH and os.path.isfile(_TOKEN_FREQ_PATH):
    with open(_TOKEN_FREQ_PATH) as f:
        token_frequencies = json.load(f)
else:
    token_frequencies = None"""
if old in text:
    trainer.write_text(text.replace(old, new))
    print("  patched llava_trainer.py")
elif "token_frequencies = None" in text and "ege_oezsoy" not in text:
    print("  llava_trainer.py already patched")
else:
    print("  WARN: llava_trainer.py unexpected content; check manually")

# 2) train_mem.py: soft-fail if flash-attn missing
train_mem = llava / "llava/train/train_mem.py"
tm = train_mem.read_text()
if "flash-attn unavailable" not in tm:
    train_mem.write_text(
        """# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
try:
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    print("[train_mem] flash-attn monkey patch applied")
except Exception as e:
    print(f"[train_mem] flash-attn unavailable ({e}); using standard attention")

from llava.train.train import train

if __name__ == "__main__":
    train()
"""
    )
    print("  patched train_mem.py")
else:
    print("  train_mem.py already patched")

# 3) llama_patch.py: allow importing upcast_layer_for_flash_attention without flash-attn
patch = llava / "llava/train/llama_patch.py"
lp = patch.read_text()
if "_HAS_FLASH_ATTN" not in lp:
    lp = lp.replace(
        """try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
except Exception:
    raise ModuleNotFoundError(
        "Please install FlashAttention first, e.g., with pip install flash-attn --no-build-isolation, Learn more at https://github.com/Dao-AILab/flash-attention#installation-and-features"
    )""",
        """try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    _HAS_FLASH_ATTN = True
except Exception:
    flash_attn_varlen_qkvpacked_func = None
    unpad_input = None
    pad_input = None
    _HAS_FLASH_ATTN = False""",
    )
    if "def replace_attn_with_flash_attn():\n    cuda_major" in lp:
        lp = lp.replace(
            "def replace_attn_with_flash_attn():\n    cuda_major",
            """def replace_attn_with_flash_attn():
    if not _HAS_FLASH_ATTN:
        raise ModuleNotFoundError(
            "Please install FlashAttention first, e.g., with pip install flash-attn --no-build-isolation"
        )
    cuda_major""",
        )
    patch.write_text(lp)
    print("  patched llama_patch.py")
else:
    print("  llama_patch.py already patched")
PY
fi

# flash-attn is required by ORacle (train_mem / llama_patch). Needs CUDA toolkit.
# Soft patches above are only a fallback if this build fails.
if command -v module >/dev/null 2>&1; then
    module load cuda/11.8.0 2>/dev/null || true
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
echo "  Installing flash-attn==2.3.4 (may take several minutes)..."
MAX_JOBS=4 pip install flash-attn==2.3.4 --no-build-isolation || \
    echo "  WARNING: flash-attn install failed — training will use standard attention fallback"

echo "  Verifying imports..."
python - <<'PY'
import torch, transformers, peft, bitsandbytes, deepspeed
import llava
print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"  transformers={transformers.__version__}")
print(f"  peft={peft.__version__} deepspeed={deepspeed.__version__}")
print(f"  llava OK")
try:
    import flash_attn
    print(f"  flash_attn={flash_attn.__version__}")
except Exception as e:
    print(f"  flash_attn MISSING ({e})")
PY

# -----------------------------------------------------------------------
# 4. Download LLaVA-v1.5-7b base model
# -----------------------------------------------------------------------
echo "[4/4] Ensuring LLaVA-v1.5-7b weights are cached..."
python - <<'PY'
from huggingface_hub import snapshot_download
import os
cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(f"  HF cache: {cache}")
try:
    snapshot_download("liuhaotian/llava-v1.5-7b", cache_dir=cache)
    print("  LLaVA-v1.5-7b weights ready.")
except Exception as e:
    print(f"  WARNING: could not download weights now ({e})")
    print("  They will download on first training run if HF access works.")
PY

echo ""
echo "======================================"
echo "Setup complete."
echo "  conda env:  $ENV_NAME"
echo "  activate:   conda activate $ENV_NAME"
echo "  ORacle:     $ORACLE_DIR"
echo "  Next:       bash baseline_mm-or/run_all.sh"
echo "======================================"
