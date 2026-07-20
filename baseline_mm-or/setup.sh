#!/bin/bash
set -euo pipefail

# Setup script for Baseline 1 (MM-OR / ORacle architecture).
# Clones the ORacle repo, installs Python dependencies, and downloads
# the LLaVA-v1.5-7b base model weights.
#
# Usage:
#   bash baseline_mm-or/setup.sh
#
# Prerequisites:
#   - conda or pip environment with Python 3.10+
#   - CUDA toolkit compatible with PyTorch
#   - ~15 GB disk for LLaVA-7B weights

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORACLE_DIR="$SCRIPT_DIR/ORacle"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "======================================"
echo "Baseline 1 (MM-OR) Setup"
echo "======================================"

# -----------------------------------------------------------------------
# 1. Clone ORacle repository
# -----------------------------------------------------------------------
if [ -d "$ORACLE_DIR/.git" ]; then
    echo "[1/3] ORacle repo already cloned at $ORACLE_DIR"
    cd "$ORACLE_DIR" && git pull --ff-only 2>/dev/null || true
    cd "$SCRIPT_DIR"
else
    echo "[1/3] Cloning ORacle repository..."
    git clone https://github.com/egeozsoy/ORacle.git "$ORACLE_DIR"
fi

# -----------------------------------------------------------------------
# 2. Install Python dependencies
# -----------------------------------------------------------------------
echo "[2/3] Installing Python dependencies..."

# Core ORacle / LLaVA deps
pip install --quiet \
    torch torchvision \
    transformers>=4.36.0 \
    peft>=0.7.0 \
    bitsandbytes>=0.41.0 \
    accelerate>=0.25.0 \
    deepspeed>=0.12.0 \
    sentencepiece \
    tokenizers \
    einops \
    flash-attn --no-build-isolation 2>/dev/null || true

# LLaVA-specific (from ORacle's LLaVA/)
pip install --quiet \
    shortuuid \
    httpx==0.24.0 \
    "uvicorn[standard]" \
    gradio \
    markdown2 \
    numpy \
    scikit-learn \
    wandb

# Our data pipeline deps
pip install --quiet nltk rouge-score

# Install LLaVA as editable package from ORacle's fork
if [ -d "$ORACLE_DIR/LLaVA" ]; then
    echo "  Installing LLaVA from ORacle fork..."
    pip install --quiet -e "$ORACLE_DIR/LLaVA" 2>/dev/null || true
fi

echo "  Dependencies installed."

# -----------------------------------------------------------------------
# 3. Download LLaVA-v1.5-7b base model
# -----------------------------------------------------------------------
echo "[3/3] Ensuring LLaVA-v1.5-7b weights are cached..."
python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
print(f'  HF cache: {cache}')
snapshot_download('liuhaotian/llava-v1.5-7b', cache_dir=cache)
print('  LLaVA-v1.5-7b weights ready.')
" || echo "  WARNING: Could not download LLaVA weights (will download during first training run)"

echo ""
echo "======================================"
echo "Setup complete."
echo "  ORacle repo: $ORACLE_DIR"
echo "  Next: Run data pipeline, then train_phase1.sh"
echo "======================================"
