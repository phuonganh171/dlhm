#!/bin/bash
set -euo pipefail

# Setup Baseline 2 conda env (dlhm-b2), clone ORQA, install deps, cache weights.
#
# Usage (on login node, once):
#   bash baseline_orqa/setup.sh
#
# Then submit jobs — sbatch scripts activate this env automatically.

ENV_NAME="dlhm-b2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORQA_DIR="$SCRIPT_DIR/ORQA"
LLAMA_FACTORY_DIR="$ORQA_DIR/Qwen2-VL/LLaMA-Factory"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"

echo "======================================"
echo "Baseline 2 (ORQA / Qwen2-VL) Setup — env: $ENV_NAME"
echo "======================================"

# -----------------------------------------------------------------------
# 0. Conda
# -----------------------------------------------------------------------
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# -----------------------------------------------------------------------
# 1. Clone ORQA repository (official code with Qwen2-VL + ORacle pooler)
# -----------------------------------------------------------------------
if [ -d "$ORQA_DIR/.git" ]; then
    echo "[1/5] ORQA repo already cloned at $ORQA_DIR"
    (cd "$ORQA_DIR" && git pull --ff-only 2>/dev/null || true)
else
    echo "[1/5] Cloning ORQA repository..."
    # Sparse-ish: full clone but we do not need the large zipped QA data.
    git clone --depth 1 https://github.com/egeozsoy/ORQA.git "$ORQA_DIR"
fi

# -----------------------------------------------------------------------
# 2. Create conda env if missing
# -----------------------------------------------------------------------
echo "[2/5] Ensuring conda env '$ENV_NAME'..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  Env exists — activating"
else
    echo "  Creating env (Python 3.10)..."
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
echo "  Using: $(which python) ($(python --version))"

# Never build pyarrow/numpy from sdist: old pyarrow build deps pull numpy
# 1.19.x which fails on Python 3.10 (_Py_HashDouble signature change).
export PIP_ONLY_BINARY="pyarrow,numpy"
export PIP_PREFER_BINARY=1

# -----------------------------------------------------------------------
# 3. Install Python dependencies (ORQA pins)
# -----------------------------------------------------------------------
echo "[3/5] Installing Python dependencies into $ENV_NAME..."

# Fast path: env already healthy from a previous partial/full setup.
if python - <<'PY' 2>/dev/null
import torch, transformers, peft, bitsandbytes, pyarrow, numpy, datasets
from llamafactory.model.qwen2_vl.modeling_qwen2_vl import ImageEmbeddingPooler  # noqa: F401
assert torch.__version__.startswith("2.4")
assert transformers.__version__ == "4.46.1"
assert str(numpy.__version__) == "1.23.0"
print(f"  Already ready: torch={torch.__version__} cuda={torch.cuda.is_available()} pyarrow={pyarrow.__version__}")
PY
then
    echo "  Skipping dependency reinstall (env already usable)."
else
    pip install --upgrade pip setuptools wheel

    # Prefer CUDA 12.1 wheels (matches ORQA README); fall back gracefully.
    if ! python -c "import torch" 2>/dev/null; then
        pip install \
            torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
            --index-url https://download.pytorch.org/whl/cu121
    fi

    echo "  Installing binary pyarrow/numpy (avoid source builds)..."
    pip install --only-binary=:all: "numpy==1.23.0" "pyarrow==18.1.0"

    # Core ORQA / LLaMA-Factory stack (skip open3d/spconv — image-only baseline)
    pip install --only-binary=pyarrow,numpy \
        "transformers==4.46.1" \
        "qwen-vl-utils==0.0.2" \
        "numpy==1.23.0" \
        "pyarrow==18.1.0" \
        "timm==1.0.12" \
        "json-tricks==3.17.3" \
        "pytorch-lightning==2.1.2" \
        "torchinfo==1.8.0" \
        "tqdm==4.61.0" \
        "bitsandbytes==0.44.1" \
        "opencv-python==4.10.0.84" \
        "editdistance==0.8.1" \
        "wandb==0.19.0" \
        "scipy==1.12.0" \
        "peft" \
        "accelerate" \
        "datasets<=3.1.0" \
        "einops" \
        "sentencepiece" \
        "protobuf" \
        "pillow" \
        "addict" \
        "tiktoken" \
        "modelscope" \
        "pandas>=2.0.0" \
        "av" \
        "gradio>=4.0.0,<5.0.0" \
        "trl>=0.8.6,<=0.9.6" \
        "tokenizers>=0.19.0,<0.20.4" \
        "fire" \
        "tyro<0.9.0" \
        "uvicorn" \
        "fastapi" \
        "sse-starlette" \
        "matplotlib>=3.7.0" \
        "pydantic"

    # Eval / data pipeline
    pip install --only-binary=pyarrow,numpy nltk rouge-score bert-score jieba rouge-chinese

    # Install LLaMA-Factory editable without resolving deps again (already installed).
    echo "  Installing LLaMA-Factory editable (--no-deps)..."
    cd "$LLAMA_FACTORY_DIR"
    pip install -e ".[torch,metrics]" --no-build-isolation --no-deps
fi

# Always (re)apply local ORQA patches — safe if already applied.
echo "  Applying image-only ORQA patches..."
python "$SCRIPT_DIR/patches/image_only_pooler.py" "$ORQA_DIR"
echo "  Applying eval dataset path patches..."
python "$SCRIPT_DIR/patches/eval_dataset_path.py" "$ORQA_DIR"
echo "  Applying optional pc/audio aligner patch..."
python "$SCRIPT_DIR/patches/optional_pc_audio.py" "$ORQA_DIR"
echo "  Applying collator hierarchy-id patch..."
python "$SCRIPT_DIR/patches/collator_skip_hierarchy_ids.py" "$ORQA_DIR"
# flash-attn (ORQA recommends 2.6.1; soft-fail if build fails)
# Keep pip nvidia-* lib dirs on LD_LIBRARY_PATH: loading a cluster CUDA module
# alone can hide libcudnn.so.9 that torch resolves via ~/.local nvidia wheels.
if ! python -c "import flash_attn" 2>/dev/null; then
    NVIDIA_LIB_ROOTS=()
    while IFS= read -r d; do
        [ -n "$d" ] && NVIDIA_LIB_ROOTS+=("$d")
    done < <(python - <<'PY'
import glob, os, site
roots = []
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    roots.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print("\n".join(dict.fromkeys(roots)))
PY
)
    NVIDIA_LIBS=$(IFS=:; echo "${NVIDIA_LIB_ROOTS[*]:-}")
    if command -v module >/dev/null 2>&1; then
        module load cuda/12.1.1 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda/11.8.0 2>/dev/null || true
    fi
    if [ -n "${CUDA_HOME:-}" ]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="${NVIDIA_LIBS:+$NVIDIA_LIBS:}$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    elif [ -n "${NVIDIA_LIBS:-}" ]; then
        export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    echo "  Installing flash-attn==2.6.1 (may take several minutes)..."
    MAX_JOBS=4 pip install flash-attn==2.6.1 --no-build-isolation || \
        echo "  WARNING: flash-attn install failed — set flash_attn: auto in the YAML if needed"
else
    echo "  flash-attn already installed — skipping"
fi

echo "  Verifying imports..."
python - <<'PY'
import torch, transformers, peft, bitsandbytes
print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"  transformers={transformers.__version__}")
print(f"  peft={peft.__version__}")
from llamafactory.model.qwen2_vl.modeling_qwen2_vl import ImageEmbeddingPooler
print("  ORQA ImageEmbeddingPooler OK")
try:
    import flash_attn
    print(f"  flash_attn={flash_attn.__version__}")
except Exception as e:
    print(f"  flash_attn MISSING ({e})")
PY

# -----------------------------------------------------------------------
# 4. Ensure dataset_info has orqa entry (already in upstream)
# -----------------------------------------------------------------------
echo "[4/5] Checking dataset_info.json orqa entry..."
python - <<PY
import json
from pathlib import Path
p = Path(r"$LLAMA_FACTORY_DIR") / "data" / "dataset_info.json"
info = json.loads(p.read_text())
assert "orqa" in info, "missing orqa dataset entry"
print("  orqa dataset entry OK")
PY

# -----------------------------------------------------------------------
# 5. Cache Qwen2-VL-2B weights
# -----------------------------------------------------------------------
echo "[5/5] Ensuring Qwen2-VL-2B-Instruct weights are cached..."
python - <<'PY'
from huggingface_hub import snapshot_download
import os
cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(f"  HF cache: {cache}")
try:
    snapshot_download("Qwen/Qwen2-VL-2B-Instruct", cache_dir=cache)
    print("  Qwen2-VL-2B-Instruct weights ready.")
except Exception as e:
    print(f"  WARNING: could not download weights now ({e})")
    print("  They will download on first training run if HF access works.")
PY

echo ""
echo "======================================"
echo "Setup complete."
echo "  conda env:  $ENV_NAME"
echo "  activate:   conda activate $ENV_NAME"
echo "  ORQA:       $ORQA_DIR"
echo "  Next:       bash baseline_orqa/run_all.sh"
echo "======================================"
