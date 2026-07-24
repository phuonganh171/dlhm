#!/usr/bin/env bash
# Shared CUDA / nvidia library setup for Baseline 2 scripts.
# Source AFTER: conda activate dlhm-b2
#
# Torch's cuDNN lives in pip nvidia-* wheels (often ~/.local). Loading a
# cluster CUDA module alone can hide libcudnn.so.9 — always prepend those
# lib dirs before CUDA_HOME/lib64.

_b2_nvidia_lib_dirs() {
  python - <<'PY' 2>/dev/null || true
import glob, os, site
roots = []
cands = []
try:
    cands.extend(site.getsitepackages())
except Exception:
    pass
try:
    cands.append(site.getusersitepackages())
except Exception:
    pass
for sp in cands:
    roots.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(":".join(dict.fromkeys(roots)))
PY
}

if command -v module >/dev/null 2>&1; then
  module load cuda/12.1.1 2>/dev/null \
    || module load cuda/12.1 2>/dev/null \
    || module load cuda/11.8.0 2>/dev/null \
    || true
fi

_NVIDIA_LIBS="$(_b2_nvidia_lib_dirs)"
_PREFIX=""
if [ -n "${_NVIDIA_LIBS:-}" ]; then
  _PREFIX="${_NVIDIA_LIBS}:"
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:${PATH:-}"
  _PREFIX="${_PREFIX}${CUDA_HOME}/lib64:"
fi
export LD_LIBRARY_PATH="${_PREFIX}${LD_LIBRARY_PATH:-}"
unset _NVIDIA_LIBS _PREFIX
