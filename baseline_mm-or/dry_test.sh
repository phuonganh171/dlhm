#!/usr/bin/env bash
# Small dry test for Baseline 1: conda env + data prep + script logic.
# Usage: bash baseline_mm-or/dry_test.sh
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$WORKDIR/baseline_mm-or"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="dlhm-b1"
DRY_DIR="/tmp/${USER}_b1_dry_$$"
SAMPLES_TINY="$DRY_DIR/samples"
DATA_DIR="$DRY_DIR/data"
N_SAMPLES="${N_SAMPLES:-32}"

cd "$WORKDIR"
mkdir -p "$SAMPLES_TINY" "$DATA_DIR" "$DRY_DIR/logs"

# Activate the training env (same as sbatch scripts)
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8.0
fi
export LD_LIBRARY_PATH="${CUDA_HOME:+$CUDA_HOME/lib64:}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BASELINE/ORacle/LLaVA:${PYTHONPATH:-}"

echo "======================================"
echo "Baseline 1 dry test"
echo "Python: $(which python)"
echo "DRY_DIR=$DRY_DIR"
echo "======================================"

pass=0
fail=0
ok() { echo "  PASS: $1"; pass=$((pass+1)); }
bad() { echo "  FAIL: $1"; fail=$((fail+1)); }

# ---------------------------------------------------------------------------
# 0. Conda env imports
# ---------------------------------------------------------------------------
echo "[0/7] Checking conda env '$ENV_NAME' imports..."
if python -c "
import torch, transformers, peft, bitsandbytes, deepspeed, llava, wandb
assert transformers.__version__.startswith('4.31')
print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()}')
print(f'  transformers={transformers.__version__} peft={peft.__version__}')
print(f'  deepspeed={deepspeed.__version__} llava+wandb OK')
"; then
  ok "conda env imports"
else
  bad "conda env imports"
fi

# ---------------------------------------------------------------------------
# 1. Tiny JSONL subset
# ---------------------------------------------------------------------------
echo "[1/7] Building tiny JSONL subset ($N_SAMPLES samples)..."
python - <<PY
import json
from pathlib import Path
n = $N_SAMPLES
src = Path("data_pipeline/samples/train.jsonl")
out_dir = Path("$SAMPLES_TINY")
out_dir.mkdir(parents=True, exist_ok=True)
rows = []
with open(src) as f:
    for i, line in enumerate(f):
        if i >= n:
            break
        rows.append(json.loads(line))
assert rows, "no samples read"

def azure_views(sample):
    """Rewrite to camera01–04 sharing one frame id (azure-style; ORacle 4-view)."""
    tp = sample.get("tp_id") or sample["id"].rsplit("/", 1)[-1]
    try:
        fid = int(tp)
    except ValueError:
        fid = 0
    sample["image"] = [
        f"colorimage/camera{c:02d}_colorimage-{fid:06d}.jpg" for c in range(1, 5)
    ]
    return sample

rows = [azure_views(r) for r in rows]
for split, data in [("train", rows[: max(1, n*3//4)]), ("val", rows[max(1, n*3//4):]), ("test", rows[: max(1, n//4)])]:
    if not data:
        data = rows[:1]
    with open(out_dir / f"{split}.jsonl", "w") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(data)} -> {out_dir / (split + '.jsonl')}")
PY
ok "tiny JSONL subset"

# ---------------------------------------------------------------------------
# 2. Convert to LLaVA JSON
# ---------------------------------------------------------------------------
echo "[2/7] Converting to LLaVA JSON..."
python "$BASELINE/convert_to_llava_json.py" \
  --samples-dir "$SAMPLES_TINY" \
  --output-dir "$DATA_DIR" \
  --processed-root /tmp/fake_mmor_dry \
  --allow-missing-images \
  --splits train val

python - <<PY
import json, re
from pathlib import Path
mem = re.compile(r"<memory_start>.*?<memory_end>", re.DOTALL)
data_dir = Path("$DATA_DIR")
for name in ["train_no_memory.json", "train_with_memory.json"]:
    p = data_dir / name
    assert p.exists(), f"missing {p}"
    data = json.load(open(p))
    assert len(data) > 0
    # Paths must be relative to MM-OR_processed (survive per-job NAS remounts)
    assert all(not str(x["image"][0]).startswith("/") for x in data if x.get("image"))
    assert all("/" in str(x["image"][0]) for x in data if x.get("image"))
    # Azure multi-view: camera01–camera04 (ORacle pooler max = 4)
    assert all(len(x["image"]) == 4 for x in data if x.get("image"))
    assert all("camera04" in x["image"][-1] for x in data if x.get("image"))
    has_mem = [bool(mem.search(s["conversations"][0]["value"])) for s in data]
    if "no_memory" in name:
        assert not any(has_mem), "no_memory still has memory blocks"
    else:
        print(f"  {name}: {sum(has_mem)}/{len(data)} with memory")
    print(f"  {name}: {len(data)} samples OK")
print("convert checks OK")
PY
ok "convert_to_llava_json"

# ---------------------------------------------------------------------------
# 2b. Missing-image filter drops samples when files are absent
# ---------------------------------------------------------------------------
echo "[2b/7] Checking missing-image filter..."
python - <<PY
import json, tempfile
from pathlib import Path
import subprocess, sys
samples = Path("$SAMPLES_TINY")
raw = [json.loads(l) for l in open(samples / "train.jsonl") if l.strip()]
assert raw
# Point processed root at empty dir → all samples missing → convert must error
empty = Path("$DRY_DIR/empty_mmor")
empty.mkdir(exist_ok=True)
out = Path("$DRY_DIR/filter_out")
out.mkdir(exist_ok=True)
rc = subprocess.call([
    sys.executable, "$BASELINE/convert_to_llava_json.py",
    "--samples-dir", str(samples),
    "--output-dir", str(out),
    "--processed-root", str(empty),
    "--splits", "train",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
assert rc != 0, "expected convert to fail when all images missing"
print("  filter rejects empty mount OK")
PY
ok "missing-image filter"

# ---------------------------------------------------------------------------
# 3. Phase1 data-prep behavior: delete with_memory after convert
# ---------------------------------------------------------------------------
echo "[3/7] Simulating Phase 1 data-prep (must remove *_with_memory.json)..."
PHASE1_DATA="$DRY_DIR/phase1_data"
mkdir -p "$PHASE1_DATA"
python "$BASELINE/convert_to_llava_json.py" \
  --samples-dir "$SAMPLES_TINY" \
  --output-dir "$PHASE1_DATA" \
  --processed-root /tmp/fake_mmor_dry \
  --allow-missing-images \
  --splits train val >/dev/null
rm -f "$PHASE1_DATA/train_with_memory.json" "$PHASE1_DATA/val_with_memory.json"
test -f "$PHASE1_DATA/train_no_memory.json"
test ! -f "$PHASE1_DATA/train_with_memory.json"
ok "phase1 deletes with_memory"

# ---------------------------------------------------------------------------
# 4. Phase2 would rebuild with_memory because missing
# ---------------------------------------------------------------------------
echo "[4/7] Simulating Phase 2 rebuild of with_memory..."
test ! -f "$PHASE1_DATA/train_with_memory.json"
python "$BASELINE/convert_to_llava_json.py" \
  --samples-dir "$SAMPLES_TINY" \
  --output-dir "$PHASE1_DATA" \
  --processed-root /tmp/fake_mmor_dry \
  --allow-missing-images \
  --splits train >/dev/null
test -f "$PHASE1_DATA/train_with_memory.json"
ok "phase2 can rebuild with_memory"

# ---------------------------------------------------------------------------
# 5. Eval smoke (identity predictions)
# ---------------------------------------------------------------------------
echo "[5/7] Eval smoke test..."
PRED="$DRY_DIR/pred_identity.jsonl"
python - <<PY
import json
from pathlib import Path
src = Path("$SAMPLES_TINY/test.jsonl")
out = Path("$PRED")
with open(src) as f, open(out, "w") as g:
    for line in f:
        s = json.loads(line)
        g.write(json.dumps({
            "id": s["id"],
            "pred_l0": s["gt_l0"],
            "pred_l1": s["gt_l1"],
            "pred_l2": s["gt_l2"],
        }) + "\n")
PY
WANDB_MODE=disabled python "$BASELINE/eval_predictions.py" \
  --gt "$SAMPLES_TINY/test.jsonl" \
  --predictions "$PRED" \
  --names "dry_identity" \
  --model-info "dry_test" \
  --project "dlhm-hierarchy-baselines-dry" \
  --no-bertscore > "$DRY_DIR/logs/eval_out.txt" 2>&1 || true
test -f "${PRED%.jsonl}_results.json"
python - <<PY
import json
r = json.load(open("${PRED%.jsonl}_results.json"))
assert r["matched"] > 0
assert abs(r["l0"]["bleu"] - 1.0) < 1e-6
print("  identity BLEU=1.0 OK, matched=", r["matched"])
PY
ok "eval_predictions identity"

# ---------------------------------------------------------------------------
# 6. ORacle training flags + train_mem import + inference compile
# ---------------------------------------------------------------------------
echo "[6/7] Checking ORacle train entrypoint + flags..."
python - <<PY
from pathlib import Path
text = Path("$BASELINE/ORacle/LLaVA/llava/train/train.py").read_text()
for n in ["mv_type", "do_img_order_augment", "unfreeze_n_vision_tower_layers",
          "curriculum_learning_weights", "bits", "lora_r"]:
    assert n in text, f"missing {n}"
print("  ORacle flags OK")

from data_pipeline.assemble import parse_model_output
assert parse_model_output("L0: a | L1: b | L2: c") == ("a", "b", "c")
print("  parse_model_output OK")

# Import the training module (flash-attn patch may warn — that's OK)
import sys
sys.path.insert(0, "$BASELINE/ORacle/LLaVA")
try:
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    print("  flash_attn monkey patch importable")
except Exception as e:
    print(f"  flash_attn patch skipped ({type(e).__name__}: {e})")

from llava.train import train as train_mod
print("  llava.train importable")

compile(Path("$BASELINE/inference.py").read_text(), "inference.py", "exec")
print("  inference.py compiles OK")
PY
ok "oracle train imports"

# ---------------------------------------------------------------------------
# 7. Script syntax + conda activate lines present
# ---------------------------------------------------------------------------
echo "[7/7] Checking shell scripts..."
bash -n "$BASELINE/train_phase1.sh"
bash -n "$BASELINE/train_phase2.sh"
bash -n "$BASELINE/run_eval.sh"
bash -n "$BASELINE/run_all.sh"
bash -n "$BASELINE/setup.sh"
for f in train_phase1.sh train_phase2.sh run_eval.sh run_all.sh; do
  grep -q 'conda activate' "$BASELINE/$f"
  grep -q 'cuda/11.8.0' "$BASELINE/$f"
done
ok "shell syntax + conda/cuda wiring"

echo "======================================"
echo "Dry test done: $pass passed, $fail failed"
echo "Artifacts: $DRY_DIR"
echo "======================================"
exit $fail
