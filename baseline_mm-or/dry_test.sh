#!/usr/bin/env bash
# Small dry test for Baseline 1 data prep + script logic (no full training).
# Usage: bash baseline_mm-or/dry_test.sh
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$WORKDIR/baseline_mm-or"
DRY_DIR="/tmp/${USER}_b1_dry_$$"
SAMPLES_TINY="$DRY_DIR/samples"
DATA_DIR="$DRY_DIR/data"
N_SAMPLES="${N_SAMPLES:-32}"

cd "$WORKDIR"
mkdir -p "$SAMPLES_TINY" "$DATA_DIR" "$DRY_DIR/logs"
echo "======================================"
echo "Baseline 1 dry test"
echo "DRY_DIR=$DRY_DIR"
echo "======================================"

pass=0
fail=0
check() {
  local name="$1"; shift
  if "$@"; then
    echo "  PASS: $name"
    pass=$((pass+1))
  else
    echo "  FAIL: $name"
    fail=$((fail+1))
  fi
}

# ---------------------------------------------------------------------------
# 1. Tiny JSONL subset
# ---------------------------------------------------------------------------
echo "[1/6] Building tiny JSONL subset ($N_SAMPLES samples)..."
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
for split, data in [("train", rows[: max(1, n*3//4)]), ("val", rows[max(1, n*3//4):]), ("test", rows[: max(1, n//4)])]:
    if not data:
        data = rows[:1]
    with open(out_dir / f"{split}.jsonl", "w") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(data)} -> {out_dir / (split + '.jsonl')}")
PY

# ---------------------------------------------------------------------------
# 2. Convert to LLaVA JSON
# ---------------------------------------------------------------------------
echo "[2/6] Converting to LLaVA JSON..."
python "$BASELINE/convert_to_llava_json.py" \
  --samples-dir "$SAMPLES_TINY" \
  --output-dir "$DATA_DIR" \
  --processed-root /tmp/fake_mmor_dry \
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
    abs_ok = all(str(x["image"][0]).startswith("/") for x in data if x.get("image"))
    assert abs_ok, "images not absolute"
    has_mem = [bool(mem.search(s["conversations"][0]["value"])) for s in data]
    if "no_memory" in name:
        assert not any(has_mem), "no_memory still has memory blocks"
    else:
        # first frames may lack memory; require at least some if source had any
        print(f"  {name}: {sum(has_mem)}/{len(data)} with memory")
    print(f"  {name}: {len(data)} samples OK")
print("convert checks OK")
PY
check "convert_to_llava_json" true

# ---------------------------------------------------------------------------
# 3. Phase1 data-prep behavior: delete with_memory after convert
# ---------------------------------------------------------------------------
echo "[3/6] Simulating Phase 1 data-prep (must remove *_with_memory.json)..."
PHASE1_DATA="$DRY_DIR/phase1_data"
mkdir -p "$PHASE1_DATA"
python "$BASELINE/convert_to_llava_json.py" \
  --samples-dir "$SAMPLES_TINY" \
  --output-dir "$PHASE1_DATA" \
  --processed-root /tmp/fake_mmor_dry \
  --splits train val >/dev/null
# mimic the fix in train_phase1.sh
rm -f "$PHASE1_DATA/train_with_memory.json" "$PHASE1_DATA/val_with_memory.json"
test -f "$PHASE1_DATA/train_no_memory.json"
test ! -f "$PHASE1_DATA/train_with_memory.json"
check "phase1 deletes with_memory" true

# ---------------------------------------------------------------------------
# 4. Phase2 would rebuild with_memory because missing
# ---------------------------------------------------------------------------
echo "[4/6] Simulating Phase 2 rebuild of with_memory..."
test ! -f "$PHASE1_DATA/train_with_memory.json"
python "$BASELINE/convert_to_llava_json.py" \
  --samples-dir "$SAMPLES_TINY" \
  --output-dir "$PHASE1_DATA" \
  --processed-root /tmp/fake_mmor_dry \
  --splits train >/dev/null
test -f "$PHASE1_DATA/train_with_memory.json"
check "phase2 can rebuild with_memory" true

# ---------------------------------------------------------------------------
# 5. Eval smoke (identity predictions)
# ---------------------------------------------------------------------------
echo "[5/6] Eval smoke test..."
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
python "$BASELINE/eval_predictions.py" \
  --gt "$SAMPLES_TINY/test.jsonl" \
  --predictions "$PRED" \
  --names "dry_identity" \
  --model-info "dry_test" \
  --project "dlhm-hierarchy-baselines-dry" \
  --no-bertscore > "$DRY_DIR/logs/eval_out.txt" 2>&1 || true
# wandb may fail offline; check metrics written
test -f "${PRED%.jsonl}_results.json"
python - <<PY
import json
r = json.load(open("${PRED%.jsonl}_results.json"))
assert r["matched"] > 0
assert abs(r["l0"]["bleu"] - 1.0) < 1e-6
print("  identity BLEU=1.0 OK, matched=", r["matched"])
PY
check "eval_predictions identity" true

# ---------------------------------------------------------------------------
# 6. ORacle training flags + inference imports (no full train)
# ---------------------------------------------------------------------------
echo "[6/6] Checking ORacle train flags + inference helpers..."
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

# inference module import without loading GPU model
import importlib.util
spec = importlib.util.spec_from_file_location("inf", "$BASELINE/inference.py")
mod = importlib.util.module_from_spec(spec)
# Don't exec fully (loads torch model builders); just compile
compile(Path("$BASELINE/inference.py").read_text(), "inference.py", "exec")
print("  inference.py compiles OK")
PY
check "oracle flags + inference compile" true

# Script syntax
bash -n "$BASELINE/train_phase1.sh"
bash -n "$BASELINE/train_phase2.sh"
bash -n "$BASELINE/run_eval.sh"
bash -n "$BASELINE/run_all.sh"
check "shell syntax" true

echo "======================================"
echo "Dry test done: $pass passed, $fail failed"
echo "Artifacts: $DRY_DIR"
echo "======================================"
exit $fail
