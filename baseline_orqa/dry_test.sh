#!/usr/bin/env bash
# Small dry test for Baseline 2: data prep + script logic (no GPU training).
# Usage: bash baseline_orqa/dry_test.sh
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$WORKDIR/baseline_orqa"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="dlhm-b2"
DRY_DIR="/tmp/${USER}_b2_dry_$$"
SAMPLES_TINY="$DRY_DIR/samples"
DATA_DIR="$DRY_DIR/data"
N_SAMPLES="${N_SAMPLES:-32}"

cd "$WORKDIR"
mkdir -p "$SAMPLES_TINY" "$DATA_DIR" "$DRY_DIR/logs"

pass=0
fail=0
ok() { echo "  PASS: $1"; pass=$((pass+1)); }
bad() { echo "  FAIL: $1"; fail=$((fail+1)); }

echo "======================================"
echo "Baseline 2 dry test"
echo "DRY_DIR=$DRY_DIR"
echo "======================================"

# ---------------------------------------------------------------------------
# 0. Optional conda env imports (skip if env not set up yet)
# ---------------------------------------------------------------------------
echo "[0/6] Checking conda env '$ENV_NAME' (optional)..."
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda activate "$ENV_NAME"
  export PYTHONPATH="$BASELINE/ORQA/Qwen2-VL/LLaMA-Factory/src:${BASELINE}/ORQA:${PYTHONPATH:-}"
  if python -c "
import torch, transformers, peft
print(f'  torch={torch.__version__} transformers={transformers.__version__}')
from llamafactory.model.qwen2_vl.modeling_qwen2_vl import ImageEmbeddingPooler
print('  ImageEmbeddingPooler OK')
"; then
    ok "conda env imports"
  else
    bad "conda env imports"
  fi
else
  echo "  SKIP: env '$ENV_NAME' missing — run setup.sh later"
  ok "conda env skipped"
fi

# ---------------------------------------------------------------------------
# 1. Tiny JSONL subset
# ---------------------------------------------------------------------------
echo "[1/6] Building tiny JSONL subset ($N_SAMPLES samples)..."
python3 - <<PY
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
assert rows, "no samples read — build data_pipeline samples first"

def azure_views(sample):
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
# 2. Convert to Qwen2-VL QA JSON
# ---------------------------------------------------------------------------
echo "[2/6] Converting to Qwen2-VL QA JSON..."
python3 "$BASELINE/convert_to_qwen_json.py" \
  --samples-dir "$SAMPLES_TINY" \
  --output-dir "$DATA_DIR" \
  --processed-root /tmp/fake_mmor_dry \
  --allow-missing-images \
  --relative-images \
  --augment-views \
  --splits train val

python3 - <<PY
import json, re
from pathlib import Path
data_dir = Path("$DATA_DIR")
mem = re.compile(r"<memory_start>.*?<memory_end>", re.DOTALL)
for name in ["train_no_memory.json", "train_with_memory.json"]:
    p = data_dir / name
    assert p.exists(), f"missing {p}"
    data = json.load(open(p))
    assert len(data) > 0
    s0 = data[0]
    assert "messages" in s0 and "images" in s0
    assert s0.get("pc", None) == ""
    assert s0.get("audio", None) == ""
    assert s0["messages"][0]["role"] == "user"
    n_img_tok = s0["messages"][0]["content"].count("<image>")
    assert n_img_tok == len(s0["images"]), (n_img_tok, len(s0["images"]))
    assert 1 <= len(s0["images"]) <= 4
    assert "L0:" in s0["messages"][1]["content"]
    has_mem = bool(mem.search(s0["messages"][0]["content"]))
    if "no_memory" in name:
        assert not has_mem, "no_memory still has memory blocks"
    print(f"  {name}: {len(data)} samples, views={len(s0['images'])}, mem={has_mem} OK")
PY
ok "convert_to_qwen_json"

# ---------------------------------------------------------------------------
# 2b. Missing-image filter
# ---------------------------------------------------------------------------
echo "[2b/6] Checking missing-image filter..."
python3 - <<PY
import subprocess, sys
from pathlib import Path
empty = Path("$DRY_DIR/empty_mmor")
empty.mkdir(exist_ok=True)
out = Path("$DRY_DIR/filter_out")
out.mkdir(exist_ok=True)
rc = subprocess.call([
    sys.executable, "$BASELINE/convert_to_qwen_json.py",
    "--samples-dir", "$SAMPLES_TINY",
    "--output-dir", str(out),
    "--processed-root", str(empty),
    "--splits", "train",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
assert rc != 0, "expected convert to fail when all images missing"
print("  filter rejects empty mount OK")
PY
ok "missing-image filter"

# ---------------------------------------------------------------------------
# 3. Curriculum YAML (phase1 + phase2)
# ---------------------------------------------------------------------------
echo "[3/6] Checking Phase 1/2 YAML (ORQA curriculum + val)..."
python3 - <<PY
from pathlib import Path
p1 = Path("$BASELINE/configs/hierarchy_lora_sft_phase1.yaml").read_text()
p2 = Path("$BASELINE/configs/hierarchy_lora_sft_phase2.yaml").read_text()
assert "unfreeze_last_n_vision_tower_layers: 8" in p1
assert "previous_model_weights" not in p1
assert "b2_phase1_hierarchy_no_memory" in p1
assert "do_eval: true" in p1 and "do_eval: true" in p2
assert "eval_data_json_file: __EVAL_DATA_JSON__" in p1
assert "image_dir: __IMAGE_DIR__" in p1 and "image_dir: __IMAGE_DIR__" in p2
assert "eval_strategy: steps" in p1
assert "previous_model_weights: __PREVIOUS_WEIGHTS__" in p2
assert "unfreeze_last_n_vision_tower_layers: null" in p2
assert "b2_phase2_hierarchy_with_memory" in p2
out = (
    p2.replace("__DATA_JSON__", "/tmp/train.json")
      .replace("__EVAL_DATA_JSON__", "/tmp/val.json")
      .replace("__CACHE_DIR__", "/tmp/cache/train.arrow")
      .replace("__EVAL_CACHE_DIR__", "/tmp/cache/val.arrow")
      .replace("__IMAGE_DIR__", "/tmp/mmor")
      .replace("__OUTPUT_DIR__", "/tmp/out")
      .replace("__PREVIOUS_WEIGHTS__", "/tmp/phase1/checkpoint-1")
)
assert "__PREVIOUS_WEIGHTS__" not in out
assert "__EVAL_DATA_JSON__" not in out
assert "__IMAGE_DIR__" not in out
assert "previous_model_weights: /tmp/phase1/checkpoint-1" in out
assert "cache_file_name: /tmp/cache/train.arrow" in out
assert "eval_cache_file_name: /tmp/cache/val.arrow" in out
assert "image_dir: /tmp/mmor" in out
print("  Phase1/Phase2 YAML OK")
PY
ok "curriculum YAML"

# ---------------------------------------------------------------------------
# 4. Eval smoke (identity predictions)
# ---------------------------------------------------------------------------
echo "[4/6] Eval smoke test..."
PRED="$DRY_DIR/pred_identity.jsonl"
python3 - <<PY
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
WANDB_MODE=disabled python3 "$BASELINE/eval_predictions.py" \
  --gt "$SAMPLES_TINY/test.jsonl" \
  --predictions "$PRED" \
  --names "dry_identity" \
  --model-info "dry_test" \
  --project "dlhm-hierarchy-baselines-dry" \
  --no-bertscore > "$DRY_DIR/logs/eval_out.txt" 2>&1 || true
test -f "${PRED%.jsonl}_results.json"
python3 - <<PY
import json
r = json.load(open("${PRED%.jsonl}_results.json"))
assert r["matched"] > 0
assert abs(r["l0"]["bleu"] - 1.0) < 1e-6
print("  identity BLEU=1.0 OK, matched=", r["matched"])
PY
ok "eval_predictions identity"

# ---------------------------------------------------------------------------
# 5. Parse + compile inference
# ---------------------------------------------------------------------------
echo "[5/6] Checking parse_model_output + inference compile..."
python3 - <<PY
from pathlib import Path
from data_pipeline.assemble import parse_model_output
assert parse_model_output("L0: a | L1: b | L2: c") == ("a", "b", "c")
compile(Path("$BASELINE/inference.py").read_text(), "inference.py", "exec")
compile(Path("$BASELINE/convert_to_qwen_json.py").read_text(), "convert.py", "exec")
compile(Path("$BASELINE/patches/image_only_pooler.py").read_text(), "patch.py", "exec")
compile(Path("$BASELINE/patches/eval_dataset_path.py").read_text(), "eval_patch.py", "exec")
compile(Path("$BASELINE/patches/optional_pc_audio.py").read_text(), "pc_patch.py", "exec")
compile(Path("$BASELINE/patches/collator_skip_hierarchy_ids.py").read_text(), "collator_patch.py", "exec")
print("  compile OK")
PY
ok "parse + compile"

# ---------------------------------------------------------------------------
# 6. Shell syntax
# ---------------------------------------------------------------------------
echo "[6/6] Checking shell scripts..."
bash -n "$BASELINE/train_phase1.sh"
bash -n "$BASELINE/train_phase2.sh"
bash -n "$BASELINE/run_eval.sh"
bash -n "$BASELINE/run_all.sh"
bash -n "$BASELINE/setup.sh"
for f in train_phase1.sh train_phase2.sh run_eval.sh run_all.sh; do
  grep -q 'conda activate' "$BASELINE/$f"
  grep -q 'dlhm-b2' "$BASELINE/$f"
  grep -q 'lib_cuda_env.sh' "$BASELINE/$f"
done
bash -n "$BASELINE/lib_cuda_env.sh"
grep -q 'previous_model_weights\|PHASE1\|phase1' "$BASELINE/train_phase2.sh"
grep -q 'train_phase1.sh' "$BASELINE/run_all.sh"
grep -q 'train_phase2.sh' "$BASELINE/run_all.sh"
ok "shell syntax + conda wiring"

echo "======================================"
echo "Dry test done: $pass passed, $fail failed"
echo "Artifacts: $DRY_DIR"
echo "======================================"
exit $fail
