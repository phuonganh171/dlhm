# Baseline 2 — ORQA (Qwen2-VL-2B)

Hierarchy prediction baseline adapted from
[ORQA](https://github.com/egeozsoy/ORQA) / [arXiv:2505.12890](https://arxiv.org/abs/2505.12890).

## Architecture

```
Multi-view RGB (≤4 Azure cameras)
  → Qwen2-VL ViT
  → ORacle ImageEmbeddingPooler → 578 tokens
  → Qwen2-VL-2B LLM (LoRA / QLoRA)
  → "L0: … | L1: … | L2: …"
```

## Training (matches ORQA paper curriculum)

Same recipe as official ORQA base → Temp:

| Phase | Data | Vision | Checkpoint |
|-------|------|--------|------------|
| **1 (base)** | no memory | unfreeze last 8 ViT layers | `checkpoints/phase1_no_memory` |
| **2 (Temp)** | with memory | ViT frozen; `previous_model_weights` = Phase 1 | `checkpoints/phase2_with_memory` |

Each phase: 1 epoch, `lr=1e-4`, batch 4 (QLoRA 4-bit).

## Setup (once, login node)

```bash
bash baseline_orqa/setup.sh
```

## Run

```bash
bash baseline_orqa/run_all.sh          # phase1 → phase2 → eval
# Or:
sbatch baseline_orqa/train_phase1.sh
sbatch baseline_orqa/train_phase2.sh
sbatch baseline_orqa/run_eval.sh

bash baseline_orqa/dry_test.sh         # offline sanity checks
```

## Layout

| Path | Role |
|------|------|
| `setup.sh` | Clone ORQA + conda env `dlhm-b2` |
| `convert_to_qwen_json.py` | JSONL → `{no,with}_memory` Qwen QA JSON |
| `configs/hierarchy_lora_sft_phase1.yaml` | Base (Falsetemp) + val every 500 steps |
| `configs/hierarchy_lora_sft_phase2.yaml` | Temp + curriculum + val |
| `train_phase1.sh` / `train_phase2.sh` | SLURM training (train + val JSON) |
| `patches/eval_dataset_path.py` | Separate `eval_data_json_file` for LLaMA-Factory |
| `patches/optional_pc_audio.py` | Allow missing `pc`/`audio` columns (image-only) |
| `patches/collator_skip_hierarchy_ids.py` | Skip ORQA take/tp parsing for hierarchy ids |
| `inference.py` | Autoregressive / GT-memory inference |
| `run_eval.sh` | SLURM eval + wandb |
| `patches/image_only_pooler.py` | Stub PointTransformer (no spconv) |

Training data from shared `data_pipeline/` (same hierarchy samples as Baseline 1).
