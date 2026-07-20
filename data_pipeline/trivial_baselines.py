"""
Trivial baselines for hierarchy prediction (metric floor).

Three baselines:
  1. **Majority class**: Predict the most common L0/L1/L2 text from training.
  2. **Copy-last**: Repeat the previous frame's prediction within the L2 segment.
  3. **Random**: Sample a random L0/L1 from training; L2 from segment's training set.

Usage::

    python -m data_pipeline.trivial_baselines \
        --train-samples data_pipeline/samples/train.jsonl \
        --test-samples data_pipeline/samples/test.jsonl \
        --output-dir data_pipeline/trivial_results
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# 1. Majority class
# ---------------------------------------------------------------------------

def majority_class_baseline(
    train_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Predict the most common L0/L1/L2 from training for every test frame."""
    l0_counter: Counter = Counter()
    l1_counter: Counter = Counter()
    l2_counter: Counter = Counter()

    for s in train_samples:
        l0_counter[s["gt_l0"]] += 1
        l1_counter[s["gt_l1"]] += 1
        l2_counter[s["gt_l2"]] += 1

    majority_l0 = l0_counter.most_common(1)[0][0] if l0_counter else ""
    majority_l1 = l1_counter.most_common(1)[0][0] if l1_counter else ""
    majority_l2 = l2_counter.most_common(1)[0][0] if l2_counter else ""

    logger.info("Majority L0: %s (%d)", majority_l0, l0_counter[majority_l0])
    logger.info("Majority L1: %s (%d)", majority_l1, l1_counter[majority_l1])
    logger.info("Majority L2: %s (%d)", majority_l2, l2_counter[majority_l2])

    predictions = []
    for s in test_samples:
        predictions.append({
            "id": s["id"],
            "role": s["role"],
            "tp_id": s["tp_id"],
            "l2_segment_id": s["gt_l2_seg_id"],
            "pred_l0": majority_l0,
            "pred_l1": majority_l1,
            "pred_l2": majority_l2,
            "gt_l0": s["gt_l0"],
            "gt_l1": s["gt_l1"],
            "gt_l2": s["gt_l2"],
        })
    return predictions


# ---------------------------------------------------------------------------
# 2. Copy-last
# ---------------------------------------------------------------------------

def copy_last_baseline(
    test_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Within each (role, L2 segment), repeat the previous frame's GT label.

    The first frame in each segment gets a blank prediction (no history).
    This measures how much the labels change between consecutive frames.
    """
    # Group by (role, l2_segment_id), preserving order
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in test_samples:
        key = f"{s['role']}_{s['gt_l2_seg_id']}"
        groups.setdefault(key, []).append(s)

    predictions = []
    for key, frames in groups.items():
        frames.sort(key=lambda s: s["tp_id"])
        prev_l0 = ""
        prev_l1 = ""
        prev_l2 = ""

        for s in frames:
            predictions.append({
                "id": s["id"],
                "role": s["role"],
                "tp_id": s["tp_id"],
                "l2_segment_id": s["gt_l2_seg_id"],
                "pred_l0": prev_l0,
                "pred_l1": prev_l1,
                "pred_l2": prev_l2,
                "gt_l0": s["gt_l0"],
                "gt_l1": s["gt_l1"],
                "gt_l2": s["gt_l2"],
            })
            prev_l0 = s["gt_l0"]
            prev_l1 = s["gt_l1"]
            prev_l2 = s["gt_l2"]

    return predictions


# ---------------------------------------------------------------------------
# 3. Random
# ---------------------------------------------------------------------------

def random_baseline(
    train_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample random L0/L1/L2 descriptions from the training set."""
    rng = random.Random(seed)

    all_l0 = [s["gt_l0"] for s in train_samples]
    all_l1 = [s["gt_l1"] for s in train_samples]
    all_l2 = [s["gt_l2"] for s in train_samples]

    predictions = []
    for s in test_samples:
        predictions.append({
            "id": s["id"],
            "role": s["role"],
            "tp_id": s["tp_id"],
            "l2_segment_id": s["gt_l2_seg_id"],
            "pred_l0": rng.choice(all_l0),
            "pred_l1": rng.choice(all_l1),
            "pred_l2": rng.choice(all_l2),
            "gt_l0": s["gt_l0"],
            "gt_l1": s["gt_l1"],
            "gt_l2": s["gt_l2"],
        })
    return predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute trivial baselines for hierarchy prediction"
    )
    parser.add_argument(
        "--train-samples", type=Path, required=True,
        help="Training samples JSONL (for majority/random)",
    )
    parser.add_argument(
        "--test-samples", type=Path, required=True,
        help="Test samples JSONL",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write baseline prediction JSONLs",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    train = load_jsonl(args.train_samples)
    test = load_jsonl(args.test_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Train samples: %d, Test samples: %d", len(train), len(test))

    # 1. Majority class
    majority_preds = majority_class_baseline(train, test)
    out = args.output_dir / "majority_class.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in majority_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Wrote %d majority-class predictions to %s", len(majority_preds), out)

    # 2. Copy-last
    copy_preds = copy_last_baseline(test)
    out = args.output_dir / "copy_last.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in copy_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Wrote %d copy-last predictions to %s", len(copy_preds), out)

    # 3. Random
    rand_preds = random_baseline(train, test, seed=args.seed)
    out = args.output_dir / "random.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in rand_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Wrote %d random predictions to %s", len(rand_preds), out)

    # Quick summary: exact match rates for each baseline
    from .evaluate import exact_match_accuracy

    for name, preds in [("majority", majority_preds), ("copy_last", copy_preds), ("random", rand_preds)]:
        for level in ["l0", "l1", "l2"]:
            acc = exact_match_accuracy(preds, test, level)
            logger.info("%s %s exact-match: %.4f", name, level, acc)


if __name__ == "__main__":
    main()
