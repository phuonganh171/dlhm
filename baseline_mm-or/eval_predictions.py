#!/usr/bin/env python3
"""
Evaluate Baseline 1 predictions and log to wandb.

Usage::

    python baseline_mm-or/eval_predictions.py \
        --gt data_pipeline/samples/test.jsonl \
        --predictions predictions/pred_autoregressive.jsonl \
                      predictions/pred_gt_memory.jsonl \
        --names b1_autoregressive b1_gt_memory \
        --project dlhm-hierarchy-baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.evaluate import (
    evaluate_per_frame,
    exact_match_accuracy,
    load_jsonl,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and log baseline predictions")
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth JSONL")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True, help="Prediction JSONL files")
    parser.add_argument("--names", type=str, nargs="+", required=True, help="Names for each prediction set")
    parser.add_argument("--model-info", type=str, default="", help="Model path for logging")
    parser.add_argument("--project", type=str, default="dlhm-hierarchy-baselines")
    parser.add_argument("--no-bertscore", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    gt = load_jsonl(args.gt)
    logger.info("Ground truth: %d samples", len(gt))

    try:
        import wandb
        use_wandb = True
    except ImportError:
        logger.warning("wandb not installed — printing metrics to stdout only")
        use_wandb = False

    for pred_path, name in zip(args.predictions, args.names):
        logger.info("Evaluating: %s (%s)", name, pred_path)
        preds = load_jsonl(pred_path)

        results = evaluate_per_frame(preds, gt, compute_bert=not args.no_bertscore)
        results["l2_exact_match"] = exact_match_accuracy(preds, gt, "l2")

        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        print(json.dumps(results, indent=2))

        if use_wandb:
            run = wandb.init(
                project=args.project,
                name=name,
                config={"model": args.model_info, "memory_mode": name},
                reinit=True,
            )
            flat: dict = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}/{kk}"] = vv
                else:
                    flat[k] = v
            wandb.log(flat)

            table_data = []
            for lv in ["l0", "l1", "l2"]:
                lv_data = results.get(lv, {})
                if isinstance(lv_data, dict):
                    table_data.append([
                        lv.upper(),
                        lv_data.get("bleu", 0),
                        lv_data.get("rouge1", 0),
                        lv_data.get("rougeL", 0),
                        lv_data.get("bertscore_f1", None),
                    ])
            table = wandb.Table(
                columns=["Level", "BLEU", "ROUGE-1", "ROUGE-L", "BERTScore F1"],
                data=table_data,
            )
            wandb.log({"metrics_table": table})
            run.finish()

        out_path = str(pred_path).replace(".jsonl", "_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
