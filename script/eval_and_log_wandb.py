"""
Evaluate trivial baseline predictions and log metrics to W&B.

Reads baseline prediction JSONLs from --results-dir, evaluates each
against --gt (ground truth), and logs all metrics as a W&B summary table.

Usage (standalone):
    python script/eval_and_log_wandb.py \
        --results-dir data_pipeline/trivial_results \
        --gt data_pipeline/samples/test.jsonl \
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
    compute_bleu,
    compute_rouge,
    exact_match_accuracy,
    load_jsonl,
)

logger = logging.getLogger(__name__)


def evaluate_baseline(
    preds: list[dict],
    gt: list[dict],
    compute_bert: bool = False,
) -> dict:
    """Compute all metrics for one baseline."""
    gt_by_id = {g["id"]: g for g in gt}

    levels = ["l0", "l1", "l2"]
    refs = {lv: [] for lv in levels}
    hyps = {lv: [] for lv in levels}

    for p in preds:
        g = gt_by_id.get(p["id"])
        if g is None:
            continue
        for lv in levels:
            refs[lv].append(g.get(f"gt_{lv}", ""))
            hyps[lv].append(p.get(f"pred_{lv}", ""))

    n = len(refs["l0"])
    if n == 0:
        return {"matched": 0}

    results: dict = {"matched": n}
    for lv in levels:
        bleu_scores = [compute_bleu(r, h) for r, h in zip(refs[lv], hyps[lv])]
        rouge_scores = [compute_rouge(r, h) for r, h in zip(refs[lv], hyps[lv])]

        results[f"{lv}/bleu"] = sum(bleu_scores) / len(bleu_scores)
        results[f"{lv}/rouge1"] = sum(s["rouge1"] for s in rouge_scores) / len(rouge_scores)
        results[f"{lv}/rougeL"] = sum(s["rougeL"] for s in rouge_scores) / len(rouge_scores)
        results[f"{lv}/exact_match"] = exact_match_accuracy(preds, gt, lv)

    if compute_bert:
        from data_pipeline.evaluate import compute_bertscore_batch
        for lv in levels:
            f1s = compute_bertscore_batch(refs[lv], hyps[lv])
            results[f"{lv}/bertscore_f1"] = sum(f1s) / len(f1s)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--project", type=str, default="dlhm-hierarchy-baselines")
    parser.add_argument("--no-bertscore", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    gt = load_jsonl(args.gt)
    logger.info("Ground truth: %d samples", len(gt))

    baseline_files = sorted(args.results_dir.glob("*.jsonl"))
    if not baseline_files:
        logger.error("No .jsonl files in %s", args.results_dir)
        sys.exit(1)

    # Try to import wandb; fall back to stdout-only if not available
    try:
        import wandb
        use_wandb = True
    except ImportError:
        logger.warning("wandb not installed — printing metrics to stdout only")
        use_wandb = False

    all_results: dict[str, dict] = {}
    for pred_file in baseline_files:
        name = pred_file.stem
        logger.info("Evaluating: %s", name)
        preds = load_jsonl(pred_file)
        metrics = evaluate_baseline(preds, gt, compute_bert=not args.no_bertscore)
        all_results[name] = metrics

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k:30s} {v:.4f}")
            else:
                print(f"  {k:30s} {v}")

    # Log to wandb: one run per baseline
    if use_wandb:
        for name, metrics in all_results.items():
            run = wandb.init(
                project=args.project,
                name=name,
                config={"baseline": name, "test_samples": len(gt)},
                reinit=True,
            )
            wandb.log(metrics)

            # Also log a summary table
            table_data = []
            for lv in ["l0", "l1", "l2"]:
                table_data.append([
                    lv.upper(),
                    metrics.get(f"{lv}/bleu", 0),
                    metrics.get(f"{lv}/rouge1", 0),
                    metrics.get(f"{lv}/rougeL", 0),
                    metrics.get(f"{lv}/exact_match", 0),
                    metrics.get(f"{lv}/bertscore_f1", None),
                ])
            table = wandb.Table(
                columns=["Level", "BLEU", "ROUGE-1", "ROUGE-L", "Exact Match", "BERTScore F1"],
                data=table_data,
            )
            wandb.log({"metrics_table": table})
            run.finish()

        logger.info("All baselines logged to wandb project '%s'", args.project)
    else:
        # Write results to JSON file as fallback
        out = args.results_dir / "eval_results.json"
        out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
