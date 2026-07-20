"""
Evaluation harness for hierarchy prediction.

Compares predicted vs. ground-truth hierarchy at two granularities:

1. **Per-frame metrics**: At each timepoint, compare predicted text against
   GT text for L0, L1, L2 using BERTScore, BLEU, ROUGE.

2. **Segment-level metrics** (optional): After assembly, compare predicted
   segment boundaries/descriptions against GT segments.

Usage::

    python -m data_pipeline.evaluate \
        --predictions predictions.jsonl \
        --ground-truth data_pipeline/samples/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_bert_score = None
_rouge_scorer = None
_bleu = None


def _load_bertscore():
    global _bert_score
    if _bert_score is None:
        from bert_score import score as bert_score_fn
        _bert_score = bert_score_fn
    return _bert_score


def _load_rouge():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rs
        _rouge_scorer = rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    return _rouge_scorer


def _load_bleu():
    global _bleu
    if _bleu is None:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        _bleu = (sentence_bleu, SmoothingFunction().method1)
    return _bleu


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute smoothed sentence-level BLEU."""
    sentence_bleu, smooth = _load_bleu()
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE-1 and ROUGE-L F1."""
    scorer = _load_rouge()
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_bertscore_batch(
    references: List[str],
    hypotheses: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 64,
) -> List[float]:
    """Compute BERTScore F1 for a batch of (ref, hyp) pairs."""
    bert_score_fn = _load_bertscore()
    if not references:
        return []
    P, R, F1 = bert_score_fn(
        hypotheses, references,
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
    )
    return F1.tolist()


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_per_frame(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    compute_bert: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate per-frame predictions against ground truth.

    Both inputs are lists of dicts with keys:
      - id (or: take, role, tp_id)
      - pred_l0 / gt_l0
      - pred_l1 / gt_l1
      - pred_l2 / gt_l2

    Returns a dict with per-level and overall metrics.
    """
    # Build GT index by id
    gt_by_id: Dict[str, Dict[str, Any]] = {}
    for g in ground_truth:
        gt_by_id[g["id"]] = g

    # Collect pairs
    levels = ["l0", "l1", "l2"]
    refs = {level: [] for level in levels}
    hyps = {level: [] for level in levels}
    matched = 0

    for pred in predictions:
        pid = pred["id"]
        gt = gt_by_id.get(pid)
        if gt is None:
            continue
        matched += 1
        for level in levels:
            ref_text = gt.get(f"gt_{level}", "")
            hyp_text = pred.get(f"pred_{level}", "")
            refs[level].append(ref_text)
            hyps[level].append(hyp_text)

    if matched == 0:
        logger.warning("No matching predictions found — check IDs")
        return {"matched": 0}

    logger.info("Evaluating %d matched frames", matched)

    results: Dict[str, Any] = {"matched": matched, "total_predictions": len(predictions)}

    for level in levels:
        level_results: Dict[str, float] = {}

        # BLEU
        bleu_scores = [
            compute_bleu(r, h) for r, h in zip(refs[level], hyps[level])
        ]
        level_results["bleu"] = sum(bleu_scores) / len(bleu_scores)

        # ROUGE
        rouge_scores = [
            compute_rouge(r, h) for r, h in zip(refs[level], hyps[level])
        ]
        level_results["rouge1"] = sum(s["rouge1"] for s in rouge_scores) / len(rouge_scores)
        level_results["rougeL"] = sum(s["rougeL"] for s in rouge_scores) / len(rouge_scores)

        # BERTScore
        if compute_bert:
            bert_f1 = compute_bertscore_batch(refs[level], hyps[level])
            level_results["bertscore_f1"] = sum(bert_f1) / len(bert_f1)

        results[level] = level_results

    return results


# ---------------------------------------------------------------------------
# Exact-match accuracy (useful for L2 within a segment)
# ---------------------------------------------------------------------------

def exact_match_accuracy(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    level: str = "l2",
) -> float:
    """Fraction of frames where predicted text exactly matches GT."""
    gt_by_id = {g["id"]: g for g in ground_truth}
    correct = 0
    total = 0
    for pred in predictions:
        gt = gt_by_id.get(pred["id"])
        if gt is None:
            continue
        total += 1
        if pred.get(f"pred_{level}", "").strip() == gt.get(f"gt_{level}", "").strip():
            correct += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hierarchy predictions")
    parser.add_argument(
        "--predictions", type=Path, required=True,
        help="JSONL with per-frame predictions (keys: id, pred_l0, pred_l1, pred_l2)",
    )
    parser.add_argument(
        "--ground-truth", type=Path, required=True,
        help="JSONL with ground truth (keys: id, gt_l0, gt_l1, gt_l2)",
    )
    parser.add_argument(
        "--no-bertscore", action="store_true",
        help="Skip BERTScore (faster, no GPU needed)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write results JSON to this path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    predictions = load_jsonl(args.predictions)
    ground_truth = load_jsonl(args.ground_truth)

    results = evaluate_per_frame(
        predictions, ground_truth,
        compute_bert=not args.no_bertscore,
    )

    # Add exact-match for L2
    results["l2_exact_match"] = exact_match_accuracy(predictions, ground_truth, "l2")

    # Print results
    print(json.dumps(results, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
