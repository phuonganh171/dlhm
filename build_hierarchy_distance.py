"""
Build a role-aware hierarchical representation of surgical scene graphs
using sentence embeddings and constrained agglomerative clustering.

Level 0: same as build_hierarchy_qwen32b.py -- deterministic segmentation by
         identical predicate sets with debounce.
Level 1: constrained agglomerative clustering with cosine distance between
         segment description embeddings, at threshold T1 (default 0.6).
Level 2: same clustering on level-1 groups at threshold T2 (default 0.85).

Distance metric: each segment's description string (e.g. "head surgeon: is
drilling patient; is holding drill") is embedded with a sentence transformer,
and cosine distance between embeddings measures how different two segments are.
No hardcoded categories, no hand-designed features -- the model understands
word semantics out of the box.

Usage:
    source /tmp/nhatvu/.venv/bin/activate &&
    python3 build_hierarchy_distance.py \
        --take_dir mm-or/MM-OR_data/MM-OR_processed/001_PKA \
        --start_tp 001114 \
        --max_frames 600 \
        --level2 \
        --t1 0.6 --t2 0.85 \
        --output hierarchy_output/001_PKA_hierarchy_distance.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from build_hierarchy_qwen32b import (
    build_level0,
    analyze_debounce,
    _log_debounce_report,
)
from scene_graph_utils import (
    humanize,
    humanize_pred,
    load_frame_map,
    load_relation_labels,
    original_timestamp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence embedding model
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """Thin wrapper around a sentence transformer for segment descriptions."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model ready (dim=%d).", self.model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_distances(embeddings: np.ndarray) -> List[float]:
    """Cosine distance between each consecutive pair of L2-normalised embeddings."""
    dots = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return (1.0 - dots).tolist()


# ---------------------------------------------------------------------------
# Constrained agglomerative clustering (adjacent-only merges)
# ---------------------------------------------------------------------------

def constrained_agglomerative(
    distances: List[float],
    threshold: float,
) -> List[List[int]]:
    """
    Cluster N items (with N-1 adjacent distances) by repeatedly merging
    the closest adjacent pair until the minimum distance exceeds `threshold`.

    Returns a list of groups, each being a list of original item indices.
    """
    n = len(distances) + 1
    if n <= 1:
        return [list(range(n))]

    clusters: List[List[int]] = [[i] for i in range(n)]
    dists = list(distances)

    while len(clusters) > 1:
        min_dist = min(dists)
        if min_dist > threshold:
            break

        idx = dists.index(min_dist)
        clusters[idx] = clusters[idx] + clusters[idx + 1]
        del clusters[idx + 1]
        del dists[idx]

    return clusters


# ---------------------------------------------------------------------------
# Description helpers
# ---------------------------------------------------------------------------

def _auto_describe_group(
    role: str,
    child_segments: List[Dict[str, Any]],
) -> str:
    """Template-based summary for a group of segments."""
    role_name = humanize(role)
    all_preds: set = set()
    for seg in child_segments:
        for p in seg["active_predicates"]:
            all_preds.add(tuple(p))

    if not all_preds:
        return f"{role_name}: idle / no active interactions"

    parts = sorted(
        f"{humanize_pred(pred)} {humanize(obj)}" for pred, obj in all_preds
    )
    return f"{role_name}: {'; '.join(parts)}"


# ---------------------------------------------------------------------------
# Level-1: embedding-based grouping of L0 segments
# ---------------------------------------------------------------------------

def build_level1_distance(
    hierarchy: Dict[str, Any],
    emb_model: EmbeddingModel,
    threshold: float = 0.6,
) -> None:
    """Group each role's L0 segments into L1 action steps using embedding distance."""
    for role, role_data in hierarchy["roles"].items():
        l0_segs = role_data["level0_segments"]

        if len(l0_segs) <= 1:
            role_data["level1_segments"] = [{
                "segment_id": f"{role}_L1_000",
                "role": role,
                "level": 1,
                "role_human": humanize(role),
                "segment_ids": [s["segment_id"] for s in l0_segs],
                "time_start": l0_segs[0]["time_start"] if l0_segs else None,
                "time_end": l0_segs[-1]["time_end"] if l0_segs else None,
                "summary": l0_segs[0]["description"] if l0_segs else "No activity",
            }]
            role_data["num_level1_segments"] = 1
            logger.info("  %-25s  1 level-1 segment (trivial)", humanize(role))
            continue

        descriptions = [s["description"] for s in l0_segs]
        embeddings = emb_model.embed(descriptions)
        dists = cosine_distances(embeddings)

        groups = constrained_agglomerative(dists, threshold)

        level1_segs = []
        for i, group_idxs in enumerate(groups):
            children = [l0_segs[j] for j in group_idxs]
            t_start = children[0]["time_start"]
            t_end = children[-1]["time_end"]
            level1_segs.append({
                "segment_id": f"{role}_L1_{i:03d}",
                "role": role,
                "level": 1,
                "role_human": humanize(role),
                "segment_ids": [c["segment_id"] for c in children],
                "time_start": t_start,
                "time_end": t_end,
                "duration_s": (
                    (t_end - t_start + 1) if t_start is not None and t_end is not None else None
                ),
                "summary": _auto_describe_group(role, children),
            })

        role_data["level1_segments"] = level1_segs
        role_data["num_level1_segments"] = len(level1_segs)
        logger.info(
            "  %-25s  %3d L0 -> %2d L1 (threshold=%.2f)",
            humanize(role), len(l0_segs), len(level1_segs), threshold,
        )


# ---------------------------------------------------------------------------
# Level-2: embedding-based grouping of L1 segments into phases
# ---------------------------------------------------------------------------

def build_level2_distance(
    hierarchy: Dict[str, Any],
    emb_model: EmbeddingModel,
    threshold: float = 0.85,
) -> None:
    """Group each role's L1 segments into L2 phases using embedding distance."""
    for role, role_data in hierarchy["roles"].items():
        l1_segs = role_data.get("level1_segments", [])
        l0_lookup = {s["segment_id"]: s for s in role_data["level0_segments"]}

        if len(l1_segs) <= 1:
            role_data["level2_segments"] = [{
                "segment_id": f"{role}_L2_000",
                "role": role,
                "level": 2,
                "role_human": humanize(role),
                "child_ids": [s["segment_id"] for s in l1_segs],
                "time_start": l1_segs[0]["time_start"] if l1_segs else None,
                "time_end": l1_segs[-1]["time_end"] if l1_segs else None,
                "summary": l1_segs[0].get("summary", "No activity") if l1_segs else "No activity",
            }]
            role_data["num_level2_segments"] = 1
            logger.info("  %-25s  1 level-2 segment (trivial)", humanize(role))
            continue

        descriptions = [s["summary"] for s in l1_segs]
        embeddings = emb_model.embed(descriptions)
        dists = cosine_distances(embeddings)

        groups = constrained_agglomerative(dists, threshold)

        level2_segs = []
        for i, group_idxs in enumerate(groups):
            children = [l1_segs[j] for j in group_idxs]
            all_l0 = []
            for ch in children:
                for cid in ch.get("segment_ids", []):
                    if cid in l0_lookup:
                        all_l0.append(l0_lookup[cid])

            t_start = children[0]["time_start"]
            t_end = children[-1]["time_end"]
            level2_segs.append({
                "segment_id": f"{role}_L2_{i:03d}",
                "role": role,
                "level": 2,
                "role_human": humanize(role),
                "child_ids": [c["segment_id"] for c in children],
                "time_start": t_start,
                "time_end": t_end,
                "duration_s": (
                    (t_end - t_start + 1) if t_start is not None and t_end is not None else None
                ),
                "summary": _auto_describe_group(role, all_l0),
            })

        role_data["level2_segments"] = level2_segs
        role_data["num_level2_segments"] = len(level2_segs)
        logger.info(
            "  %-25s  %2d L1 -> %2d L2 (threshold=%.2f)",
            humanize(role), len(l1_segs), len(level2_segs), threshold,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build role-aware hierarchy using embedding distances."
    )
    parser.add_argument(
        "--take_dir", type=Path,
        default=Path("mm-or/MM-OR_data/MM-OR_processed/001_PKA"),
    )
    parser.add_argument("--start_tp", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--debounce", type=int, default=2)
    parser.add_argument(
        "--output", type=Path,
        default=Path("hierarchy_output/001_PKA_hierarchy_distance.json"),
    )
    parser.add_argument(
        "--level1", action="store_true",
        help="Run level-1 embedding-based grouping.",
    )
    parser.add_argument(
        "--level2", action="store_true",
        help="Run level-2 embedding-based grouping (implies --level1).",
    )
    parser.add_argument(
        "--t1", type=float, default=0.30,
        help="Cosine distance threshold for level-1 (default: 0.30).",
    )
    parser.add_argument(
        "--t2", type=float, default=0.45,
        help="Cosine distance threshold for level-2 (default: 0.45).",
    )
    parser.add_argument(
        "--embed_model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: all-MiniLM-L6-v2).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Loading frame map from %s ...", args.take_dir)
    frame_map = load_frame_map(args.take_dir)
    logger.info("%d timepoints in frame map.", len(frame_map))

    logger.info("Loading relation labels ...")
    entries = load_relation_labels(args.take_dir, frame_map)
    logger.info("%d relation-label timepoints loaded.", len(entries))

    if args.start_tp is not None:
        before = len(entries)
        entries = [(tp, t) for tp, t in entries if tp >= args.start_tp]
        logger.info("Starting at %s: kept %d/%d entries.", args.start_tp, len(entries), before)

    if args.max_frames is not None:
        entries = entries[:args.max_frames]
        logger.info("Using %d timepoints (--max_frames %d).", len(entries), args.max_frames)

    logger.info("Building level-0 segmentation (debounce=%d) ...", args.debounce)
    hierarchy = build_level0(entries, frame_map, debounce=args.debounce)

    tp_start = entries[0][0] if entries else None
    tp_end = entries[-1][0] if entries else None
    hierarchy["metadata"].update({
        "take_dir": str(args.take_dir),
        "tp_range": [tp_start, tp_end],
        "num_timepoints": len(entries),
        "time_range": [
            original_timestamp(frame_map, tp_start) if tp_start else None,
            original_timestamp(frame_map, tp_end) if tp_end else None,
        ],
        "method": "embedding_cosine_distance",
        "embed_model": args.embed_model,
    })

    if args.debounce > 1:
        logger.info("Running debounce analysis ...")
        debounce_report = analyze_debounce(entries, frame_map, debounce=args.debounce)
        hierarchy["debounce_analysis"] = debounce_report
        _log_debounce_report(debounce_report)

    if args.level2:
        args.level1 = True

    emb_model = None
    if args.level1:
        logger.info("")
        emb_model = EmbeddingModel(args.embed_model)

        logger.info("")
        logger.info("=== Building Level-1 (embedding distance, threshold=%.2f) ===", args.t1)
        build_level1_distance(hierarchy, emb_model, threshold=args.t1)
        total_l1 = sum(
            d.get("num_level1_segments", 0) for d in hierarchy["roles"].values()
        )
        hierarchy["metadata"]["total_level1_segments"] = total_l1
        hierarchy["metadata"]["level1_threshold"] = args.t1

    if args.level2:
        logger.info("")
        logger.info("=== Building Level-2 (embedding distance, threshold=%.2f) ===", args.t2)
        build_level2_distance(hierarchy, emb_model, threshold=args.t2)
        total_l2 = sum(
            d.get("num_level2_segments", 0) for d in hierarchy["roles"].values()
        )
        hierarchy["metadata"]["total_level2_segments"] = total_l2
        hierarchy["metadata"]["level2_threshold"] = args.t2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hierarchy, indent=2))
    logger.info("Hierarchy saved to %s", args.output)


if __name__ == "__main__":
    main()
