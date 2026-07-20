"""
Build per-frame, per-role training samples from hierarchy JSONs + MM-OR images.

Each sample is scoped to one L2 segment and contains:
  - Image paths for all camera views at that timepoint
  - Role name
  - Temporal memory (from preceding frames within the L2 segment)
  - Ground-truth (L0, L1, L2) labels

Outputs a JSON-lines file per split (train / val / test), with each line
formatted as a LLaVA-style conversation sample.

Usage::

    python -m data_pipeline.build_samples --processed-root /path/to/MM-OR_processed
    python -m data_pipeline.build_samples --split train --take 001_PKA
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    ALL_TAKES,
    HIERARCHY_DIR,
    SAMPLES_DIR,
    SPLIT_MAP,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    get_processed_root,
)
from .hierarchy_utils import (
    HierarchyIndex,
    load_frame_map,
    resolve_image_paths,
)
from .temporal_memory import (
    TemporalMemoryBuilder,
    augment_memory,
    format_memory_string,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "<image>\n"
    "Role: {role_human}\n"
    "{memory}"
    "Describe the current activity hierarchy for this role."
)

PROMPT_TEMPLATE_NO_MEMORY = (
    "<image>\n"
    "Role: {role_human}\n"
    "Describe the current activity hierarchy for this role."
)

ANSWER_TEMPLATE = "L0: {l0} | L1: {l1} | L2: {l2}"


# ---------------------------------------------------------------------------
# Sample building
# ---------------------------------------------------------------------------

def build_samples_for_l2_segment(
    take: str,
    role: str,
    l2_seg_id: str,
    hier_idx: HierarchyIndex,
    frame_map: Optional[Dict[str, Any]],
    take_dir: Path,
    augment: bool = True,
    dry_run: bool = False,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """
    Build all training samples for one (role, L2 segment) pair.

    Walks frames chronologically, building temporal memory from GT states.

    If dry_run=True, image paths are set to placeholder strings (no
    frame_map required).
    """
    rng = rng or random.Random()
    tps = hier_idx.l2_timepoints(role, l2_seg_id)
    if not tps:
        return []

    memory_builder = TemporalMemoryBuilder()
    memory_builder.reset()

    samples: List[Dict[str, Any]] = []
    for tp_id in tps:
        labels = hier_idx.frame_labels(role, tp_id)
        if labels is None:
            continue

        # Get memory state BEFORE this frame, then advance
        mem_state = memory_builder.step(
            tp_id, labels.l0_description, labels.l1_summary
        )

        # Resolve camera image paths (relative to take_dir)
        if dry_run:
            image_paths = [f"colorimage/camera{c:02d}_colorimage-{tp_id}.jpg" for c in [1, 2, 3, 4]]
        else:
            image_paths = resolve_image_paths(take_dir, frame_map, tp_id)  # type: ignore[arg-type]
            if not image_paths:
                continue

        # Format memory string (with augmentation during training)
        if augment:
            memory_str = augment_memory(mem_state, rng)
        else:
            memory_str = format_memory_string(mem_state)

        # Build prompt
        if memory_str:
            prompt = PROMPT_TEMPLATE.format(
                role_human=labels.role_human,
                memory=memory_str + "\n",
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_MEMORY.format(
                role_human=labels.role_human,
            )

        answer = ANSWER_TEMPLATE.format(
            l0=labels.l0_description,
            l1=labels.l1_summary,
            l2=labels.l2_summary,
        )

        sample = {
            "id": f"{take}/{role}/{l2_seg_id}/{tp_id}",
            "image": image_paths,
            "take": take,
            "role": role,
            "role_human": labels.role_human,
            "l2_segment_id": l2_seg_id,
            "tp_id": tp_id,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ],
            "gt_l0": labels.l0_description,
            "gt_l1": labels.l1_summary,
            "gt_l2": labels.l2_summary,
            "gt_l0_seg_id": labels.l0_segment_id,
            "gt_l1_seg_id": labels.l1_segment_id,
            "gt_l2_seg_id": labels.l2_segment_id,
        }
        samples.append(sample)

    return samples


def build_samples_for_take(
    take: str,
    hier_idx: HierarchyIndex,
    processed_root: Path,
    augment: bool = True,
    dry_run: bool = False,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Build all samples for one take (all roles, all L2 segments)."""
    rng = rng or random.Random()
    take_dir = processed_root / take

    frame_map: Optional[Dict[str, Any]] = None
    if not dry_run:
        try:
            frame_map = load_frame_map(take_dir)
        except FileNotFoundError:
            logger.warning(
                "No timestamp_to_pcd_and_frames_list.json for %s — skipping. "
                "Is MM-OR_processed mounted?", take
            )
            return []

    all_samples: List[Dict[str, Any]] = []
    for role in hier_idx.role_names:
        for l2_seg in hier_idx.iter_l2_segments(role):
            samples = build_samples_for_l2_segment(
                take=take,
                role=role,
                l2_seg_id=l2_seg.segment_id,
                hier_idx=hier_idx,
                frame_map=frame_map,
                take_dir=take_dir,
                augment=augment,
                dry_run=dry_run,
                rng=rng,
            )
            all_samples.extend(samples)

    logger.info(
        "Take %s: %d roles, %d samples",
        take, len(hier_idx.role_names), len(all_samples),
    )
    return all_samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-frame training samples from hierarchy JSONs."
    )
    parser.add_argument(
        "--processed-root", type=Path, default=None,
        help="MM-OR_processed root (default: env or auto-detect)",
    )
    parser.add_argument(
        "--hierarchy-dir", type=Path, default=HIERARCHY_DIR,
        help="Directory with hierarchy JSONs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SAMPLES_DIR,
        help="Output directory for sample JSONLs",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all",
        help="Which split to build (default: all)",
    )
    parser.add_argument(
        "--take", type=str, default=None,
        help="Build only this take (overrides --split)",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable temporal augmentation (deterministic memory)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for augmentation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip image path resolution (no MM-OR_processed needed). "
             "Uses placeholder image paths. Useful for testing.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    processed_root = get_processed_root(args.processed_root)
    logger.info("MM-OR processed root: %s", processed_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Determine which takes to process
    if args.take:
        takes = [args.take]
    elif args.split == "all":
        takes = ALL_TAKES
    elif args.split == "train":
        takes = SPLIT_TRAIN
    elif args.split == "val":
        takes = SPLIT_VAL
    else:
        takes = SPLIT_TEST

    # Group by split for output
    split_samples: Dict[str, List[Dict[str, Any]]] = {
        "train": [], "val": [], "test": [],
    }

    augment = not args.no_augment

    for take in takes:
        hier_path = args.hierarchy_dir / f"{take}_hierarchy_qwen27b.json"
        if not hier_path.exists():
            logger.warning("No hierarchy JSON for %s — skipping", take)
            continue

        hier_idx = HierarchyIndex.from_file(hier_path)
        samples = build_samples_for_take(
            take, hier_idx, processed_root,
            augment=augment, dry_run=args.dry_run, rng=rng,
        )

        split_name = SPLIT_MAP.get(take, "train")
        if args.take:
            split_name = SPLIT_MAP.get(take, "custom")
        split_samples.setdefault(split_name, []).extend(samples)

    # Write output files
    total = 0
    for split_name, samples in split_samples.items():
        if not samples:
            continue
        out_path = args.output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info("Wrote %d samples to %s", len(samples), out_path)
        total += len(samples)

    logger.info("Total samples: %d", total)


if __name__ == "__main__":
    main()
