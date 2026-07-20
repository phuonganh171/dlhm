#!/usr/bin/env python3
"""
Convert data_pipeline JSONL samples to LLaVA-compatible JSON arrays.

Reads the per-frame samples produced by ``data_pipeline.build_samples`` and
writes two JSON files per split:

- ``{split}_no_memory.json``  — prompts with memory stripped (Phase 1)
- ``{split}_with_memory.json`` — prompts preserved as-is (Phase 2)

Image paths are made absolute by prepending the MM-OR processed root.

Usage::

    python baseline_mm-or/convert_to_llava_json.py \
        --samples-dir data_pipeline/samples \
        --output-dir  baseline_mm-or/data \
        --processed-root /path/to/MM-OR_processed
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_pipeline.config import get_processed_root

logger = logging.getLogger(__name__)

MEMORY_PATTERN = re.compile(
    r"<memory_start>\n.*?<memory_end>\n?",
    re.DOTALL,
)


def strip_memory(text: str) -> str:
    """Remove the <memory_start>...<memory_end> block from a prompt."""
    return MEMORY_PATTERN.sub("", text)


def make_absolute_images(
    image_paths: List[str],
    take: str,
    processed_root: Path,
) -> List[str]:
    """Prepend processed_root/take/ to relative image paths."""
    base = processed_root / take
    absolute = []
    for p in image_paths:
        if Path(p).is_absolute():
            absolute.append(p)
        else:
            absolute.append(str(base / p))
    return absolute


def convert_split(
    jsonl_path: Path,
    output_dir: Path,
    split_name: str,
    processed_root: Path,
) -> None:
    """Read a JSONL split and write two LLaVA JSON files."""
    samples_raw: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples_raw.append(json.loads(line))

    logger.info("Loaded %d samples from %s", len(samples_raw), jsonl_path)

    samples_with_memory: List[Dict[str, Any]] = []
    samples_no_memory: List[Dict[str, Any]] = []

    for s in samples_raw:
        take = s.get("take", s["id"].split("/")[0])
        abs_images = make_absolute_images(s["image"], take, processed_root)

        base_sample = {
            "id": s["id"],
            "image": abs_images,
        }

        # With memory — keep conversations as-is
        wm = dict(base_sample)
        wm["conversations"] = s["conversations"]
        samples_with_memory.append(wm)

        # No memory — strip memory block from human prompt
        nm = dict(base_sample)
        convs_stripped = []
        for turn in s["conversations"]:
            if turn["from"] == "human":
                convs_stripped.append({
                    "from": "human",
                    "value": strip_memory(turn["value"]),
                })
            else:
                convs_stripped.append(turn)
        nm["conversations"] = convs_stripped
        samples_no_memory.append(nm)

    output_dir.mkdir(parents=True, exist_ok=True)

    wm_path = output_dir / f"{split_name}_with_memory.json"
    nm_path = output_dir / f"{split_name}_no_memory.json"

    with open(wm_path, "w", encoding="utf-8") as f:
        json.dump(samples_with_memory, f, ensure_ascii=False, indent=None)
    logger.info("Wrote %d samples → %s", len(samples_with_memory), wm_path)

    with open(nm_path, "w", encoding="utf-8") as f:
        json.dump(samples_no_memory, f, ensure_ascii=False, indent=None)
    logger.info("Wrote %d samples → %s", len(samples_no_memory), nm_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert data_pipeline JSONL to LLaVA JSON arrays"
    )
    parser.add_argument(
        "--samples-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data_pipeline" / "samples",
        help="Directory with {split}.jsonl files (default: data_pipeline/samples)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory for LLaVA JSON files (default: baseline_mm-or/data)",
    )
    parser.add_argument(
        "--processed-root", type=Path, default=None,
        help="MM-OR_processed root for absolute image paths",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Which splits to convert (default: train val)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    processed_root = get_processed_root(args.processed_root)
    logger.info("MM-OR processed root: %s", processed_root)

    for split in args.splits:
        jsonl_path = args.samples_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            logger.warning("Skipping %s — %s not found", split, jsonl_path)
            continue
        convert_split(jsonl_path, args.output_dir, split, processed_root)

    logger.info("Done.")


if __name__ == "__main__":
    main()
