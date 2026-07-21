#!/usr/bin/env python3
"""
Convert data_pipeline JSONL samples to LLaVA-compatible JSON arrays.

Reads the per-frame samples produced by ``data_pipeline.build_samples`` and
writes two JSON files per split:

- ``{split}_no_memory.json``  — prompts with memory stripped (Phase 1)
- ``{split}_with_memory.json`` — prompts preserved as-is (Phase 2)

Image paths are stored relative to MM-OR_processed as ``{take}/{rel}``
(e.g. ``001_PKA/colorimage/camera01_....jpg``). Training must pass
``--image_folder $MM_OR_PROCESSED_ROOT`` so paths survive per-job NAS remounts.

By default, samples whose colorimage files are missing under ``--processed-root``
are skipped (so training does not crash on incomplete NAS frames).

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
from typing import Any, Dict, List, Tuple

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


def make_relative_images(image_paths: List[str], take: str) -> List[str]:
    """Return paths relative to MM-OR_processed: ``{take}/{rel}``."""
    relative: List[str] = []
    for p in image_paths:
        path = Path(p)
        if path.is_absolute():
            # Strip any .../MM-OR_processed/ prefix if present; else keep as-is
            parts = path.parts
            if "MM-OR_processed" in parts:
                idx = parts.index("MM-OR_processed")
                relative.append(str(Path(*parts[idx + 1 :])))
            else:
                relative.append(str(path))
        else:
            relative.append(str(Path(take) / path))
    return relative


def missing_images(rel_images: List[str], processed_root: Path) -> List[str]:
    """Return relative paths that are not files under processed_root."""
    return [p for p in rel_images if not (processed_root / p).is_file()]


def convert_split(
    jsonl_path: Path,
    output_dir: Path,
    split_name: str,
    processed_root: Path,
    require_existing_images: bool = True,
) -> Tuple[int, int]:
    """Read a JSONL split and write two LLaVA JSON files.

    Returns (kept, skipped) sample counts.
    """
    samples_raw: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples_raw.append(json.loads(line))

    logger.info("Loaded %d samples from %s", len(samples_raw), jsonl_path)

    samples_with_memory: List[Dict[str, Any]] = []
    samples_no_memory: List[Dict[str, Any]] = []
    skipped = 0

    for s in samples_raw:
        take = s.get("take", s["id"].split("/")[0])
        rel_images = make_relative_images(s["image"], take)

        if require_existing_images:
            missing = missing_images(rel_images, processed_root)
            if missing:
                skipped += 1
                if skipped <= 5:
                    logger.warning(
                        "Skipping %s — missing %d image(s), e.g. %s",
                        s.get("id", "?"),
                        len(missing),
                        missing[0],
                    )
                continue

        base_sample = {
            "id": s["id"],
            "image": rel_images,
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

    if require_existing_images and skipped:
        logger.info(
            "Skipped %d/%d samples with missing colorimage files under %s",
            skipped,
            len(samples_raw),
            processed_root,
        )

    if require_existing_images and not samples_no_memory:
        raise RuntimeError(
            f"No usable samples for {split_name} after filtering missing images "
            f"under {processed_root}. Is the NAS mount correct?"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    wm_path = output_dir / f"{split_name}_with_memory.json"
    nm_path = output_dir / f"{split_name}_no_memory.json"

    with open(wm_path, "w", encoding="utf-8") as f:
        json.dump(samples_with_memory, f, ensure_ascii=False, indent=None)
    logger.info("Wrote %d samples → %s", len(samples_with_memory), wm_path)

    with open(nm_path, "w", encoding="utf-8") as f:
        json.dump(samples_no_memory, f, ensure_ascii=False, indent=None)
    logger.info("Wrote %d samples → %s", len(samples_no_memory), nm_path)

    return len(samples_no_memory), skipped


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
        help="MM-OR_processed root used to resolve/verify image paths",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Do not skip samples when colorimage files are missing "
             "(for dry tests without a real NAS mount)",
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
    require_existing = not args.allow_missing_images
    logger.info("MM-OR processed root: %s", processed_root)
    logger.info(
        "Require existing images: %s",
        require_existing,
    )

    for split in args.splits:
        jsonl_path = args.samples_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            logger.warning("Skipping %s — %s not found", split, jsonl_path)
            continue
        convert_split(
            jsonl_path,
            args.output_dir,
            split,
            processed_root,
            require_existing_images=require_existing,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
