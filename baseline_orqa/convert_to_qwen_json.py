#!/usr/bin/env python3
"""
Convert data_pipeline JSONL samples to ORQA / Qwen2-VL chat (QA) JSON.

Writes two files per split (ORQA curriculum = base then temporal):

- ``{split}_no_memory.json``  — memory blocks stripped (Phase 1 / base)
- ``{split}_with_memory.json`` — memory preserved (Phase 2 / Temp)

Format (LLaMA-Factory sharegpt / orqa dataset)::

- ``messages``: user / assistant turns (QA)
- ``images``: absolute paths under ``--processed-root`` (or relative
  ``{take}/colorimage/...`` when ``--relative-images`` is set)
- ``pc`` / ``audio``: empty strings (ORQA schema; unused for image-only)
- One ``<image>`` token per view in the user content

Training-time view augmentation (matches ORQA multimodal drop):
- Shuffle views
- Independently drop each view with probability ``--view-drop-prob`` (default 0.5)
- Keep at least one view

Usage::

    python baseline_orqa/convert_to_qwen_json.py \\
        --samples-dir data_pipeline/samples \\
        --output-dir  baseline_orqa/data \\
        --processed-root /path/to/MM-OR_processed \\
        --splits train val \\
        --augment-views
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_pipeline.config import get_processed_root

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<image>"
LEADING_IMAGE_RE = re.compile(r"^(?:<image>\s*)+")
MEMORY_PATTERN = re.compile(
    r"<memory_start>\n.*?<memory_end>\n?",
    re.DOTALL,
)


def strip_leading_image_tokens(text: str) -> str:
    """Remove leading ``<image>`` markers from a LLaVA-style human prompt."""
    return LEADING_IMAGE_RE.sub("", text).lstrip("\n")


def strip_memory(text: str) -> str:
    """Remove the <memory_start>...<memory_end> block from a prompt."""
    return MEMORY_PATTERN.sub("", text)


def make_relative_images(image_paths: List[str], take: str) -> List[str]:
    """Return paths relative to MM-OR_processed: ``{take}/{rel}``."""
    relative: List[str] = []
    for p in image_paths:
        path = Path(p)
        if path.is_absolute():
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
    return [p for p in rel_images if not (processed_root / p).is_file()]


def augment_views(
    rel_images: List[str],
    rng: random.Random,
    drop_prob: float,
) -> List[str]:
    """Shuffle and randomly drop views; keep at least one."""
    views = list(rel_images)
    rng.shuffle(views)
    if drop_prob <= 0 or len(views) <= 1:
        return views
    kept = [v for v in views if rng.random() >= drop_prob]
    if not kept:
        kept = [rng.choice(views)]
    return kept


def to_qwen_sample(
    sample: Dict[str, Any],
    rel_images: List[str],
    processed_root: Path,
    relative_images: bool,
    *,
    with_memory: bool,
) -> Dict[str, Any]:
    """Build one Qwen2-VL / ORQA training sample."""
    human = ""
    assistant = ""
    for turn in sample["conversations"]:
        if turn["from"] == "human":
            human = strip_leading_image_tokens(turn["value"])
            if not with_memory:
                human = strip_memory(human)
        elif turn["from"] == "gpt":
            assistant = turn["value"]

    image_tokens = IMAGE_TOKEN * len(rel_images)
    user_content = f"{image_tokens}{human}"

    if relative_images:
        images = list(rel_images)
    else:
        images = [str((processed_root / p).resolve()) for p in rel_images]

    # ORQA dataset_info maps pc/audio columns; image-only baseline leaves them empty
    # (collator treats "" as no point cloud / no audio).
    return {
        "id": sample["id"],
        "images": images,
        "pc": "",
        "audio": "",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant},
        ],
    }


def convert_split(
    jsonl_path: Path,
    output_dir: Path,
    split_name: str,
    processed_root: Path,
    require_existing_images: bool = True,
    augment_views_flag: bool = False,
    view_drop_prob: float = 0.5,
    relative_images: bool = False,
    seed: int = 42,
) -> Tuple[int, int]:
    samples_raw: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples_raw.append(json.loads(line))

    logger.info("Loaded %d samples from %s", len(samples_raw), jsonl_path)
    rng = random.Random(seed)
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

        views = (
            augment_views(rel_images, rng, view_drop_prob)
            if augment_views_flag
            else list(rel_images)
        )

        samples_with_memory.append(
            to_qwen_sample(
                s, views, processed_root, relative_images, with_memory=True
            )
        )
        # Same view set for the no-memory twin (fair pair); memory stripped only.
        samples_no_memory.append(
            to_qwen_sample(
                s, views, processed_root, relative_images, with_memory=False
            )
        )

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
        json.dump(samples_with_memory, f, ensure_ascii=False)
    with open(nm_path, "w", encoding="utf-8") as f:
        json.dump(samples_no_memory, f, ensure_ascii=False)
    logger.info("Wrote %d samples → %s", len(samples_with_memory), wm_path)
    logger.info("Wrote %d samples → %s", len(samples_no_memory), nm_path)
    return len(samples_no_memory), skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert data_pipeline JSONL to Qwen2-VL / ORQA QA JSON"
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data_pipeline" / "samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Do not skip samples when colorimage files are missing",
    )
    parser.add_argument(
        "--relative-images",
        action="store_true",
        help="Store {take}/colorimage/... paths instead of absolute paths",
    )
    parser.add_argument(
        "--augment-views",
        action="store_true",
        help="Shuffle + randomly drop camera views (training)",
    )
    parser.add_argument(
        "--view-drop-prob",
        type=float,
        default=0.5,
        help="Per-view drop probability when --augment-views (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    processed_root = get_processed_root(args.processed_root)
    require_existing = not args.allow_missing_images
    logger.info("MM-OR processed root: %s", processed_root)
    logger.info("Require existing images: %s", require_existing)
    logger.info("Augment views: %s (drop_prob=%s)", args.augment_views, args.view_drop_prob)

    for split in args.splits:
        jsonl_path = args.samples_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            logger.warning("Skipping %s — %s not found", split, jsonl_path)
            continue
        do_aug = args.augment_views and split == "train"
        convert_split(
            jsonl_path,
            args.output_dir,
            split,
            processed_root,
            require_existing_images=require_existing,
            augment_views_flag=do_aug,
            view_drop_prob=args.view_drop_prob,
            relative_images=args.relative_images,
            seed=args.seed,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
