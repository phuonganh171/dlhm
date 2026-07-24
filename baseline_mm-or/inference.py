#!/usr/bin/env python3
"""
Sequential per-frame inference for Baseline 1 (MM-OR / ORacle architecture).

For each (role, L2 segment) in the test set:
  1. Walk frames sequentially within the L2 segment.
  2. At each frame: load multi-view images → model.generate() → parse output.
  3. Build temporal memory from the model's own predictions (autoregressive).
  4. Write predictions JSONL compatible with ``data_pipeline.evaluate``.

Supports two memory modes:
  - ``--memory-mode predicted``: autoregressive (model's own predictions)
  - ``--memory-mode gt``:        ground-truth temporal memory (oracle)

Usage::

    python baseline_mm-or/inference.py \
        --model-path baseline_mm-or/checkpoints/phase2_with_memory \
        --test-samples data_pipeline/samples/test.jsonl \
        --processed-root /path/to/MM-OR_processed \
        --output predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.assemble import parse_model_output
from data_pipeline.config import get_processed_root
from data_pipeline.temporal_memory import (
    TemporalMemoryBuilder,
    format_memory_string,
)


def resolve_model_path(model_path: str) -> Path:
    """Use ``model_path`` if it has adapters; else latest ``checkpoint-*``."""
    root = Path(model_path)
    if (root / "adapter_config.json").exists() or (root / "adapter_model.bin").exists():
        return root
    ckpts = sorted(
        root.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if ckpts:
        return ckpts[-1]
    return root


def load_model(model_path: str, model_base: str = "liuhaotian/llava-v1.5-7b"):
    """Load the fine-tuned ORacle LLaVA model with LoRA weights.

    ``load_pretrained_model`` only wires the vision tower / image processor when
    the model *name* contains ``llava`` (and LoRA when it contains ``lora``).
    Checkpoint dirs like ``phase2_with_memory`` do not, so we force those tags.
    """
    oracle_llava = Path(__file__).resolve().parent / "ORacle" / "LLaVA"
    if str(oracle_llava) not in sys.path:
        sys.path.insert(0, str(oracle_llava))

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    resolved = resolve_model_path(model_path)
    model_name = get_model_name_from_path(str(resolved))
    # Builder branches on substrings in model_name — not on config.model_type.
    if "llava" not in model_name.lower():
        model_name = f"llava_{model_name}"
    if "lora" not in model_name.lower():
        model_name = f"{model_name}_lora"

    logger.info("Loading model from %s (name=%s)", resolved, model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        str(resolved), model_base, model_name,
        load_8bit=False, load_4bit=False,
    )
    if image_processor is None:
        raise RuntimeError(
            "image_processor is None after load — vision tower was not initialized. "
            f"model_name={model_name!r} path={resolved}"
        )
    model.config.mv_type = "learned"
    model.config.tokenizer_padding_side = "left"
    model.eval()
    return tokenizer, model, image_processor, context_len


def build_prompt(
    role_human: str,
    memory_str: str,
    mm_use_im_start_end: bool = False,
) -> str:
    """Build the prompt string matching training format."""
    if mm_use_im_start_end:
        img_token = "<im_start><image><im_end>"
    else:
        img_token = "<image>"

    parts = [img_token, f"\nRole: {role_human}\n"]
    if memory_str:
        parts.append(memory_str + "\n")
    parts.append("Describe the current activity hierarchy for this role.")
    return "".join(parts)


def generate_single(
    model,
    tokenizer,
    image_processor,
    image_paths: List[str],
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Run inference on a single frame with multi-view images."""
    oracle_llava = Path(__file__).resolve().parent / "ORacle" / "LLaVA"
    if str(oracle_llava) not in sys.path:
        sys.path.insert(0, str(oracle_llava))

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, default_conversation
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )

    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except (FileNotFoundError, OSError) as e:
            logger.warning("Could not open image %s: %s", p, e)
            continue

    if not images:
        return ""

    image_tensor = process_images(images, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(model.device, dtype=torch.bfloat16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

    conv = deepcopy(default_conversation)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
        )

    output_text = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    if output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)].strip()

    return output_text


def load_test_samples(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load test samples JSONL and group by (role, l2_segment_id)."""
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def group_by_role_l2(
    samples: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Group samples by (take, role, l2_segment_id), sorted by tp_id."""
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        take = s.get("take", s["id"].split("/")[0])
        role = s["role"]
        l2_seg = s["l2_segment_id"]
        groups[(take, role, l2_seg)].append(s)

    for key in groups:
        groups[key].sort(key=lambda s: s["tp_id"])

    return groups


def filter_samples(
    samples: List[Dict[str, Any]],
    takes: Optional[List[str]] = None,
    max_groups: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Optionally keep only selected takes and/or the first N (take, role, L2)
    groups (sorted). Whole L2 groups are kept so temporal memory stays valid.
    """
    if takes:
        take_set = set(takes)
        samples = [
            s for s in samples
            if s.get("take", s["id"].split("/")[0]) in take_set
        ]
    if max_groups is None:
        return samples

    groups = group_by_role_l2(samples)
    keep_keys = set(sorted(groups.keys())[: max(0, max_groups)])
    return [
        s for s in samples
        if (
            s.get("take", s["id"].split("/")[0]),
            s["role"],
            s["l2_segment_id"],
        ) in keep_keys
    ]


def run_inference(
    model,
    tokenizer,
    image_processor,
    samples: List[Dict[str, Any]],
    processed_root: Path,
    memory_mode: str = "predicted",
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """
    Run inference on all test samples grouped by (role, L2 segment).

    For each group, walk frames sequentially, building temporal memory
    from either the model's predictions or ground-truth states.
    """
    groups = group_by_role_l2(samples)
    predictions: List[Dict[str, Any]] = []
    total_groups = len(groups)

    for group_idx, ((take, role, l2_seg), frames) in enumerate(sorted(groups.items())):
        logger.info(
            "[%d/%d] %s / %s / %s (%d frames)",
            group_idx + 1, total_groups, take, role, l2_seg, len(frames),
        )
        memory_builder = TemporalMemoryBuilder()
        memory_builder.reset()

        prev_l0: Optional[str] = None
        prev_l1: Optional[str] = None

        for frame_idx, sample in enumerate(frames):
            tp_id = sample["tp_id"]
            role_human = sample.get("role_human", role)

            # Build memory from prior predictions (or GT)
            if memory_mode == "gt" and frame_idx > 0:
                gt_l0 = sample.get("gt_l0", "")
                gt_l1 = sample.get("gt_l1", "")
                mem_state = memory_builder.step(tp_id, gt_l0, gt_l1)
            elif memory_mode == "predicted" and prev_l0 is not None:
                mem_state = memory_builder.step(tp_id, prev_l0, prev_l1 or "")
            else:
                from data_pipeline.temporal_memory import MemoryState
                mem_state = MemoryState()
                if frame_idx == 0:
                    if memory_mode == "gt":
                        gt_l0 = sample.get("gt_l0", "")
                        gt_l1 = sample.get("gt_l1", "")
                        memory_builder.step(tp_id, gt_l0, gt_l1)
                    # For predicted mode, first frame has no memory

            memory_str = format_memory_string(mem_state)
            prompt = build_prompt(
                role_human=role_human,
                memory_str=memory_str,
                mm_use_im_start_end=getattr(model.config, "mm_use_im_start_end", False),
            )

            # Resolve absolute image paths
            image_paths = []
            take_dir = processed_root / take
            for img_rel in sample.get("image", []):
                if Path(img_rel).is_absolute():
                    image_paths.append(img_rel)
                else:
                    image_paths.append(str(take_dir / img_rel))

            output_text = generate_single(
                model, tokenizer, image_processor,
                image_paths, prompt, max_new_tokens,
            )

            pred_l0, pred_l1, pred_l2 = parse_model_output(output_text)

            prev_l0 = pred_l0 if pred_l0 else sample.get("gt_l0", "")
            prev_l1 = pred_l1 if pred_l1 else sample.get("gt_l1", "")

            if memory_mode == "predicted" and frame_idx == 0:
                memory_builder.step(tp_id, prev_l0, prev_l1)

            pred = {
                "id": sample["id"],
                "take": take,
                "role": role,
                "role_human": role_human,
                "tp_id": tp_id,
                "l2_segment_id": l2_seg,
                "pred_l0": pred_l0,
                "pred_l1": pred_l1,
                "pred_l2": pred_l2,
                "raw_output": output_text,
                "gt_l0": sample.get("gt_l0", ""),
                "gt_l1": sample.get("gt_l1", ""),
                "gt_l2": sample.get("gt_l2", ""),
            }
            predictions.append(pred)

            if (frame_idx + 1) % 50 == 0:
                logger.info("  Frame %d/%d", frame_idx + 1, len(frames))

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference for Baseline 1 (MM-OR / ORacle)"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the fine-tuned LoRA checkpoint directory",
    )
    parser.add_argument(
        "--model-base", type=str, default="liuhaotian/llava-v1.5-7b",
        help="Base model name or path",
    )
    parser.add_argument(
        "--test-samples", type=Path, required=True,
        help="Test JSONL file from data_pipeline",
    )
    parser.add_argument(
        "--processed-root", type=Path, default=None,
        help="MM-OR_processed root for resolving image paths",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("predictions.jsonl"),
        help="Output predictions JSONL file",
    )
    parser.add_argument(
        "--memory-mode", choices=["predicted", "gt"], default="predicted",
        help="Temporal memory source: 'predicted' (autoregressive) or 'gt' (oracle)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum tokens to generate per frame",
    )
    parser.add_argument(
        "--takes", type=str, default="",
        help="Comma-separated take names to evaluate (default: all in the JSONL)",
    )
    parser.add_argument(
        "--max-groups", type=int, default=None,
        help="Evaluate only the first N (take, role, L2) groups after filtering",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    processed_root = get_processed_root(args.processed_root)
    logger.info("MM-OR processed root: %s", processed_root)

    logger.info("Loading model from %s ...", args.model_path)
    tokenizer, model, image_processor, context_len = load_model(
        args.model_path, args.model_base
    )
    logger.info("Model loaded. Context length: %d", context_len)

    logger.info("Loading test samples from %s ...", args.test_samples)
    samples = load_test_samples(args.test_samples)
    logger.info("Loaded %d test samples", len(samples))

    take_list = [t.strip() for t in args.takes.split(",") if t.strip()] or None
    if take_list or args.max_groups is not None:
        before = len(samples)
        samples = filter_samples(samples, takes=take_list, max_groups=args.max_groups)
        logger.info(
            "Filtered samples: %d → %d (takes=%s max_groups=%s)",
            before, len(samples), take_list or "all", args.max_groups,
        )

    predictions = run_inference(
        model, tokenizer, image_processor,
        samples, processed_root,
        memory_mode=args.memory_mode,
        max_new_tokens=args.max_new_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    logger.info("Wrote %d predictions to %s", len(predictions), args.output)

    correct_l0 = sum(1 for p in predictions if p["pred_l0"] == p["gt_l0"])
    correct_l1 = sum(1 for p in predictions if p["pred_l1"] == p["gt_l1"])
    correct_l2 = sum(1 for p in predictions if p["pred_l2"] == p["gt_l2"])
    total = len(predictions) or 1
    logger.info(
        "Quick exact-match: L0=%.1f%% L1=%.1f%% L2=%.1f%%",
        100 * correct_l0 / total, 100 * correct_l1 / total, 100 * correct_l2 / total,
    )


if __name__ == "__main__":
    main()
