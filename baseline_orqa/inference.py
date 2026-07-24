#!/usr/bin/env python3
"""
Sequential per-frame inference for Baseline 2 (ORQA / Qwen2-VL).

For each (role, L2 segment) in the test set:
  1. Walk frames sequentially within the L2 segment.
  2. At each frame: multi-view images → model.generate() → parse output.
  3. Build temporal memory from the model's own predictions (autoregressive).
  4. Write predictions JSONL compatible with ``data_pipeline.evaluate``.

Usage::

    python baseline_orqa/inference.py \\
        --model-path baseline_orqa/checkpoints/with_memory \\
        --test-samples data_pipeline/samples/test.jsonl \\
        --processed-root /path/to/MM-OR_processed \\
        --output predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORQA_ROOT = Path(__file__).resolve().parent / "ORQA"
LLAMA_FACTORY_SRC = ORQA_ROOT / "Qwen2-VL" / "LLaMA-Factory" / "src"

sys.path.insert(0, str(PROJECT_ROOT))
if LLAMA_FACTORY_SRC.is_dir():
    sys.path.insert(0, str(LLAMA_FACTORY_SRC))
if ORQA_ROOT.is_dir():
    sys.path.insert(0, str(ORQA_ROOT))

from data_pipeline.assemble import parse_model_output
from data_pipeline.config import get_processed_root
from data_pipeline.temporal_memory import (
    MemoryState,
    TemporalMemoryBuilder,
    format_memory_string,
)


def _inject_local_qwen_processors() -> None:
    """ORQA overrides transformers Qwen2-VL processors with local forks."""
    from llamafactory.model.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )
    from llamafactory.model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

    proc_mod = types.ModuleType("transformers.models.qwen2_vl.processing_qwen2_vl")
    proc_mod.Qwen2VLProcessor = Qwen2VLProcessor
    img_mod = types.ModuleType(
        "transformers.models.qwen2_vl.image_processing_qwen2_vl"
    )
    img_mod.Qwen2VLImageProcessor = Qwen2VLImageProcessor
    sys.modules["transformers.models.qwen2_vl.processing_qwen2_vl"] = proc_mod
    sys.modules["transformers.models.qwen2_vl.image_processing_qwen2_vl"] = img_mod


def _get_visual_module(model):
    """Return the vision tower whether model is bare or Peft-wrapped."""
    if hasattr(model, "visual"):
        return model.visual
    # PeftModel → base_model.model.visual
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "model") and hasattr(base.model, "visual"):
        return base.model.visual
    if base is not None and hasattr(base, "visual"):
        return base.visual
    raise AttributeError("Could not locate model.visual")


def load_hierarchy_checkpoint(model_path: str, model_base: str):
    """
    Load a LLaMA-Factory ORQA checkpoint.

    Training saves LoRA adapters + ``visual_block.pt``. Prefer Peft load when
    ``adapter_config.json`` is present; otherwise fall back to ORQA's helper.
    """
    from llamafactory.model.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLForConditionalGeneration,
    )
    from llamafactory.model.qwen2_vl.qwen2_vl_helpers import load_pretrained_model

    path = Path(model_path)
    attn = "flash_attention_2" if torch.cuda.is_available() else "eager"

    if (path / "adapter_config.json").is_file():
        from peft import PeftModel

        logger.info("Loading base %s + LoRA adapter from %s", model_base, path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_base,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attn,
        )
        model = PeftModel.from_pretrained(model, str(path))
        visual_block = path / "visual_block.pt"
        if visual_block.is_file():
            logger.info("Loading visual_block.pt")
            state = torch.load(visual_block, map_location="cpu")
            _get_visual_module(model).load_state_dict(state, strict=False)
        return model

    logger.info("Loading via ORQA helper from %s", path)
    return load_pretrained_model(str(path))


class HierarchyORQAModel:
    """Thin wrapper around ORQA's Qwen2-VL + image pooler for hierarchy inference."""

    def __init__(
        self,
        model_path: str,
        model_base: str = "Qwen/Qwen2-VL-2B-Instruct",
        fix_number_of_image_tokens: int = 578,
        image_resolution: int = 112896,
    ):
        _inject_local_qwen_processors()

        from llamafactory.data import (
            SFTDataCollatorWith4DAttentionMask,
            get_template_and_fix_tokenizer,
        )
        from llamafactory.model import load_tokenizer

        self.model = load_hierarchy_checkpoint(model_path, model_base)
        self.model.eval()
        visual = _get_visual_module(self.model)
        visual.image_pooler.fix_number_of_image_tokens = fix_number_of_image_tokens
        visual.image_pooler.use_past_visual_embeds = False

        model_args = SimpleNamespace(
            model_name_or_path=model_base,
            cache_dir=None,
            model_revision=None,
            hf_hub_token=None,
            use_fast_tokenizer=True,
            split_special_tokens=False,
            new_special_tokens=None,
            image_resolution=image_resolution,
            video_resolution=128 * 128,
            video_fps=2.0,
            video_maxlen=64,
        )
        data_args = SimpleNamespace(
            template="qwen2_vl",
            train_on_prompt=False,
            ignore_pad_token_for_loss=True,
            tool_format=None,
            fix_number_of_image_tokens=fix_number_of_image_tokens,
            use_past_visual_embeds=False,
        )
        tokenizer_module = load_tokenizer(model_args)
        tokenizer_module["tokenizer"].padding_side = "left"
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        self.mm_plugin = self.template.mm_plugin
        self.data_collator = SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            label_pad_token_id=-100,
            block_diag_attn=False,
            attn_implementation=self.model.config._attn_implementation,
            compute_dtype=torch.bfloat16,
            **tokenizer_module,
        )

    def generate(
        self,
        image_paths: List[str],
        prompt_text: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Run one forward generate; ``prompt_text`` must already include ``<image>`` tokens."""
        messages = [{"role": "user", "content": prompt_text}]
        images = list(image_paths)
        processed_messages = self.mm_plugin.process_messages(
            messages, images, [], self.processor
        )
        input_ids, _ = self.template.encode_oneturn(
            self.tokenizer, processed_messages, system=None, tools=None
        )
        features = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "images": images,
            "videos": [],
            "pc": None,
            "audio": None,
            "id": "infer",
        }
        batch_features = self.data_collator([features])
        for k, v in batch_features.items():
            if isinstance(v, torch.Tensor):
                batch_features[k] = v.to(self.model.device)

        multimodal_extras = getattr(batch_features, "_multimodal_extras", None)
        if multimodal_extras:
            for k, v in multimodal_extras.items():
                batch_features.data[k] = v

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        with torch.inference_mode():
            output_ids = self.model.generate(**batch_features, **gen_kwargs)

        text = self.tokenizer.decode(
            output_ids[0, batch_features.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return text.replace("<|im_end|>", "").strip()


def build_prompt(role_human: str, memory_str: str, n_images: int) -> str:
    """Match training format: N ``<image>`` tokens + role + optional memory + question."""
    image_tokens = "<image>" * max(1, n_images)
    parts = [image_tokens, f"Role: {role_human}\n"]
    if memory_str:
        parts.append(memory_str + "\n")
    parts.append("Describe the current activity hierarchy for this role.")
    return "".join(parts)


def load_test_samples(jsonl_path: Path) -> List[Dict[str, Any]]:
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
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        take = s.get("take", s["id"].split("/")[0])
        groups[(take, s["role"], s["l2_segment_id"])].append(s)
    for key in groups:
        groups[key].sort(key=lambda s: s["tp_id"])
    return groups


def filter_samples(
    samples: List[Dict[str, Any]],
    takes: Optional[List[str]] = None,
    max_groups: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if takes:
        take_set = set(takes)
        samples = [
            s
            for s in samples
            if s.get("take", s["id"].split("/")[0]) in take_set
        ]
    if max_groups is None:
        return samples
    groups = group_by_role_l2(samples)
    keep_keys = set(sorted(groups.keys())[: max(0, max_groups)])
    return [
        s
        for s in samples
        if (
            s.get("take", s["id"].split("/")[0]),
            s["role"],
            s["l2_segment_id"],
        )
        in keep_keys
    ]


def resolve_image_paths(
    sample: Dict[str, Any],
    take: str,
    processed_root: Path,
) -> List[str]:
    take_dir = processed_root / take
    paths: List[str] = []
    for img_rel in sample.get("image", []):
        p = Path(img_rel)
        if p.is_absolute():
            paths.append(str(p))
        else:
            paths.append(str(take_dir / img_rel))
    # Drop missing files (keep at least nothing → caller skips)
    existing = [p for p in paths if Path(p).is_file()]
    return existing


def run_inference(
    model: HierarchyORQAModel,
    samples: List[Dict[str, Any]],
    processed_root: Path,
    memory_mode: str = "predicted",
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    groups = group_by_role_l2(samples)
    predictions: List[Dict[str, Any]] = []
    total_groups = len(groups)

    for group_idx, ((take, role, l2_seg), frames) in enumerate(sorted(groups.items())):
        logger.info(
            "[%d/%d] %s / %s / %s (%d frames)",
            group_idx + 1,
            total_groups,
            take,
            role,
            l2_seg,
            len(frames),
        )
        memory_builder = TemporalMemoryBuilder()
        memory_builder.reset()
        prev_l0: Optional[str] = None
        prev_l1: Optional[str] = None

        for frame_idx, sample in enumerate(frames):
            tp_id = sample["tp_id"]
            role_human = sample.get("role_human", role)

            if memory_mode == "gt" and frame_idx > 0:
                mem_state = memory_builder.step(
                    tp_id, sample.get("gt_l0", ""), sample.get("gt_l1", "")
                )
            elif memory_mode == "predicted" and prev_l0 is not None:
                mem_state = memory_builder.step(tp_id, prev_l0, prev_l1 or "")
            else:
                mem_state = MemoryState()
                if frame_idx == 0 and memory_mode == "gt":
                    memory_builder.step(
                        tp_id, sample.get("gt_l0", ""), sample.get("gt_l1", "")
                    )

            memory_str = format_memory_string(mem_state)
            image_paths = resolve_image_paths(sample, take, processed_root)
            if not image_paths:
                logger.warning("No images for %s — skipping", sample["id"])
                continue

            prompt = build_prompt(role_human, memory_str, len(image_paths))
            output_text = model.generate(
                image_paths, prompt, max_new_tokens=max_new_tokens
            )
            pred_l0, pred_l1, pred_l2 = parse_model_output(output_text)

            prev_l0 = pred_l0 if pred_l0 else sample.get("gt_l0", "")
            prev_l1 = pred_l1 if pred_l1 else sample.get("gt_l1", "")
            if memory_mode == "predicted" and frame_idx == 0:
                memory_builder.step(tp_id, prev_l0, prev_l1)

            predictions.append(
                {
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
            )

            if (frame_idx + 1) % 50 == 0:
                logger.info("  Frame %d/%d", frame_idx + 1, len(frames))

    return predictions


def find_checkpoint(model_path: Path) -> Path:
    """Prefer latest checkpoint-* under model_path if present."""
    if (model_path / "adapter_config.json").exists() or (
        model_path / "model.safetensors"
    ).exists() or (model_path / "visual_block.pt").exists():
        return model_path
    ckpts = sorted(
        model_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if ckpts:
        return ckpts[-1]
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference for Baseline 2 (ORQA / Qwen2-VL)"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--model-base", type=str, default="Qwen/Qwen2-VL-2B-Instruct"
    )
    parser.add_argument("--test-samples", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument(
        "--memory-mode", choices=["predicted", "gt"], default="predicted"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--takes", type=str, default="")
    parser.add_argument("--max-groups", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    processed_root = get_processed_root(args.processed_root)
    model_path = find_checkpoint(Path(args.model_path))
    logger.info("MM-OR processed root: %s", processed_root)
    logger.info("Loading model from %s ...", model_path)

    model = HierarchyORQAModel(
        str(model_path),
        model_base=args.model_base,
    )

    samples = load_test_samples(args.test_samples)
    logger.info("Loaded %d test samples", len(samples))

    take_list = [t.strip() for t in args.takes.split(",") if t.strip()] or None
    if take_list or args.max_groups is not None:
        before = len(samples)
        samples = filter_samples(samples, takes=take_list, max_groups=args.max_groups)
        logger.info(
            "Filtered samples: %d → %d (takes=%s max_groups=%s)",
            before,
            len(samples),
            take_list or "all",
            args.max_groups,
        )

    predictions = run_inference(
        model,
        samples,
        processed_root,
        memory_mode=args.memory_mode,
        max_new_tokens=args.max_new_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    logger.info("Wrote %d predictions to %s", len(predictions), args.output)
    total = len(predictions) or 1
    correct_l0 = sum(1 for p in predictions if p["pred_l0"] == p["gt_l0"])
    correct_l1 = sum(1 for p in predictions if p["pred_l1"] == p["gt_l1"])
    correct_l2 = sum(1 for p in predictions if p["pred_l2"] == p["gt_l2"])
    logger.info(
        "Quick exact-match: L0=%.1f%% L1=%.1f%% L2=%.1f%%",
        100 * correct_l0 / total,
        100 * correct_l1 / total,
        100 * correct_l2 / total,
    )


if __name__ == "__main__":
    main()
