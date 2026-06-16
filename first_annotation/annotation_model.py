"""
Pseudo-annotation of atomic actions using scene graphs + Qwen3-32B.

For each 5-second window of scene graphs, the LLM produces a single-sentence
summary describing the atomic action / activity occurring in the OR.

Qwen3-32B is loaded in 4-bit quantization (bitsandbytes NF4) so it fits on a
single 24 GB GPU.  The model's "thinking" mode is left enabled for better
reasoning; <think>...</think> blocks are stripped from the final output.

Usage:
    cd /mnt/home/nhatvu/dlhm && HF_HOME=/tmp/nhatvu/hf_cache \
    /tmp/nhatvu/.venv/bin/python3 annotation_model.py \
    --max_frames 600 \
    --output annotation_output/annotations_001_PKA_600.jsonl \
    --log_file /tmp/nhatvu/run.log \
    --batch_size 2

"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from scene_graph_utils import (
    ENTITY_NAMES,
    PREDICATE_NAMES,
    build_chat_input,
    frame_info,
    humanize,
    humanize_pred,
    load_frame_map,
    load_model,
    load_relation_labels,
    make_windows,
    original_timestamp,
    run_inference_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
'''
  You are a surgical activity annotator for operating room scenes. You receive 5 consecutive scene graphs (one per second) from a 5-second segment. Each scene graph is a list of
  (subject, predicate, object) triplets describing the relationships between people, instruments, and equipment in the OR.
  
  Predicate types:
  - Active (prioritize these): Manipulating, Assisting, Holding, Preparing, Touching, ...
  - Spatial (mention only if relevant): CloseTo
  
  Rules:
  - Describe the atomic action in exactly one or two grammatically complete English sentences, no preamble, no prefix 
  - If relationships CHANGE: describe the transition (who started/stopped doing what)
  - If relationships are CONSTANT: describe the ongoing activity
  - A triplet that DISAPPEARS means that action STOPPED (e.g., "Holding, instrument" disappearing = released the instrument)
  - A triplet that APPEARS means that action STARTED
  - Each triplet is independent — do NOT combine separate triplets into one causal action (e.g., "Manipulating, mako_robot" and "Holding, instrument" are two separate facts, not 
  "manipulating the robot to hold the instrument")
  - Do NOT speculate or add reasoning not directly supported by the triplets
  - Prioritize active predicates over spatial ones
  - If only spatial predicates (CloseTo) remain with no change, describe positions briefly
  - Do not make more than 2 sentences.
  - NEVER output raw triplets, lists, brackets, or structured data. Always output natural language only.

  Example 1 — Transition:
  Input:
  Throughout (t1-t5):
  [("mps","Manipulating","mako_robot"), ("anest","Touching","ae")]
  Changes:
  - t1-t2: ("nurse","CloseTo","anest")
  - t3-t5: ("nurse","Assisting","mps")
  Output: The robot technician operates the Mako robot as the scrub nurse transitions from the anaesthetist's side to assisting the robot technician.

  Example 2 — Constant:
  Input:
  Throughout (t1-t5):
  [("mps","Manipulating","mako_robot"), ("anest","Touching","ae"), ("nurse","Assisting","mps")]
  Changes: none
  Output: The robot technician continues operating the Mako robot with the scrub nurse assisting, while the anaesthetist monitors the anaesthesia equipment.

  Example 3 — Disappearing relationship:
  Input:
  Throughout (t1-t5):
  [("nurse","CloseTo","instrument_table"), ("mps","Manipulating","mako_robot")]
  Changes:
  - t1-t4: ("mps","Holding","instrument")
  - t5: (removed)
  Output: The robot technician releases the surgical instrument while continuing to operate the Mako robotic arm, with the scrub nurse remaining at the instrument table.

'''
)


def format_window_prompt(
    window: list[tuple[str, list]],
    frame_map: Dict[str, Dict[str, Any]],
) -> str:
    """
    Build the user message for one 5-second window.
    """
    lines = []
    for tp_id, triplets in window:
        if tp_id not in frame_map:
            rel_str = "(no frame mapping)"
        elif not triplets:
            rel_str = "(no relations annotated)"
        else:
            rels = "; ".join(
                f"{humanize(s)} {humanize_pred(p)} {humanize(o)}"
                for s, p, o in triplets
            )
            rel_str = rels
        ts = original_timestamp(frame_map, tp_id)
        ts_label = f"{ts}s" if ts is not None else "unmapped"
        lines.append(f"  t={ts_label}: {rel_str}")

    t_start = original_timestamp(frame_map, window[0][0])
    t_end = original_timestamp(frame_map, window[-1][0])
    if t_start is not None and t_end is not None:
        range_str = f"t={t_start}s – t={t_end}s"
    else:
        range_str = f"tp={window[0][0]} – tp={window[-1][0]}"

    scene_text = "\n".join(lines)
    return (
        f"Scene graph annotations for a 5-second window ({range_str}):\n"
        f"{scene_text}\n\n"
        "Atomic action summary:"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def annotate(
    take_dir: Path,
    model_name_or_path: str,
    hf_token: Optional[str],
    window_size: int,
    start_tp: Optional[str],
    max_frames: Optional[int],
    batch_size: int,
    output_path: Path,
):
    # --- Load data ---
    logger.info("Loading frame map from %s ...", take_dir)
    frame_map = load_frame_map(take_dir)
    logger.info("%d timepoints in timestamp_to_pcd_and_frames_list.json.", len(frame_map))

    logger.info("Loading relation labels from %s ...", take_dir)
    entries = load_relation_labels(take_dir, frame_map)
    logger.info("%d relation-label timepoints loaded.", len(entries))

    if start_tp is not None:
        before = len(entries)
        entries = [(tp_id, triplets) for tp_id, triplets in entries if tp_id >= start_tp]
        t0 = original_timestamp(frame_map, entries[0][0]) if entries else None
        logger.info(
            "Starting at timepoint %s: kept %d/%d entries (t=%s).",
            start_tp,
            len(entries),
            before,
            t0 if t0 is not None else "?",
        )

    if max_frames is not None:
        entries = entries[:max_frames]
        logger.info("Using %d timepoints (--max_frames %d).", len(entries), max_frames)

    windows = make_windows(entries, window_size)
    logger.info("%d windows of %ds each.", len(windows), window_size)

    # --- Load model ---
    model, tokenizer = load_model(model_name_or_path, hf_token)

    from evaluate_annotations import check_annotation

    # --- Inference in batches ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    n_windows = len(windows)

    for batch_start in tqdm(range(0, n_windows, batch_size), desc="Annotating", unit="batch"):
        batch_windows = windows[batch_start : batch_start + batch_size]

        prompts = [
            build_chat_input(tokenizer, SYSTEM_PROMPT, format_window_prompt(w, frame_map))
            for w in batch_windows
        ]

        summaries = run_inference_batch(model, tokenizer, prompts)

        for window, summary in zip(batch_windows, summaries):
            tp_ids = [tp_id for tp_id, _ in window]
            orig_ts = [original_timestamp(frame_map, tp_id) for tp_id in tp_ids]
            colorimage_frames = [frame_info(frame_map, tp_id) for tp_id in tp_ids]
            scene_graphs = [triplets for _, triplets in window]

            record = {
                "window_start_tp":       tp_ids[0],
                "window_end_tp":         tp_ids[-1],
                "original_timestamps":   orig_ts,
                "colorimage_frames":     colorimage_frames,
                "window_size_s":         len(window),
                "scene_graphs":          scene_graphs,
                "atomic_action_summary": summary,
                "issues":                check_annotation({
                    "scene_graphs":          scene_graphs,
                    "atomic_action_summary": summary,
                }),
            }
            results.append(record)

        done = min(batch_start + batch_size, n_windows)
        logger.info(
            "Batch %d/%d done — window %d/%d | last summary: %s%s",
            done // batch_size,
            (n_windows + batch_size - 1) // batch_size,
            done,
            n_windows,
            repr(summaries[-1][:80]),
            f" | issues: {results[-1]['issues']}" if results[-1]["issues"] else "",
        )

    # --- Save annotations (with issues embedded) ---
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    logger.info("Done. %d annotations saved to %s", len(results), output_path)

    # --- Auto-evaluate summary ---
    from evaluate_annotations import evaluate_all, print_report
    logger.info("Running automated quality checks ...")
    report = evaluate_all(results)
    print_report(report)

    report_json_path = output_path.with_name(output_path.stem + "_evaluation.json")
    report_json_path.write_text(json.dumps(report, indent=2))
    logger.info("Evaluation JSON saved to %s", report_json_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate atomic-action pseudo-annotations from MM-OR scene graphs."
    )
    parser.add_argument(
        "--take_dir",
        type=Path,
        default=Path("mm-or/MM-OR_data/MM-OR_processed/001_PKA"),
        help="Path to the take directory (contains relation_labels/, etc.).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="HuggingFace model ID or local path to the model.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="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",
        help="HuggingFace access token (required for gated models like Llama 3.1).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Number of seconds (timepoints) per window (default: 5).",
    )
    parser.add_argument(
        "--start_tp",
        type=str,
        default=None,
        help="Start at this relation-label timepoint id (e.g. 001114 for surgery).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Number of timepoints to process from start_tp (default: all remaining).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Inference batch size (lower if OOM; default: 2 for 32B model on 24 GB GPU).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotation_output/annotations_001_PKA.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Optional file to write logs to (in addition to stdout).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import torch, random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info("Random seed set to %d", args.seed)

    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
        logger.info("Logging to %s", args.log_file)
    annotate(
        take_dir=args.take_dir,
        model_name_or_path=args.model,
        hf_token=args.hf_token,
        window_size=args.window_size,
        start_tp=args.start_tp,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        output_path=args.output,
    )

