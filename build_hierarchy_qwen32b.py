"""
Build a role-aware hierarchical representation of surgical scene graphs.

Phase 1:
    Level-0 segmentation -- for each role, find maximal contiguous spans where
    the set of active predicates stays the same.  A debounce window absorbs
    1-frame flickers caused by detector noise.

Usage:
    source /tmp/nhatvu/.venv/bin/activate &&
    python3 build_hierarchy_qwen32b.py \
        --take_dir mm-or/MM-OR_data/MM-OR_processed/001_PKA \
        --start_tp 001114 \
        --max_frames 600 \
        --level2 \
        --model Qwen/Qwen3-32B \
        --output hierarchy_output/001_PKA_hierarchy.json
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from scene_graph_utils import (
    ACTIVE_PREDICATES,
    ROLE_ENTITIES,
    TOOL_ENTITIES,
    VERB_TOOL,
    humanize,
    humanize_pred,
    load_frame_map,
    load_relation_labels,
    load_robot_phase,
    load_screen_summaries,
    original_timestamp,
)

# Synthetic roles whose timelines come from phase signals (not scene-graph triplets).
ROBOT_ROLES = {"robot_setup", "robot_monitor"}

# Tool/instrument roles, built by inverting the scene graph on the object slot.
TOOL_ROLES = set(TOOL_ENTITIES)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-role triplet extraction
# ---------------------------------------------------------------------------

def extract_role_triplets(
    entries: List[Tuple[str, list]],
) -> Dict[str, List[Tuple[str, FrozenSet[Tuple[str, str]]]]]:
    """
    Decompose the global scene graph timeline into per-role timelines.

    Returns {role: [(tp_id, frozenset of (predicate, object) tuples), ...]}.
    Only active predicates are included (CloseTo, LyingOn filtered out).
    Only entities in ROLE_ENTITIES appear as roles (people, not objects).
    """
    roles_seen: set = set()
    for _, triplets in entries:
        for subj, pred, obj in triplets:
            if subj in ROLE_ENTITIES:
                roles_seen.add(subj)

    role_timelines: Dict[str, List[Tuple[str, FrozenSet[Tuple[str, str]]]]] = {}
    for role in sorted(roles_seen):
        timeline = []
        for tp_id, triplets in entries:
            present = any(subj == role for subj, pred, obj in triplets)
            if not present:
                continue
            active = frozenset(
                (pred, obj)
                for subj, pred, obj in triplets
                if subj == role and pred in ACTIVE_PREDICATES
            )
            timeline.append((tp_id, active))
        role_timelines[role] = timeline

    return role_timelines


def extract_tool_timelines(
    entries: List[Tuple[str, list]],
) -> Dict[str, List[Tuple[str, FrozenSet[Tuple[str, str]]]]]:
    """
    Decompose the global scene graph into per-tool timelines (inverse of
    extract_role_triplets: group by the *object* slot instead of the subject).

    For each tool, a timepoint is included whenever the tool is referenced --
    either as the object of a relation, or as the implied instrument of a mapped
    action verb (VERB_TOOL).  The active state is a frozenset of (predicate,
    subject) pairs, i.e. *who* is acting on / with the tool:

      - handling/touching:  (head_surgeon, Holding, saw)   -> ("Holding", "head_surgeon")
      - implied action use: (head_surgeon, Sawing, patient) -> ("Sawing", "head_surgeon")

    Only tools that actually appear in the window are returned.
    """
    tool_timelines: Dict[str, List[Tuple[str, FrozenSet[Tuple[str, str]]]]] = {}

    for tool in sorted(TOOL_ENTITIES):
        timeline = []
        for tp_id, triplets in entries:
            present = False
            state = set()
            for subj, pred, obj in triplets:
                if obj == tool:
                    present = True
                    if pred in ACTIVE_PREDICATES:
                        state.add((pred, subj))
                if VERB_TOOL.get(pred) == tool:
                    present = True
                    state.add((pred, subj))
            if present:
                timeline.append((tp_id, frozenset(state)))
        if timeline:
            tool_timelines[tool] = timeline

    return tool_timelines


def build_robot_role_timelines(
    entries: List[Tuple[str, list]],
    frame_map: Dict[str, Dict[str, Any]],
    take_dir: Path,
) -> Dict[str, List[Tuple[str, FrozenSet[Tuple[str, str]]]]]:
    """
    Construct synthetic per-timepoint state timelines for the robot roles, in the
    same (tp_id, frozenset of (key, value)) format used for human roles so they
    can flow through the identical level-0/1/2 machinery.

    - robot_setup:   physical robot setup phase (keyed directly by timepoint id)
    - robot_monitor: Mako on-screen navigation phase + current_step
    """
    tp_ids = [tp for tp, _ in entries]
    timelines: Dict[str, List[Tuple[str, FrozenSet[Tuple[str, str]]]]] = {}

    robot_phase = load_robot_phase(take_dir)
    if robot_phase:
        tl = []
        for tp in tp_ids:
            phase = robot_phase.get(tp)
            if phase:
                tl.append((tp, frozenset({("Phase", phase)})))
        if tl:
            timelines["robot_setup"] = tl

    screen = load_screen_summaries(take_dir, frame_map)
    if screen:
        tl = []
        for tp in tp_ids:
            phase, step = screen.get(tp, (None, None))
            state = set()
            if phase:
                state.add(("Phase", phase))
            if step:
                state.add(("Step", step))
            if state:
                tl.append((tp, frozenset(state)))
        if tl:
            timelines["robot_monitor"] = tl

    return timelines


# ---------------------------------------------------------------------------
# Level-0 segmentation with debounce
# ---------------------------------------------------------------------------

def segment_level0(
    timeline: List[Tuple[str, FrozenSet[Tuple[str, str]]]],
    debounce: int = 2,
) -> List[Dict[str, Any]]:
    """
    Segment a single role's timeline into level-0 segments.

    A level-0 segment is a maximal contiguous span where the set of active
    (predicate, object) pairs is identical.

    The debounce parameter absorbs transient changes lasting fewer than
    `debounce` frames -- if the triplet set changes for < debounce frames
    and then reverts, the blip is merged into the surrounding segment.

    Returns a list of segment dicts (no summary yet -- that comes at level-1).
    """
    if not timeline:
        return []

    smoothed = _debounce_timeline(timeline, debounce)

    segments: List[Dict[str, Any]] = []
    seg_start_idx = 0
    current_set = smoothed[0][1]

    for i in range(1, len(smoothed)):
        if smoothed[i][1] != current_set:
            segments.append(_make_segment(smoothed, seg_start_idx, i - 1, current_set))
            seg_start_idx = i
            current_set = smoothed[i][1]

    segments.append(_make_segment(smoothed, seg_start_idx, len(smoothed) - 1, current_set))
    return segments


def _debounce_timeline(
    timeline: List[Tuple[str, FrozenSet[Tuple[str, str]]]],
    debounce: int,
) -> List[Tuple[str, FrozenSet[Tuple[str, str]]]]:
    """
    Smooth out transient flickers shorter than `debounce` frames.

    If a run of frames with a different triplet set is shorter than `debounce`,
    replace it with the surrounding (previous) set.
    """
    if debounce <= 1 or len(timeline) < 3:
        return list(timeline)

    result = list(timeline)

    i = 0
    while i < len(result):
        run_start = i
        run_set = result[i][1]
        while i < len(result) and result[i][1] == run_set:
            i += 1
        run_len = i - run_start

        if run_len < debounce and run_start > 0:
            prev_set = result[run_start - 1][1]
            for j in range(run_start, min(i, len(result))):
                result[j] = (result[j][0], prev_set)

    return result


def _make_segment(
    timeline: List[Tuple[str, FrozenSet[Tuple[str, str]]]],
    start_idx: int,
    end_idx: int,
    active_set: FrozenSet[Tuple[str, str]],
) -> Dict[str, Any]:
    """Build a level-0 segment dict from timeline indices."""
    tp_ids = [timeline[i][0] for i in range(start_idx, end_idx + 1)]
    return {
        "tp_start": tp_ids[0],
        "tp_end": tp_ids[-1],
        "duration_frames": len(tp_ids),
        "active_predicates": sorted([list(t) for t in active_set]),
    }


# ---------------------------------------------------------------------------
# Build full level-0 hierarchy
# ---------------------------------------------------------------------------

def build_level0(
    entries: List[Tuple[str, list]],
    frame_map: Dict[str, Dict[str, Any]],
    debounce: int = 2,
    take_dir: Optional[Path] = None,
    include_tools: bool = True,
) -> Dict[str, Any]:
    """
    Build the complete level-0 hierarchy for all roles.

    Returns a dict ready to be serialised as JSON:
    {
        "roles": {
            "head_surgeon": {
                "level0_segments": [ ... ],
                "num_segments": N,
            },
            ...
        },
        "metadata": { ... }
    }
    """
    role_timelines = extract_role_triplets(entries)

    if take_dir is not None:
        role_timelines.update(
            build_robot_role_timelines(entries, frame_map, take_dir)
        )

    if include_tools:
        role_timelines.update(extract_tool_timelines(entries))

    roles_output = {}
    total_segments = 0

    for role, timeline in sorted(role_timelines.items()):
        segments = segment_level0(timeline, debounce=debounce)

        for i, seg in enumerate(segments):
            seg["segment_id"] = f"{role}_L0_{i:03d}"
            seg["role"] = role
            seg["level"] = 0
            seg["role_human"] = humanize(role)

            ts_start = original_timestamp(frame_map, seg["tp_start"])
            ts_end = original_timestamp(frame_map, seg["tp_end"])
            seg["time_start"] = ts_start
            seg["time_end"] = ts_end
            if ts_start is not None and ts_end is not None:
                seg["duration_s"] = ts_end - ts_start + 1
            else:
                seg["duration_s"] = None

            if role in ROBOT_ROLES:
                seg["description"] = _auto_describe_robot(role, seg["active_predicates"])
            elif role in TOOL_ROLES:
                seg["description"] = _auto_describe_tool(role, seg["active_predicates"])
            else:
                seg["description"] = _auto_describe_l0(role, seg["active_predicates"])

        roles_output[role] = {
            "level0_segments": segments,
            "num_segments": len(segments),
        }
        total_segments += len(segments)

        logger.info(
            "  %-20s  %3d level-0 segments",
            humanize(role),
            len(segments),
        )

    return {
        "roles": roles_output,
        "metadata": {
            "total_roles": len(roles_output),
            "total_level0_segments": total_segments,
            "debounce_frames": debounce,
        },
    }


def analyze_debounce(
    entries: List[Tuple[str, list]],
    frame_map: Dict[str, Dict[str, Any]],
    debounce: int = 2,
) -> Dict[str, Any]:
    """
    Compare segmentation with and without debounce, categorise every absorbed
    1-frame segment into noise types, and return a summary report.

    Categories:
      FLICKER         -- A->B->A, state reverts to what it was before
      PRE_TRANSITION  -- gap->B->C, new state appears 1 frame early
      DROPOUT         -- A->nothing->C, detector lost tracking for 1 frame
      POST_ACTIVITY   -- A->B->idle, previous state lingers 1 extra frame
      UNIQUE          -- A->B->C where all three differ (potential real transition)
    """
    role_timelines = extract_role_triplets(entries)

    # Build timestamp -> per-role active-predicate lookup
    ts_role_active: Dict[int, Dict[str, FrozenSet]] = defaultdict(dict)
    for role, timeline in role_timelines.items():
        for tp_id, active_set in timeline:
            ts = original_timestamp(frame_map, tp_id)
            if ts is not None:
                ts_role_active[ts][role] = active_set

    categories: Dict[str, list] = defaultdict(list)
    total_absorbed = 0

    for role, timeline in role_timelines.items():
        raw_segs = segment_level0(timeline, debounce=1)
        for seg in raw_segs:
            if seg["duration_frames"] != 1:
                continue

            tp = seg["tp_start"]
            ts = original_timestamp(frame_map, tp)
            if ts is None:
                continue

            total_absorbed += 1
            current = ts_role_active.get(ts, {}).get(role, frozenset())
            before = ts_role_active.get(ts - 1, {}).get(role, frozenset())
            after = ts_role_active.get(ts + 1, {}).get(role, frozenset())

            if before == after:
                cat = "FLICKER"
            elif len(current) == 0:
                cat = "DROPOUT"
            elif len(before) == 0 and len(after) > 0:
                cat = "PRE_TRANSITION"
            elif len(after) == 0:
                cat = "POST_ACTIVITY"
            else:
                cat = "UNIQUE"

            categories[cat].append({
                "role": role,
                "time": ts,
                "before": sorted([list(t) for t in before]),
                "current": sorted([list(t) for t in current]),
                "after": sorted([list(t) for t in after]),
            })

    # Also compute segment count diff
    seg_counts_raw = {}
    seg_counts_deb = {}
    for role, timeline in role_timelines.items():
        seg_counts_raw[role] = len(segment_level0(timeline, debounce=1))
        seg_counts_deb[role] = len(segment_level0(timeline, debounce=debounce))

    report = {
        "debounce": debounce,
        "segments_without_debounce": sum(seg_counts_raw.values()),
        "segments_with_debounce": sum(seg_counts_deb.values()),
        "segments_removed": sum(seg_counts_raw.values()) - sum(seg_counts_deb.values()),
        "one_frame_segments_analyzed": total_absorbed,
        "categories": {
            cat: {"count": len(items), "percent": round(100 * len(items) / max(total_absorbed, 1), 1)}
            for cat, items in sorted(categories.items(), key=lambda x: -len(x[1]))
        },
        "per_role": {
            role: {"raw": seg_counts_raw[role], "debounced": seg_counts_deb[role],
                   "removed": seg_counts_raw[role] - seg_counts_deb[role]}
            for role in sorted(seg_counts_raw)
        },
    }
    return report


def _log_debounce_report(report: Dict[str, Any]) -> None:
    """Pretty-print the debounce analysis report."""
    logger.info("")
    logger.info("=== Debounce Analysis (debounce=%d) ===", report["debounce"])
    logger.info(
        "Segments: %d (raw) -> %d (debounced), %d removed (%.0f%%)",
        report["segments_without_debounce"],
        report["segments_with_debounce"],
        report["segments_removed"],
        100 * report["segments_removed"] / max(report["segments_without_debounce"], 1),
    )
    logger.info("")
    logger.info("Per-role breakdown:")
    for role, counts in report["per_role"].items():
        if counts["removed"] > 0:
            logger.info(
                "  %-25s %3d -> %3d  (-%d)",
                humanize(role), counts["raw"], counts["debounced"], counts["removed"],
            )
    logger.info("")
    logger.info("1-frame segment categories (%d total):", report["one_frame_segments_analyzed"])
    for cat, info in report["categories"].items():
        logger.info("  %-20s %3d  (%5.1f%%)", cat, info["count"], info["percent"])


def _auto_describe_l0(role: str, active_predicates: list) -> str:
    """Generate a short automatic description for a level-0 segment."""
    role_name = humanize(role)
    if not active_predicates:
        return f"{role_name}: no active interactions"
    parts = []
    for pred, obj in active_predicates:
        parts.append(f"{humanize_pred(pred)} {humanize(obj)}")
    return f"{role_name}: {'; '.join(parts)}"


def _auto_describe_robot(role: str, active_predicates: list) -> str:
    """
    Describe a robot-role level-0 segment whose 'active_predicates' encode the
    phase (and optional current step) rather than scene-graph predicates.
    """
    role_name = humanize(role)
    state = {k: v for k, v in active_predicates}
    phase = state.get("Phase")
    step = state.get("Step")
    if phase and step:
        body = f"{phase} ({step})"
    elif phase:
        body = phase
    else:
        body = "idle"
    return f"{role_name}: {body}"


def _auto_describe_tool(role: str, active_predicates: list) -> str:
    """
    Describe a tool-role level-0 segment, whose 'active_predicates' encode
    (predicate, subject) pairs -- i.e. who is handling / using the tool.
    """
    role_name = humanize(role)
    if not active_predicates:
        return f"{role_name}: not in use"
    parts = []
    for pred, subj in active_predicates:
        parts.append(f"{humanize(subj)} {humanize_pred(pred)}")
    return f"{role_name}: {'; '.join(parts)}"


# ---------------------------------------------------------------------------
# Level-1: LLM-based grouping of level-0 segments into action steps
# ---------------------------------------------------------------------------

LEVEL1_SYSTEM_PROMPT = """\
You are a surgical workflow analyst. Always respond in English. \
You will receive a chronological list of \
level-0 activity segments for one person during a surgical procedure. Each \
segment represents a period where that person's active interactions stayed \
constant.

Your task: group consecutive segments into coherent "action steps". An action \
step is a sequence of related atomic activities that together accomplish a \
sub-task (e.g., "drilling the femur", "calibrating the robot", "preparing \
instruments").

Rules:
- Groups MUST be consecutive -- no skipping or reordering segments.
- Every segment must belong to exactly one group.
- Each group gets a 1-sentence natural language summary.
- Short idle gaps between related active segments should be included in the \
same group (the person briefly paused but continued the same task).
- Long idle periods (>60s with no active interactions) should generally be \
their own group or a boundary between groups.
- Output ONLY valid JSON, no other text.

Output format (JSON array):
[
  {
    "segment_ids": ["<first_seg_id>", ..., "<last_seg_id>"],
    "summary": "One sentence describing the action step."
  },
  ...
]"""


def _lazy_import_llm():
    """Import LLM utilities only when needed (avoids torch dependency for level-0 only runs)."""
    from scene_graph_utils import (
        build_chat_input,
        load_model,
        run_inference_batch,
        strip_think_tags,
    )
    return build_chat_input, load_model, run_inference_batch, strip_think_tags


def _format_l0_for_prompt(role: str, segments: List[Dict[str, Any]]) -> str:
    """Format a role's level-0 segments as the user message for the LLM."""
    role_name = humanize(role)
    lines = [f"Level-0 segments for {role_name}:\n"]
    for seg in segments:
        t_range = f"t={seg['time_start']}s-{seg['time_end']}s ({seg['duration_s']}s)"
        lines.append(f"  [{seg['segment_id']}] {t_range}: {seg['description']}")
    lines.append("\nGroup these into action steps. Output JSON only.")
    return "\n".join(lines)


def _clean_segment_ids(groups: List[Dict]) -> List[Dict]:
    """Strip brackets and whitespace from segment IDs (7B models copy them from the prompt)."""
    for g in groups:
        if "segment_ids" in g:
            g["segment_ids"] = [
                sid.strip().strip("[]") for sid in g["segment_ids"]
            ]
    return groups


def _parse_level1_json(raw_text: str) -> Optional[List[Dict]]:
    """Extract and parse JSON array from LLM output, tolerating markdown fences."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return _clean_segment_ids(result)
    except json.JSONDecodeError:
        pass
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            if isinstance(result, list):
                return _clean_segment_ids(result)
        except json.JSONDecodeError:
            pass
    return None



def _validate_level1_groups(
    groups: List[Dict], segment_ids: List[str], role: str,
) -> List[str]:
    """Check that groups cover all segments exactly once in order. Returns issues."""
    issues = []
    seen_ids = []
    for g in groups:
        ids = g.get("segment_ids", [])
        if not ids:
            issues.append(f"Empty group found for {role}")
            continue
        seen_ids.extend(ids)
        if not g.get("summary"):
            issues.append(f"Missing summary in group ending at {ids[-1]}")

    if seen_ids != segment_ids:
        missing = set(segment_ids) - set(seen_ids)
        extra = set(seen_ids) - set(segment_ids)
        if missing:
            issues.append(f"{role}: missing segments {missing}")
        if extra:
            issues.append(f"{role}: extra/unknown segments {extra}")
        if not missing and not extra and seen_ids != segment_ids:
            issues.append(f"{role}: segments out of order")
    return issues


_CHUNK_SIZE = 30
_CHUNK_OVERLAP = 3


def _call_llm_for_l1(
    build_chat_input, run_inference_batch, strip_think_tags,
    tokenizer, model, role: str, segments: List[Dict],
) -> Optional[List[Dict]]:
    """Send segments to LLM and return parsed groups, or None on failure."""
    user_msg = _format_l0_for_prompt(role, segments)
    prompt = build_chat_input(tokenizer, LEVEL1_SYSTEM_PROMPT, user_msg, enable_thinking=False)
    results = run_inference_batch(model, tokenizer, [prompt], max_new_tokens=4096)
    raw = results[0]
    cleaned = strip_think_tags(raw)
    groups = _parse_level1_json(cleaned)
    if groups is None:
        logger.warning("  %-25s  FAILED to parse LLM JSON output", humanize(role))
        logger.warning("  Raw output: %s", repr(raw[:300]))
    return groups


def _group_chunks(
    build_chat_input, run_inference_batch, strip_think_tags,
    tokenizer, model, role: str, l0_segs: List[Dict],
) -> Optional[List[Dict]]:
    """Split large segment lists into overlapping chunks, merge results."""
    all_groups: List[Dict] = []
    start = 0
    chunk_idx = 0

    while start < len(l0_segs):
        end = min(start + _CHUNK_SIZE, len(l0_segs))
        chunk = l0_segs[start:end]
        logger.info("    chunk %d: segments %d-%d (%d segs)",
                     chunk_idx, start, end - 1, len(chunk))

        groups = _call_llm_for_l1(
            build_chat_input, run_inference_batch, strip_think_tags,
            tokenizer, model, role, chunk,
        )
        if groups is None:
            return None

        if all_groups and _CHUNK_OVERLAP > 0:
            overlap_ids = {s["segment_id"] for s in l0_segs[start:start + _CHUNK_OVERLAP]}
            while all_groups and overlap_ids & set(all_groups[-1].get("segment_ids", [])):
                all_groups.pop()

        all_groups.extend(groups)
        start = end - _CHUNK_OVERLAP if end < len(l0_segs) else end
        chunk_idx += 1

    return all_groups


def build_level1(
    hierarchy: Dict[str, Any],
    model,
    tokenizer,
) -> None:
    """
    Use the LLM to group each role's level-0 segments into level-1 action steps.
    Modifies the hierarchy dict in-place, adding 'level1_segments' to each role.
    """
    build_chat_input, _, run_inference_batch, strip_think_tags = _lazy_import_llm()

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
            logger.info("  %-25s  1 level-1 segment (trivial, <=1 L0)", humanize(role))
            continue

        if len(l0_segs) > _CHUNK_SIZE:
            logger.info("  %-25s  %d L0 segments -> chunking (%d per chunk)",
                         humanize(role), len(l0_segs), _CHUNK_SIZE)
            groups = _group_chunks(
                build_chat_input, run_inference_batch, strip_think_tags,
                tokenizer, model, role, l0_segs,
            )
        else:
            groups = _call_llm_for_l1(
                build_chat_input, run_inference_batch, strip_think_tags,
                tokenizer, model, role, l0_segs,
            )

        if groups is None:
            role_data["level1_segments"] = []
            role_data["num_level1_segments"] = 0
            continue

        seg_ids = [s["segment_id"] for s in l0_segs]
        issues = _validate_level1_groups(groups, seg_ids, role)
        if issues:
            for issue in issues:
                logger.warning("  L1 validation: %s", issue)

        level1_segs = []
        for i, group in enumerate(groups):
            child_ids = group.get("segment_ids", [])
            children = [s for s in l0_segs if s["segment_id"] in child_ids]
            level1_segs.append({
                "segment_id": f"{role}_L1_{i:03d}",
                "role": role,
                "level": 1,
                "role_human": humanize(role),
                "segment_ids": child_ids,
                "time_start": children[0]["time_start"] if children else None,
                "time_end": children[-1]["time_end"] if children else None,
                "duration_s": (
                    (children[-1]["time_end"] - children[0]["time_start"] + 1)
                    if children and children[0]["time_start"] is not None
                    and children[-1]["time_end"] is not None
                    else None
                ),
                "summary": group.get("summary", ""),
            })

        role_data["level1_segments"] = level1_segs
        role_data["num_level1_segments"] = len(level1_segs)
        logger.info("  %-25s  %2d level-1 segments", humanize(role), len(level1_segs))


# ---------------------------------------------------------------------------
# Level-2: LLM-based grouping of level-1 segments into surgical phases
# ---------------------------------------------------------------------------

LEVEL2_SYSTEM_PROMPT = """\
You are a surgical workflow analyst. Always respond in English. \
You will receive a chronological list of \
level-1 "action step" segments for one person during a surgical procedure. Each \
action step groups several atomic activities that accomplish a sub-task.

Your task: group consecutive action steps into high-level "surgical phases". \
A phase represents a major stage of the procedure from this person's perspective \
(e.g., "patient preparation", "femoral bone work", "robot-assisted implant \
placement", "wound closure").

Rules:
- Groups MUST be consecutive -- no skipping or reordering segments.
- Every segment must belong to exactly one group.
- Each group gets a 1-sentence natural language summary describing the phase.
- A role with few action steps may have only 1-2 phases -- that is fine.
- Output ONLY valid JSON, no other text.

Output format (JSON array):
[
  {
    "segment_ids": ["<first_L1_id>", ..., "<last_L1_id>"],
    "summary": "One sentence describing the surgical phase."
  },
  ...
]"""


def _format_l1_for_prompt(role: str, segments: List[Dict[str, Any]]) -> str:
    """Format a role's level-1 segments as the user message for the level-2 LLM."""
    role_name = humanize(role)
    lines = [f"Level-1 action steps for {role_name}:\n"]
    for seg in segments:
        t_start = seg.get("time_start", "?")
        t_end = seg.get("time_end", "?")
        dur = seg.get("duration_s", "?")
        n_children = len(seg.get("segment_ids", []))
        lines.append(
            f"  [{seg['segment_id']}] t={t_start}s-{t_end}s ({dur}s, "
            f"{n_children} L0 segments): {seg.get('summary', '')}"
        )
    lines.append("\nGroup these into surgical phases. Output JSON only.")
    return "\n".join(lines)


def build_level2(
    hierarchy: Dict[str, Any],
    model,
    tokenizer,
) -> None:
    """
    Use the LLM to group each role's level-1 segments into level-2 surgical phases.
    Modifies the hierarchy dict in-place, adding 'level2_segments' to each role.
    """
    build_chat_input, _, run_inference_batch, strip_think_tags = _lazy_import_llm()

    for role, role_data in hierarchy["roles"].items():
        l1_segs = role_data.get("level1_segments", [])

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
            logger.info("  %-25s  1 level-2 segment (trivial, <=1 L1)", humanize(role))
            continue

        user_msg = _format_l1_for_prompt(role, l1_segs)
        prompt = build_chat_input(tokenizer, LEVEL2_SYSTEM_PROMPT, user_msg, enable_thinking=False)
        results = run_inference_batch(model, tokenizer, [prompt], max_new_tokens=4096)
        raw_output = results[0]
        cleaned = strip_think_tags(raw_output)
        groups = _parse_level1_json(cleaned)

        if groups is None:
            logger.warning("  %-25s  FAILED to parse L2 JSON output", humanize(role))
            logger.warning("  Raw output: %s", repr(raw_output[:300]))
            role_data["level2_segments"] = []
            role_data["num_level2_segments"] = 0
            continue

        l1_ids = [s["segment_id"] for s in l1_segs]
        issues = _validate_level1_groups(groups, l1_ids, role)
        if issues:
            for issue in issues:
                logger.warning("  L2 validation: %s", issue)

        level2_segs = []
        for i, group in enumerate(groups):
            child_ids = group.get("segment_ids", [])
            children = [s for s in l1_segs if s["segment_id"] in child_ids]
            level2_segs.append({
                "segment_id": f"{role}_L2_{i:03d}",
                "role": role,
                "level": 2,
                "role_human": humanize(role),
                "child_ids": child_ids,
                "time_start": children[0]["time_start"] if children else None,
                "time_end": children[-1]["time_end"] if children else None,
                "duration_s": (
                    (children[-1]["time_end"] - children[0]["time_start"] + 1)
                    if children and children[0]["time_start"] is not None
                    and children[-1]["time_end"] is not None
                    else None
                ),
                "summary": group.get("summary", ""),
            })

        role_data["level2_segments"] = level2_segs
        role_data["num_level2_segments"] = len(level2_segs)
        logger.info("  %-25s  %2d level-2 segments", humanize(role), len(level2_segs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build role-aware hierarchical scene graph representation."
    )
    parser.add_argument(
        "--take_dir", type=Path,
        default=Path("mm-or/MM-OR_data/MM-OR_processed/001_PKA"),
    )
    parser.add_argument(
        "--start_tp", type=str, default=None,
        help="Start at this relation-label timepoint id (e.g. 001114).",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Number of timepoints to process from start_tp.",
    )
    parser.add_argument(
        "--debounce", type=int, default=2,
        help="Frames required for a change to count (absorbs flickers, default: 2).",
    )
    parser.add_argument(
        "--no_robot_roles", action="store_true",
        help="Disable the synthetic robot_setup / robot_monitor roles.",
    )
    parser.add_argument(
        "--no_tool_roles", action="store_true",
        help="Disable the tool/instrument roles (saw, drill, hammer, ...).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("hierarchy_output/001_PKA_hierarchy.json"),
    )
    parser.add_argument(
        "--level1", action="store_true",
        help="Run level-1 grouping using an LLM (requires GPU).",
    )
    parser.add_argument(
        "--level2", action="store_true",
        help="Run level-2 grouping (implies --level1, requires GPU).",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-32B",
        help="HuggingFace model ID (used with --level1/--level2).",
    )
    parser.add_argument(
        "--hf_token", type=str,
        default="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",
        help="HuggingFace access token.",
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
    hierarchy = build_level0(
        entries, frame_map, debounce=args.debounce,
        take_dir=None if args.no_robot_roles else args.take_dir,
        include_tools=not args.no_tool_roles,
    )

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
    })

    # --- Debounce quality analysis ---
    if args.debounce > 1:
        logger.info("Running debounce analysis ...")
        debounce_report = analyze_debounce(entries, frame_map, debounce=args.debounce)
        hierarchy["debounce_analysis"] = debounce_report
        _log_debounce_report(debounce_report)

    # --level2 implies --level1
    if args.level2:
        args.level1 = True

    # --- Level-1: LLM-based grouping ---
    model = tokenizer = None
    if args.level1:
        logger.info("")
        logger.info("=== Building Level-1 (LLM grouping) ===")
        _, load_model, _, _ = _lazy_import_llm()
        model, tokenizer = load_model(args.model, args.hf_token)
        build_level1(hierarchy, model, tokenizer)

        total_l1 = sum(
            d.get("num_level1_segments", 0) for d in hierarchy["roles"].values()
        )
        hierarchy["metadata"]["total_level1_segments"] = total_l1

    # --- Level-2: LLM-based phase grouping ---
    if args.level2:
        logger.info("")
        logger.info("=== Building Level-2 (LLM phase grouping) ===")
        if model is None:
            _, load_model, _, _ = _lazy_import_llm()
            model, tokenizer = load_model(args.model, args.hf_token)
        build_level2(hierarchy, model, tokenizer)

        total_l2 = sum(
            d.get("num_level2_segments", 0) for d in hierarchy["roles"].values()
        )
        hierarchy["metadata"]["total_level2_segments"] = total_l2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hierarchy, indent=2))
    logger.info("Hierarchy saved to %s", args.output)


if __name__ == "__main__":
    main()
