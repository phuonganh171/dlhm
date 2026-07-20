"""
Assemble per-frame model predictions into a hierarchy JSON.

Takes a stream of (role, timepoint, l0_pred, l1_pred, l2_pred) tuples
and groups consecutive identical predictions into segments, producing
the same JSON structure as hierarchy_output/*.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PredictedFrame:
    """One per-frame prediction."""
    tp_id: str
    role: str
    role_human: str
    l0: str
    l1: str
    l2: str
    time: Optional[int] = None  # original_timestamp if available


def _group_consecutive(
    frames: List[PredictedFrame],
    key_fn,
) -> List[Tuple[str, int, int, List[PredictedFrame]]]:
    """
    Group consecutive frames where key_fn returns the same value.

    Returns list of (key_value, start_idx, end_idx, frame_list).
    """
    if not frames:
        return []

    groups: List[Tuple[str, int, int, List[PredictedFrame]]] = []
    current_key = key_fn(frames[0])
    start_idx = 0
    group_frames = [frames[0]]

    for i, frame in enumerate(frames[1:], 1):
        k = key_fn(frame)
        if k == current_key:
            group_frames.append(frame)
        else:
            groups.append((current_key, start_idx, i - 1, group_frames))
            current_key = k
            start_idx = i
            group_frames = [frame]

    groups.append((current_key, start_idx, len(frames) - 1, group_frames))
    return groups


def assemble_l2_segment(
    frames: List[PredictedFrame],
    role: str,
    l2_segment_idx: int = 0,
) -> Dict[str, Any]:
    """
    Assemble per-frame predictions for one (role, L2 segment) into a
    hierarchy subtree.

    Parameters
    ----------
    frames : sorted by timepoint, all belonging to the same L2 segment.
    role : role identifier.
    l2_segment_idx : index for generating segment IDs.

    Returns dict with level0_segments, level1_segments, level2_segments.
    """
    if not frames:
        return {"level0_segments": [], "level1_segments": [], "level2_segments": []}

    role_human = frames[0].role_human

    # L2: majority vote across all frames (should be consistent)
    l2_votes: Dict[str, int] = {}
    for f in frames:
        l2_votes[f.l2] = l2_votes.get(f.l2, 0) + 1
    l2_summary = max(l2_votes, key=l2_votes.get)  # type: ignore[arg-type]

    # Group by L1
    l1_groups = _group_consecutive(frames, lambda f: f.l1)

    l0_segments: List[Dict[str, Any]] = []
    l1_segments: List[Dict[str, Any]] = []
    l0_counter = 0
    l1_counter = 0

    for l1_text, l1_start, l1_end, l1_frames in l1_groups:
        l1_seg_id = f"{role}_L1_{l2_segment_idx:03d}_{l1_counter:03d}"
        l1_child_l0_ids: List[str] = []

        # Within this L1, group by L0
        l0_groups = _group_consecutive(l1_frames, lambda f: f.l0)

        for l0_text, l0_start_rel, l0_end_rel, l0_frames in l0_groups:
            l0_seg_id = f"{role}_L0_{l0_counter:03d}"
            l0_child_l0_ids_inner = l0_seg_id
            l1_child_l0_ids.append(l0_seg_id)

            l0_segments.append({
                "segment_id": l0_seg_id,
                "role": role,
                "role_human": role_human,
                "level": 0,
                "tp_start": l0_frames[0].tp_id,
                "tp_end": l0_frames[-1].tp_id,
                "description": l0_text,
                "time_start": l0_frames[0].time if l0_frames[0].time else 0,
                "time_end": l0_frames[-1].time if l0_frames[-1].time else 0,
                "duration_frames": len(l0_frames),
            })
            l0_counter += 1

        l1_segments.append({
            "segment_id": l1_seg_id,
            "role": role,
            "role_human": role_human,
            "level": 1,
            "segment_ids": l1_child_l0_ids,
            "summary": l1_text,
            "time_start": l1_frames[0].time if l1_frames[0].time else 0,
            "time_end": l1_frames[-1].time if l1_frames[-1].time else 0,
        })
        l1_counter += 1

    l2_seg_id = f"{role}_L2_{l2_segment_idx:03d}"
    l2_segment = {
        "segment_id": l2_seg_id,
        "role": role,
        "role_human": role_human,
        "level": 2,
        "child_ids": [s["segment_id"] for s in l1_segments],
        "summary": l2_summary,
        "time_start": frames[0].time if frames[0].time else 0,
        "time_end": frames[-1].time if frames[-1].time else 0,
    }

    return {
        "level0_segments": l0_segments,
        "level1_segments": l1_segments,
        "level2_segments": [l2_segment],
    }


def assemble_full_hierarchy(
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Assemble all per-frame predictions into a full hierarchy JSON.

    Parameters
    ----------
    predictions : list of dicts, each with keys:
        - role, role_human, tp_id, pred_l0, pred_l1, pred_l2, l2_segment_id
        - optionally: time (original_timestamp)

    Returns a dict matching the hierarchy_output JSON schema.
    """
    # Group by (role, l2_segment_id)
    groups: Dict[Tuple[str, str], List[PredictedFrame]] = {}
    for p in predictions:
        key = (p["role"], p["l2_segment_id"])
        frame = PredictedFrame(
            tp_id=p["tp_id"],
            role=p["role"],
            role_human=p.get("role_human", p["role"]),
            l0=p["pred_l0"],
            l1=p["pred_l1"],
            l2=p["pred_l2"],
            time=p.get("time"),
        )
        groups.setdefault(key, []).append(frame)

    # Sort each group by timepoint
    for key in groups:
        groups[key].sort(key=lambda f: f.tp_id)

    # Assemble per role
    roles_data: Dict[str, Dict[str, Any]] = {}
    for (role, l2_seg_id) in sorted(groups.keys()):
        role_frames = groups[(role, l2_seg_id)]
        if role not in roles_data:
            roles_data[role] = {
                "level0_segments": [],
                "level1_segments": [],
                "level2_segments": [],
            }

        # Extract L2 segment index from segment ID (e.g., "role_L2_002" → 2)
        try:
            l2_idx = int(l2_seg_id.split("_L2_")[1])
        except (IndexError, ValueError):
            l2_idx = len(roles_data[role]["level2_segments"])

        subtree = assemble_l2_segment(role_frames, role, l2_idx)

        roles_data[role]["level0_segments"].extend(subtree["level0_segments"])
        roles_data[role]["level1_segments"].extend(subtree["level1_segments"])
        roles_data[role]["level2_segments"].extend(subtree["level2_segments"])

    return {"roles": roles_data}


def parse_model_output(text: str) -> Tuple[str, str, str]:
    """
    Parse a model's text output into (l0, l1, l2) strings.

    Expected format: "L0: <text> | L1: <text> | L2: <text>"
    """
    parts = text.split("|")
    l0 = l1 = l2 = ""
    for part in parts:
        part = part.strip()
        if part.startswith("L0:"):
            l0 = part[3:].strip()
        elif part.startswith("L1:"):
            l1 = part[3:].strip()
        elif part.startswith("L2:"):
            l2 = part[3:].strip()
    return l0, l1, l2
