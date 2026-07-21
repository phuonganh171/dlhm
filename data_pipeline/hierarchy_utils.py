"""
Utilities for loading and indexing hierarchy JSONs.

Provides:
  - HierarchyIndex: per-role, per-frame lookup of (L0, L1, L2) labels
  - Frame-map loading and colorimage path resolution
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ALL_CAMERAS,
    AZURE_FRAME_KEY,
    COLORIMAGE_TEMPLATE,
    HIERARCHY_DIR,
    HIERARCHY_PATTERN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class L0Segment:
    segment_id: str
    role: str
    role_human: str
    tp_start: str          # zero-padded, e.g. "000872"
    tp_end: str
    description: str
    time_start: int        # original_timestamp (seconds)
    time_end: int

    @property
    def tp_start_int(self) -> int:
        return int(self.tp_start)

    @property
    def tp_end_int(self) -> int:
        return int(self.tp_end)


@dataclass
class L1Segment:
    segment_id: str
    role: str
    role_human: str
    child_l0_ids: List[str]
    summary: str
    time_start: int
    time_end: int


@dataclass
class L2Segment:
    segment_id: str
    role: str
    role_human: str
    child_l1_ids: List[str]
    summary: str
    time_start: int
    time_end: int


@dataclass
class FrameLabels:
    """Ground-truth labels for a single (role, timepoint) pair."""
    tp_id: str
    role: str
    role_human: str
    l0_description: str
    l1_summary: str
    l2_summary: str
    l0_segment_id: str
    l1_segment_id: str
    l2_segment_id: str


@dataclass
class RoleHierarchy:
    """All segments for one role in one take."""
    role: str
    role_human: str
    l0_segments: List[L0Segment] = field(default_factory=list)
    l1_segments: List[L1Segment] = field(default_factory=list)
    l2_segments: List[L2Segment] = field(default_factory=list)
    # fast lookup indexes (built by HierarchyIndex)
    l0_by_id: Dict[str, L0Segment] = field(default_factory=dict)
    l1_by_id: Dict[str, L1Segment] = field(default_factory=dict)
    l2_by_id: Dict[str, L2Segment] = field(default_factory=dict)
    # L0 segment_id → parent L1 segment_id
    l0_to_l1: Dict[str, str] = field(default_factory=dict)
    # L1 segment_id → parent L2 segment_id
    l1_to_l2: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HierarchyIndex
# ---------------------------------------------------------------------------

class HierarchyIndex:
    """
    Index a hierarchy JSON for fast per-frame label lookup.

    Usage::

        idx = HierarchyIndex.from_file("hierarchy_output/001_PKA_hierarchy_qwen27b.json")
        labels = idx.frame_labels("head_surgeon", "000872")
        # labels.l0_description, labels.l1_summary, labels.l2_summary
    """

    def __init__(self, data: dict, take: str = ""):
        self.take = take
        self.metadata = data.get("metadata", {})
        self.roles: Dict[str, RoleHierarchy] = {}
        self._tp_to_l0: Dict[str, Dict[int, L0Segment]] = {}  # role → {tp_int → L0}

        for role_key, role_data in data.get("roles", {}).items():
            rh = self._parse_role(role_key, role_data)
            self.roles[role_key] = rh
            self._build_tp_index(rh)

    # ----- construction helpers -----

    @classmethod
    def from_file(cls, path: Path | str) -> "HierarchyIndex":
        path = Path(path)
        take = path.stem.replace("_hierarchy_qwen27b", "")
        data = json.loads(path.read_bytes().decode("utf-8"))
        return cls(data, take=take)

    @classmethod
    def from_take(cls, take: str, hierarchy_dir: Path | None = None) -> "HierarchyIndex":
        hdir = hierarchy_dir or HIERARCHY_DIR
        path = hdir / HIERARCHY_PATTERN.format(take=take)
        return cls.from_file(path)

    # ----- parsing -----

    @staticmethod
    def _parse_role(role_key: str, data: dict) -> RoleHierarchy:
        first_l0 = data.get("level0_segments", [{}])[0] if data.get("level0_segments") else {}
        role_human = first_l0.get("role_human", role_key)

        l0s = [
            L0Segment(
                segment_id=s["segment_id"],
                role=role_key,
                role_human=s.get("role_human", role_human),
                tp_start=s["tp_start"],
                tp_end=s["tp_end"],
                description=s["description"],
                time_start=s["time_start"],
                time_end=s["time_end"],
            )
            for s in data.get("level0_segments", [])
        ]

        l1s = [
            L1Segment(
                segment_id=s["segment_id"],
                role=role_key,
                role_human=s.get("role_human", role_human),
                child_l0_ids=s.get("segment_ids", []),
                summary=s["summary"],
                time_start=s["time_start"],
                time_end=s["time_end"],
            )
            for s in data.get("level1_segments", [])
        ]

        l2s = [
            L2Segment(
                segment_id=s["segment_id"],
                role=role_key,
                role_human=s.get("role_human", role_human),
                child_l1_ids=s.get("child_ids", []),
                summary=s["summary"],
                time_start=s["time_start"],
                time_end=s["time_end"],
            )
            for s in data.get("level2_segments", [])
        ]

        rh = RoleHierarchy(role=role_key, role_human=role_human,
                           l0_segments=l0s, l1_segments=l1s, l2_segments=l2s)
        rh.l0_by_id = {s.segment_id: s for s in l0s}
        rh.l1_by_id = {s.segment_id: s for s in l1s}
        rh.l2_by_id = {s.segment_id: s for s in l2s}

        for l1 in l1s:
            for child_id in l1.child_l0_ids:
                rh.l0_to_l1[child_id] = l1.segment_id
        for l2 in l2s:
            for child_id in l2.child_l1_ids:
                rh.l1_to_l2[child_id] = l2.segment_id

        return rh

    def _build_tp_index(self, rh: RoleHierarchy) -> None:
        """Map every timepoint int → its L0 segment for O(1) lookup."""
        tp_map: Dict[int, L0Segment] = {}
        for l0 in rh.l0_segments:
            for tp in range(l0.tp_start_int, l0.tp_end_int + 1):
                tp_map[tp] = l0
        self._tp_to_l0[rh.role] = tp_map

    # ----- public API -----

    def frame_labels(self, role: str, tp_id: str) -> Optional[FrameLabels]:
        """
        Return (L0, L1, L2) ground-truth labels for one (role, timepoint).

        Returns None if the role or timepoint is not covered.
        """
        rh = self.roles.get(role)
        if rh is None:
            return None

        tp_int = int(tp_id)
        l0 = self._tp_to_l0.get(role, {}).get(tp_int)
        if l0 is None:
            return None

        l1_id = rh.l0_to_l1.get(l0.segment_id)
        if l1_id is None:
            return None
        l1 = rh.l1_by_id[l1_id]

        l2_id = rh.l1_to_l2.get(l1_id)
        if l2_id is None:
            return None
        l2 = rh.l2_by_id[l2_id]

        return FrameLabels(
            tp_id=tp_id,
            role=role,
            role_human=rh.role_human,
            l0_description=l0.description,
            l1_summary=l1.summary,
            l2_summary=l2.summary,
            l0_segment_id=l0.segment_id,
            l1_segment_id=l1.segment_id,
            l2_segment_id=l2.segment_id,
        )

    def l2_timepoints(self, role: str, l2_segment_id: str) -> List[str]:
        """
        Return all timepoint IDs (zero-padded) that fall within a given
        L2 segment for a role, in chronological order.

        Walks L2 → child L1s → child L0s → expand tp_start..tp_end.
        """
        rh = self.roles.get(role)
        if rh is None:
            return []
        l2 = rh.l2_by_id.get(l2_segment_id)
        if l2 is None:
            return []

        tps: List[int] = []
        for l1_id in l2.child_l1_ids:
            l1 = rh.l1_by_id.get(l1_id)
            if l1 is None:
                continue
            for l0_id in l1.child_l0_ids:
                l0 = rh.l0_by_id.get(l0_id)
                if l0 is None:
                    continue
                tps.extend(range(l0.tp_start_int, l0.tp_end_int + 1))

        tps_sorted = sorted(set(tps))
        return [f"{t:06d}" for t in tps_sorted]

    def iter_l2_segments(self, role: str) -> List[L2Segment]:
        """Return all L2 segments for a role."""
        rh = self.roles.get(role)
        return rh.l2_segments if rh else []

    @property
    def role_names(self) -> List[str]:
        return list(self.roles.keys())


# ---------------------------------------------------------------------------
# Frame-map and image-path resolution
# ---------------------------------------------------------------------------

def load_frame_map(take_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Index-based mapping: tp_id (zero-padded) → {camera_key: frame_id, ...}.

    Replicates scene_graph_utils.load_frame_map but lives here so the
    pipeline doesn't depend on the root-level module.
    """
    fpath = take_dir / "timestamp_to_pcd_and_frames_list.json"
    data = json.loads(fpath.read_bytes().decode("utf-8"))
    return {f"{i:06d}": entry[1] for i, entry in enumerate(data)}


def resolve_image_paths(
    take_dir: Path,
    frame_map: Dict[str, Dict[str, Any]],
    tp_id: str,
    cameras: List[str] | None = None,
) -> List[str]:
    """
    Return colorimage paths for all Azure Kinect views at a timepoint.

    Uses frame_map[tp][azure] as the shared frame id for camera01–camera04
    (relative to take_dir), e.g. ``colorimage/camera01_colorimage-000329.jpg``.
    """
    _ = take_dir  # paths are relative; existence is checked by the caller
    cameras = cameras or ALL_CAMERAS
    raw = frame_map.get(tp_id)
    if raw is None:
        return []

    frame_id = raw.get(AZURE_FRAME_KEY)
    if isinstance(frame_id, dict):
        frame_id = frame_id.get("frame_id", frame_id.get("id"))
    if frame_id is None:
        return []

    fid = int(frame_id)
    return [
        COLORIMAGE_TEMPLATE.format(cam=cam_id, frame_id=fid)
        for cam_id in cameras
    ]


def all_colorimages_exist(take_dir: Path, rel_paths: List[str]) -> bool:
    """True iff every path relative to take_dir exists as a file."""
    return bool(rel_paths) and all((take_dir / p).is_file() for p in rel_paths)


def resolve_image_paths_absolute(
    take_dir: Path,
    frame_map: Dict[str, Dict[str, Any]],
    tp_id: str,
    cameras: List[str] | None = None,
) -> List[str]:
    """Same as resolve_image_paths but returns absolute paths."""
    rel = resolve_image_paths(take_dir, frame_map, tp_id, cameras)
    return [str(take_dir / p) for p in rel]
