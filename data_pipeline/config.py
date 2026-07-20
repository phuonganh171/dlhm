"""
Shared configuration for the data pipeline.

Paths, data splits, camera keys, and other constants used across all
pipeline stages (sample building, evaluation, trivial baselines).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Hierarchy output directory
# ---------------------------------------------------------------------------

HIERARCHY_DIR = PROJECT_ROOT / "hierarchy_output"
HIERARCHY_PATTERN = "{take}_hierarchy_qwen27b.json"

# ---------------------------------------------------------------------------
# Camera mapping  (same as scene_graph_utils.COLORIMAGE_CAMERA_KEYS)
# ---------------------------------------------------------------------------

CAMERA_KEYS: Dict[str, str] = {
    "azure": "camera01",
    "simstation": "camera02",
    "trackercam": "camera03",
    "tracker": "camera04",
}

ALL_CAMERAS: List[str] = list(CAMERA_KEYS.values())

# ---------------------------------------------------------------------------
# Image filename template
# ---------------------------------------------------------------------------

COLORIMAGE_TEMPLATE = "colorimage/{cam}_colorimage-{frame_id:06d}.jpg"

# ---------------------------------------------------------------------------
# Data split — MM-OR official split by surgery session
#
# The 18 annotated takes span surgeries 1-11 plus extra short sessions.
# Verify against the MM-OR paper's appendix before final training.
# ---------------------------------------------------------------------------

SPLIT_TRAIN: List[str] = [
    "001_PKA", "002_PKA", "003_TKA", "004_PKA", "005_TKA",
    "006_PKA", "007_TKA", "008_PKA", "009_TKA",
]

SPLIT_VAL: List[str] = [
    "010_PKA", "011_TKA", "013_PKA",
]

SPLIT_TEST: List[str] = [
    "014_PKA", "033_PKA", "035_PKA", "036_PKA", "037_TKA", "038_TKA",
]

ALL_TAKES: List[str] = SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST

SPLIT_MAP: Dict[str, str] = {}
for _t in SPLIT_TRAIN:
    SPLIT_MAP[_t] = "train"
for _t in SPLIT_VAL:
    SPLIT_MAP[_t] = "val"
for _t in SPLIT_TEST:
    SPLIT_MAP[_t] = "test"

# ---------------------------------------------------------------------------
# MM-OR processed root resolution (reuses mm_or_dataset logic)
# ---------------------------------------------------------------------------


def get_processed_root(explicit: Path | None = None) -> Path:
    """Return MM-OR_processed root, preferring explicit arg → env → default."""
    if explicit is not None:
        return Path(explicit)
    if env := os.environ.get("MM_OR_PROCESSED_ROOT"):
        return Path(env)
    local = PROJECT_ROOT / "mm-or" / "MM-OR_data" / "MM-OR_processed"
    if local.is_dir() and (local / "001_PKA").is_dir():
        return local
    nas_default = Path(f"/tmp/{os.environ.get('USER', 'user')}/nas_mount/MM-OR_data/MM-OR_processed")
    if nas_default.is_dir():
        return nas_default
    return local


# ---------------------------------------------------------------------------
# Default output directory for generated samples
# ---------------------------------------------------------------------------

SAMPLES_DIR = PROJECT_ROOT / "data_pipeline" / "samples"

# ---------------------------------------------------------------------------
# Temporal memory defaults
# ---------------------------------------------------------------------------

SHORT_TERM_WINDOW = 5  # last N changelog entries for short-term memory
