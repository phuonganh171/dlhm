"""
MM-OR dataset path helpers.

Supports both the local repo layout (mm-or/MM-OR_data/MM-OR_processed) and
NAS-mounted data (rclone mount of nas:ge42faj/MM-OR_data/MM-OR_processed).

NAS differences handled here:
  - No take_transcripts/ on NAS (SRT is optional)
  - Split sessions use suffixed ids (012_1_PKA, 012_2_PKA) for robot phase /
    screen summaries while the take folder may be 012_PKA
  - Combined zip folders (015-018_PKA, …) have color images but no scene graphs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

METADATA_DIR_NAMES = frozenset({
    "screen_summaries",
    "take_transcripts",
    "take_timestamp_to_robot_phase",
    "take_audios",
})

_TAKE_DIR_RE = re.compile(r"^\d{3}(?:[-_]\d{3})*(?:_\d+)?_(?:PKA|TKA)$")


def get_processed_root(explicit: Optional[Path] = None) -> Path:
    """Return MM-OR_processed root, preferring explicit arg then env then NAS mount."""
    if explicit is not None:
        return Path(explicit)
    if env := os.environ.get("MM_OR_PROCESSED_ROOT"):
        return Path(env)
    nas_default = Path("/tmp/nhatvu/nas_mount/MM-OR_data/MM-OR_processed")
    if _looks_like_processed_root(nas_default):
        return nas_default
    return Path("mm-or/MM-OR_data/MM-OR_processed")


def _looks_like_processed_root(path: Path) -> bool:
    return path.is_dir() and (path / "001_PKA").is_dir()


def _take_suffix(take_name: str) -> str:
    """Return PKA or TKA from a take name like 012_PKA."""
    return take_name.rsplit("_", 1)[-1]


def _take_number_prefix(take_name: str) -> str:
    """Return leading surgery number(s), e.g. '012' from '012_PKA'."""
    return take_name.split("_", 1)[0]


def is_take_dir_name(name: str) -> bool:
    return bool(_TAKE_DIR_RE.match(name))


def resolve_take_dir(processed_root: Path, take_name: str) -> Path:
    return processed_root / take_name


def is_processable_take(take_dir: Path, min_labels: int = 10) -> bool:
    """True when the take has scene graphs and a frame map."""
    labels_dir = take_dir / "relation_labels"
    ts_file = take_dir / "timestamp_to_pcd_and_frames_list.json"
    if not labels_dir.is_dir() or not ts_file.is_file():
        return False
    n_labels = sum(1 for _ in labels_dir.glob("[0-9]*.json"))
    return n_labels >= min_labels


def list_processable_takes(
    processed_root: Optional[Path] = None,
    min_labels: int = 10,
) -> List[str]:
    """List take folder names that have relation_labels + frame map."""
    root = get_processed_root(processed_root)
    if not root.is_dir():
        logger.warning("Processed root does not exist: %s", root)
        return []

    takes: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in METADATA_DIR_NAMES:
            continue
        if not is_take_dir_name(child.name):
            continue
        if is_processable_take(child, min_labels=min_labels):
            takes.append(child.name)
    return takes


def iter_robot_phase_files(processed_root: Path, take_name: str) -> Iterator[Path]:
    """
    Yield robot-phase JSON files for a take.

    Prefers <take>.json; falls back to split-session files such as
    012_1_PKA.json / 012_2_PKA.json for take 012_PKA.
    """
    rp_dir = processed_root / "take_timestamp_to_robot_phase"
    if not rp_dir.is_dir():
        return

    primary = rp_dir / f"{take_name}.json"
    if primary.is_file():
        yield primary
        return

    suffix = _take_suffix(take_name)
    num_prefix = _take_number_prefix(take_name)
    seen: set = set()
    for candidate in sorted(rp_dir.glob(f"{num_prefix}*_{suffix}.json")):
        if candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            yield candidate


def iter_screen_summary_dirs(processed_root: Path, take_name: str) -> Iterator[Path]:
    """
    Yield screen-summary directories for a take.

    Prefers screen_summaries/<take>/; falls back to split dirs like
    screen_summaries/012_1_PKA/ for take 012_PKA.
    """
    ss_root = processed_root / "screen_summaries"
    if not ss_root.is_dir():
        return

    primary = ss_root / take_name
    if primary.is_dir():
        yield primary
        return

    suffix = _take_suffix(take_name)
    num_prefix = _take_number_prefix(take_name)
    seen: set = set()
    for candidate in sorted(ss_root.iterdir()):
        if not candidate.is_dir():
            continue
        name = candidate.name
        if name in seen:
            continue
        if name.startswith(f"{num_prefix}") and name.endswith(f"_{suffix}"):
            seen.add(name)
            yield candidate


def resolve_transcript_srt(processed_root: Path, take_name: str) -> Optional[Path]:
    """Return SRT path if present (often missing on NAS)."""
    for candidate in (
        processed_root / "take_transcripts" / f"{take_name}.srt",
        processed_root.parent / "take_transcripts" / f"{take_name}.srt",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_merged_robot_phase(processed_root: Path, take_name: str) -> dict:
    merged: dict = {}
    for fpath in iter_robot_phase_files(processed_root, take_name):
        try:
            merged.update(json.loads(fpath.read_bytes().decode("utf-8")))
        except Exception as exc:
            logger.warning("Failed to read robot phase %s: %s", fpath, exc)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="MM-OR dataset path utilities")
    parser.add_argument(
        "--root", type=Path, default=None,
        help="MM-OR_processed root (default: env MM_OR_PROCESSED_ROOT or auto-detect)",
    )
    parser.add_argument(
        "--list-takes", action="store_true",
        help="Print processable take names, one per line",
    )
    parser.add_argument(
        "--min-labels", type=int, default=10,
        help="Minimum relation-label files required (default: 10)",
    )
    parser.add_argument("take", nargs="?", help="Show resolved paths for one take")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = get_processed_root(args.root)

    if args.list_takes:
        for name in list_processable_takes(root, min_labels=args.min_labels):
            print(name)
        return

    if args.take:
        take_dir = resolve_take_dir(root, args.take)
        print(f"processed_root: {root}")
        print(f"take_dir:       {take_dir}")
        print(f"processable:    {is_processable_take(take_dir, args.min_labels)}")
        print(f"srt:            {resolve_transcript_srt(root, args.take)}")
        print("robot_phase:")
        for p in iter_robot_phase_files(root, args.take):
            print(f"  {p}")
        print("screen_summaries:")
        for p in iter_screen_summary_dirs(root, args.take):
            print(f"  {p}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
