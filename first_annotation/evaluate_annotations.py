"""
Automated quality checks for atomic-action annotation JSONL files.

Usage:
    python evaluate_annotations.py \
        --input annotation_output/annotations_001_PKA_600.jsonl

    # also write flagged windows to a file:
    python evaluate_annotations.py \
        --input annotation_output/annotations_001_PKA_600.jsonl \
        --output annotation_output/evaluation_report.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scene_graph_utils import ENTITY_NAMES, PREDICATE_NAMES

# Human name the LLM was told to use → expected raw entity id.
# Built directly from ENTITY_NAMES — no extra aliases needed.
ROLE_CHECKS: dict = {human.lower(): raw_id for raw_id, human in ENTITY_NAMES.items()}

# Any predicate that isn't purely spatial/static is considered "active".
# Derived directly from PREDICATE_NAMES so it stays in sync automatically.
PASSIVE_PREDICATES = {"CloseTo", "LyingOn"}
ACTIVE_PREDICATES = set(PREDICATE_NAMES.keys()) - PASSIVE_PREDICATES


# ---------------------------------------------------------------------------
# Per-window checker
# ---------------------------------------------------------------------------

def check_annotation(window: dict) -> list[str]:
    issues = []
    summary = window.get("atomic_action_summary", "").strip()
    graphs = window["scene_graphs"]
    sl = summary.lower()

    # ------------------------------------------------------------------ #
    # 1. FORMAT
    # ------------------------------------------------------------------ #

    if not summary:
        issues.append("FORMAT: empty summary")
        return issues  # no point running further checks

    sentences = [s.strip() for s in summary.split(".") if s.strip()]
    if len(sentences) > 2:
        issues.append(f"FORMAT: {len(sentences)} sentences (expected ≤2)")

    # ------------------------------------------------------------------ #
    # 2. ENTITY COVERAGE — active actors should be named
    # ------------------------------------------------------------------ #

    active_entities: set[str] = set()
    for graph in graphs:
        for subj, pred, obj in graph:
            if pred in ACTIVE_PREDICATES:
                active_entities.add(subj)

    for entity_id in active_entities:
        human = ENTITY_NAMES.get(entity_id, entity_id).lower()
        # Accept any substring match (e.g. "robot technician" matches "robot technician (MPS)")
        if not any(part in sl for part in human.split("(")):
            issues.append(
                f"COVERAGE: active entity '{entity_id}' ({human}) not mentioned in summary"
            )

    # ------------------------------------------------------------------ #
    # 3. CHANGE DETECTION
    # ------------------------------------------------------------------ #

    graph_sets = [frozenset(tuple(t) for t in g) for g in graphs]
    has_changes = any(graph_sets[i] != graph_sets[i + 1] for i in range(len(graph_sets) - 1))

    if has_changes:
        # Check all consecutive pairs so mid-window transitions aren't missed
        ever_appeared:    set = set()
        ever_disappeared: set = set()
        for i in range(len(graph_sets) - 1):
            ever_disappeared |= graph_sets[i] - graph_sets[i + 1]
            ever_appeared    |= graph_sets[i + 1] - graph_sets[i]

        def action_mentioned(subj: str, pred: str) -> bool:
            """True if the summary references the action or actor in any common word form."""
            pred_word = PREDICATE_NAMES[pred].replace("is ", "")  # e.g. "assisting"
            # Also match root form: "assisting" → "assist", "drilling" → "drill"
            pred_root = pred_word.rstrip("ing").rstrip("e")  # "assist", "drill", "saw"
            subj_word = ENTITY_NAMES.get(subj, subj).lower()
            return any(w in sl for w in (pred_word, pred_root, subj_word))

        for subj, pred, obj in ever_disappeared:
            if pred in ACTIVE_PREDICATES:
                if not action_mentioned(subj, pred):
                    issues.append(
                        f"CHANGE: ({subj}, {pred}, {obj}) stopped but not reflected in summary"
                    )

        for subj, pred, obj in ever_appeared:
            if pred in ACTIVE_PREDICATES:
                if not action_mentioned(subj, pred):
                    issues.append(
                        f"CHANGE: ({subj}, {pred}, {obj}) started but not reflected in summary"
                    )

    # ------------------------------------------------------------------ #
    # 4. HALLUCINATION — named roles not present in scene graphs
    # ------------------------------------------------------------------ #

    all_entities: set[str] = set()
    for graph in graphs:
        for subj, pred, obj in graph:
            all_entities.add(subj)
            all_entities.add(obj)

    for role_name, entity_id in ROLE_CHECKS.items():
        if role_name in sl and entity_id not in all_entities:
            issues.append(
                f"HALLUCINATION: '{role_name}' mentioned but '{entity_id}' absent from scene graphs"
            )

    return issues


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_all(records: list[dict]) -> dict:
    stats: dict = defaultdict(int)
    flagged = []

    for i, window in enumerate(records):
        # Re-use pre-computed issues if already embedded in the record
        issues = window.get("issues") or check_annotation(window)
        for issue in issues:
            category = issue.split(":")[0]
            stats[category] += 1
        if issues:
            flagged.append({
                "window_index":      i,
                "window_start_tp":   window.get("window_start_tp"),
                "window_end_tp":     window.get("window_end_tp"),
                "original_timestamps": window.get("original_timestamps"),
                "atomic_action_summary": window.get("atomic_action_summary"),
                "issues":            issues,
            })

    return {
        "total_windows":        len(records),
        "windows_with_issues":  len(flagged),
        "issue_breakdown":      dict(sorted(stats.items(), key=lambda x: -x[1])),
        "flagged_windows":      flagged,
    }


def format_report_text(report: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  Annotation Quality Report")
    lines.append("=" * 60)
    lines.append(f"  Total windows    : {report['total_windows']}")
    pct = 100 * report["windows_with_issues"] // max(report["total_windows"], 1)
    lines.append(f"  Windows flagged  : {report['windows_with_issues']} ({pct}%)")
    lines.append("")
    lines.append("  Issue breakdown:")
    for category, count in report["issue_breakdown"].items():
        lines.append(f"    {category:<20} {count}")
    lines.append("")
    lines.append(f"  All flagged windows ({len(report['flagged_windows'])}):")
    for entry in report["flagged_windows"]:
        ts = entry["original_timestamps"]
        t_range = f"t={ts[0]}s–{ts[-1]}s" if ts else "?"
        lines.append(f"\n  [{entry['window_index']:>3}] {t_range}")
        lines.append(f"        Summary : {entry['atomic_action_summary'][:120]!r}")
        for issue in entry["issues"]:
            lines.append(f"        !  {issue}")
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def print_report(report: dict) -> None:
    print(format_report_text(report))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated quality checks on annotation JSONL."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input .jsonl annotation file.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to save full JSON report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    records = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]

    report = evaluate_all(records)
    print_report(report)

    # Save full JSON report
    json_path = args.output or args.input.with_name(args.input.stem + "_evaluation.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    print(f"JSON report saved to: {json_path}")
