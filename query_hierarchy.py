"""
Query-answering system over the role-aware hierarchical scene graph.

Loads the hierarchy JSON produced by build_hierarchy.py, retrieves relevant
segments based on a user query (role, time range, granularity), and uses
Qwen3-32B (or any HF model) to generate a natural-language answer.

Usage (interactive):
    source /tmp/nhatvu/.venv/bin/activate && \
    python3 query_hierarchy.py \
        --hierarchy hierarchy_output/001_PKA_hierarchy.json \
        --model Qwen/Qwen3-32B

Usage (single query):
    python3 query_hierarchy.py \
        --hierarchy hierarchy_output/001_PKA_hierarchy.json \
        --model Qwen/Qwen3-32B \
        --query "What did the head surgeon do in the first 15 minutes?"
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scene_graph_utils import humanize, ENTITY_NAMES, ROLE_ENTITIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hierarchy loader
# ---------------------------------------------------------------------------

def load_hierarchy(path: Path) -> Dict[str, Any]:
    """Load the hierarchy JSON file."""
    data = json.loads(path.read_text())
    logger.info(
        "Loaded hierarchy: %d roles, time range %s",
        data["metadata"]["total_roles"],
        data["metadata"].get("time_range", "unknown"),
    )
    return data


# ---------------------------------------------------------------------------
# Query parser
# ---------------------------------------------------------------------------

ROLE_KEYWORDS: Dict[str, str] = {}
for _entity_id, _human_name in ENTITY_NAMES.items():
    if _entity_id in ROLE_ENTITIES:
        for token in _human_name.lower().split():
            ROLE_KEYWORDS[token] = _entity_id
        ROLE_KEYWORDS[_entity_id.lower()] = _entity_id
        ROLE_KEYWORDS[_human_name.lower()] = _entity_id
ROLE_KEYWORDS.update({
    "surgeon": "head_surgeon",
    "head surgeon": "head_surgeon",
    "assistant": "assistant_surgeon",
    "scrub nurse": "nurse",
    "scrub": "nurse",
    "circulator": "circulator",
    "anaesthetist": "anest",
    "anesthetist": "anest",
    "robot": "mps",
    "mps": "mps",
    "technician": "mps",
})

LEVEL_KEYWORDS = {
    0: {"detail", "detailed", "exactly", "atomic", "fine", "fine-grained",
        "level-0", "level0", "l0", "moment", "second"},
    1: {"step", "steps", "action", "actions", "task", "tasks",
        "level-1", "level1", "l1", "sub-task"},
    2: {"phase", "phases", "overview", "summary", "high-level", "general",
        "level-2", "level2", "l2", "stage", "stages", "broad"},
}

_TIME_PATTERN = re.compile(
    r"(?:t\s*=?\s*|at\s+|from\s+|between\s+|time\s+)?"
    r"(\d+)\s*s?"
    r"(?:\s*[-–to]+\s*(\d+)\s*s?)?",
    re.IGNORECASE,
)
_MINUTE_PATTERN = re.compile(
    r"(?:first|last|initial|final)?\s*(\d+)\s*min(?:ute)?s?",
    re.IGNORECASE,
)


def parse_query(query: str, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured intent from a natural-language query.

    Returns {
        "roles": [list of role entity ids, or empty for all],
        "level": int or None (auto-detect),
        "time_start": int or None,
        "time_end": int or None,
        "raw_query": str,
    }
    """
    q_lower = query.lower()

    # --- Roles ---
    roles = []
    available_roles = set(hierarchy["roles"].keys())
    for phrase, entity_id in sorted(ROLE_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if phrase in q_lower and entity_id in available_roles and entity_id not in roles:
            roles.append(entity_id)

    interaction_keywords = {"interaction", "between", "together", "cooperat", "collaborat"}
    is_cross_role = any(kw in q_lower for kw in interaction_keywords)

    # --- Level ---
    level = None
    for lvl, keywords in LEVEL_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            level = lvl
            break

    # --- Time range ---
    time_range = hierarchy["metadata"].get("time_range", [None, None])
    global_start = time_range[0] if time_range else None
    time_start = None
    time_end = None

    minute_match = _MINUTE_PATTERN.search(query)
    if minute_match:
        minutes = int(minute_match.group(1))
        if "last" in q_lower or "final" in q_lower:
            if time_range and time_range[1]:
                time_end = time_range[1]
                time_start = time_range[1] - minutes * 60
        else:
            if global_start is not None:
                time_start = global_start
                time_end = global_start + minutes * 60
    else:
        time_match = _TIME_PATTERN.search(query)
        if time_match:
            time_start = int(time_match.group(1))
            if time_match.group(2):
                time_end = int(time_match.group(2))

    return {
        "roles": roles,
        "level": level,
        "time_start": time_start,
        "time_end": time_end,
        "is_cross_role": is_cross_role,
        "raw_query": query,
    }


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def _segments_in_range(
    segments: List[Dict], time_start: Optional[int], time_end: Optional[int],
) -> List[Dict]:
    """Filter segments overlapping the given time range."""
    if time_start is None and time_end is None:
        return segments
    result = []
    for seg in segments:
        seg_start = seg.get("time_start")
        seg_end = seg.get("time_end")
        if seg_start is None or seg_end is None:
            continue
        if time_end is not None and seg_start > time_end:
            continue
        if time_start is not None and seg_end < time_start:
            continue
        result.append(seg)
    return result


def retrieve(
    hierarchy: Dict[str, Any],
    parsed: Dict[str, Any],
) -> List[Dict]:
    """
    Retrieve relevant segments from the hierarchy based on parsed query intent.
    """
    roles = parsed["roles"] or list(hierarchy["roles"].keys())
    level = parsed["level"]

    if level is None:
        level = 1
        if parsed["time_start"] is not None and parsed["time_end"] is not None:
            span = parsed["time_end"] - parsed["time_start"]
            if span <= 30:
                level = 0
            elif span >= 600:
                level = 2

    level_key_map = {
        0: "level0_segments",
        1: "level1_segments",
        2: "level2_segments",
    }
    seg_key = level_key_map.get(level, "level1_segments")

    results = []
    for role in roles:
        role_data = hierarchy["roles"].get(role)
        if not role_data:
            continue
        segments = role_data.get(seg_key, [])
        if not segments and level == 2:
            segments = role_data.get("level1_segments", [])
        if not segments:
            segments = role_data.get("level0_segments", [])

        filtered = _segments_in_range(
            segments, parsed["time_start"], parsed["time_end"],
        )
        results.extend(filtered)

    results.sort(key=lambda s: (s.get("time_start") or 0, s.get("role", "")))
    return results


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def format_context(segments: List[Dict], parsed: Dict[str, Any]) -> str:
    """Format retrieved segments into a context string for the LLM."""
    if not segments:
        return "No relevant segments found for this query."

    lines = []
    current_role = None
    for seg in segments:
        role = seg.get("role", "unknown")
        if role != current_role:
            current_role = role
            lines.append(f"\n--- {humanize(role)} ---")

        level = seg.get("level", "?")
        t_start = seg.get("time_start", "?")
        t_end = seg.get("time_end", "?")
        dur = seg.get("duration_s", "?")
        sid = seg.get("segment_id", "?")

        summary = seg.get("summary", seg.get("description", ""))
        child_key = "segment_ids" if "segment_ids" in seg else "child_ids"
        children = seg.get(child_key, [])

        lines.append(
            f"  [L{level}] {sid} | t={t_start}s–{t_end}s ({dur}s)"
            f"{f' | {len(children)} children' if children else ''}"
            f" | {summary}"
        )

        if level == 0:
            preds = seg.get("active_predicates", [])
            if preds:
                pred_str = "; ".join(f"{p} {o}" for p, o in preds)
                lines.append(f"         predicates: {pred_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """\
You are a surgical procedure analyst. You will be given context from a \
hierarchical activity log of a surgical procedure. The context contains \
segments at different granularity levels:
- Level 0: atomic states (constant interactions for a period)
- Level 1: action steps (groups of atomic states forming a sub-task)
- Level 2: surgical phases (high-level stages of the procedure)

Each segment belongs to a specific role (head surgeon, nurse, etc.) and has \
a time range and summary.

Answer the user's question based ONLY on the provided context. Be concise \
and specific. Reference time ranges when relevant. If the context doesn't \
contain enough information, say so."""


def generate_answer(
    query: str,
    context: str,
    model,
    tokenizer,
    build_chat_input,
    run_inference_batch,
    strip_think_tags,
) -> str:
    """Generate an answer using the LLM given context and query."""
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"
    prompt = build_chat_input(tokenizer, QA_SYSTEM_PROMPT, user_msg, enable_thinking=False)
    results = run_inference_batch(model, tokenizer, [prompt], max_new_tokens=1024)
    return strip_think_tags(results[0])


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def run_interactive(hierarchy: Dict[str, Any], model, tokenizer,
                    build_chat_input, run_inference_batch, strip_think_tags):
    """Interactive query loop."""
    print("\n" + "=" * 60)
    print("  Surgical Hierarchy QA System")
    print("  Type a question, or 'quit' to exit.")
    print("  Commands:  :roles  :stats  :time")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if query == ":roles":
            for role, data in hierarchy["roles"].items():
                n0 = data.get("num_segments", 0)
                n1 = data.get("num_level1_segments", 0)
                n2 = data.get("num_level2_segments", 0)
                print(f"  {humanize(role):30s}  L0={n0:3d}  L1={n1:3d}  L2={n2:3d}")
            continue

        if query == ":stats":
            meta = hierarchy["metadata"]
            print(f"  Time range: {meta.get('time_range')}")
            print(f"  Timepoints: {meta.get('num_timepoints')}")
            print(f"  Total L0: {meta.get('total_level0_segments')}")
            print(f"  Total L1: {meta.get('total_level1_segments', 'N/A')}")
            print(f"  Total L2: {meta.get('total_level2_segments', 'N/A')}")
            continue

        if query == ":time":
            tr = hierarchy["metadata"].get("time_range", [None, None])
            if tr and tr[0] is not None:
                dur = tr[1] - tr[0]
                print(f"  {tr[0]}s – {tr[1]}s  ({dur}s = {dur // 60}m {dur % 60}s)")
            continue

        parsed = parse_query(query, hierarchy)
        segments = retrieve(hierarchy, parsed)

        print(f"\n  [Retrieved {len(segments)} segments | "
              f"roles={[humanize(r) for r in (parsed['roles'] or ['all'])]}"
              f" | level={parsed['level'] or 'auto'}"
              f" | time={parsed['time_start']}–{parsed['time_end']}]\n")

        if not segments:
            print("  No matching segments found. Try a different query.\n")
            continue

        context = format_context(segments, parsed)
        answer = generate_answer(
            query, context, model, tokenizer,
            build_chat_input, run_inference_batch, strip_think_tags,
        )
        print(f"\n{answer}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the role-aware surgical hierarchy."
    )
    parser.add_argument(
        "--hierarchy", type=Path, required=True,
        help="Path to hierarchy JSON file.",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-32B",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--hf_token", type=str,
        default="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",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query (non-interactive mode). Omit for interactive.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hierarchy = load_hierarchy(args.hierarchy)

    from scene_graph_utils import (
        build_chat_input,
        load_model,
        run_inference_batch,
        strip_think_tags,
    )

    logger.info("Loading model ...")
    model, tokenizer = load_model(args.model, args.hf_token)

    if args.query:
        parsed = parse_query(args.query, hierarchy)
        segments = retrieve(hierarchy, parsed)
        context = format_context(segments, parsed)
        answer = generate_answer(
            args.query, context, model, tokenizer,
            build_chat_input, run_inference_batch, strip_think_tags,
        )
        print(f"\nAnswer:\n{answer}\n")
    else:
        run_interactive(
            hierarchy, model, tokenizer,
            build_chat_input, run_inference_batch, strip_think_tags,
        )


if __name__ == "__main__":
    main()
