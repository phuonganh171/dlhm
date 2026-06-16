"""
Generate a self-contained HTML viewer for the role-aware hierarchical scene graph.

Displays an expandable tree per role: Level-2 (phases) → Level-1 (action steps) → Level-0 (atomic states).
Includes a timeline bar chart showing segment durations, search/filter, and role tabs.

Usage:
    python3 visualize_hierarchy.py \
        --input  hierarchy_output/001_PKA_hierarchy.json \
        --output hierarchy_output/001_PKA_hierarchy.html

    python3 -m http.server 8080 --directory /mnt/home/nhatvu/dlhm/hierarchy_output
"""

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List

from scene_graph_utils import ENTITY_NAMES, PREDICATE_NAMES, TOOL_ENTITIES


def humanize(entity: str) -> str:
    return ENTITY_NAMES.get(entity, entity.replace("_", " "))


PREDICATE_COLORS = {
    "Manipulating": "#3b82f6", "Calibrating": "#8b5cf6", "Preparing": "#f59e0b",
    "Assisting": "#10b981", "Holding": "#06b6d4", "Touching": "#64748b",
    "Drilling": "#ef4444", "Sawing": "#dc2626", "Hammering": "#b91c1c",
    "Suturing": "#ec4899", "Cutting": "#f97316", "Cementing": "#84cc16",
    "Cleaning": "#22d3ee", "Scanning": "#a78bfa",
    # Robot-monitor / robot-setup pseudo-predicates (phase + current step)
    "Phase": "#a855f7", "Step": "#f59e0b",
}

LEVEL_COLORS = {0: "#64748b", 1: "#3b82f6", 2: "#8b5cf6"}


def _pred_badge(pred: str, obj: str, *, tool_role: bool = False) -> str:
    color = PREDICATE_COLORS.get(pred, "#94a3b8")
    if pred == "Phase":
        label = humanize(obj)
    elif pred == "Step":
        label = f"&#9656; {humanize(obj)}"
    elif tool_role:
        label = f"{humanize(obj)} {PREDICATE_NAMES.get(pred, pred)}"
    else:
        label = f"{PREDICATE_NAMES.get(pred, pred)} {humanize(obj)}"
    return (
        f'<span class="pred-badge" style="background:{color}15;color:{color};'
        f'border-color:{color}40">{label}</span>'
    )


def _time_fmt(seconds: Any) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _render_l0_segment(seg: Dict) -> str:
    preds = seg.get("active_predicates", [])
    tool_role = seg.get("role") in TOOL_ENTITIES
    pred_html = (
        " ".join(_pred_badge(p, o, tool_role=tool_role) for p, o in preds)
        if preds else '<em class="idle">idle</em>'
    )
    dur = seg.get("duration_s", "?")
    t0, t1 = seg.get("time_start", "?"), seg.get("time_end", "?")
    return (
        f'<div class="l0-seg" data-text="{escape(seg.get("description", ""))}">'
        f'<span class="seg-id">L0</span>'
        f'<span class="seg-time">{t0}s–{t1}s ({dur}s)</span>'
        f'<span class="seg-preds">{pred_html}</span>'
        f'</div>'
    )


def _render_l1_segment(seg: Dict, l0_lookup: Dict[str, Dict]) -> str:
    summary = escape(seg.get("summary", ""))
    dur = seg.get("duration_s", "?")
    t0, t1 = seg.get("time_start", "?"), seg.get("time_end", "?")
    child_ids = seg.get("segment_ids", [])

    children_html = ""
    for cid in child_ids:
        child = l0_lookup.get(cid)
        if child:
            children_html += _render_l0_segment(child)

    return (
        f'<details class="l1-group" data-text="{summary.lower()}">'
        f'<summary>'
        f'<span class="seg-id">L1</span>'
        f'<span class="seg-time">{t0}s–{t1}s ({dur}s)</span>'
        f'<span class="seg-summary">{summary}</span>'
        f'<span class="child-count">{len(child_ids)} segments</span>'
        f'</summary>'
        f'<div class="children">{children_html}</div>'
        f'</details>'
    )


def _render_l2_segment(seg: Dict, l1_lookup: Dict[str, Dict], l0_lookup: Dict[str, Dict]) -> str:
    summary = escape(seg.get("summary", ""))
    dur = seg.get("duration_s", "?")
    t0, t1 = seg.get("time_start", "?"), seg.get("time_end", "?")
    child_ids = seg.get("child_ids", seg.get("segment_ids", []))

    children_html = ""
    for cid in child_ids:
        child = l1_lookup.get(cid)
        if child:
            children_html += _render_l1_segment(child, l0_lookup)

    return (
        f'<details class="l2-group" data-text="{summary.lower()}">'
        f'<summary>'
        f'<span class="seg-id l2">L2</span>'
        f'<span class="seg-time">{t0}s–{t1}s ({dur}s)</span>'
        f'<span class="seg-summary">{summary}</span>'
        f'<span class="child-count">{len(child_ids)} steps</span>'
        f'</summary>'
        f'<div class="children">{children_html}</div>'
        f'</details>'
    )


def _render_role(role: str, role_data: Dict, global_start: int, global_end: int) -> str:
    l0_segs = role_data.get("level0_segments", [])
    l1_segs = role_data.get("level1_segments", [])
    l2_segs = role_data.get("level2_segments", [])

    l0_lookup = {s["segment_id"]: s for s in l0_segs}
    l1_lookup = {s["segment_id"]: s for s in l1_segs}

    n0 = len(l0_segs)
    n1 = len(l1_segs)
    n2 = len(l2_segs)

    # Timeline bar
    total_dur = max(global_end - global_start, 1)
    bars_html = ""
    for seg in l0_segs:
        t0 = seg.get("time_start", global_start)
        t1 = seg.get("time_end", global_start)
        if t0 is None or t1 is None:
            continue
        left_pct = 100 * (t0 - global_start) / total_dur
        width_pct = max(100 * (t1 - t0) / total_dur, 0.2)
        preds = seg.get("active_predicates", [])
        color = "#475569"
        if preds:
            color = PREDICATE_COLORS.get(preds[0][0], "#475569")
        bars_html += (
            f'<div class="timeline-bar" style="left:{left_pct:.2f}%;'
            f'width:{width_pct:.2f}%;background:{color}" '
            f'title="{escape(seg.get("description", ""))}"></div>'
        )

    # Tree content
    tree_html = ""
    if l2_segs:
        for seg in l2_segs:
            tree_html += _render_l2_segment(seg, l1_lookup, l0_lookup)
    elif l1_segs:
        for seg in l1_segs:
            tree_html += _render_l1_segment(seg, l0_lookup)
    else:
        for seg in l0_segs:
            tree_html += _render_l0_segment(seg)

    role_name = humanize(role)
    return (
        f'<div class="role-panel" id="role-{role}" data-role="{role}">'
        f'<div class="role-header">'
        f'<h2>{role_name}</h2>'
        f'<div class="role-stats">'
        f'<span class="stat">L0: {n0}</span>'
        f'<span class="stat">L1: {n1}</span>'
        f'<span class="stat">L2: {n2}</span>'
        f'</div></div>'
        f'<div class="timeline-track">'
        f'<div class="timeline-label">{_time_fmt(global_start)}</div>'
        f'<div class="timeline-container">{bars_html}</div>'
        f'<div class="timeline-label">{_time_fmt(global_end)}</div>'
        f'</div>'
        f'<div class="tree-content">{tree_html}</div>'
        f'</div>'
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --surface2: #273549;
    --border: #334155; --text: #e2e8f0; --muted: #94a3b8;
    --accent: #38bdf8; --l0: #64748b; --l1: #3b82f6; --l2: #8b5cf6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Inter','Segoe UI',sans-serif; font-size: 14px; line-height: 1.6; }}
  header {{ padding: 20px 28px; border-bottom: 1px solid var(--border); }}
  header h1 {{ font-size: 18px; font-weight: 700; color: var(--accent); }}
  header p {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}

  .controls {{ padding: 12px 28px; display: flex; gap: 10px; align-items: center; border-bottom: 1px solid var(--border); background: var(--surface); position: sticky; top: 0; z-index: 10; flex-wrap: wrap; }}
  .controls input {{ flex: 1; min-width: 200px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; color: var(--text); font-size: 13px; outline: none; }}
  .controls input:focus {{ border-color: var(--accent); }}
  .controls button {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 5px 12px; color: var(--text); cursor: pointer; font-size: 12px; white-space: nowrap; }}
  .controls button:hover {{ background: var(--border); }}
  .controls button.active {{ border-color: var(--accent); color: var(--accent); }}

  .role-tabs {{ padding: 10px 28px; display: flex; gap: 6px; flex-wrap: wrap; border-bottom: 1px solid var(--border); }}
  .role-tab {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 4px 14px; color: var(--muted); cursor: pointer; font-size: 12px; font-weight: 500; transition: all 0.15s; }}
  .role-tab:hover {{ color: var(--text); border-color: var(--accent); }}
  .role-tab.active {{ background: var(--accent); color: var(--bg); border-color: var(--accent); }}

  .container {{ padding: 16px 28px; }}

  .role-panel {{ display: none; }}
  .role-panel.visible {{ display: block; }}

  .role-header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }}
  .role-header h2 {{ font-size: 16px; font-weight: 600; color: var(--accent); text-transform: capitalize; }}
  .role-stats {{ display: flex; gap: 8px; }}
  .stat {{ font-size: 11px; color: var(--muted); background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; }}

  .timeline-track {{ display: flex; align-items: center; gap: 8px; margin-bottom: 16px; padding: 8px 0; }}
  .timeline-label {{ font-size: 10px; color: var(--muted); font-family: monospace; white-space: nowrap; }}
  .timeline-container {{ flex: 1; height: 18px; background: var(--surface); border-radius: 4px; position: relative; overflow: hidden; border: 1px solid var(--border); }}
  .timeline-bar {{ position: absolute; top: 2px; height: 14px; border-radius: 2px; opacity: 0.8; min-width: 1px; }}

  .tree-content {{ display: flex; flex-direction: column; gap: 4px; }}

  details {{ border: 1px solid var(--border); border-radius: 6px; background: var(--surface); }}
  details > summary {{ padding: 8px 12px; cursor: pointer; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; list-style: none; }}
  details > summary::-webkit-details-marker {{ display: none; }}
  details > summary::before {{ content: '▸'; color: var(--muted); font-size: 12px; transition: transform 0.15s; }}
  details[open] > summary::before {{ transform: rotate(90deg); }}
  details > .children {{ padding: 4px 4px 4px 24px; display: flex; flex-direction: column; gap: 3px; }}

  .l2-group {{ margin-bottom: 6px; }}
  .l2-group > .children {{ padding: 6px 6px 6px 24px; }}
  .l1-group {{ background: var(--surface2); }}

  .seg-id {{ font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 3px; color: white; }}
  .seg-id {{ background: var(--l0); }}
  .l1-group > summary .seg-id {{ background: var(--l1); }}
  .l2-group > summary .seg-id.l2 {{ background: var(--l2); }}
  .seg-time {{ font-size: 11px; color: var(--muted); font-family: monospace; }}
  .seg-summary {{ font-size: 13px; flex: 1; }}
  .child-count {{ font-size: 10px; color: var(--muted); background: var(--bg); padding: 1px 6px; border-radius: 3px; }}
  .seg-preds {{ display: flex; gap: 4px; flex-wrap: wrap; }}

  .l0-seg {{ padding: 5px 10px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; border-radius: 4px; background: var(--bg); border: 1px solid transparent; }}
  .l0-seg:hover {{ border-color: var(--border); }}

  .pred-badge {{ font-size: 11px; font-weight: 500; padding: 1px 7px; border-radius: 4px; border: 1px solid; white-space: nowrap; }}
  .idle {{ color: var(--muted); font-size: 12px; }}

  .hidden {{ display: none !important; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>{subtitle}</p>
</header>
<div class="controls">
  <input type="text" id="search" placeholder="Search summaries, predicates…" oninput="filterTree()"/>
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
</div>
<div class="role-tabs" id="role-tabs">{role_tabs}</div>
<div class="container" id="container">{panels}</div>
<script>
const tabs = document.querySelectorAll('.role-tab');
const panels = document.querySelectorAll('.role-panel');
function showRole(role) {{
  tabs.forEach(t => t.classList.toggle('active', t.dataset.role === role));
  panels.forEach(p => p.classList.toggle('visible', p.dataset.role === role));
}}
tabs.forEach(t => t.addEventListener('click', () => showRole(t.dataset.role)));
if (tabs.length) showRole(tabs[0].dataset.role);

function expandAll() {{
  document.querySelectorAll('.role-panel.visible details').forEach(d => d.open = true);
}}
function collapseAll() {{
  document.querySelectorAll('.role-panel.visible details').forEach(d => d.open = false);
}}
function filterTree() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.role-panel.visible details, .role-panel.visible .l0-seg').forEach(el => {{
    const text = (el.dataset.text || el.textContent || '').toLowerCase();
    const match = !q || text.includes(q);
    el.classList.toggle('hidden', !match);
    if (match && q) {{
      let parent = el.closest('details');
      while (parent) {{ parent.open = true; parent.classList.remove('hidden'); parent = parent.parentElement.closest('details'); }}
    }}
  }});
}}
</script>
</body>
</html>"""


def _filter_unmapped_segments(hierarchy: Dict) -> Dict:
    """Remove L0 segments whose timepoints have no frame-map entry (time_start/time_end is None).

    Also drops any L1/L2 segments that become empty after the L0 filtering,
    and updates child-id lists so they stay consistent.
    """
    for role_data in hierarchy["roles"].values():
        l0_segs = role_data.get("level0_segments", [])
        filtered_l0 = [s for s in l0_segs if s.get("time_start") is not None and s.get("time_end") is not None]
        removed_ids = {s["segment_id"] for s in l0_segs} - {s["segment_id"] for s in filtered_l0}
        role_data["level0_segments"] = filtered_l0
        role_data["num_segments"] = len(filtered_l0)

        if removed_ids:
            for l1 in role_data.get("level1_segments", []):
                l1["segment_ids"] = [sid for sid in l1.get("segment_ids", []) if sid not in removed_ids]
            role_data["level1_segments"] = [
                s for s in role_data.get("level1_segments", []) if s.get("segment_ids")
            ]
            role_data["num_level1_segments"] = len(role_data.get("level1_segments", []))

            kept_l1_ids = {s["segment_id"] for s in role_data.get("level1_segments", [])}
            for l2 in role_data.get("level2_segments", []):
                key = "child_ids" if "child_ids" in l2 else "segment_ids"
                l2[key] = [sid for sid in l2.get(key, []) if sid in kept_l1_ids]
            role_data["level2_segments"] = [
                s for s in role_data.get("level2_segments", [])
                if s.get("child_ids") or s.get("segment_ids")
            ]
            role_data["num_level2_segments"] = len(role_data.get("level2_segments", []))

    return hierarchy


def build_report(input_path: Path, output_path: Path) -> None:
    hierarchy = json.loads(input_path.read_text())
    hierarchy = _filter_unmapped_segments(hierarchy)
    meta = hierarchy.get("metadata", {})
    time_range = meta.get("time_range", [0, 0])
    global_start = time_range[0] or 0
    global_end = time_range[1] or 0

    role_tabs_html = ""
    panels_html = ""
    for role in sorted(hierarchy["roles"].keys()):
        role_data = hierarchy["roles"][role]
        role_name = humanize(role)
        role_tabs_html += f'<div class="role-tab" data-role="{role}">{role_name}</div>'
        panels_html += _render_role(role, role_data, global_start, global_end)

    subtitle = (
        f"{meta.get('total_roles', '?')} roles · "
        f"L0={meta.get('total_level0_segments', '?')} · "
        f"L1={meta.get('total_level1_segments', 'N/A')} · "
        f"L2={meta.get('total_level2_segments', 'N/A')} · "
        f"time {_time_fmt(global_start)}–{_time_fmt(global_end)}"
    )

    html = HTML_TEMPLATE.format(
        title=f"Hierarchy — {input_path.stem}",
        subtitle=subtitle,
        role_tabs=role_tabs_html,
        panels=panels_html,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise hierarchy JSON as HTML.")
    parser.add_argument("--input", type=Path, required=True, help="Hierarchy JSON file.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or args.input.with_suffix(".html")
    build_report(args.input, output)
