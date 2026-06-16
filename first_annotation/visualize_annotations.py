"""
Generate a self-contained HTML report from an annotations JSONL file.

Usage:
    python visualize_annotations.py \
        --input  annotation_output/annotations_001_PKA_600.jsonl \
        --output annotation_output/annotations_001_PKA_600.html
        
    python3 -m http.server 8080 --directory /mnt/home/nhatvu/dlhm/annotation_output    
"""

import argparse
import json
from pathlib import Path
from typing import List
from annotation_model import ENTITY_NAMES, PREDICATE_NAMES


def humanize(entity: str) -> str:
    return ENTITY_NAMES.get(entity, entity.replace("_", " "))


def humanize_pred(pred: str) -> str:
    return PREDICATE_NAMES.get(pred, pred)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PREDICATE_COLORS = {
    "Manipulating": "#3b82f6",   # blue
    "Calibrating":  "#8b5cf6",   # violet
    "Preparing":    "#f59e0b",   # amber
    "Assisting":    "#10b981",   # emerald
    "Holding":      "#06b6d4",   # cyan
    "Touching":     "#64748b",   # slate
    "CloseTo":      "#94a3b8",   # light slate
    "LyingOn":      "#e2e8f0",   # very light (patient/table)
    "Drilling":     "#ef4444",   # red
    "Sawing":       "#dc2626",   # dark red
    "Hammering":    "#b91c1c",   # darker red
    "Suturing":     "#ec4899",   # pink
    "Cutting":      "#f97316",   # orange
    "Cementing":    "#84cc16",   # lime
    "Cleaning":     "#22d3ee",   # light cyan
    "Scanning":     "#a78bfa",   # purple
}

def pred_color(p: str) -> str:
    return PREDICATE_COLORS.get(p, "#94a3b8")

def dominant_predicate(scene_graphs: list) -> str:
    """Most active (non-CloseTo, non-LyingOn) predicate in the window."""
    counts: dict = {}
    for sg in scene_graphs:
        for _, pred, _ in sg:
            if pred not in ("CloseTo", "LyingOn"):
                counts[pred] = counts.get(pred, 0) + 1
    if counts:
        return max(counts, key=lambda k: counts[k])
    return "CloseTo"

def render_triplet(s: str, p: str, o: str) -> str:
    color = pred_color(p)
    return (
        f'<span class="entity">{humanize(s)}</span> '
        f'<span class="pred" style="background:{color}20;color:{color};border-color:{color}40">'
        f'{humanize_pred(p)}</span> '
        f'<span class="entity">{humanize(o)}</span>'
    )

def render_window(idx: int, record: dict) -> str:
    ts = record["original_timestamps"]
    t_start, t_end = ts[0], ts[-1]
    summary = record.get("atomic_action_summary", "").strip()
    scene_graphs = record["scene_graphs"]
    dom_pred = dominant_predicate(scene_graphs)
    border_color = pred_color(dom_pred)

    # Scene graph rows: one row per second
    sg_rows = []
    for i, (t, sg) in enumerate(zip(ts, scene_graphs)):
        triplets_html = " &nbsp;·&nbsp; ".join(render_triplet(s, p, o) for s, p, o in sg) if sg else "<em>—</em>"
        sg_rows.append(
            f'<tr><td class="ts-cell">t={t}s</td><td class="rel-cell">{triplets_html}</td></tr>'
        )
    sg_table = "\n".join(sg_rows)

    summary_html = (
        f'<div class="summary">{summary}</div>'
        if summary else
        '<div class="summary empty">— no summary generated —</div>'
    )

    issues = record.get("issues", [])
    issues_html = ""
    if issues:
        issue_items = "\n".join(f'<div class="issue">⚠ {i}</div>' for i in issues)
        issues_html = f'<div class="issues">{issue_items}</div>'

    return f"""
<div class="window-card" style="border-left:4px solid {border_color}">
  <div class="card-header">
    <span class="win-id">Window #{idx + 1}</span>
    <span class="time-range">t = {t_start}s – {t_end}s &nbsp;·&nbsp; frames {record['window_start_tp']}–{record['window_end_tp']}</span>
    <span class="badge" style="background:{border_color}20;color:{border_color};border-color:{border_color}40">{dom_pred}</span>
  </div>
  <table class="sg-table">{sg_table}</table>
  {summary_html}
  {issues_html}
</div>
"""

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

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
    --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6; }}
  header {{ padding: 24px 32px; border-bottom: 1px solid var(--border); }}
  header h1 {{ font-size: 20px; font-weight: 700; color: var(--accent); }}
  header p {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .controls {{ padding: 16px 32px; display: flex; gap: 12px; align-items: center; border-bottom: 1px solid var(--border); background: var(--surface); position: sticky; top: 0; z-index: 10; }}
  .controls input {{ flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; color: var(--text); font-size: 13px; outline: none; }}
  .controls input:focus {{ border-color: var(--accent); }}
  .controls label {{ color: var(--muted); font-size: 13px; white-space: nowrap; }}
  .stats {{ padding: 8px 32px; color: var(--muted); font-size: 12px; }}
  .container {{ padding: 16px 32px; display: flex; flex-direction: column; gap: 12px; }}
  .window-card {{ background: var(--surface); border-radius: 8px; border: 1px solid var(--border); padding: 14px 16px; transition: background 0.1s; }}
  .window-card:hover {{ background: var(--surface2); }}
  .card-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }}
  .win-id {{ font-weight: 700; color: var(--accent); font-size: 13px; min-width: 80px; }}
  .time-range {{ color: var(--muted); font-size: 12px; font-family: monospace; }}
  .badge {{ font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; border: 1px solid; margin-left: auto; }}
  .sg-table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 12px; }}
  .sg-table tr {{ border-bottom: 1px solid var(--border); }}
  .sg-table tr:last-child {{ border-bottom: none; }}
  .ts-cell {{ color: var(--muted); font-family: monospace; white-space: nowrap; padding: 3px 10px 3px 0; width: 60px; vertical-align: top; }}
  .rel-cell {{ padding: 3px 0; }}
  .entity {{ color: var(--text); }}
  .pred {{ font-size: 11px; font-weight: 600; padding: 1px 6px; border-radius: 4px; border: 1px solid; margin: 0 2px; }}
  .summary {{ background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 13px; color: var(--text); line-height: 1.5; }}
  .summary.empty {{ color: var(--muted); font-style: italic; }}
  .issues {{ margin-top: 6px; display: flex; flex-direction: column; gap: 3px; }}
  .issue {{ font-size: 11px; padding: 3px 8px; border-radius: 4px; background: #7f1d1d30; border: 1px solid #f8717140; color: #fca5a5; font-family: monospace; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>{subtitle}</p>
</header>
<div class="controls">
  <label>Search</label>
  <input type="text" id="search" placeholder="Filter by summary text, entity, predicate…" oninput="filterCards()"/>
</div>
<div class="stats" id="stats"></div>
<div class="container" id="container">
{cards}
</div>
<script>
  const cards = Array.from(document.querySelectorAll('.window-card'));
  const stats = document.getElementById('stats');
  function updateStats(visible) {{
    stats.textContent = visible + ' / {total} windows shown';
  }}
  function filterCards() {{
    const q = document.getElementById('search').value.toLowerCase();
    let visible = 0;
    cards.forEach(c => {{
      const match = !q || c.textContent.toLowerCase().includes(q);
      c.classList.toggle('hidden', !match);
      if (match) visible++;
    }});
    updateStats(visible);
  }}
  updateStats({total});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report(input_path: Path, output_path: Path) -> None:
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found in input file.")
        return

    ts_all = [r["original_timestamps"] for r in records]
    t_start = ts_all[0][0]
    t_end   = ts_all[-1][-1]
    take_name = input_path.stem

    cards_html = "\n".join(render_window(i, r) for i, r in enumerate(records))

    html = HTML_TEMPLATE.format(
        title=f"Annotations — {take_name}",
        subtitle=(
            f"{len(records)} windows · "
            f"t={t_start}s – t={t_end}s · "
            f"window size = {records[0]['window_size_s']}s"
        ),
        cards=cards_html,
        total=len(records),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to: {output_path}  ({len(records)} windows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise annotation JSONL as HTML.")
    parser.add_argument("--input",  type=Path, required=True,  help="Input .jsonl file.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .html file (default: same folder as input, same stem + .html).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or args.input.with_suffix(".html")
    build_report(args.input, output)


