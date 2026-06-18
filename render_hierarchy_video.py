#!/usr/bin/env python3
"""
Render a synchronized HTML "video" player with hierarchy levels:
- Left: Level-2 phases → Level-1 steps → Level-0 atomic segments + per-frame scene graph
- Right: multi-camera frame view (1 frame/second)

The player uses the 2-level hierarchy output from build_hierarchy_qwen32b.py
and loads per-frame scene graphs from the relation_labels directory.

Usage:
    python3 render_hierarchy_video.py \
    --hierarchy hierarchy_output/001_PKA_hierarchy.json \
    --srt mm-or/MM-OR_data/MM-OR_processed/take_transcripts/001_PKA.srt \
    --colorimage_dir mm-or/MM-OR_data/MM-OR_processed/001_PKA/colorimage \
    --output hierarchy_video_sync_qwen32b.html

    python3 -m http.server 8080 --directory /mnt/home/nhatvu/dlhm
    http://localhost:8080/hierarchy_video_sync.html
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import re

from scene_graph_utils import (
    ENTITY_NAMES,
    PREDICATE_NAMES,
    ROLE_ENTITIES,
    TOOL_ENTITIES,
    load_frame_map,
    load_relation_labels,
    original_timestamp,
)


PREDICATE_COLORS = {
    "Manipulating": "#3b82f6", "Calibrating": "#8b5cf6", "Preparing": "#f59e0b",
    "Assisting": "#10b981", "Holding": "#06b6d4", "Touching": "#64748b",
    "Drilling": "#ef4444", "Sawing": "#dc2626", "Hammering": "#b91c1c",
    "Suturing": "#ec4899", "Cutting": "#f97316", "Cementing": "#84cc16",
    "Cleaning": "#22d3ee", "Scanning": "#a78bfa",
    "CloseTo": "#475569", "LyingOn": "#374151",
    # Robot-monitor / robot-setup pseudo-predicates (phase + current step)
    "Phase": "#a855f7", "Step": "#f59e0b",
}

# Synthetic roles whose timeline comes from the robot monitor logs, not the
# scene-graph triplets (so they never appear as a triplet subject).
ROBOT_ROLES = {"robot_setup", "robot_monitor"}

# Tool/instrument roles appear as the *object* of triplets, not the subject.
TOOL_ROLES = set(TOOL_ENTITIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create HTML player for hierarchy + camera frames."
    )
    parser.add_argument(
        "--hierarchy", type=Path,
        default=Path("hierarchy_output/001_PKA_hierarchy.json"),
        help="Path to hierarchy JSON from build_hierarchy_qwen32b.py.",
    )
    parser.add_argument(
        "--colorimage_dir", type=Path,
        default=Path("mm-or/MM-OR_data/MM-OR_processed/001_PKA/colorimage"),
        help="Directory containing color image frames.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("hierarchy_video_sync.html"),
        help="Output HTML path.",
    )
    parser.add_argument(
        "--cameras", type=str,
        default="camera01,camera02,camera03,camera04,camera05",
        help="Comma-separated camera ids.",
    )
    parser.add_argument(
        "--simstation_dir", type=Path,
        default=Path("mm-or/simstation"),
        help="Directory containing simstation (robot monitor screen) frames.",
    )
    parser.add_argument(
        "--srt", type=Path, default=None,
        help="Path to SRT transcript file for subtitle display.",
    )
    parser.add_argument(
        "--autoplay", action="store_true",
        help="Start playback automatically.",
    )
    parser.add_argument(
        "--web_root", type=Path, default=None,
        help="HTTP server root for frame paths.",
    )
    return parser.parse_args()


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse an SRT file into a sorted list of {start, end, text} entries (seconds)."""
    content = srt_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        ts_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1],
        )
        if not ts_match:
            continue
        g = [int(x) for x in ts_match.groups()]
        start_s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end_s = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        text = " ".join(lines[2:]).strip()
        entries.append({"start": round(start_s, 1), "end": round(end_s, 1), "text": text})
    return entries


def infer_web_root(output: Path, colorimage_dir: Path) -> Path:
    output = output.resolve()
    colorimage_dir = colorimage_dir.resolve()
    for candidate in (Path.cwd().resolve(), output.parent.resolve()):
        try:
            output.relative_to(candidate)
            colorimage_dir.relative_to(candidate)
            return candidate
        except ValueError:
            continue
    return output.parent.resolve()


def color_dir_for_web(colorimage_dir: Path, web_root: Path) -> str:
    rel = os.path.relpath(colorimage_dir.resolve(), web_root.resolve())
    return Path(rel).as_posix()


def build_camera_grid_html(cameras: list[str]) -> str:
    cells = []
    for cam in cameras:
        cells.append(
            f"""        <div class="camera-cell" data-camera="{cam}">
          <div class="camera-label">{cam}</div>
          <img class="camera-frame" alt="{cam}" />
        </div>"""
        )
    return "\n".join(cells)


def build_simstation_map(hierarchy: dict) -> dict[int, str]:
    """
    Build a mapping from original_timestamp (colorimage second) to the
    simstation frame id string, so the player can load the correct monitor
    screen image for each time.
    """
    meta = hierarchy.get("metadata", {})
    take_dir = Path(meta["take_dir"])
    frame_map = load_frame_map(take_dir)

    ts_to_sim: dict[int, str] = {}
    for tp_id, info in frame_map.items():
        ts = info.get("original_timestamp")
        sim = info.get("simstation")
        if ts is not None and sim is not None:
            ts_to_sim[ts] = sim
    return ts_to_sim


def load_scene_graphs(hierarchy: dict) -> dict[int, list]:
    """
    Load per-frame scene graphs from relation_labels, keyed by original_timestamp.
    Only loads frames within the hierarchy's time range.
    """
    meta = hierarchy.get("metadata", {})
    take_dir = Path(meta["take_dir"])
    tp_range = meta.get("tp_range", [None, None])

    frame_map = load_frame_map(take_dir)
    entries = load_relation_labels(take_dir, frame_map)

    if tp_range[0]:
        entries = [(tp, t) for tp, t in entries if tp >= tp_range[0]]
    if tp_range[1]:
        entries = [(tp, t) for tp, t in entries if tp <= tp_range[1]]

    sg_by_time: dict[int, list] = {}
    for tp_id, triplets in entries:
        ts = original_timestamp(frame_map, tp_id)
        if ts is not None:
            sg_by_time[ts] = triplets

    return sg_by_time


def _filter_unmapped_segments(hierarchy: dict) -> dict:
    """Remove L0 segments whose timepoints have no frame-map entry (time_start/time_end is None).

    Also drops any L1/L2 segments that become empty after L0 filtering,
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


def _robot_label(seg: dict) -> str:
    """Compact 'Phase · Step' label for a robot-role L0 segment."""
    state = {k: v for k, v in seg.get("active_predicates", [])}
    phase = state.get("Phase")
    step = state.get("Step")
    if phase and step:
        return f"{phase} \u00b7 {step}"
    return phase or "idle"


def _build_robot_timeline(hierarchy: dict, role: str) -> list:
    """
    Flatten a robot role's L0 segments into [t_start, t_end, label] triples on
    the colorimage (original_timestamp) time axis, ready for time-synced lookup
    in the player.  Returns [] if the role is absent.
    """
    role_data = hierarchy["roles"].get(role)
    if not role_data:
        return []
    out = []
    for seg in role_data.get("level0_segments", []):
        t0 = seg.get("time_start")
        t1 = seg.get("time_end")
        if t0 is None or t1 is None:
            continue
        out.append([t0, t1, _robot_label(seg)])
    return out


def _order_roles(roles) -> list:
    """
    Order role ids by category -- people first, then robot, then tools, then
    anything else -- and alphabetically within each category.
    """
    def rank(role: str) -> int:
        if role in ROLE_ENTITIES:
            return 0
        if role in ROBOT_ROLES:
            return 1
        if role in TOOL_ROLES:
            return 2
        return 3

    return sorted(roles, key=lambda r: (rank(r), r))


def build_hierarchy_data(hierarchy: dict) -> dict:
    """Extract the compact data needed by the frontend JS."""
    hierarchy = _filter_unmapped_segments(hierarchy)
    meta = hierarchy.get("metadata", {})
    time_range = meta.get("time_range", [0, 0])
    roles_data = {}

    for role, role_data in hierarchy["roles"].items():
        l0_segs = role_data.get("level0_segments", [])
        l1_segs = role_data.get("level1_segments", [])
        l2_segs = role_data.get("level2_segments", [])

        roles_data[role] = {
            "role_human": l0_segs[0]["role_human"] if l0_segs else role.replace("_", " "),
            "level0": l0_segs,
            "level1": l1_segs,
            "level2": l2_segs,
            "is_robot": role in ROBOT_ROLES,
            "is_tool": role in TOOL_ROLES,
        }

    return {
        "roles": roles_data,
        "role_order": _order_roles(roles_data.keys()),
        "time_range": time_range,
        "metadata": meta,
        # Always-on monitor overlay, synced to the colorimage timeline.
        "monitor_timeline": _build_robot_timeline(hierarchy, "robot_monitor"),
    }


def build_html(
    hierarchy: dict,
    scene_graphs: dict[int, list],
    color_dir: str,
    output: Path,
    serve_dir: Path,
    cameras: list[str],
    autoplay: bool,
    simstation_dir: str = "",
    simstation_map: dict[int, str] | None = None,
    srt_entries: list[dict] | None = None,
) -> str:
    data = build_hierarchy_data(hierarchy)
    data_json = json.dumps(data, ensure_ascii=False)
    pred_colors_json = json.dumps(PREDICATE_COLORS)
    entity_names_json = json.dumps(ENTITY_NAMES, ensure_ascii=False)
    predicate_names_json = json.dumps(PREDICATE_NAMES, ensure_ascii=False)
    sg_json = json.dumps(scene_graphs, ensure_ascii=False)
    transcript_json = json.dumps(srt_entries or [], ensure_ascii=False)
    simstation_map_json = json.dumps(
        {str(k): v for k, v in (simstation_map or {}).items()},
        ensure_ascii=False,
    )
    html_name = output.name
    serve_dir_posix = serve_dir.resolve().as_posix()
    autoplay_js = "true" if autoplay else "false"
    cameras_json = json.dumps(cameras)
    camera_grid_html = build_camera_grid_html(cameras)

    time_range = data["time_range"]
    t_start = time_range[0] or 0
    t_end = time_range[1] or 0

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hierarchy + Frame Player</title>
  <style>
    :root {{
      --bg: #0f1220;
      --card: #1a1f35;
      --surface: #232a45;
      --text: #ecf0ff;
      --muted: #a8b1d9;
      --accent: #53a8ff;
      --l2: #8b5cf6;
      --l1: #3b82f6;
      --l0: #64748b;
      --ok: #58d68d;
      --warn: #f5b041;
      --danger: #ff6b6b;
      --border: #2b3356;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0; font-family: 'Inter', Arial, sans-serif;
      background: var(--bg); color: var(--text);
      height: 100vh; overflow: hidden;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 500px 1fr;
      height: 100vh;
    }}
    .left-panel {{
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--border);
      overflow: hidden;
    }}
    .right-panel {{
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    .panel-header {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      background: var(--card);
      flex-shrink: 0;
    }}
    .panel-header h2 {{ font-size: 14px; font-weight: 700; margin: 0; }}
    .panel-header .subtitle {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}

    .monitor-status {{
      margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap;
    }}
    .monitor-status .mon-chip {{
      display: inline-flex; align-items: center; gap: 5px;
      padding: 2px 8px; border-radius: 5px;
      background: var(--surface); border: 1px solid var(--border);
    }}
    .monitor-status .mon-key {{
      font-size: 9px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.05em; color: var(--muted);
    }}
    .monitor-status .mon-val {{ font-size: 11px; font-weight: 600; color: var(--l2); }}
    .monitor-status .mon-chip.setup .mon-val {{ color: var(--warn); }}

    .controls-bar {{
      display: flex; align-items: center; gap: 6px; padding: 6px 14px;
      background: var(--card); border-bottom: 1px solid var(--border);
      flex-shrink: 0; flex-wrap: wrap;
    }}
    button {{
      border: 1px solid #44518b; background: #1e2645; color: var(--text);
      padding: 5px 9px; border-radius: 5px; cursor: pointer; font-size: 11px;
      white-space: nowrap;
    }}
    button:hover {{ background: #2a3562; }}
    button.active {{ border-color: var(--accent); color: var(--accent); }}
    input[type="range"] {{ flex: 1; min-width: 80px; }}
    .time-display {{
      font-family: monospace; font-size: 12px; color: var(--accent);
      white-space: nowrap;
    }}

    .role-tabs {{
      display: flex; gap: 3px; padding: 6px 14px;
      border-bottom: 1px solid var(--border); flex-shrink: 0;
      overflow-x: auto; background: var(--card);
    }}
    .role-tab {{
      padding: 3px 8px; border-radius: 4px; font-size: 10px;
      font-weight: 600; cursor: pointer; white-space: nowrap;
      background: var(--surface); border: 1px solid var(--border);
      color: var(--muted); transition: all 0.15s;
    }}
    .role-tab:hover {{ color: var(--text); border-color: var(--accent); }}
    .role-tab.active {{ background: var(--accent); color: var(--bg); border-color: var(--accent); }}

    /* Window (L2 phase) info box */
    .window-info {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      background: var(--card);
      flex-shrink: 0;
    }}
    .window-title {{
      display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
    }}
    .window-title .badge {{
      font-size: 9px; font-weight: 700; padding: 2px 6px;
      border-radius: 3px; color: white; background: var(--l2);
    }}
    .window-title .win-num {{
      font-size: 12px; color: var(--muted);
    }}
    .window-summary {{
      font-size: 13px; line-height: 1.4; padding: 8px 10px;
      border-left: 3px solid var(--l2); background: rgba(139,92,246,0.06);
      border-radius: 4px; margin-bottom: 8px;
    }}
    .window-meta {{
      font-size: 10px; color: var(--muted); display: flex; gap: 12px; flex-wrap: wrap;
    }}
    .window-progress {{
      margin-top: 6px;
    }}
    .window-progress-label {{ font-size: 10px; color: var(--muted); margin-bottom: 3px; }}
    .window-progress-bar {{
      width: 100%; height: 8px; border-radius: 4px;
      background: var(--surface); overflow: hidden; border: 1px solid var(--border);
    }}
    .window-progress-fill {{
      height: 100%; width: 0%;
      background: linear-gradient(90deg, var(--l2), #b794f6);
      transition: width 100ms linear;
    }}

    /* Scene graph panel */
    .sg-panel {{
      flex-shrink: 0;
      border-bottom: 1px solid var(--border);
      max-height: 160px;
      overflow-y: auto;
      padding: 8px 14px;
      background: rgba(26,31,53,0.7);
    }}
    .sg-title {{
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.05em; color: var(--muted); margin-bottom: 5px;
    }}
    .sg-triplets {{
      display: flex; flex-wrap: wrap; gap: 3px;
    }}
    .sg-triplet {{
      font-size: 10px; padding: 2px 5px; border-radius: 3px;
      background: var(--surface); border: 1px solid var(--border);
      display: inline-flex; align-items: center; gap: 2px;
    }}
    .sg-subj {{ font-weight: 600; color: var(--ok); }}
    .sg-pred {{ font-weight: 600; padding: 0 3px; border-radius: 2px; }}
    .sg-obj {{ color: var(--warn); }}
    .sg-empty {{ color: var(--muted); font-style: italic; font-size: 11px; }}

    /* Transcript status in right panel header */
    .transcript-status {{
      margin-top: 6px; padding: 4px 8px;
      background: rgba(83,168,255,0.08);
      border: 1px solid rgba(83,168,255,0.2);
      border-radius: 5px;
      max-height: 48px; overflow-y: auto;
    }}
    .transcript-status .transcript-line {{
      font-size: 11px; line-height: 1.3; color: #e8ecff;
    }}
    .transcript-status .transcript-empty {{
      font-size: 10px; color: var(--muted); font-style: italic;
    }}

    /* Hierarchy content (L1 steps + L0 segments for current window) */
    .hierarchy-content {{
      flex: 1; overflow-y: auto; padding: 8px 14px;
    }}
    .section-label {{
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.05em; color: var(--muted); margin-bottom: 6px;
    }}

    .step-block {{
      margin-bottom: 4px; border: 1px solid var(--border);
      border-radius: 5px; overflow: hidden; transition: border-color 0.2s;
    }}
    .step-block.active {{ border-color: var(--l1); background: rgba(59,130,246,0.06); }}
    .step-header {{
      padding: 5px 8px; background: var(--surface);
      display: flex; align-items: center; gap: 5px;
      cursor: pointer;
    }}
    .step-header:hover {{ background: #2a3358; }}
    .step-summary {{ font-size: 10px; flex: 1; }}
    .phase-badge {{
      font-size: 9px; font-weight: 700; padding: 2px 5px;
      border-radius: 3px; color: white; flex-shrink: 0;
    }}
    .phase-badge.l2 {{ background: var(--l2); }}
    .phase-badge.l1 {{ background: var(--l1); }}
    .phase-badge.l0 {{ background: var(--l0); }}
    .phase-time {{
      font-size: 9px; color: var(--muted); font-family: monospace; flex-shrink: 0;
    }}
    .phase-dur {{ font-size: 9px; color: var(--muted); flex-shrink: 0; }}
    .step-body {{ display: none; padding: 3px 6px 4px 18px; }}
    .step-block.expanded .step-body {{ display: block; }}

    .l0-item {{
      padding: 3px 6px; font-size: 10px; border-radius: 3px;
      margin: 2px 0; display: flex; align-items: center; gap: 4px;
      transition: background 0.15s; border-left: 2px solid transparent;
    }}
    .l0-item.active {{ background: rgba(100,116,139,0.2); border-left-color: var(--l0); }}
    .l0-time {{ font-size: 9px; color: var(--muted); font-family: monospace; flex-shrink: 0; }}
    .l0-desc {{ flex: 1; color: var(--text); }}
    .l0-dur {{ font-size: 9px; color: var(--muted); flex-shrink: 0; }}
    .pred-badge {{
      font-size: 9px; font-weight: 500; padding: 1px 5px;
      border-radius: 3px; border: 1px solid; white-space: nowrap;
    }}
    .idle {{ color: var(--muted); font-style: italic; font-size: 10px; }}

    /* Camera grid */
    .camera-grid {{
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr 1fr;
      gap: 5px;
      padding: 6px;
      background: #0a0e1d;
      min-height: 0;
    }}
    .camera-cell {{
      display: flex; flex-direction: column;
      min-height: 0; border: 1px solid #3a4574;
      border-radius: 5px; overflow: hidden; background: #12182c;
    }}
    .camera-label {{
      font-size: 9px; font-weight: 700; letter-spacing: 0.04em;
      text-transform: uppercase; color: var(--muted);
      padding: 2px 6px; background: #1a2240;
      border-bottom: 1px solid #3a4574; flex-shrink: 0;
    }}
    .camera-frame {{
      flex: 1; width: 100%; min-height: 0;
      object-fit: contain; background: #000;
    }}
    .camera-cell.span-full {{ grid-column: 1 / -1; }}
    .camera-cell.monitor-cell {{ border-color: var(--l2); }}
    .camera-cell.monitor-cell .camera-label {{ background: #1e1040; color: var(--l2); }}

    .foot {{
      font-size: 10px; color: var(--muted); padding: 6px 14px;
      border-top: 1px solid var(--border); flex-shrink: 0;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <!-- LEFT PANEL -->
    <div class="left-panel">
      <div class="panel-header">
        <h2>Hierarchy Video Player</h2>
        <div class="subtitle">Window = L2 Phase | L1 Steps | L0 Atomic | Scene Graph</div>
      </div>
      <div class="controls-bar">
        <button id="playBtn">Play</button>
        <button id="prevBtn">-1s</button>
        <button id="nextBtn">+1s</button>
        <button id="prevWinBtn">◀ Prev Phase</button>
        <button id="nextWinBtn">Next Phase ▶</button>
        <button id="speedBtn">1x</button>
        <input id="timeline" type="range" min="0" max="0" value="0" step="1" />
        <span class="time-display" id="timeDisplay">0s</span>
      </div>
      <div class="role-tabs" id="roleTabs"></div>
      <div class="window-info" id="windowInfo">
        <div class="window-title">
          <span class="badge">L2</span>
          <span class="win-num" id="winNum">Phase 1 / ?</span>
        </div>
        <div class="window-summary" id="winSummary">—</div>
        <div class="window-meta">
          <span id="winTime">?</span>
          <span id="winDur">?</span>
        </div>
        <div class="window-progress">
          <div class="window-progress-label" id="winProgressLabel">0 / 0s</div>
          <div class="window-progress-bar">
            <div class="window-progress-fill" id="winProgressFill"></div>
          </div>
        </div>
      </div>
      <div class="sg-panel">
        <div class="sg-title">Scene Graph at <span id="sgTime">t=?</span></div>
        <div class="sg-triplets" id="sgTriplets">
          <span class="sg-empty">No data</span>
        </div>
      </div>
      <div class="hierarchy-content" id="hierarchyContent">
        <div class="section-label">L1 Steps & L0 Segments in this Phase</div>
      </div>
      <div class="foot">
        Serve: <code>python3 -m http.server 8080 --directory {serve_dir_posix}</code> |
        Open: <code>http://localhost:8080/{html_name}</code>
      </div>
    </div>

    <!-- RIGHT PANEL -->
    <div class="right-panel">
      <div class="panel-header">
        <h2>Multi-Camera View</h2>
        <div class="subtitle" id="frameInfo">\u2014</div>
        <div class="monitor-status" id="monitorStatus"></div>
        <div class="transcript-status" id="transcriptStatus">
          <span class="transcript-empty">\u2014 silence \u2014</span>
        </div>
      </div>
      <div id="cameraGrid" class="camera-grid">
{camera_grid_html}
        <div class="camera-cell monitor-cell" data-camera="simstation">
          <div class="camera-label">Robot Monitor Screen</div>
          <img class="camera-frame" alt="simstation" />
        </div>
      </div>
    </div>
  </div>

  <script>
    const hierarchyData = {data_json};
    const sceneGraphs = {sg_json};
    const colorDir = {json.dumps(color_dir)};
    const cameras = {cameras_json};
    const predColors = {pred_colors_json};
    const entityNames = {entity_names_json};
    const predicateNames = {predicate_names_json};
    const timeStart = {t_start};
    const timeEnd = {t_end};
    const monitorTimeline = hierarchyData.monitor_timeline || [];
    const simstationDir = {json.dumps(simstation_dir)};
    const simstationMap = {simstation_map_json};
    const transcriptEntries = {transcript_json};

    const allValidTimes = Object.keys(sceneGraphs).map(Number).sort((a,b) => a - b);

    // Per-role valid timestamps: only frames where the role appears as a subject
    const roleTimesCache = {{}};
    function buildRoleValidTimes(role) {{
      if (roleTimesCache[role]) return roleTimesCache[role];
      const rd = hierarchyData.roles[role];
      let times;
      if (rd && rd.is_robot) {{
        // Robot roles (monitor logs) never appear as a scene-graph subject;
        // they are valid across the whole colorimage timeline.
        times = allValidTimes.slice();
      }} else if (rd && rd.is_tool) {{
        // Tool roles appear as the object (or implied by an action verb), not
        // the subject -- include any frame where the tool is referenced.
        times = allValidTimes.filter(t => {{
          const triplets = sceneGraphs[String(t)] || [];
          return triplets.some(([s, p, o]) => o === role || s === role);
        }});
        if (!times.length) times = allValidTimes.slice();
      }} else {{
        times = allValidTimes.filter(t => {{
          const triplets = sceneGraphs[String(t)] || [];
          return triplets.some(([s, p, o]) => s === role);
        }});
      }}
      roleTimesCache[role] = times;
      return times;
    }}

    function labelAtTime(timeline, t) {{
      for (let i = 0; i < timeline.length; i++) {{
        if (t >= timeline[i][0] && t <= timeline[i][1]) return timeline[i][2];
      }}
      return null;
    }}

    function renderMonitorStatus() {{
      const el = document.getElementById('monitorStatus');
      if (!el) return;
      const mon = labelAtTime(monitorTimeline, currentTime);
      let html = '';
      if (monitorTimeline.length) {{
        html += `<span class="mon-chip"><span class="mon-key">Robot Monitor</span>` +
          `<span class="mon-val">${{mon ? escapeHtml(mon) : '\\u2014'}}</span></span>`;
      }}
      el.innerHTML = html;
    }}

    let validTimes = allValidTimes;

    function nextValidTime(t) {{
      for (let i = 0; i < validTimes.length; i++) {{ if (validTimes[i] > t) return validTimes[i]; }}
      return t;
    }}
    function prevValidTime(t) {{
      for (let i = validTimes.length - 1; i >= 0; i--) {{ if (validTimes[i] < t) return validTimes[i]; }}
      return t;
    }}
    function nearestValidTime(t) {{
      if (!validTimes.length) return t;
      let best = validTimes[0];
      for (const v of validTimes) {{ if (Math.abs(v - t) < Math.abs(best - t)) best = v; }}
      return best;
    }}
    function nextValidTimeInRange(t, lo, hi) {{
      const nxt = nextValidTime(t);
      return nxt <= hi ? nxt : null;
    }}
    function prevValidTimeInRange(t, lo, hi) {{
      const prv = prevValidTime(t);
      return prv >= lo ? prv : null;
    }}
    function firstValidTimeInRange(lo, hi) {{
      for (const v of validTimes) {{ if (v >= lo && v <= hi) return v; }}
      return null;
    }}

    let currentTime = allValidTimes.length ? allValidTimes[0] : timeStart;
    let currentWindowIdx = 0;
    let timer = null;
    let playing = {autoplay_js};
    let speed = 1;
    let activeRole = null;

    // Per-role L2 windows (the main navigation unit)
    let windows = [];  // array of L2 segments for active role
    let l1Lookup = {{}};
    let l0Lookup = {{}};

    // Init camera grid (5th spans)
    (function() {{
      const cells = document.querySelectorAll('.camera-cell');
      if (cells.length === 5) cells[4].classList.add('span-full');
    }})();

    // Build role tabs
    (function() {{
      const container = document.getElementById('roleTabs');
      const roles = hierarchyData.role_order || Object.keys(hierarchyData.roles).sort();
      roles.forEach((role, i) => {{
        const rd = hierarchyData.roles[role];
        const tab = document.createElement('div');
        tab.className = 'role-tab' + (i === 0 ? ' active' : '');
        tab.dataset.role = role;
        tab.textContent = rd.role_human;
        tab.addEventListener('click', () => selectRole(role));
        container.appendChild(tab);
      }});
      if (roles.length) selectRole(roles[0]);
    }})();

    function selectRole(role) {{
      activeRole = role;
      validTimes = buildRoleValidTimes(role);
      document.querySelectorAll('.role-tab').forEach(t =>
        t.classList.toggle('active', t.dataset.role === role)
      );
      const rd = hierarchyData.roles[role];
      windows = rd.level2 || [];
      const l1segs = rd.level1 || [];
      const l0segs = rd.level0 || [];
      l1Lookup = {{}};
      l1segs.forEach(s => l1Lookup[s.segment_id] = s);
      l0Lookup = {{}};
      l0segs.forEach(s => l0Lookup[s.segment_id] = s);

      // Reset to first window
      currentWindowIdx = 0;
      if (windows.length) {{
        const ws = windows[0].time_start || timeStart;
        const we = windows[0].time_end || timeEnd;
        currentTime = firstValidTimeInRange(ws, we) ?? ws;
      }}
      updateTimeline();
      renderWindow();
      syncUI();
    }}

    function updateTimeline() {{
      const slider = document.getElementById('timeline');
      if (!windows.length) {{ slider.min = timeStart; slider.max = timeEnd; return; }}
      const win = windows[currentWindowIdx];
      slider.min = String(win.time_start || timeStart);
      slider.max = String(win.time_end || timeEnd);
      slider.value = String(currentTime);
    }}

    function findWindowForTime(t) {{
      for (let i = 0; i < windows.length; i++) {{
        if (t >= windows[i].time_start && t <= windows[i].time_end) return i;
      }}
      return currentWindowIdx;
    }}

    function escapeHtml(text) {{
      return String(text).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}

    function humanizeEntity(id) {{
      return entityNames[id] || id.replace(/_/g, ' ');
    }}

    function humanizePred(id) {{
      return predicateNames[id] || id;
    }}

    function predBadgeHtml(pred, obj, toolRole) {{
      const color = predColors[pred] || '#94a3b8';
      const label = toolRole
        ? `${{humanizeEntity(obj)}} ${{humanizePred(pred)}}`
        : `${{humanizePred(pred)}} ${{humanizeEntity(obj)}}`;
      return `<span class="pred-badge" style="background:${{color}}20;color:${{color}};border-color:${{color}}50">${{escapeHtml(label)}}</span>`;
    }}

    function renderSceneGraph() {{
      const triplets = sceneGraphs[String(currentTime)] || [];
      document.getElementById('sgTime').textContent = `t=${{currentTime}}s`;
      const container = document.getElementById('sgTriplets');
      if (!triplets.length) {{
        container.innerHTML = '<span class="sg-empty">No relations at this frame</span>';
        return;
      }}
      container.innerHTML = triplets.map(([s, p, o]) => {{
        const color = predColors[p] || '#94a3b8';
        return `<span class="sg-triplet">` +
          `<span class="sg-subj">${{escapeHtml(s)}}</span>` +
          `<span class="sg-pred" style="background:${{color}}25;color:${{color}}">${{escapeHtml(p)}}</span>` +
          `<span class="sg-obj">${{escapeHtml(o)}}</span></span>`;
      }}).join('');
    }}

    function renderTranscript() {{
      const container = document.getElementById('transcriptStatus');
      const t = currentTime;
      const active = transcriptEntries.filter(e => t >= e.start && t <= e.end);
      if (!active.length) {{
        container.innerHTML = '<span class="transcript-empty">\u2014 silence \u2014</span>';
      }} else {{
        container.innerHTML = active.map(e =>
          `<div class="transcript-line">"${{escapeHtml(e.text)}}"</div>`
        ).join('');
      }}
    }}

    function renderWindow() {{
      if (!windows.length) {{
        document.getElementById('winNum').textContent = 'No phases';
        document.getElementById('winSummary').textContent = '\u2014';
        document.getElementById('winTime').textContent = '';
        document.getElementById('winDur').textContent = '';
        document.getElementById('hierarchyContent').innerHTML = '';
        return;
      }}

      const win = windows[currentWindowIdx];
      document.getElementById('winNum').textContent =
        `Phase ${{currentWindowIdx + 1}} / ${{windows.length}}`;
      document.getElementById('winSummary').textContent = win.summary || '—';
      document.getElementById('winTime').textContent =
        `${{win.time_start}}s – ${{win.time_end}}s`;
      document.getElementById('winDur').textContent =
        `Duration: ${{win.duration_s || '?'}}s`;

      // Render L1 steps and L0 segments for this window
      const childIds = win.child_ids || win.segment_ids || [];
      let html = '<div class="section-label">L1 Steps & L0 Segments in this Phase</div>';

      const isToolRole = !!(activeRole && hierarchyData.roles[activeRole]?.is_tool);

      childIds.forEach(cid => {{
        const step = l1Lookup[cid];
        if (!step) return;
        const stepChildIds = step.segment_ids || [];
        let atomicHtml = '';
        stepChildIds.forEach(sid => {{
          const seg = l0Lookup[sid];
          if (!seg) return;
          const preds = (seg.active_predicates || []);
          const predsHtml = preds.length
            ? preds.map(([p,o]) => predBadgeHtml(p, o, isToolRole)).join(' ')
            : '<span class="idle">idle</span>';
          const dur = seg.duration_s != null ? seg.duration_s + 's' : '?';
          atomicHtml += `<div class="l0-item" data-ts="${{seg.time_start}}" data-te="${{seg.time_end}}">` +
            `<span class="phase-badge l0">L0</span>` +
            `<span class="l0-time">${{seg.time_start}}\u2013${{seg.time_end}}</span>` +
            `<span class="l0-desc">${{predsHtml}}</span>` +
            `<span class="l0-dur">${{dur}}</span></div>`;
        }});
        const dur = step.duration_s != null ? step.duration_s + 's' : '?';
        html += `<div class="step-block expanded" data-ts="${{step.time_start}}" data-te="${{step.time_end}}">` +
          `<div class="step-header" onclick="this.parentElement.classList.toggle('expanded')">` +
          `<span class="phase-badge l1">L1</span>` +
          `<span class="phase-time">${{step.time_start}}\u2013${{step.time_end}}</span>` +
          `<span class="step-summary">${{escapeHtml(step.summary || '')}}</span>` +
          `<span class="phase-dur">${{dur}}</span></div>` +
          `<div class="step-body">${{atomicHtml}}</div></div>`;
      }});

      document.getElementById('hierarchyContent').innerHTML = html;
    }}

    function padded(num, width = 6) {{
      return String(num).padStart(width, '0');
    }}

    function buildFramePath(camera, timestamp) {{
      return `${{colorDir}}/${{camera}}_colorimage-${{padded(timestamp)}}.jpg`;
    }}

    function syncUI() {{
      if (!windows.length) return;
      const win = windows[currentWindowIdx];
      const wStart = win.time_start || timeStart;
      const wEnd = win.time_end || timeEnd;

      if (currentTime < wStart) currentTime = wStart;
      if (currentTime > wEnd) currentTime = wEnd;

      document.getElementById('timeDisplay').textContent = `${{currentTime}}s`;
      document.getElementById('timeline').value = String(currentTime);

      // Window progress
      const wDur = Math.max(wEnd - wStart, 1);
      const elapsed = currentTime - wStart;
      const pct = (elapsed / wDur) * 100;
      document.getElementById('winProgressFill').style.width = pct + '%';
      document.getElementById('winProgressLabel').textContent =
        `${{elapsed}}s / ${{wDur}}s into phase`;

      document.getElementById('frameInfo').textContent =
        `t=${{currentTime}}s | frame ${{padded(currentTime)}} | Phase ${{currentWindowIdx + 1}}/${{windows.length}}`;

      // Scene graph
      renderSceneGraph();

      // Transcript subtitle
      renderTranscript();

      // Robot monitor / setup status (always synced to the frame)
      renderMonitorStatus();

      // Highlight active L1 and L0
      document.querySelectorAll('.step-block').forEach(el => {{
        const ts = Number(el.dataset.ts), te = Number(el.dataset.te);
        const active = currentTime >= ts && currentTime <= te;
        el.classList.toggle('active', active);
      }});
      document.querySelectorAll('.l0-item').forEach(el => {{
        const ts = Number(el.dataset.ts), te = Number(el.dataset.te);
        el.classList.toggle('active', currentTime >= ts && currentTime <= te);
      }});

      // Scroll active L0 into view
      const activeL0 = document.querySelector('.l0-item.active');
      if (activeL0) activeL0.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});

      // Camera frames
      document.querySelectorAll('.camera-cell').forEach(cell => {{
        const cam = cell.dataset.camera;
        const img = cell.querySelector('.camera-frame');
        let path;
        if (cam === 'simstation') {{
          const simFrame = simstationMap[String(currentTime)];
          path = simFrame
            ? `${{simstationDir}}/camera01_${{simFrame}}.jpg`
            : '';
        }} else {{
          path = buildFramePath(cam, currentTime);
        }}
        if (path && img.getAttribute('data-path') !== path) {{
          img.setAttribute('data-path', path);
          img.src = path;
        }}
      }});
    }}

    function tick() {{
      if (!windows.length) return;
      const win = windows[currentWindowIdx];
      const wEnd = win.time_end || timeEnd;
      const nxt = nextValidTimeInRange(currentTime, win.time_start || timeStart, wEnd);
      if (nxt == null) {{
        // No more valid frames in this window -- advance to next
        if (currentWindowIdx < windows.length - 1) {{
          currentWindowIdx++;
          const nw = windows[currentWindowIdx];
          const first = firstValidTimeInRange(nw.time_start || timeStart, nw.time_end || timeEnd);
          currentTime = first ?? (nw.time_start || timeStart);
          updateTimeline();
          renderWindow();
          syncUI();
        }} else {{
          stopPlayback();
        }}
        return;
      }}
      currentTime = nxt;
      syncUI();
    }}

    function startPlayback() {{
      if (timer) return;
      timer = setInterval(tick, 1000 / speed);
      playing = true;
      document.getElementById('playBtn').textContent = 'Pause';
    }}

    function stopPlayback() {{
      if (timer) {{ clearInterval(timer); timer = null; }}
      playing = false;
      document.getElementById('playBtn').textContent = 'Play';
    }}

    function togglePlayback() {{
      if (playing) stopPlayback(); else startPlayback();
    }}

    function cycleSpeed() {{
      const speeds = [1, 2, 4, 8, 16];
      const idx = (speeds.indexOf(speed) + 1) % speeds.length;
      speed = speeds[idx];
      document.getElementById('speedBtn').textContent = speed + 'x';
      if (playing) {{ clearInterval(timer); timer = setInterval(tick, 1000 / speed); }}
    }}

    function goToPrevWindow() {{
      if (currentWindowIdx > 0) {{
        currentWindowIdx--;
        const nw = windows[currentWindowIdx];
        const first = firstValidTimeInRange(nw.time_start || timeStart, nw.time_end || timeEnd);
        currentTime = first ?? (nw.time_start || timeStart);
        updateTimeline();
        renderWindow();
        syncUI();
      }}
    }}

    function goToNextWindow() {{
      if (currentWindowIdx < windows.length - 1) {{
        currentWindowIdx++;
        const nw = windows[currentWindowIdx];
        const first = firstValidTimeInRange(nw.time_start || timeStart, nw.time_end || timeEnd);
        currentTime = first ?? (nw.time_start || timeStart);
        updateTimeline();
        renderWindow();
        syncUI();
      }}
    }}

    document.getElementById('playBtn').addEventListener('click', togglePlayback);
    document.getElementById('prevBtn').addEventListener('click', () => {{
      if (!windows.length) return;
      const wStart = windows[currentWindowIdx].time_start || timeStart;
      const prv = prevValidTimeInRange(currentTime, wStart, windows[currentWindowIdx].time_end || timeEnd);
      if (prv != null) {{
        currentTime = prv;
        syncUI();
      }} else if (currentWindowIdx > 0) {{
        goToPrevWindow();
      }}
    }});
    document.getElementById('nextBtn').addEventListener('click', () => {{
      if (!windows.length) return;
      const wEnd = windows[currentWindowIdx].time_end || timeEnd;
      const nxt = nextValidTimeInRange(currentTime, windows[currentWindowIdx].time_start || timeStart, wEnd);
      if (nxt != null) {{
        currentTime = nxt;
        syncUI();
      }} else if (currentWindowIdx < windows.length - 1) {{
        goToNextWindow();
      }}
    }});
    document.getElementById('prevWinBtn').addEventListener('click', goToPrevWindow);
    document.getElementById('nextWinBtn').addEventListener('click', goToNextWindow);
    document.getElementById('speedBtn').addEventListener('click', cycleSpeed);
    document.getElementById('timeline').addEventListener('input', (e) => {{
      currentTime = nearestValidTime(Number(e.target.value)); syncUI();
    }});

    document.addEventListener('keydown', (e) => {{
      if (e.target.tagName === 'INPUT') return;
      if (e.key === ' ' || e.key === 'k') {{ e.preventDefault(); togglePlayback(); }}
      else if (e.key === 'ArrowLeft') {{
        e.preventDefault();
        const wS = windows[currentWindowIdx]?.time_start || timeStart;
        const wE = windows[currentWindowIdx]?.time_end || timeEnd;
        const prv = prevValidTimeInRange(currentTime, wS, wE);
        if (prv != null) {{ currentTime = prv; syncUI(); }}
        else if (currentWindowIdx > 0) goToPrevWindow();
      }}
      else if (e.key === 'ArrowRight') {{
        e.preventDefault();
        const wS = windows[currentWindowIdx]?.time_start || timeStart;
        const wE = windows[currentWindowIdx]?.time_end || timeEnd;
        const nxt = nextValidTimeInRange(currentTime, wS, wE);
        if (nxt != null) {{ currentTime = nxt; syncUI(); }}
        else if (currentWindowIdx < windows.length - 1) goToNextWindow();
      }}
      else if (e.key === 'ArrowUp' || e.key === 'p') {{ e.preventDefault(); goToPrevWindow(); }}
      else if (e.key === 'ArrowDown' || e.key === 'n') {{ e.preventDefault(); goToNextWindow(); }}
    }});

    syncUI();
    if (playing) startPlayback();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    hierarchy = json.loads(args.hierarchy.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    web_root = args.web_root.resolve() if args.web_root else infer_web_root(args.output, args.colorimage_dir)
    color_dir = color_dir_for_web(args.colorimage_dir, web_root)
    simstation_dir = color_dir_for_web(args.simstation_dir, web_root) if args.simstation_dir.exists() else ""
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    print("Loading per-frame scene graphs...")
    scene_graphs = load_scene_graphs(hierarchy)
    print(f"  {len(scene_graphs)} frames with scene graph data")

    print("Building simstation frame map...")
    simstation_map = build_simstation_map(hierarchy)
    print(f"  {len(simstation_map)} timestamps mapped to simstation frames")

    srt_entries = None
    if args.srt and args.srt.exists():
        print(f"Loading transcript from {args.srt} ...")
        srt_entries = parse_srt(args.srt)
        print(f"  {len(srt_entries)} transcript entries loaded.")

    html = build_html(
        hierarchy=hierarchy,
        scene_graphs=scene_graphs,
        color_dir=color_dir,
        output=args.output,
        serve_dir=web_root,
        cameras=cameras,
        autoplay=args.autoplay,
        simstation_dir=simstation_dir,
        simstation_map=simstation_map,
        srt_entries=srt_entries,
    )
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")
    html_url = Path(os.path.relpath(args.output.resolve(), web_root)).as_posix()
    print(f"Frame URL prefix: {color_dir}/")
    print("Start HTTP server:")
    print(f"  python3 -m http.server 8080 --directory {web_root}")
    print(f"  http://localhost:8080/{html_url}")


if __name__ == "__main__":
    main()
