#!/usr/bin/env python3
"""
Render a synchronized HTML "video" player:
- Left: 5-second summary window details
- Right: frame image (1 frame/second)

The player uses MM-OR naming convention:
  {camera}_colorimage-{frame_id}.jpg
where frame_id is the Azure/colorimage frame number (e.g. 000329),
not the relation-label timepoint index (000000).
Summaries align with original_timestamp starting at --start_timestamp (329).

Writes summary_video_sync.html to the project root (dlhm). Serve HTTP from dlhm:
  python3 -m http.server 8080 --directory .
  http://localhost:8080/summary_video_sync.html
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from visualize_annotations import PREDICATE_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side HTML player for frames and 5s summaries."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("annotation_output/annotations_001_PKA_600.jsonl"),
        help="Path to JSONL summary file.",
    )
    parser.add_argument(
        "--colorimage_dir",
        type=Path,
        default=Path("mm-or/MM-OR_data/MM-OR_processed/001_PKA/colorimage"),
        help="Directory containing color image frames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("summary_video_sync.html"),
        help="Output HTML path (default: project root).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="camera01,camera02,camera03,camera04,camera05",
        help="Comma-separated camera ids (e.g. camera01,camera02,...).",
    )
    parser.add_argument(
        "--start_timestamp",
        type=int,
        default=329,
        help="Real timestamp corresponding to timepoint 000000.",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Start playback automatically.",
    )
    parser.add_argument(
        "--web_root",
        type=Path,
        default=None,
        help="HTTP server root for frame paths (default: auto-detect from output + colorimage).",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No annotation records found in {path}")
    return rows


def infer_start_timestamp(annotations: list[dict]) -> int | None:
    """First original_timestamp in JSONL (for fallback when a second is unmapped)."""
    for item in annotations:
        for ts in item.get("original_timestamps") or []:
            if ts is not None:
                return int(ts)
        for fr in item.get("colorimage_frames") or []:
            if fr and fr.get("original_timestamp") is not None:
                return int(fr["original_timestamp"])
    return None


def infer_web_root(output: Path, colorimage_dir: Path) -> Path:
    """Pick a directory that contains both the HTML output and colorimage frames."""
    output = output.resolve()
    colorimage_dir = colorimage_dir.resolve()
    for candidate in (Path.cwd().resolve(), output.parent.resolve(), output.parent.parent.resolve()):
        try:
            output.relative_to(candidate)
            colorimage_dir.relative_to(candidate)
            return candidate
        except ValueError:
            continue
    return output.parent.resolve()


def parse_cameras(cameras_arg: str) -> list[str]:
    return [c.strip() for c in cameras_arg.split(",") if c.strip()]


def color_dir_for_web(colorimage_dir: Path, web_root: Path) -> str:
    """Path used in <img src>, relative to web_root (HTTP server --directory)."""
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


def build_html(
    annotations: list[dict],
    color_dir: str,
    output: Path,
    serve_dir: Path,
    cameras: list[str],
    start_timestamp: int,
    autoplay: bool,
) -> str:
    # Keep only frontend fields needed by JS.
    compact = [
        {
            "window_start_tp": int(item["window_start_tp"]),
            "window_end_tp": int(item["window_end_tp"]),
            "original_timestamps": item["original_timestamps"],
            "colorimage_frames": item.get("colorimage_frames", []),
            "scene_graphs": item.get("scene_graphs", []),
            "summary": item["atomic_action_summary"],
            "issues": item.get("issues", []),
        }
        for item in annotations
    ]
    data_json = json.dumps(compact, ensure_ascii=False)
    pred_colors_json = json.dumps(PREDICATE_COLORS)
    html_name = output.name
    serve_dir_posix = serve_dir.resolve().as_posix()
    autoplay_js = "true" if autoplay else "false"
    cameras_json = json.dumps(cameras)
    camera_grid_html = build_camera_grid_html(cameras)
    cameras_label = ", ".join(cameras)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Summary + Frame Player</title>
  <style>
    :root {{
      --bg: #0f1220;
      --card: #1a1f35;
      --text: #ecf0ff;
      --muted: #a8b1d9;
      --accent: #53a8ff;
      --ok: #58d68d;
      --warn: #f5b041;
      --danger: #ff6b6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 16px; font-family: Arial, sans-serif;
      background: var(--bg); color: var(--text);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 0.9fr) minmax(480px, 1.4fr);
      gap: 14px;
      min-height: calc(100vh - 32px);
    }}
    .panel {{
      background: var(--card);
      border-radius: 10px;
      padding: 14px;
      border: 1px solid #2b3356;
      overflow: hidden;
    }}
    .title {{ font-size: 18px; font-weight: 700; margin-bottom: 12px; }}
    .meta {{
      font-size: 14px; color: var(--muted); margin-bottom: 10px;
      display: grid; gap: 5px;
    }}
    .badge {{
      display: inline-block; padding: 4px 8px; border-radius: 999px;
      font-size: 12px; font-weight: 700; margin-right: 6px;
      border: 1px solid transparent;
    }}
    .start {{ background: rgba(88,214,141,0.14); color: var(--ok); border-color: var(--ok); }}
    .middle {{ background: rgba(83,168,255,0.14); color: var(--accent); border-color: var(--accent); }}
    .end {{ background: rgba(245,176,65,0.14); color: var(--warn); border-color: var(--warn); }}
    .summary-box {{
      margin-top: 8px; font-size: 17px; line-height: 1.45;
      padding: 12px; border-left: 4px solid var(--accent);
      background: rgba(83,168,255,0.08); border-radius: 8px;
      min-height: 130px;
    }}
    .progress-wrap {{ margin-top: 12px; }}
    .progress-label {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    .progress {{
      width: 100%; height: 14px; border-radius: 20px;
      background: #2a335d; overflow: hidden; border: 1px solid #3a4574;
    }}
    .progress-fill {{
      height: 100%; width: 0%;
      background: linear-gradient(90deg, var(--accent), #8fbcff);
      transition: width 120ms linear;
    }}
    .camera-grid {{
      height: calc(100vh - 100px);
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr 1fr;
      gap: 8px;
      border: 2px solid #2f3a67;
      border-radius: 10px;
      padding: 8px;
      background: #0a0e1d;
      transition: border-color 120ms linear, box-shadow 120ms linear;
    }}
    .camera-grid.start-border {{ border-color: var(--ok); box-shadow: 0 0 0 2px rgba(88,214,141,0.25) inset; }}
    .camera-grid.end-border {{ border-color: var(--warn); box-shadow: 0 0 0 2px rgba(245,176,65,0.25) inset; }}
    .camera-cell {{
      display: flex;
      flex-direction: column;
      min-height: 0;
      border: 1px solid #3a4574;
      border-radius: 8px;
      overflow: hidden;
      background: #12182c;
    }}
    .camera-label {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      padding: 4px 8px;
      background: #1a2240;
      border-bottom: 1px solid #3a4574;
      flex-shrink: 0;
    }}
    .camera-frame {{
      flex: 1;
      width: 100%;
      min-height: 0;
      object-fit: contain;
      background: #000;
    }}
    .camera-cell.span-full {{ grid-column: 1 / -1; }}
    .controls {{
      margin-top: 10px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    }}
    button {{
      border: 1px solid #44518b; background: #1e2645; color: var(--text);
      padding: 8px 12px; border-radius: 8px; cursor: pointer;
    }}
    button:hover {{ background: #2a3562; }}
    input[type="range"] {{ width: 300px; }}
    .left-panel {{
      max-height: calc(100vh - 32px);
      overflow-y: auto;
    }}
    .sg-section {{
      margin-top: 14px;
      border-top: 1px solid #2b3356;
      padding-top: 12px;
    }}
    .sg-heading {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .sg-entity {{
      font-family: ui-monospace, monospace;
      font-size: 12px;
      color: var(--text);
    }}
    .sg-pred {{
      font-family: ui-monospace, monospace;
      font-size: 11px;
      font-weight: 600;
      padding: 1px 6px;
      border-radius: 4px;
      border: 1px solid;
      margin: 0 2px;
    }}
    .sg-empty {{ color: var(--muted); font-style: italic; font-size: 13px; }}
    .sg-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .sg-table tr {{ border-bottom: 1px solid #2b3356; }}
    .sg-table tr:last-child {{ border-bottom: none; }}
    .sg-ts {{
      color: var(--muted);
      font-family: monospace;
      white-space: nowrap;
      padding: 6px 10px 6px 0;
      vertical-align: top;
      width: 72px;
    }}
    .sg-rels {{ padding: 6px 0; line-height: 1.55; }}
    .sg-dot {{ color: var(--muted); }}
    .issues-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 8px;
    }}
    .issue-item {{
      font-size: 11px;
      padding: 6px 10px;
      border-radius: 6px;
      background: rgba(127, 29, 29, 0.25);
      border: 1px solid rgba(248, 113, 113, 0.35);
      color: #fca5a5;
      line-height: 1.45;
      font-family: ui-monospace, monospace;
    }}
    .foot {{ font-size: 12px; color: var(--muted); margin-top: 8px; }}
    .missing {{ color: var(--danger); font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel left-panel">
      <div class="title">5s Summary Timeline</div>
      <div class="meta">
        <div id="windowLine"></div>
        <div id="timeLine"></div>
        <div id="frameLine"></div>
      </div>
      <div>
        <span id="stateBadge" class="badge middle">MIDDLE OF WINDOW</span>
      </div>
      <div id="summaryText" class="summary-box"></div>
      <div class="progress-wrap">
        <div class="progress-label" id="progressLabel"></div>
        <div class="progress"><div id="progressFill" class="progress-fill"></div></div>
      </div>
      <div class="controls">
        <button id="playBtn">Play</button>
        <button id="prevBtn">-1s</button>
        <button id="nextBtn">+1s</button>
        <button id="jumpPrevWindowBtn">Prev Window</button>
        <button id="jumpNextWindowBtn">Next Window</button>
        <input id="timeline" type="range" min="0" max="0" value="0" step="1" />
      </div>
      <div class="sg-section">
        <div class="sg-heading">Scene graphs (5s window)</div>
        <table class="sg-table">
          <tbody id="sceneGraphWindowBody"></tbody>
        </table>
        <div id="issuesBox" class="issues-list" style="display:none"></div>
      </div>
      <div class="foot">
        Window start/end is highlighted with badge + frame border color.<br />
        Start timestamp = {start_timestamp}, cameras = {cameras_label}, playback = 1 FPS.<br />
        <strong>Serve from:</strong> <code>{serve_dir_posix}</code><br />
        <code>python3 -m http.server 8080 --directory {serve_dir_posix}</code><br />
        Open: <code>http://localhost:8080/{html_name}</code>
      </div>
      <div id="missingMsg" class="missing"></div>
    </section>

    <section class="panel">
      <div class="title">Multi-Camera View ({len(cameras)} cameras)</div>
      <div id="cameraGrid" class="camera-grid">
{camera_grid_html}
      </div>
    </section>
  </div>

  <script>
    const annotations = {data_json};
    const colorDir = {json.dumps(color_dir)};
    const cameras = {cameras_json};
    const startTimestamp = {start_timestamp};
    const predColors = {pred_colors_json};

    // 5th camera spans full width on bottom row
    (function initCameraGrid() {{
      const grid = document.getElementById('cameraGrid');
      const cells = grid.querySelectorAll('.camera-cell');
      if (cells.length === 5) cells[4].classList.add('span-full');
    }})();
    let currentGlobalSecond = 0;
    let lastRenderedWindow = -1;
    let timer = null;
    let playing = {autoplay_js};

    const maxSecond = annotations.length * 5 - 1;
    const timeline = document.getElementById('timeline');
    timeline.max = String(maxSecond);

    function getWindowIndex(globalSecond) {{
      return Math.floor(globalSecond / 5);
    }}

    function getOffsetInWindow(globalSecond) {{
      return globalSecond % 5;
    }}

    function padded(num, width = 6) {{
      return String(num).padStart(width, '0');
    }}

    /** Wall-clock second for this offset (camera01 filename uses this). */
    function originalTimestampAt(record, offset) {{
      const tsList = record.original_timestamps || [];
      if (tsList[offset] != null) return tsList[offset];
      const frames = record.colorimage_frames || [];
      if (frames[offset] && frames[offset].original_timestamp != null) {{
        return frames[offset].original_timestamp;
      }}
      for (let i = 0; i < 5; i++) {{
        if (tsList[i] != null) return tsList[i] + (offset - i);
        if (frames[i] && frames[i].original_timestamp != null) {{
          return frames[i].original_timestamp + (offset - i);
        }}
      }}
      return null;
    }}

    function buildFramePath(camera, globalSecond) {{
      const winIdx = getWindowIndex(globalSecond);
      const offset = getOffsetInWindow(globalSecond);
      const record = annotations[winIdx];
      const frames = record.colorimage_frames || [];

      // camera01 (Azure): frame id == original_timestamp, not relation-label tp id
      if (camera === 'camera01') {{
        let ts = originalTimestampAt(record, offset);
        if (ts == null) ts = startTimestamp + globalSecond;
        return `${{colorDir}}/${{camera}}_colorimage-${{padded(ts)}}.jpg`;
      }}

      let frameId = null;
      if (frames[offset] && frames[offset][camera]) {{
        frameId = frames[offset][camera];
      }}
      if (!frameId) {{
        for (let i = 0; i < 5; i++) {{
          if (frames[i] && frames[i][camera]) {{
            const base = parseInt(frames[i][camera], 10);
            frameId = padded(base + (offset - i));
            break;
          }}
        }}
      }}
      if (!frameId) {{
        frameId = padded(startTimestamp + globalSecond);
      }}
      return `${{colorDir}}/${{camera}}_colorimage-${{frameId}}.jpg`;
    }}

    function predicateColor(pred) {{
      return predColors[pred] || '#94a3b8';
    }}

    function renderTriplet(s, p, o) {{
      const color = predicateColor(p);
      return (
        `<span class="sg-entity">${{escapeHtml(s)}}</span> ` +
        `<span class="sg-pred" style="background:${{color}}33;color:${{color}};border-color:${{color}}66">` +
        `${{escapeHtml(p)}}</span> ` +
        `<span class="sg-entity">${{escapeHtml(o)}}</span>`
      );
    }}

    function escapeHtml(text) {{
      return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }}

    function renderSceneGraphWindow(record, winIdx) {{
      const sceneGraphs = record.scene_graphs || [];
      const timestamps = record.original_timestamps || [];
      const tbody = document.getElementById('sceneGraphWindowBody');
      tbody.innerHTML = sceneGraphs.map((sg, i) => {{
        const t = timestamps[i] ?? '?';
        const rels = sg.length
          ? sg.map(([s, p, o]) => renderTriplet(s, p, o)).join(' <span class="sg-dot">·</span> ')
          : '<span class="sg-empty">—</span>';
        return `<tr><td class="sg-ts">t=${{t}}s</td><td class="sg-rels">${{rels}}</td></tr>`;
      }}).join('');

      const issuesBox = document.getElementById('issuesBox');
      const issues = record.issues || [];
      if (!issues.length) {{
        issuesBox.style.display = 'none';
        issuesBox.innerHTML = '';
        return;
      }}
      issuesBox.style.display = 'flex';
      issuesBox.innerHTML =
        '<div class="sg-heading">Issues</div>' +
        issues.map((issue) => `<div class="issue-item">⚠ ${{escapeHtml(issue)}}</div>`).join('');
    }}

    function syncUI() {{
      if (currentGlobalSecond < 0) currentGlobalSecond = 0;
      if (currentGlobalSecond > maxSecond) currentGlobalSecond = maxSecond;

      const winIdx = getWindowIndex(currentGlobalSecond);
      const offset = getOffsetInWindow(currentGlobalSecond);
      const record = annotations[winIdx];
      const isStart = offset === 0;
      const isEnd = offset === 4;

      const tsList = record.original_timestamps || [];
      const ts = tsList[offset] ?? null;
      const tStart = tsList[0] ?? null;
      const tEnd = tsList[tsList.length - 1] ?? null;
      const tsLabel = ts !== null ? `${{ts}}s` : 'unmapped';
      const winRange =
        tStart !== null && tEnd !== null
          ? `${{tStart}}s -> ${{tEnd}}s`
          : `tp ${{record.window_start_tp}}-${{record.window_end_tp}}`;

      document.getElementById('windowLine').textContent =
        `Window ${{winIdx + 1}} / ${{annotations.length}} (5s each)`;
      document.getElementById('timeLine').textContent =
        `Current timestamp: ${{tsLabel}} | Window: ${{winRange}}`;
      const ogTs = originalTimestampAt(record, offset);
      const frameLabel = ogTs != null ? `t=${{ogTs}} (${{padded(ogTs)}})` : 'unmapped';
      document.getElementById('frameLine').textContent =
        `In window: ${{offset + 1}}/5 | ${{cameras[0]}}: ${{frameLabel}}`;
      document.getElementById('summaryText').textContent = record.summary;
      if (winIdx !== lastRenderedWindow) {{
        renderSceneGraphWindow(record, winIdx);
        lastRenderedWindow = winIdx;
      }}
      document.getElementById('progressLabel').textContent = `Window progress: ${{offset + 1}} / 5`;
      document.getElementById('progressFill').style.width = `${{(offset + 1) * 20}}%`;
      timeline.value = String(currentGlobalSecond);

      const badge = document.getElementById('stateBadge');
      badge.className = 'badge';
      if (isStart) {{
        badge.classList.add('start');
        badge.textContent = 'WINDOW START';
      }} else if (isEnd) {{
        badge.classList.add('end');
        badge.textContent = 'WINDOW END';
      }} else {{
        badge.classList.add('middle');
        badge.textContent = 'MIDDLE OF WINDOW';
      }}

      const cameraGrid = document.getElementById('cameraGrid');
      cameraGrid.classList.remove('start-border', 'end-border');
      if (isStart) cameraGrid.classList.add('start-border');
      if (isEnd) cameraGrid.classList.add('end-border');

      const missing = document.getElementById('missingMsg');
      const missingPaths = [];
      let pending = cameras.length;

      function reportMissing() {{
        if (missingPaths.length === 0) {{
          missing.textContent = '';
          return;
        }}
        missing.textContent =
          `Missing ${{missingPaths.length}} frame(s): ` +
          missingPaths.map((p) => p.split('/').pop()).join(', ');
      }}

      document.querySelectorAll('.camera-cell').forEach((cell) => {{
        const cam = cell.dataset.camera;
        const img = cell.querySelector('.camera-frame');
        const imgPath = buildFramePath(cam, currentGlobalSecond);
        img.onload = () => {{
          pending -= 1;
          if (pending === 0) reportMissing();
        }};
        img.onerror = () => {{
          if (!missingPaths.includes(imgPath)) missingPaths.push(imgPath);
          pending -= 1;
          reportMissing();
        }};
        img.src = imgPath;
      }});
    }}

    function tick() {{
      if (currentGlobalSecond >= maxSecond) {{
        stopPlayback();
        return;
      }}
      currentGlobalSecond += 1;
      syncUI();
    }}

    function startPlayback() {{
      if (timer) return;
      timer = setInterval(tick, 1000); // 1 frame per second
      playing = true;
      document.getElementById('playBtn').textContent = 'Pause';
    }}

    function stopPlayback() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
      }}
      playing = false;
      document.getElementById('playBtn').textContent = 'Play';
    }}

    function togglePlayback() {{
      if (playing) stopPlayback();
      else startPlayback();
    }}

    document.getElementById('playBtn').addEventListener('click', togglePlayback);
    document.getElementById('prevBtn').addEventListener('click', () => {{
      currentGlobalSecond = Math.max(0, currentGlobalSecond - 1);
      syncUI();
    }});
    document.getElementById('nextBtn').addEventListener('click', () => {{
      currentGlobalSecond = Math.min(maxSecond, currentGlobalSecond + 1);
      syncUI();
    }});
    document.getElementById('jumpPrevWindowBtn').addEventListener('click', () => {{
      const prev = Math.max(0, getWindowIndex(currentGlobalSecond) - 1);
      currentGlobalSecond = prev * 5;
      syncUI();
    }});
    document.getElementById('jumpNextWindowBtn').addEventListener('click', () => {{
      const next = Math.min(annotations.length - 1, getWindowIndex(currentGlobalSecond) + 1);
      currentGlobalSecond = next * 5;
      syncUI();
    }});
    timeline.addEventListener('input', (e) => {{
      currentGlobalSecond = Number(e.target.value);
      syncUI();
    }});

    syncUI();
    if (playing) startPlayback();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    web_root = args.web_root.resolve() if args.web_root else infer_web_root(args.output, args.colorimage_dir)
    color_dir = color_dir_for_web(args.colorimage_dir, web_root)

    inferred = infer_start_timestamp(annotations)
    start_ts = inferred if inferred is not None else args.start_timestamp
    if inferred is not None and args.start_timestamp != 329 and args.start_timestamp != inferred:
        print(
            f"Note: --start_timestamp {args.start_timestamp} ignored for mapped seconds; "
            f"JSONL begins at t={inferred}."
        )

    cameras = parse_cameras(args.cameras)
    html = build_html(
        annotations=annotations,
        color_dir=color_dir,
        output=args.output,
        serve_dir=web_root,
        cameras=cameras,
        start_timestamp=start_ts,
        autoplay=args.autoplay,
    )
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")
    html_url = Path(os.path.relpath(args.output.resolve(), web_root)).as_posix()
    print(f"Frame URL prefix: {color_dir}/")
    print("Start HTTP server from project root:")
    print(f"  cd {web_root}")
    print(f"  python3 -m http.server 8080 --directory .")
    print(f"  http://localhost:8080/{html_url}")


if __name__ == "__main__":
    main()
