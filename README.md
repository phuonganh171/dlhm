# DLHM — Hierarchy Annotation

## Run the annotation pipeline

```bash
bash script/run.sh [TAKE]    # one take (default: 001_PKA)
bash script/run.sh --all     # all takes with scene graphs
```

`run.sh` sets up the Python env, mounts the dataset (NAS by default), then for each take runs. Cloning the repo does not give NAS access — that needs a configured rclone `nas` remote; otherwise use `USE_NAS=0` or set `MM_OR_PROCESSED_ROOT` to a local copy.

1. **`build_hierarchy_qwen.py`** — builds level-0 / level-1 / level-2 hierarchy annotations with Qwen3.5-27B  
2. **`visualize_hierarchy.py`** — static HTML hierarchy viewer  
3. **`render_hierarchy_video.py`** — video-synced HTML player (with frame images via `viewer_links/`)

**Outputs** (per take, e.g. `001_PKA`):

| File | Description |
|------|-------------|
| `hierarchy_output/<TAKE>_hierarchy_qwen27b.json` | Hierarchy annotation JSON |
| `hierarchy_output/<TAKE>_hierarchy_qwen27b.html` | Static hierarchy HTML viewer |
| `<TAKE>_hierarchy_video_sync_qwen27b.html` | Interactive video-synced player |

## Interactive web viewer

```bash
bash script/serve_viewer.sh
```

Serves the video-synced hierarchy HTML on port **8080**. Open the printed URL in your browser. Run `run.sh` first so the HTML and frame links exist.

## Edit annotations

```bash
bash script/serve_editor.sh
```

Starts the hierarchy annotation editor on port **8081**. Open `http://localhost:8081/hierarchy_editor.html` to load, edit, and save hierarchy JSON under `hierarchy_output/`. Edits are written back to the JSON; an edit-history log is kept alongside each file, and viewers can be regenerated from the updated annotations.
