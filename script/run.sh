#!/bin/bash
set -e

# Usage:
#   ./run.sh [TAKE]              Process one take (default: 001_PKA)
#   ./run.sh --all               Process all takes with scene graphs on NAS/local root
#   USE_NAS=0 ./run.sh 001_PKA   Use local mm-or copy instead of NAS mount
#
# Environment (HF_TOKEN can be overridden by exporting before running):
#   MM_OR_PROCESSED_ROOT  Override dataset root (auto-set when NAS is mounted)
#   NAS_REMOTE            rclone remote (default: nas:ge42faj)
#   NAS_MOUNT             Local mount point (default: /tmp/nhatvu/nas_mount — outside repo)
#   USE_NAS               1 = mount NAS (default), 0 = use local mm-or copy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export HF_TOKEN="${HF_TOKEN:-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}"

USE_NAS="${USE_NAS:-1}"
NAS_REMOTE="${NAS_REMOTE:-nas:ge42faj}"
NAS_MOUNT="${NAS_MOUNT:-/tmp/nhatvu/nas_mount}"
VIEWER_LINKS="$WORKDIR/viewer_links"
LOCAL_PROCESSED_ROOT="$WORKDIR/mm-or/MM-OR_data/MM-OR_processed"

VENV=/tmp/nhatvu/.venv312
HF_CACHE="${HF_HOME:-/tmp/nhatvu/hf_cache}"
PYTHON=$VENV/bin/python3
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
HIER_DIR=$WORKDIR/hierarchy_output
LOG=/tmp/nhatvu/run.log

# ---------------------------------------------------------------------------
# Dataset root: NAS mount or local copy
# ---------------------------------------------------------------------------
setup_dataset_root() {
    if [ -n "${MM_OR_PROCESSED_ROOT:-}" ] && [ -d "$MM_OR_PROCESSED_ROOT" ]; then
        echo "[data] Using MM_OR_PROCESSED_ROOT=$MM_OR_PROCESSED_ROOT"
        return 0
    fi

    if [ "$USE_NAS" = "1" ]; then
        # shellcheck source=mount_nas.sh
        source "$SCRIPT_DIR/mount_nas.sh"
        mount_nas
        echo "[data] NAS dataset root: $MM_OR_PROCESSED_ROOT"
        return 0
    fi

    MM_OR_PROCESSED_ROOT="$LOCAL_PROCESSED_ROOT"
    export MM_OR_PROCESSED_ROOT
    echo "[data] Local dataset root: $MM_OR_PROCESSED_ROOT"
}

list_all_takes() {
    MM_OR_PROCESSED_ROOT="$MM_OR_PROCESSED_ROOT" "$PYTHON" "$WORKDIR/mm_or_dataset.py" --list-takes
}

# Symlinks under viewer_links/ point at NAS data in /tmp — no repo disk usage.
setup_viewer_links() {
    local TAKE="$1"
    local TAKE_DIR="$2"
    local LINK_DIR="$VIEWER_LINKS/$TAKE"
    mkdir -p "$LINK_DIR"
    ln -sfn "$TAKE_DIR/colorimage" "$LINK_DIR/colorimage"
    if [ -d "$TAKE_DIR/simstation" ]; then
        ln -sfn "$TAKE_DIR/simstation" "$LINK_DIR/simstation"
    fi
    echo "[viewer] Symlinks: $LINK_DIR -> $TAKE_DIR"
}

run_one_take() {
    local TAKE="$1"
    local TAKE_DIR="$MM_OR_PROCESSED_ROOT/$TAKE"
    local SRT="$MM_OR_PROCESSED_ROOT/take_transcripts/${TAKE}.srt"
    local HIER_JSON="$HIER_DIR/${TAKE}_hierarchy_qwen27b.json"
    local HIER_HTML="$HIER_DIR/${TAKE}_hierarchy_qwen27b.html"
    local VIDEO_HTML="$WORKDIR/${TAKE}_hierarchy_video_sync_qwen27b.html"

    if [ ! -d "$TAKE_DIR/relation_labels" ]; then
        echo "[skip] $TAKE: no relation_labels at $TAKE_DIR" >&2
        return 0
    fi

    echo "[run] take=$TAKE dir=$TAKE_DIR"

    local SRT_ARGS=()
    if [ -f "$SRT" ]; then
        SRT_ARGS=(--srt "$SRT")
        echo "[run] transcript: $SRT"
    else
        echo "[run] transcript: (none — OK on NAS)"
    fi

    echo "[hierarchy] Building level-0 + level-1 + level-2 ..."
    cd "$WORKDIR"
    HF_HOME=$HF_CACHE $PYTHON build_hierarchy_qwen.py \
        --take_dir "$TAKE_DIR" \
        --start_tp 000000 \
        --level2 \
        --model Qwen/Qwen3.5-27B \
        --hf_token "$HF_TOKEN" \
        "${SRT_ARGS[@]}" \
        --output "$HIER_JSON"

    echo "[visualize] Generating hierarchy HTML viewer ..."
    $PYTHON visualize_hierarchy.py \
        --input  "$HIER_JSON" \
        --output "$HIER_HTML"

    echo "[render] Generating hierarchy video player ..."
    setup_viewer_links "$TAKE" "$TAKE_DIR"
    local VIEWER_TAKE_DIR="$VIEWER_LINKS/$TAKE"
    local SIM_ARGS=()
    if [ -d "$VIEWER_TAKE_DIR/simstation" ]; then
        SIM_ARGS=(--simstation_dir "$VIEWER_TAKE_DIR/simstation")
    fi
    $PYTHON render_hierarchy_video.py \
        --hierarchy "$HIER_JSON" \
        --colorimage_dir "$VIEWER_TAKE_DIR/colorimage" \
        "${SIM_ARGS[@]}" \
        --web_root "$WORKDIR" \
        --output "$VIDEO_HTML"

    echo "[done] $TAKE"
    echo "  JSON  : $HIER_JSON"
    echo "  HTML  : $HIER_HTML"
    echo "  Video : $VIDEO_HTML"
}

# ---------------------------------------------------------------------------
# 1. Setup virtual environment
# ---------------------------------------------------------------------------
ensure_venv() {
    if [ ! -f "$PYTHON" ]; then
        echo "[setup] Creating Python 3.12 venv at $VENV via uv ..."
        if [ ! -f "$UV" ]; then
            echo "[setup] uv not found at $UV; install it first (pip install uv)." >&2
            exit 1
        fi
        $UV python install 3.12
        $UV venv --python 3.12 "$VENV"
    else
        echo "[setup] Venv already exists at $VENV"
    fi

    echo "[setup] Ensuring Python dependencies ..."
    $UV pip install --python "$PYTHON" \
        "transformers>=5.2.0" torch accelerate tqdm bitsandbytes sentence-transformers
}

ensure_venv

mkdir -p "$HIER_DIR" /tmp/nhatvu
setup_dataset_root

# ---------------------------------------------------------------------------
# 2. Run pipeline
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--all" ]; then
    echo "[run] Processing all takes with scene graphs ..."
    mapfile -t ALL_TAKES < <(list_all_takes)
    echo "[run] Found ${#ALL_TAKES[@]} takes: ${ALL_TAKES[*]}"
    for TAKE in "${ALL_TAKES[@]}"; do
        echo ""
        echo "============================================"
        run_one_take "$TAKE"
        echo "============================================"
    done
else
    TAKE="${1:-001_PKA}"
    run_one_take "$TAKE"
fi

echo ""
echo "All done."
