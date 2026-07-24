#!/bin/bash
# Shared rclone NAS mount helpers for Baseline 2 scripts.
# Expects: RCLONE, NAS_REMOTE, NAS_MOUNT, WORKDIR (optional, for logs)

b2_mount_nas() {
    local max_attempts="${1:-3}"
    local wait_s="${2:-90}"
    local attempt log i
    mkdir -p "$NAS_MOUNT"
    export MM_OR_PROCESSED_ROOT="$NAS_MOUNT/MM-OR_data/MM-OR_processed"
    log="${WORKDIR:-.}/logs/rclone_mount_b2_${SLURM_JOB_ID:-$$}.log"
    mkdir -p "$(dirname "$log")"

    for attempt in $(seq 1 "$max_attempts"); do
        echo "[nas] Mount attempt $attempt/$max_attempts → $NAS_MOUNT"
        if mountpoint -q "$NAS_MOUNT" 2>/dev/null; then
            fusermount -uz "$NAS_MOUNT" 2>/dev/null || true
            sleep 1
        fi
        "$RCLONE" mount "$NAS_REMOTE" "$NAS_MOUNT" \
            --vfs-cache-mode full \
            --dir-cache-time 72h \
            --poll-interval 1m \
            --log-file "$log" \
            --log-level INFO \
            --daemon
        for i in $(seq 1 "$wait_s"); do
            if [ -d "$MM_OR_PROCESSED_ROOT/001_PKA" ]; then
                echo "[nas] Ready: $MM_OR_PROCESSED_ROOT (after ${i}s)"
                return 0
            fi
            sleep 1
        done
        echo "[nas] WARN: not ready after ${wait_s}s (log: $log)" >&2
        tail -20 "$log" 2>/dev/null || true
        fusermount -uz "$NAS_MOUNT" 2>/dev/null || true
        sleep 5
    done
    echo "[nas] ERROR: mount failed after $max_attempts attempts" >&2
    return 1
}

b2_unmount_nas() {
    echo "[cleanup] Unmounting NAS..."
    fusermount -uz "$NAS_MOUNT" 2>/dev/null || true
    # Only remove empty mount dir (stable path may be reused)
    rmdir "$NAS_MOUNT" 2>/dev/null || true
}
