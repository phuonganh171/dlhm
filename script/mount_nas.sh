#!/bin/bash
# Source or run this script to mount the MM-OR NAS via rclone.
#
# Usage:
#   source script/mount_nas.sh
#   # or
#   bash script/mount_nas.sh

NAS_REMOTE="${NAS_REMOTE:-nas:ge42faj}"
NAS_MOUNT="${NAS_MOUNT:-/tmp/nhatvu/nas_mount}"
MM_OR_PROCESSED_ROOT="${MM_OR_PROCESSED_ROOT:-$NAS_MOUNT/MM-OR_data/MM-OR_processed}"

mount_nas() {
    if mountpoint -q "$NAS_MOUNT" 2>/dev/null; then
        echo "[nas] Already mounted at $NAS_MOUNT"
        export MM_OR_PROCESSED_ROOT
        return 0
    fi

    if ! command -v rclone >/dev/null 2>&1; then
        echo "[nas] ERROR: rclone not found on PATH" >&2
        return 1
    fi

    mkdir -p "$NAS_MOUNT"
    echo "[nas] Mounting $NAS_REMOTE -> $NAS_MOUNT ..."
    rclone mount "$NAS_REMOTE" "$NAS_MOUNT" \
        --vfs-cache-mode full \
        --dir-cache-time 72h \
        --poll-interval 1m \
        --daemon

    # Wait until the processed root is visible
    for _ in $(seq 1 30); do
        if [ -d "$MM_OR_PROCESSED_ROOT/001_PKA" ]; then
            echo "[nas] Ready: $MM_OR_PROCESSED_ROOT"
            export MM_OR_PROCESSED_ROOT
            return 0
        fi
        sleep 1
    done

    echo "[nas] ERROR: mount timed out waiting for $MM_OR_PROCESSED_ROOT" >&2
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    mount_nas
fi
