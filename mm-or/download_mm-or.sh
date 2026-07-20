#!/bin/bash
set -e

# Usage: ./script.sh [--dry-run]
# If the --dry-run flag is provided, the script will only validate that the URLs are reachable.
# Without the flag, it will download and extract the files.

# Check for the dry-run flag
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=1
    echo "Running in dry-run mode. No files will be downloaded or unzipped."
else
    DRY_RUN=0
fi

# Base URL for the dataset zip files
BASE_URL="https://www.campar.in.tum.de/public_datasets/2025_cvpr_oezsoy/MM-OR"

# List of zip files to process
# FILES=(
#     "001_PKA.zip"
#     "002_PKA.zip"
#     "003_TKA.zip"
#     "004_PKA.zip"
#     "005_TKA.zip"
#     "006_PKA.zip"
#     "007_TKA.zip"
#     "008_PKA.zip"
#     "009_TKA.zip"
#     "010_PKA.zip"
#     "011_TKA.zip"
#     "012_PKA.zip"
#     "013_PKA.zip"
#     "014_PKA.zip"
#     "015-018_PKA.zip"
#     "019-022_PKA.zip"
#     "023-032_PKA.zip"
#     "033_PKA.zip"
#     "035_PKA.zip"
#     "036_PKA.zip"
#     "037_TKA.zip"
#     "038_TKA.zip"
#     "take_jsons.zip"
#     "take_point_clouds_sparse.zip"
#     "take_timestamp_to_next_action.zip"
#     "take_timestamp_to_robot_phase.zip"
#     "take_timestamp_to_sterility_breach.zip"
#     "take_tracks.zip"
#     "take_transcripts.zip"
#     "screen_summaries.zip"
#     "take_audios.zip"
#     "4D-OR_pcd_sparse.zip"
# )


FILES=(
    "001_PKA.zip"
    "002_PKA.zip"
    "003_TKA.zip"
    "004_PKA.zip"
    "005_TKA.zip"
    "006_PKA.zip"
    "007_TKA.zip"
    "008_PKA.zip"
    "009_TKA.zip"
    "010_PKA.zip"
    "011_TKA.zip"
    "012_PKA.zip"
    "013_PKA.zip"
    "014_PKA.zip"
    "015-018_PKA.zip"
    "019-022_PKA.zip"
    "023-032_PKA.zip"
    "033_PKA.zip"
    "035_PKA.zip"
    "036_PKA.zip"
    "037_TKA.zip"
    "038_TKA.zip"
    "take_jsons.zip"
    "take_point_clouds_sparse.zip"
    "take_timestamp_to_next_action.zip"
    "take_timestamp_to_robot_phase.zip"
    "take_timestamp_to_sterility_breach.zip"
    "take_tracks.zip"
    "take_transcripts.zip"
    "screen_summaries.zip"
    "take_audios.zip"
    "4D-OR_pcd_sparse.zip"
)

# Dry-run mode: validate URL reachability
if [ $DRY_RUN -eq 1 ]; then
    echo "Starting dry-run: Validating reachability of dataset zip files..."
    reachable_count=0
    unreachable_count=0

    for file in "${FILES[@]}"; do
        URL="${BASE_URL}/${file}"
        echo "Checking ${URL} ..."
        if wget --spider "$URL" &>/dev/null; then
            echo "Success: ${file} is reachable."
            ((reachable_count++))
        else
            echo "Error: ${file} is NOT reachable."
            ((unreachable_count++))
        fi
    done

    echo "Dry-run completed."
    echo "Reachable files: $reachable_count"
    echo "Unreachable files: $unreachable_count"
    exit 0
fi

# Actual download and extraction mode

# Create target directories if they do not exist
TARGET_DIR="MM-OR_data"
MARKERS_DIR="markers"
mkdir -p "$TARGET_DIR"
mkdir -p "$MARKERS_DIR"

echo "Starting actual download and extraction process..."

for file in "${FILES[@]}"; do
    marker="${MARKERS_DIR}/${file}.downloaded"
    if [ -f "$marker" ]; then
        echo "$file has already been downloaded and extracted. Skipping."
        continue
    fi

    echo "Downloading $file..."
    wget -c "${BASE_URL}/${file}" -O "$file"

    echo "Unzipping $file into $TARGET_DIR..."
    unzip -o "$file" -d "$TARGET_DIR"

    # Remove the zip file after extraction
    rm "$file"

    # Create a marker file to indicate successful processing of this zip
    touch "$marker"
done

echo "All files have been processed and unzipped into the $TARGET_DIR directory."
