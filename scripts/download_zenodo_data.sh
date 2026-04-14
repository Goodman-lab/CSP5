#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Download and extract CSP5 Zenodo dataset bundle.

Usage:
  ./download_zenodo_data.sh [options]

Options:
  --archive PATH      Archive path (default: ./CSP5_data.tar.gz)
  --extract-dir PATH  Directory to extract into (default: .)
  --force-download    Re-download archive even if it already exists
  --force-extract     Remove existing zenodo_csp5_upload before extraction
  -h, --help          Show this help
USAGE
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARCHIVE="$ROOT/CSP5_data.tar.gz"
EXTRACT_DIR="$ROOT"
FORCE_DOWNLOAD=0
FORCE_EXTRACT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive)
      ARCHIVE="$2"
      shift 2
      ;;
    --extract-dir)
      EXTRACT_DIR="$2"
      shift 2
      ;;
    --force-download)
      FORCE_DOWNLOAD=1
      shift
      ;;
    --force-extract)
      FORCE_EXTRACT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

URL="https://zenodo.org/api/records/19486118/files/CSP5_data.tar.gz/content"
EXPECTED_MD5="86e1c0986ac8ebd23e3281a7497e1142"
ARCHIVE_DIR=$(cd "$(dirname "$ARCHIVE")" && pwd)
ARCHIVE_PATH="$ARCHIVE_DIR/$(basename "$ARCHIVE")"
mkdir -p "$ARCHIVE_DIR"
mkdir -p "$EXTRACT_DIR"

if [[ "$FORCE_DOWNLOAD" == "1" || ! -f "$ARCHIVE_PATH" ]]; then
  echo "[info] Downloading $URL"
  curl -L --fail --retry 3 --retry-delay 2 -o "$ARCHIVE_PATH" "$URL"
fi

echo "$EXPECTED_MD5  $ARCHIVE_PATH" | md5sum -c

TARGET_DIR="$EXTRACT_DIR/zenodo_csp5_upload"
if [[ -d "$TARGET_DIR" ]]; then
  if [[ "$FORCE_EXTRACT" == "1" ]]; then
    rm -rf "$TARGET_DIR"
  else
    echo "[info] Existing extract found at $TARGET_DIR (use --force-extract to replace)"
  fi
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "[info] Extracting archive to $EXTRACT_DIR"
  tar -xzf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"
fi

for required in \
  "$TARGET_DIR/data/nmrexp/NMRexp_with_ids.parquet" \
  "$TARGET_DIR/data/splits/NMRexp-13C-scaffold-doi_split.jsonl" \
  "$TARGET_DIR/data/splits/NMRexp-1H-scaffold-doi_split.jsonl" \
  "$TARGET_DIR/data/assigned/Exp22K_13C_entries.pkl" \
  "$TARGET_DIR/data/assigned/DFT8K_1H_entries.pkl" \
  "$TARGET_DIR/data/splits/CSP5-13C-scaffold-doi_split.json" \
  "$TARGET_DIR/data/splits/CSP5-1H-scaffold-doi_split.json"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required extracted file: $required" >&2
    exit 1
  fi
done

echo "[info] Zenodo data ready at $TARGET_DIR"
