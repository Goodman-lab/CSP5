#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Build CASCADE ensemble shards from Zenodo NMRexp data.

Usage:
  ./build_zenodo_ensemble_shards.sh [options]

Options:
  --data-root PATH        Root containing zenodo_csp5_upload (default: ./zenodo_csp5_upload)
  --output-root PATH      Root for output ensemble dirs (default: ./data)
  --targets CSV           Targets to build: 13C,1H (default: 13C,1H)
  --max-mols N            Limit molecules per target (default: 0 = full)
  --num-confs N           Number of conformers to attempt (default: 20)
  --max-confs N           Keep at most this many conformers (default: 10)
  --min-confs N           Require at least this many conformers (default: 1)
  --shard-size N          Molecules per shard (default: 10000)
  --batch-read-size N     Parquet batch size (default: 2000)
  --embed-threads N       RDKit embedding threads (default: 1)
  --max-embed-tries N     RDKit embedding retries (default: 5)
  --mmff-max-iters N      MMFF optimization iterations (default: 200)
  --python BIN            Python executable (default: python)
  -h, --help              Show this help
USAGE
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="$ROOT/zenodo_csp5_upload"
OUTPUT_ROOT="$ROOT/data"
TARGETS="13C,1H"
MAX_MOLS=0
NUM_CONFS=20
MAX_CONFS=10
MIN_CONFS=1
SHARD_SIZE=10000
BATCH_READ_SIZE=2000
EMBED_THREADS=1
MAX_EMBED_TRIES=5
MMFF_MAX_ITERS=200
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --targets)
      TARGETS="$2"
      shift 2
      ;;
    --max-mols)
      MAX_MOLS="$2"
      shift 2
      ;;
    --num-confs)
      NUM_CONFS="$2"
      shift 2
      ;;
    --max-confs)
      MAX_CONFS="$2"
      shift 2
      ;;
    --min-confs)
      MIN_CONFS="$2"
      shift 2
      ;;
    --shard-size)
      SHARD_SIZE="$2"
      shift 2
      ;;
    --batch-read-size)
      BATCH_READ_SIZE="$2"
      shift 2
      ;;
    --embed-threads)
      EMBED_THREADS="$2"
      shift 2
      ;;
    --max-embed-tries)
      MAX_EMBED_TRIES="$2"
      shift 2
      ;;
    --mmff-max-iters)
      MMFF_MAX_ITERS="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
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

if [[ ! -f "$DATA_ROOT/data/nmrexp/NMRexp_with_ids.parquet" ]]; then
  echo "[error] Missing Zenodo parquet under $DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
IFS=',' read -r -a target_arr <<< "$TARGETS"
for raw in "${target_arr[@]}"; do
  target=$(echo "$raw" | xargs)
  case "$target" in
    13C)
      target_lower="13c"
      manifest="$DATA_ROOT/data/splits/NMRexp-13C-scaffold-doi_split.jsonl"
      ;;
    1H)
      target_lower="1h"
      manifest="$DATA_ROOT/data/splits/NMRexp-1H-scaffold-doi_split.jsonl"
      ;;
    *)
      echo "[error] Unsupported target in --targets: $target" >&2
      exit 1
      ;;
  esac

  if [[ ! -f "$manifest" ]]; then
    echo "[error] Missing manifest for $target: $manifest" >&2
    exit 1
  fi

  out_dir="$OUTPUT_ROOT/cascade_nmrexp_${target_lower}_ensembles_scaffold_doi"
  cmd=(
    "$PYTHON_BIN" "$ROOT/src/cascade_nmr/NMRexp_PaiNN/build_conformer_ensembles.py"
    --parquet_path "$DATA_ROOT/data/nmrexp/NMRexp_with_ids.parquet"
    --output_dir "$out_dir"
    --manifest_path "$manifest"
    --target "$target"
    --require_peaks
    --num_confs "$NUM_CONFS"
    --max_confs "$MAX_CONFS"
    --min_confs "$MIN_CONFS"
    --shard_size "$SHARD_SIZE"
    --batch_read_size "$BATCH_READ_SIZE"
    --embed_threads "$EMBED_THREADS"
    --max_embed_tries "$MAX_EMBED_TRIES"
    --mmff_max_iters "$MMFF_MAX_ITERS"
    --precompute-edges
    --store_weights
  )
  if [[ "$MAX_MOLS" -gt 0 ]]; then
    cmd+=(--max_mols "$MAX_MOLS")
  fi

  echo "[info] Building $target ensemble shards -> $out_dir"
  "${cmd[@]}"

done

echo "[info] Ensemble shard build complete. Output root: $OUTPUT_ROOT"
