#!/usr/bin/env bash
# process -> generate -> render on the airgapped server.
#
# Usage: bash scripts/run_all.sh [--dry-run] [--out-dir DIR] [RAW_JSONL]
#   RAW_JSONL   fetched snapshot (default: data/raw/raw_articles.jsonl)
#   --dry-run   skip the LLM; extractive summaries only (fully offline)
#   --out-dir   write ranked_clusters.json, top10.json, grounding_report.json, digest.html here
#               (default: data/processed + data/output)
# Env: EMBED_MODEL_PATH (required), LLM_BASE_URL / LLM_MODEL / LLM_API_KEY (unless --dry-run), PYTHON.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
RAW="$ROOT/data/raw/raw_articles.jsonl"
PROCESSED_DIR="$ROOT/data/processed"
OUTPUT_DIR="$ROOT/data/output"
GEN_FLAGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) GEN_FLAGS+=(--dry-run); shift ;;
    --out-dir) PROCESSED_DIR="$2"; OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) RAW="$1"; shift ;;
  esac
done

[[ -f "$RAW" ]] || { echo "raw snapshot not found: $RAW" >&2; exit 1; }
[[ -n "${EMBED_MODEL_PATH:-}" ]] || { echo "EMBED_MODEL_PATH is not set" >&2; exit 1; }
if [[ ${#GEN_FLAGS[@]} -eq 0 && -z "${LLM_BASE_URL:-}" ]]; then
  echo "LLM_BASE_URL is not set (use --dry-run for extractive summaries)" >&2; exit 1
fi

CLUSTERS="$PROCESSED_DIR/ranked_clusters.json"
TOP10="$OUTPUT_DIR/top10.json"
REPORT="$OUTPUT_DIR/grounding_report.json"
HTML="$OUTPUT_DIR/digest.html"
START=$SECONDS

echo "== process  ($RAW)"
"$PY" "$ROOT/scripts/process.py" --in "$RAW" --out "$CLUSTERS"
echo "== generate ${GEN_FLAGS[*]:-}"
"$PY" "$ROOT/scripts/generate.py" --in "$CLUSTERS" --out "$TOP10" --report "$REPORT" ${GEN_FLAGS[@]+"${GEN_FLAGS[@]}"}
echo "== render"
"$PY" "$ROOT/scripts/render.py" --in "$TOP10" --out "$HTML" --raw "$RAW"

echo
echo "Done in $((SECONDS - START))s -> $HTML"
echo "Audit trail: $REPORT"
