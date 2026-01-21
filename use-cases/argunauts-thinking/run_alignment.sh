#!/usr/bin/env bash

set -euo pipefail

# Run the argunauts-thinking alignment using Synthetic Data Kit.
#
# Usage:
#   bash run_alignment.sh <mode> <split>
#
# where:
#   <mode>  is either 'a' or 'b' (for argunauts_config_a.yaml / argunauts_config_b.yaml)
#   <split> is a dataset split name like 'train', 'validation', or 'test'
#
# This script assumes that prepare_subset.py has already created a file
#   argunauts_<split>_conversations.json
# in the same directory as this script.

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <mode:a|b> <split>" >&2
  exit 1
fi

MODE="$1"
SPLIT="$2"

if [[ "$MODE" != "a" && "$MODE" != "b" ]]; then
  echo "Error: mode must be 'a' or 'b', got '$MODE'" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSON="${SCRIPT_DIR}/argunauts_${SPLIT}_conversations.json"
if [[ ! -f "$INPUT_JSON" ]]; then
  echo "Error: input file not found: $INPUT_JSON" >&2
  echo "Run prepare_subset.py first to create it." >&2
  exit 1
fi

if [[ "$MODE" == "a" ]]; then
  CONFIG="${SCRIPT_DIR}/argunauts_config_a.yaml"
else
  CONFIG="${SCRIPT_DIR}/argunauts_config_b.yaml"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config file not found: $CONFIG" >&2
  exit 1
fi

OUT_DIR="${SCRIPT_DIR}/aligned_mode_${MODE}_${SPLIT}"
BASENAME="$(basename "$INPUT_JSON" .json)"

echo "Running alignment mode $MODE on split $SPLIT..."
synthetic-data-kit \
  -c "$CONFIG" \
  create "$INPUT_JSON" \
  --type cot-enhance \
  -o "$OUT_DIR"

ENHANCED_JSON="${OUT_DIR}/${BASENAME}_enhanced.json"
if [[ ! -f "$ENHANCED_JSON" ]]; then
  echo "Error: expected enhanced file not found: $ENHANCED_JSON" >&2
  exit 1
fi

FINAL_JSON="${SCRIPT_DIR}/aligned_mode_${MODE}_${SPLIT}.json"
cp "$ENHANCED_JSON" "$FINAL_JSON"

echo "Wrote aligned file: $FINAL_JSON"