#!/usr/bin/env bash

set -euo pipefail

DEBUG=0
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG=1
  shift || true
  echo "[debug] Running in debug mode: n=5, verbose output, separate data_debug dir"
fi

# Orchestrate alignment for deep-argmap / argunauts-thinking style datasets
# using Synthetic Data Kit.
#
# This script coordinates the following steps for a set of dataset configs
# and splits:
#   1. Prepare raw JSON subsets via prepare_subset.py
#   2. Assign each example to a (mode, model) combination via
#      assign_modes_and_models.py
#   3. Run synthetic-data-kit create for each (config, split, mode, model)
#   4. Merge per-group outputs back into one aligned split per config via
#      merge_per_config.py
#
# It is designed to be idempotent: if an expected intermediate or final file
# already exists, the corresponding step is skipped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$DEBUG" -eq 1 ]]; then
  DATA_DIR="${SCRIPT_DIR}/data_debug"
  set -x
else
  DATA_DIR="${SCRIPT_DIR}/data"
fi

RAW_DIR="${DATA_DIR}/raw"
ASSIGNED_DIR="${DATA_DIR}/assigned"
ALIGNED_DIR="${DATA_DIR}/aligned"
MERGED_DIR="${DATA_DIR}/merged"

# Base HF dataset identifier for the deep-argmap corpus. Adjust this to the
# actual dataset repo id you use.
BASE_DATASET_ID="DebateLabKIT/argunauts-thinking"

# Dataset configurations we want to process.
CONFIGS=(
  "deepa2-aaac01-thinking"
  "deepa2-aaac02-thinking"
  "deepa2-aaac03-thinking"
  "deepa2-folly-thinking"
)

# Splits and target sizes per split.
SPLITS=("train" "validation" "test")
TRAIN_N=15000
EVAL_N=600

if [[ "$DEBUG" -eq 1 ]]; then
  TRAIN_N=5
  EVAL_N=5
fi

# Alignment modes and their corresponding Synthetic Data Kit configs.
MODES=("a" "b")
MODE_CONFIG_A="${SCRIPT_DIR}/argunauts_config_a.yaml"
MODE_CONFIG_B="${SCRIPT_DIR}/argunauts_config_b.yaml"

# Models to use for transformation. These are passed via the --model flag and
# are not tied to the config files.
MODELS=(
  "kit.gpt-oss-120b"
  "kit.mixtral-8x22b-instruct"
  "kit.qwen3-vl-235b-a22b-instruct"
)

mkdir -p "$RAW_DIR" "$ASSIGNED_DIR" "$ALIGNED_DIR" "$MERGED_DIR"

CREATE_EXTRA_ARGS=()
if [[ "$DEBUG" -eq 1 ]]; then
  CREATE_EXTRA_ARGS+=(--verbose)
fi

# Helper to select the appropriate config for a given mode.
get_mode_config() {
  local mode="$1"
  if [[ "$mode" == "a" ]]; then
    echo "$MODE_CONFIG_A"
  else
    echo "$MODE_CONFIG_B"
  fi
}

# 1) Prepare raw subsets per (config, split)
for config in "${CONFIGS[@]}"; do
  for split in "${SPLITS[@]}"; do
    raw_out="${RAW_DIR}/${config}_${split}_raw.json"

    if [[ -f "$raw_out" ]]; then
      echo "[prepare] Skipping existing raw subset: $raw_out"
      continue
    fi

    # Decide target n based on split
    if [[ "$split" == "train" ]]; then
      n="$TRAIN_N"
    else
      n="$EVAL_N"
    fi

    echo "[prepare] Preparing subset for config=$config, split=$split (n=$n)"
    python "${SCRIPT_DIR}/prepare_subset.py" \
      --dataset "$BASE_DATASET_ID" \
      --config-name "$config" \
      --split "$split" \
      --n "$n" \
      --output "$raw_out" || {
        echo "[prepare] Warning: failed to prepare subset for $config/$split" >&2
      }
  done
done

# 2) Assign modes and models per (config, split)
for config in "${CONFIGS[@]}"; do
  for split in "${SPLITS[@]}"; do
    raw_in="${RAW_DIR}/${config}_${split}_raw.json"

    if [[ ! -f "$raw_in" ]]; then
      echo "[assign] Raw subset missing, skipping: $raw_in"
      continue
    fi

    # Check if at least one assigned file already exists; if so, assume
    # assignment has been performed for this (config, split).
    assigned_glob="${ASSIGNED_DIR}/${config}_${split}_mode-*_model-*.json"
    if compgen -G "$assigned_glob" > /dev/null; then
      echo "[assign] Skipping assignment for $config/$split (files already present)"
      continue
    fi

    echo "[assign] Assigning modes/models for config=$config, split=$split"
    python "${SCRIPT_DIR}/assign_modes_and_models.py" \
      --input "$raw_in" \
      --modes "${MODES[@]}" \
      --models "${MODELS[@]}" \
      --output-dir "$ASSIGNED_DIR" \
      --seed 42
  done
done

# 3) Run Synthetic Data Kit create for each (config, split, mode, model)
for config in "${CONFIGS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for mode in "${MODES[@]}"; do
      for model in "${MODELS[@]}"; do
        safe_model=${model//\//-}
        assigned_in="${ASSIGNED_DIR}/${config}_${split}_mode-${mode}_model-${safe_model}.json"

        if [[ ! -f "$assigned_in" ]]; then
          echo "[create] Assigned input missing, skipping: $assigned_in"
          continue
        fi

        mode_config="$(get_mode_config "$mode")"
        if [[ ! -f "$mode_config" ]]; then
          echo "[create] Mode config not found for mode=$mode: $mode_config" >&2
          exit 1
        fi

        out_dir="${ALIGNED_DIR}/${config}/${split}/mode-${mode}_model-${safe_model}"
        mkdir -p "$out_dir"

        # Expected enhanced file name follows the Synthetic Data Kit naming
        # convention: <basename>_enhanced.json
        base_name="$(basename "$assigned_in" .json)"
        enhanced_json="${out_dir}/${base_name}_enhanced.json"

        if [[ -f "$enhanced_json" ]]; then
          echo "[create] Skipping existing enhanced file: $enhanced_json"
          continue
        fi

        echo "[create] Running SD-Kit for config=$config, split=$split, mode=$mode, model=$model"
        synthetic-data-kit \
          -c "$mode_config" \
          create "$assigned_in" \
          --type cot-enhance \
          --model "$model" \
          --output-dir "$out_dir" \
          "${CREATE_EXTRA_ARGS[@]}"
      done
    done
  done
done

# 4) Merge per-group outputs back into a single aligned split per config
for config in "${CONFIGS[@]}"; do
  for split in "${SPLITS[@]}"; do
    merged_out="${MERGED_DIR}/${config}-aligned_${split}.json"

    if [[ -f "$merged_out" ]]; then
      echo "[merge] Skipping existing merged file: $merged_out"
      continue
    fi

    # Only attempt merge if we have some aligned data for this (config, split)
    aligned_base="${ALIGNED_DIR}/${config}/${split}"
    if [[ ! -d "$aligned_base" ]]; then
      echo "[merge] No aligned data directory for $config/$split, skipping"
      continue
    fi

    echo "[merge] Merging aligned groups for config=$config, split=$split"
    python "${SCRIPT_DIR}/merge_per_config.py" \
      --config "$config" \
      --split "$split" \
      --input-root "$ALIGNED_DIR" \
      --output "$merged_out" || {
        echo "[merge] Warning: merge failed for $config/$split" >&2
      }
  done
done

echo "All steps completed. Merged aligned splits are in: $MERGED_DIR"
