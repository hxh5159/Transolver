#!/usr/bin/env bash

# Shared launcher for the six standard PDE benchmarks. Dataset scripts source
# this file; it is not intended to be executed directly.

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RUN_SCRIPT_DIR/../../.." && pwd)"
PDE_ROOT="$REPO_ROOT/PDE-Solving-StandardBenchmark"
MONITOR_ENTRY="$REPO_ROOT/LINEARNO/monitor/run.py"

DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno}"
GPU_ID="${GPU_ID:-1}"
EPOCHS="${EPOCHS:-500}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PDE_ROOT/output}"
CAPTURE_EVERY_VALIDATIONS="${CAPTURE_EVERY_VALIDATIONS:-50}"
MONITOR_MAX_SAMPLES="${MONITOR_MAX_SAMPLES:-4}"
MONITOR_MAX_POINTS="${MONITOR_MAX_POINTS:-4096}"
MONITOR_SEED="${MONITOR_SEED:-1729}"

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required dataset file is missing or empty: $path" >&2
    exit 1
  fi
}

new_experiment_dir() {
  local tag="$1"
  local stamp candidate suffix
  stamp="$(date '+%Y%m%d_%H%M%S')"
  candidate="$OUTPUT_ROOT/${stamp}_${tag}"
  suffix=0
  while [[ -e "$candidate" ]]; do
    suffix=$((suffix + 1))
    candidate="$OUTPUT_ROOT/${stamp}_${tag}_$(printf '%02d' "$suffix")"
  done
  mkdir -p "$candidate/logs"
  printf '%s\n' "$candidate"
}

run_standard_benchmark() {
  local dataset="$1"
  local tag="$2"
  local entrypoint="$3"
  local checkpoint_interval="$4"
  local visualization_interval="$5"
  shift 5

  local experiment_dir
  experiment_dir="$(new_experiment_dir "$tag")"

  echo "Dataset: $dataset"
  echo "GPU: $GPU_ID"
  echo "Dataset root: $DATA_ROOT"
  echo "Experiment output: $experiment_dir"
  echo "Monitor output: $REPO_ROOT/LINEARNO/monitor/output/$dataset"

  python "$MONITOR_ENTRY" \
    --dataset "$dataset" \
    --run-name "${tag}_train" \
    --capture-every-validations "$CAPTURE_EVERY_VALIDATIONS" \
    --max-samples-per-snapshot "$MONITOR_MAX_SAMPLES" \
    --max-points "$MONITOR_MAX_POINTS" \
    --monitor-seed "$MONITOR_SEED" \
    -- \
    "$PDE_ROOT/$entrypoint" \
    "$@" \
    --epochs "$EPOCHS" \
    --eval 0 \
    --seed "$SEED" \
    --checkpoint_interval "$checkpoint_interval" \
    --visualization_interval "$visualization_interval" \
    --output_root "$OUTPUT_ROOT" \
    --experiment_dir "$experiment_dir" \
    2>&1 | tee "$experiment_dir/logs/train_console.log"

  python "$MONITOR_ENTRY" \
    --dataset "$dataset" \
    --run-name "${tag}_eval" \
    --capture-every-validations "$CAPTURE_EVERY_VALIDATIONS" \
    --max-samples-per-snapshot "$MONITOR_MAX_SAMPLES" \
    --max-points "$MONITOR_MAX_POINTS" \
    --monitor-seed "$MONITOR_SEED" \
    -- \
    "$PDE_ROOT/$entrypoint" \
    "$@" \
    --epochs "$EPOCHS" \
    --eval 1 \
    --seed "$SEED" \
    --checkpoint_interval "$checkpoint_interval" \
    --visualization_interval "$visualization_interval" \
    --output_root "$OUTPUT_ROOT" \
    --experiment_dir "$experiment_dir" \
    --model_path "$experiment_dir/checkpoints/checkpoint_final.pth" \
    2>&1 | tee "$experiment_dir/logs/evaluation_console.log"

  echo "Completed training and evaluation: $experiment_dir"
}
