#!/usr/bin/env bash
# Train and evaluate the ShapeNetCar Transolver experiment on one GPU.
# The defaults match the repository/paper setup: nine-fold holdout,
# 200 epochs per fold, batch size 1, Adam lr 1e-3, pressure weight 0.5.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CAR_ROOT="$REPO_ROOT/Car-Design-ShapeNetCar"

DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
GPU_ID="${GPU_ID:-1}"
if [[ -n "${FOLDS:-}" ]]; then
  FOLDS_TO_RUN="$FOLDS"
elif [[ -n "${FOLD_ID:-}" ]]; then
  FOLDS_TO_RUN="$FOLD_ID"
else
  FOLDS_TO_RUN="0 1 2 3 4 5 6 7 8"
fi
EPOCHS="${EPOCHS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
VISUALIZATION_INTERVAL="${VISUALIZATION_INTERVAL:-50}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CAR_ROOT/output}"
RAW_DATA_DIR="${CAR_DATA_DIR:-$DATA_ROOT/mlcfd/training_data}"
PREPROCESSED_DIR="${CAR_PREPROCESSED_DIR:-$DATA_ROOT/mlcfd/preprocessed_data}"
if [[ -z "${PREPROCESSED:-}" ]]; then
  if find "$PREPROCESSED_DIR" -type f -name 'x.npy' -print -quit 2>/dev/null | grep -q .; then
    PREPROCESSED=1
  else
    PREPROCESSED=0
  fi
fi
if [[ ! -d "$RAW_DATA_DIR/param0" ]]; then
  echo "ERROR: missing ShapeNetCar data directory: $RAW_DATA_DIR" >&2
  exit 1
fi

RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
echo "GPU: $GPU_ID"
echo "Raw data: $RAW_DATA_DIR"
echo "Preprocessed data: $PREPROCESSED_DIR"
echo "Use preprocessed cache: $PREPROCESSED"

cd "$CAR_ROOT"
RESULT_DIRS=()
for FOLD_ID in $FOLDS_TO_RUN; do
  EXPERIMENT_DIR="$OUTPUT_ROOT/${RUN_STAMP}_car_Transolver_fold${FOLD_ID}"
  SUFFIX=0
  while [[ -e "$EXPERIMENT_DIR" ]]; do
    SUFFIX=$((SUFFIX + 1))
    EXPERIMENT_DIR="$OUTPUT_ROOT/${RUN_STAMP}_car_Transolver_fold${FOLD_ID}_${SUFFIX}"
  done
  mkdir -p "$EXPERIMENT_DIR/logs"
  RESULT_DIRS+=("$EXPERIMENT_DIR")

  echo "Car experiment fold ${FOLD_ID}: $EXPERIMENT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py \
    --cfd_model Transolver \
    --data_dir "$RAW_DATA_DIR" \
    --save_dir "$PREPROCESSED_DIR" \
    --fold_id "$FOLD_ID" \
    --gpu 0 \
    --val_iter 10 \
    --weight 0.5 \
    --lr 0.001 \
    --batch_size 1 \
    --nb_epochs "$EPOCHS" \
    --preprocessed "$PREPROCESSED" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --visualization_interval "$VISUALIZATION_INTERVAL" \
    --visualization_sample -1 \
    --seed "$SEED" \
    --output_root "$OUTPUT_ROOT" \
    --experiment_dir "$EXPERIMENT_DIR" \
    2>&1 | tee "$EXPERIMENT_DIR/logs/train_console.log"

  # A raw-data first fold creates the complete cache for subsequent folds.
  PREPROCESSED=1

  CUDA_VISIBLE_DEVICES="$GPU_ID" python main_evaluation.py \
    --cfd_model Transolver \
    --data_dir "$RAW_DATA_DIR" \
    --save_dir "$PREPROCESSED_DIR" \
    --fold_id "$FOLD_ID" \
    --gpu 0 \
    --r 0.2 \
    --nb_epochs "$EPOCHS" \
    --model_path "$EXPERIMENT_DIR/model_final.pth" \
    --experiment_dir "$EXPERIMENT_DIR" \
    2>&1 | tee "$EXPERIMENT_DIR/logs/evaluation_console.log"
done

# Aggregate the nine-fold paper metrics without changing any training output.
python - "$OUTPUT_ROOT" "${RESULT_DIRS[@]}" <<'PY'
import json
import os
import sys

import numpy as np

output_root = sys.argv[1]
experiment_dirs = sys.argv[2:]
rows = []
for directory in experiment_dirs:
    path = os.path.join(directory, 'evaluation', 'evaluation_metrics.json')
    if not os.path.isfile(path):
        raise SystemExit(f'missing evaluation metrics: {path}')
    with open(path) as file:
        rows.append(json.load(file))

def mean(key):
    return float(np.mean([row[key] for row in rows]))

summary = {
    'folds': len(rows),
    'experiments': experiment_dirs,
    'volume_velocity_relative_l2_mean': mean('volume_velocity_relative_l2_mean'),
    'surface_pressure_relative_l2_mean': mean('surface_pressure_relative_l2_mean'),
    'drag_coefficient_mape_mean': float(np.mean([
        row['drag_coefficient']['mape'] for row in rows])),
    'drag_coefficient_spearman_mean': float(np.mean([
        row['drag_coefficient']['spearman'] for row in rows])),
}
summary_path = os.path.join(
    output_root, f'{os.path.basename(experiment_dirs[0]).split("_fold")[0]}_summary.json')
with open(summary_path, 'w') as file:
    json.dump(summary, file, indent=2)
print('Car cross-fold summary:', summary_path)
PY

echo "Car training and evaluation completed for folds: $FOLDS_TO_RUN"
