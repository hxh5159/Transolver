#!/usr/bin/env bash
# Train and evaluate the standard Plasticity benchmark with Transolver.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PDE_ROOT="$REPO_ROOT/PDE-Solving-StandardBenchmark"
DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno}"
GPU_ID="${GPU_ID:-1}"
EPOCHS="${EPOCHS:-500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
VISUALIZATION_INTERVAL="${VISUALIZATION_INTERVAL:-100}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PDE_ROOT/output}"
if [[ ! -s "$DATA_ROOT/plas_N987_T20.mat" ]]; then
  echo "ERROR: missing standard Plasticity data: $DATA_ROOT/plas_N987_T20.mat" >&2
  exit 1
fi
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
EXPERIMENT_DIR="$OUTPUT_ROOT/${RUN_STAMP}_plas_Transolver"
SUFFIX=0
while [[ -e "$EXPERIMENT_DIR" ]]; do
  SUFFIX=$((SUFFIX + 1))
  EXPERIMENT_DIR="$OUTPUT_ROOT/${RUN_STAMP}_plas_Transolver_${SUFFIX}"
done
mkdir -p "$EXPERIMENT_DIR/logs"

cd "$PDE_ROOT"
python exp_plas.py \
  --gpu "$GPU_ID" \
  --model Transolver_Structured_Mesh_2D \
  --n-hidden 128 --n-heads 8 --n-layers 8 \
  --lr 0.001 --max_grad_norm 0.1 --batch-size 8 \
  --slice_num 64 --unified_pos 0 --ref 8 \
  --epochs "$EPOCHS" --eval 0 \
  --save_name plas_Transolver \
  --data_path "$DATA_ROOT/plas_N987_T20.mat" \
  --output_root "$OUTPUT_ROOT" \
  --experiment_dir "$EXPERIMENT_DIR" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --visualization_interval "$VISUALIZATION_INTERVAL" \
  --seed "$SEED" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/train_console.log"

python exp_plas.py \
  --gpu "$GPU_ID" \
  --model Transolver_Structured_Mesh_2D \
  --n-hidden 128 --n-heads 8 --n-layers 8 \
  --lr 0.001 --max_grad_norm 0.1 --batch-size 8 \
  --slice_num 64 --unified_pos 0 --ref 8 \
  --epochs "$EPOCHS" --eval 1 \
  --save_name plas_Transolver \
  --data_path "$DATA_ROOT/plas_N987_T20.mat" \
  --output_root "$OUTPUT_ROOT" \
  --experiment_dir "$EXPERIMENT_DIR" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --visualization_interval "$VISUALIZATION_INTERVAL" \
  --seed "$SEED" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/evaluation_console.log"

echo "Standard Plasticity training and evaluation completed: $EXPERIMENT_DIR"
