#!/usr/bin/env bash
# Train and evaluate the standard Navier-Stokes benchmark with Transolver.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PDE_ROOT="$REPO_ROOT/PDE-Solving-StandardBenchmark"
DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno}"
GPU_ID="${GPU_ID:-1}"
EPOCHS="${EPOCHS:-500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PDE_ROOT/output}"
DATA_FILE="$DATA_ROOT/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"
if [[ ! -s "$DATA_FILE" ]]; then
  echo "ERROR: missing standard Navier-Stokes data: $DATA_FILE" >&2
  exit 1
fi
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
EXPERIMENT_DIR="$OUTPUT_ROOT/${RUN_STAMP}_ns_Transolver"
SUFFIX=0
while [[ -e "$EXPERIMENT_DIR" ]]; do
  SUFFIX=$((SUFFIX + 1))
  EXPERIMENT_DIR="$OUTPUT_ROOT/${RUN_STAMP}_ns_Transolver_${SUFFIX}"
done
mkdir -p "$EXPERIMENT_DIR/logs"

cd "$PDE_ROOT"
python exp_ns.py \
  --gpu "$GPU_ID" \
  --model Transolver_Structured_Mesh_2D \
  --n-hidden 256 --n-heads 8 --n-layers 8 \
  --lr 0.001 --batch-size 2 \
  --slice_num 32 --unified_pos 1 --ref 8 \
  --epochs "$EPOCHS" --eval 0 \
  --save_name ns_Transolver \
  --data_path "$DATA_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --experiment_dir "$EXPERIMENT_DIR" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --seed "$SEED" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/train_console.log"

python exp_ns.py \
  --gpu "$GPU_ID" \
  --model Transolver_Structured_Mesh_2D \
  --n-hidden 256 --n-heads 8 --n-layers 8 \
  --lr 0.001 --batch-size 2 \
  --slice_num 32 --unified_pos 1 --ref 8 \
  --epochs "$EPOCHS" --eval 1 \
  --save_name ns_Transolver \
  --data_path "$DATA_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --experiment_dir "$EXPERIMENT_DIR" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --seed "$SEED" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/evaluation_console.log"

echo "Standard Navier-Stokes training and evaluation completed: $EXPERIMENT_DIR"
