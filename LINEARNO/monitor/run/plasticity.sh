#!/usr/bin/env bash
# Paper configuration: scripts/Transolver_Plas.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

require_file "$DATA_ROOT/plas_N987_T20.mat"

run_standard_benchmark plasticity plasticity_Transolver exp_plas.py \
  "${CHECKPOINT_INTERVAL:-100}" "${VISUALIZATION_INTERVAL:-100}" \
  --gpu "$GPU_ID" \
  --model Transolver_Structured_Mesh_2D \
  --n-hidden 128 --n-heads 8 --n-layers 8 \
  --mlp_ratio 1 --dropout 0 \
  --lr 0.001 --weight_decay 1e-5 --max_grad_norm 0.1 \
  --batch-size 8 --slice_num 64 \
  --unified_pos 0 --ref 8 \
  --save_name plas_Transolver --data_path "$DATA_ROOT/plas_N987_T20.mat"
