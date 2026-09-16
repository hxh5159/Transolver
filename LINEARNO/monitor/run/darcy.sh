#!/usr/bin/env bash
# Paper configuration: scripts/Transolver_Darcy.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

require_file "$DATA_ROOT/piececonst_r421_N1024_smooth1.mat"
require_file "$DATA_ROOT/piececonst_r421_N1024_smooth2.mat"

run_standard_benchmark darcy darcy_UniPDE exp_darcy.py \
  "${CHECKPOINT_INTERVAL:-100}" "${VISUALIZATION_INTERVAL:-100}" \
  --gpu "$GPU_ID" \
  --model Transolver_Structured_Mesh_2D \
  --n-hidden 128 --n-heads 8 --n-layers 8 \
  --mlp_ratio 1 --dropout 0 \
  --lr 0.001 --weight_decay 1e-5 --max_grad_norm 0.1 \
  --batch-size 4 --slice_num 64 \
  --unified_pos 1 --ref 8 --downsample 5 --ntrain 1000 \
  --save_name darcy_UniPDE --data_path "$DATA_ROOT"
