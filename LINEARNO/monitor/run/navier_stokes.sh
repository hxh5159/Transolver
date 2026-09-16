#!/usr/bin/env bash
# Paper configuration: scripts/Transolver_NS.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

require_file "$DATA_ROOT/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"

run_standard_benchmark navier_stokes navier_stokes_Transolver exp_ns.py \
  "${CHECKPOINT_INTERVAL:-50}" "${VISUALIZATION_INTERVAL:-50}" \
  --gpu "$GPU_ID" \
  --model Transolver_Structured_Mesh_2D \
  --n-hidden 256 --n-heads 8 --n-layers 8 \
  --mlp_ratio 1 --dropout 0 \
  --lr 0.001 --weight_decay 1e-5 \
  --batch-size 2 --slice_num 32 \
  --unified_pos 1 --ref 8 --downsample 1 \
  --save_name ns_Transolver --data_path "$DATA_ROOT"
