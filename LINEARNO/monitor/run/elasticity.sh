#!/usr/bin/env bash
# Paper configuration: scripts/Transolver_Elas.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

require_file "$DATA_ROOT/elasticity/Meshes/Random_UnitCell_sigma_10.npy"
require_file "$DATA_ROOT/elasticity/Meshes/Random_UnitCell_XY_10.npy"

run_standard_benchmark elasticity elasticity_Transolver exp_elas.py \
  "${CHECKPOINT_INTERVAL:-100}" "${VISUALIZATION_INTERVAL:-100}" \
  --gpu "$GPU_ID" \
  --model Transolver_Irregular_Mesh \
  --n-hidden 128 --n-heads 8 --n-layers 8 \
  --mlp_ratio 1 --dropout 0 \
  --lr 0.001 --weight_decay 1e-5 --max_grad_norm 0.1 \
  --batch-size 1 --slice_num 64 \
  --unified_pos 0 --ref 8 --downsample 5 --ntrain 1000 \
  --save_name elas_Transolver --data_path "$DATA_ROOT"
