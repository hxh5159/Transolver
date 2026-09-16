#!/usr/bin/env bash
# Paper configuration: scripts/Transolver_Airfoil.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

AIRFOIL_DATA="$DATA_ROOT/airfoil/naca"
require_file "$AIRFOIL_DATA/NACA_Cylinder_X.npy"
require_file "$AIRFOIL_DATA/NACA_Cylinder_Y.npy"
require_file "$AIRFOIL_DATA/NACA_Cylinder_Q.npy"

run_standard_benchmark standard_airfoil standard_airfoil_Transolver exp_airfoil.py \
  "${CHECKPOINT_INTERVAL:-100}" "${VISUALIZATION_INTERVAL:-100}" \
  --gpu "$GPU_ID" \
  --model Transolver_Structured_Mesh_2D \
  --n-hidden 128 --n-heads 8 --n-layers 8 \
  --mlp_ratio 1 --dropout 0 \
  --lr 0.001 --weight_decay 1e-5 --max_grad_norm 0.1 \
  --batch-size 4 --slice_num 64 \
  --unified_pos 0 --ref 8 --downsamplex 1 --downsampley 1 \
  --save_name airfoil_Transolver --data_path "$AIRFOIL_DATA"
