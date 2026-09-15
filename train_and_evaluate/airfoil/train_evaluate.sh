#!/usr/bin/env bash
# =============================================================================
# Airfoil（标准基准）单卡训练 + 评估，使用论文配置。
#
# 论文配置（scripts/Transolver_Airfoil.sh）：model=Transolver_Structured_Mesh_2D,
#   n_hidden=128, n_heads=8, n_layers=8, lr=0.001, max_grad_norm=0.1,
#   batch_size=4, slice_num=64, unified_pos=0, ref=8
#   其余用默认：epochs=500, weight_decay=1e-5
#
# 用法： bash train_evaluate.sh
# 环境变量： DATA / CONDA_ENV / GPU / SEED / EXPERIMENT_NAME
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
BENCH_DIR="$REPO/PDE-Solving-StandardBenchmark"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
DATA_PATH="$DATA/fno/airfoil/naca"   # 内含 NACA_Cylinder_{X,Y,Q}.npy

CONDA_ENV="${CONDA_ENV:-transolver}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-airfoil_Transolver}"
EXPERIMENT_DIR="$BENCH_DIR/output/$EXPERIMENT_NAME"

MODEL=Transolver_Structured_Mesh_2D
N_HIDDEN=128
N_HEADS=8
N_LAYERS=8
LR=0.001
MAX_GRAD_NORM=0.1
BATCH_SIZE=4
SLICE_NUM=64
UNIFIED_POS=0
REF=8

log() { echo "[$(date '+%F %T')] $*"; }

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || log "警告：无法激活 $CONDA_ENV，使用当前环境"
fi

if [ ! -f "$BENCH_DIR/exp_airfoil.py" ]; then echo "错误：找不到 $BENCH_DIR/exp_airfoil.py"; exit 1; fi
if [ ! -s "$DATA_PATH/NACA_Cylinder_X.npy" ]; then
  echo "错误：找不到 Airfoil 数据 $DATA_PATH/NACA_Cylinder_*.npy"
  exit 1
fi
cd "$BENCH_DIR"

COMMON_ARGS=(
  --model "$MODEL" --n-hidden "$N_HIDDEN" --n-heads "$N_HEADS" --n-layers "$N_LAYERS"
  --lr "$LR" --max_grad_norm "$MAX_GRAD_NORM" --batch-size "$BATCH_SIZE"
  --slice_num "$SLICE_NUM" --unified_pos "$UNIFIED_POS" --ref "$REF"
  --gpu "$GPU" --data_path "$DATA_PATH" --save_name "$EXPERIMENT_NAME"
  --experiment_dir "$EXPERIMENT_DIR" --seed "$SEED"
)

log "开始训练（$MODEL, GPU $GPU, seed $SEED）→ $EXPERIMENT_DIR"
python exp_airfoil.py "${COMMON_ARGS[@]}" --eval 0 || { log "训练失败"; exit 1; }

log "开始评估"
python exp_airfoil.py "${COMMON_ARGS[@]}" --eval 1 || { log "评估失败"; exit 1; }

log "完成。评估结果：$EXPERIMENT_DIR/evaluation/evaluation_metrics.json"
