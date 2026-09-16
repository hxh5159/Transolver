#!/usr/bin/env bash
# =============================================================================
# 汽车设计（ShapeNetCar / MLCFD）单折训练 + 评估（默认 fold 0）
#
# 依据：原仓库 scripts/Transolver.sh 只跑 fold_id=0（没传 --fold_id，用默认 0），
#       README 的 Get Started 也只演示这一条。所以复现论文只需跑这一折：
#       训练 fold 0 + 评估 fold 0，不用跑满 9 折取平均。
#
# 论文配置 = main.py 的默认值：nb_epochs=200, batch_size=1, lr=0.001, weight=0.5，
#   Transolver(n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2, slice_num=32, unified_pos=0)
#
# 用法： bash single_fold.sh
# 环境变量： DATA / CONDA_ENV / GPU / SEED / FOLD_ID / EXPERIMENT_NAME
# =============================================================================
set -uo pipefail

# ---------- 路径（从脚本位置推导仓库根） ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
CAR_DIR="$REPO/Car-Design-ShapeNetCar"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
DATA_DIR="$DATA/mlcfd/training_data"
SAVE_DIR="$DATA/mlcfd/preprocessed_data"

# ---------- 参数 ----------
CONDA_ENV="${CONDA_ENV:-transolver}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
FOLD_ID="${FOLD_ID:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fold_${FOLD_ID}}"
EXPERIMENT_DIR="$CAR_DIR/output/$EXPERIMENT_NAME"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 0. 激活 conda ----------
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || log "警告：无法激活 $CONDA_ENV，使用当前环境"
fi

# ---------- 1. 前置检查 ----------
if [ ! -f "$CAR_DIR/main.py" ]; then echo "错误：找不到 $CAR_DIR/main.py"; exit 1; fi
if [ ! -d "$DATA_DIR" ]; then echo "错误：找不到 $DATA_DIR"; exit 1; fi
cd "$CAR_DIR"

# ---------- 2. 训练（fold 0） ----------
log "训练 fold $FOLD_ID（GPU $GPU, seed $SEED）→ $EXPERIMENT_DIR"
python main.py \
  --cfd_model=Transolver \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --fold_id "$FOLD_ID" \
  --gpu "$GPU" \
  --seed "$SEED" \
  --experiment_dir "$EXPERIMENT_DIR" \
  || { log "训练失败"; exit 1; }

# ---------- 3. 评估（fold 0） ----------
log "评估 fold $FOLD_ID（读 $EXPERIMENT_DIR/model_final.pth）"
python main_evaluation.py \
  --cfd_model=Transolver \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --fold_id "$FOLD_ID" \
  --gpu "$GPU" \
  --experiment_dir "$EXPERIMENT_DIR" \
  || { log "评估失败"; exit 1; }

log "完成。"
log "  实验目录：  $EXPERIMENT_DIR"
log "  评估结果：  $EXPERIMENT_DIR/evaluation/evaluation_metrics.json"
log "  （含体速度/表面压力相对 L2、Cd 相对误差、Spearman 四个论文指标）"
