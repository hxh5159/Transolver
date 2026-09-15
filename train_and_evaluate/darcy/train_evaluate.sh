#!/usr/bin/env bash
# =============================================================================
# Darcy Flow（标准基准）单卡训练 + 评估，使用论文配置。
#
# 论文配置（来自 scripts/Transolver_Darcy.sh）：
#   model=Transolver_Structured_Mesh_2D, n_hidden=128, n_heads=8, n_layers=8,
#   lr=0.001, max_grad_norm=0.1, batch_size=4, slice_num=64, unified_pos=1,
#   ref=8, downsample=5
#   其余用 exp_darcy.py 默认：epochs=500, weight_decay=1e-5, ntrain=1000, ntest=200
#
# 与 exp_darcy.py 的 experiment-dir 逻辑兼容：训练输出到
#   <bench>/output/<EXPERIMENT_NAME>/（config.json、run_status.json、
#   checkpoints/checkpoint_final.pth、logs、visualizations、training_summary.json），
#   评估时 --eval 1 读 checkpoint_final.pth 并写 evaluation/evaluation_metrics.json。
#
# 用法： bash train_evaluate.sh
# 环境变量： DATA / CONDA_ENV / GPU / SEED / EXPERIMENT_NAME
# =============================================================================
set -uo pipefail

# ---------- 路径（从脚本位置推导仓库根） ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
BENCH_DIR="$REPO/PDE-Solving-StandardBenchmark"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
DATA_PATH="$DATA/fno"   # 内含 piececonst_r421_N1024_smooth1.mat / smooth2.mat

# ---------- 参数 ----------
CONDA_ENV="${CONDA_ENV:-transolver}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-darcy_Transolver}"
EXPERIMENT_DIR="$BENCH_DIR/output/$EXPERIMENT_NAME"

# ---------- 论文超参 ----------
MODEL=Transolver_Structured_Mesh_2D
N_HIDDEN=128
N_HEADS=8
N_LAYERS=8
LR=0.001
MAX_GRAD_NORM=0.1
BATCH_SIZE=4
SLICE_NUM=64
UNIFIED_POS=1
REF=8
DOWNSAMPLE=5

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 0. 激活 conda ----------
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || log "警告：无法激活 $CONDA_ENV，使用当前环境"
fi

# ---------- 1. 前置检查 ----------
if [ ! -f "$BENCH_DIR/exp_darcy.py" ]; then echo "错误：找不到 $BENCH_DIR/exp_darcy.py"; exit 1; fi
if [ ! -s "$DATA_PATH/piececonst_r421_N1024_smooth1.mat" ] || [ ! -s "$DATA_PATH/piececonst_r421_N1024_smooth2.mat" ]; then
  echo "错误：找不到 Darcy 数据 $DATA_PATH/piececonst_r421_N1024_smooth{1,2}.mat"
  exit 1
fi
cd "$BENCH_DIR"

# 训练/评估共用参数（保持两阶段模型架构完全一致）
COMMON_ARGS=(
  --model "$MODEL"
  --n-hidden "$N_HIDDEN"
  --n-heads "$N_HEADS"
  --n-layers "$N_LAYERS"
  --lr "$LR"
  --max_grad_norm "$MAX_GRAD_NORM"
  --batch-size "$BATCH_SIZE"
  --slice_num "$SLICE_NUM"
  --unified_pos "$UNIFIED_POS"
  --ref "$REF"
  --downsample "$DOWNSAMPLE"
  --gpu "$GPU"
  --data_path "$DATA_PATH"
  --save_name "$EXPERIMENT_NAME"
  --experiment_dir "$EXPERIMENT_DIR"
  --seed "$SEED"
)

# ---------- 2. 训练 ----------
log "开始训练（$MODEL, 500 epoch, GPU $GPU, seed $SEED）→ $EXPERIMENT_DIR"
python exp_darcy.py "${COMMON_ARGS[@]}" --eval 0 || { log "训练失败"; exit 1; }

# ---------- 3. 评估 ----------
log "开始评估（读取 checkpoint_final.pth）"
python exp_darcy.py "${COMMON_ARGS[@]}" --eval 1 || { log "评估失败"; exit 1; }

log "完成。"
log "  实验目录：  $EXPERIMENT_DIR"
log "  评估结果：  $EXPERIMENT_DIR/evaluation/evaluation_metrics.json"
log "  训练曲线：  $EXPERIMENT_DIR/visualizations/"
