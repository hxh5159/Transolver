#!/usr/bin/env bash
# =============================================================================
# Car-Design-ShapeNetCar 训练脚本（在远端运行）
#
# 用法：
#   bash train_car.sh                     # 默认 fold 0, gpu 0
#   bash train_car.sh --fold_id 3 --gpu 2
#
# 路径默认已按你的远端环境填好，可用环境变量覆盖：
#   REPO=/path/to/Transolver  DATA=/path/to/data  CONDA_ENV=transolver
# =============================================================================
set -uo pipefail

# ---------- 路径配置 ----------
REPO="${REPO:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/Transolver_re}"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
DATA_DIR="$DATA/mlcfd/training_data"        # 原始 VTK（param0..param8）
SAVE_DIR="$DATA/mlcfd/preprocessed_data"    # 预处理输出（npy）

PROJ_DIR="$REPO/Car-Design-ShapeNetCar"

# ---------- 训练参数 ----------
CONDA_ENV="${CONDA_ENV:-transolver}"
FOLD_ID="${FOLD_ID:-0}"
GPU="${GPU:-0}"

# 命令行参数覆盖
while [ $# -gt 0 ]; do
  case "$1" in
    --fold_id) FOLD_ID="$2"; shift 2 ;;
    --gpu)     GPU="$2";     shift 2 ;;
    *) echo "未知参数: $1（支持 --fold_id / --gpu）"; exit 1 ;;
  esac
done

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 1. 路径校验 ----------
if [ ! -d "$PROJ_DIR" ]; then
  echo "错误：项目目录不存在：$PROJ_DIR"; exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "错误：训练数据目录不存在：$DATA_DIR（应包含 param0..param8）"; exit 1
fi
n_param=$(find "$DATA_DIR" -maxdepth 1 -type d -name 'param*' | wc -l)
if [ "$n_param" -lt 1 ]; then
  echo "错误：$DATA_DIR 下没有 param* 子目录"; exit 1
fi
log "路径检查通过：$DATA_DIR（$n_param 个 param 目录）"

mkdir -p "$SAVE_DIR"

# ---------- 2. 激活 conda 环境 ----------
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || {
    echo "错误：无法激活 conda 环境 $CONDA_ENV"; exit 1
  }
  log "已激活 conda 环境：$CONDA_ENV"
else
  echo "警告：未检测到 conda，使用当前 Python 环境"
fi

# ---------- 3. 依赖自检（CUDA + torch_geometric/radius_graph） ----------
log "依赖自检..."
python - <<'PY'
import sys, torch, torch_geometric
from torch_geometric.nn import radius_graph

if not torch.cuda.is_available():
    print("ERROR: CUDA 不可用。main.py 里模型是硬编码 .cuda()，必须用 GPU。")
    print("       请更新驱动，或重装匹配驱动的 torch，例如：")
    print("       pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

try:
    radius_graph(torch.randn(8, 2), r=0.5, max_num_neighbors=8)
except Exception as e:
    print("ERROR: radius_graph 调用失败，缺少底层 C++ 扩展。")
    print("       实际错误：", e)
    print("       修复（pyg-lib 匹配当前 torch 版本）：")
    print("         pip install -U torch_geometric")
    print("         pip install pyg-lib -f https://data.pyg.org/whl/torch-%s.html" % torch.__version__)
    sys.exit(1)

print("依赖检查通过：torch", torch.__version__,
      "| torch_geometric", torch_geometric.__version__, "| CUDA ok")
PY
[ $? -ne 0 ] && exit 1

# ---------- 4. 判断是否首次预处理 ----------
n_npy=$(find "$SAVE_DIR" -name '*.npy' 2>/dev/null | wc -l)
PREPROCESSED=1
if [ "$n_npy" -eq 0 ]; then
  PREPROCESSED=0
  log "未检测到预处理数据，本次将从 VTK 首次生成（较慢，请耐心）"
else
  log "检测到预处理数据（$n_npy 个 npy），直接加载"
fi

# ---------- 5. 运行训练 ----------
cd "$PROJ_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
log "启动训练：fold_id=$FOLD_ID  gpu=$GPU  preprocessed=$PREPROCESSED"

python main.py \
  --cfd_model=Transolver \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --fold_id "$FOLD_ID" \
  --preprocessed "$PREPROCESSED"
