#!/usr/bin/env bash
# =============================================================================
# 远端路径设置脚本
#
# 用法（注意：必须用 source，不能用 bash，否则 export 不会保留到当前 shell）：
#   source path.sh
# 或把它加到 ~/.bashrc 末尾，每次打开实例自动生效：
#   echo "source <本脚本绝对路径>" >> ~/.bashrc
# =============================================================================

# ---- 仓库根（已从 Transolver 改为 Transolver_re）----
export REPO=/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/Transolver_re

# ---- 数据根（未变）----
export DATA=/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data

# ---- 标准基准（6 个 PDE benchmark，exp_*.py 的 --data_path 根）----
export FNO_DIR="$DATA/fno"

# ---- 汽车设计（ShapeNetCar / MLCFD）----
export CAR_DATA_DIR="$DATA/mlcfd/training_data"       # main.py 的 --data_dir
export CAR_SAVE_DIR="$DATA/mlcfd/preprocessed_data"   # main.py 的 --save_dir

# ---- 翼型设计（AirfRANS）----
export AIRFRANS_DATASET="$DATA/naca/Dataset"          # main.py 的 --my_path

# ---- 项目子目录 ----
export CAR_DIR="$REPO/Car-Design-ShapeNetCar"
export BENCH_DIR="$REPO/PDE-Solving-StandardBenchmark"
export AIRFOIL_DIR="$REPO/Airfoil-Design-AirfRANS"

# ---- 常用环境 ----
export CONDA_ENV=transolver

# ---- 打印确认 ----
echo "[path.sh] 远端路径已设置："
echo "  REPO            = $REPO"
echo "  DATA            = $DATA"
echo "  FNO_DIR         = $FNO_DIR"
echo "  CAR_DATA_DIR    = $CAR_DATA_DIR"
echo "  CAR_SAVE_DIR    = $CAR_SAVE_DIR"
echo "  AIRFRANS_DATASET = $AIRFRANS_DATASET"
echo "  CAR_DIR         = $CAR_DIR"
echo "  BENCH_DIR       = $BENCH_DIR"
echo "  AIRFOIL_DIR     = $AIRFOIL_DIR"
echo "  CONDA_ENV       = $CONDA_ENV"
