#!/usr/bin/env bash
# =============================================================================
# Transolver 完整依赖安装脚本
# 覆盖三个子项目：标准基准(PDE-Solving-StandardBenchmark) + 汽车(Car) + 翼型(Airfoil)
#
# 用法（在 conda 环境 transolver / Python 3.10 下）：
#   bash pip.sh
# 可选环境变量：
#   CONDA_ENV=transolver   目标 conda 环境名
#   CUDA_TAG=cu128         若需新装 torch，指定 CUDA 版本（如 cu121/cu124/cu128）；留空则复用已装 torch
#
# 关键结论（踩坑记录）：
#   - torch_geometric 的 radius_graph 需要 C++ 扩展 pyg-lib（torch_geometric>=2.5）
#     或 torch-cluster（<=2.4，已废弃）。正确做法是用最新 torch_geometric + 装 pyg-lib。
#   - pyg-lib 必须与 torch 版本严格匹配，且清华镜像没有它，必须走 pyg 官方源 data.pyg.org。
#   - 原 requirements.txt 里的 dgl / h5py 实际未被代码使用，无需安装。
#   - 原 requirements.txt 里 torch==1.10.1 太老，现代环境用 torch 2.x。
# =============================================================================
set -uo pipefail

CONDA_ENV="${CONDA_ENV:-transolver}"
CUDA_TAG="${CUDA_TAG:-}"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 0. 激活/创建 conda 环境 ----------
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    conda activate "$CONDA_ENV"
  else
    log "创建 conda 环境 $CONDA_ENV (python 3.10)"
    conda create -y -n "$CONDA_ENV" python=3.10
    conda activate "$CONDA_ENV"
  fi
  log "当前环境：$CONDA_ENV（$(python --version 2>&1)）"
else
  log "未检测到 conda，使用当前 Python 环境"
fi

# ---------- 1. 安装 PyTorch ----------
if [ -n "$CUDA_TAG" ]; then
  log "安装 torch + torchvision（CUDA: $CUDA_TAG）"
  pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
elif python -c "import torch" 2>/dev/null; then
  log "已装 torch $(python -c 'import torch;print(torch.__version__)')，跳过安装"
else
  log "错误：未检测到 torch，且未指定 CUDA_TAG。请设置 CUDA_TAG=cu128 后重跑（或先手动装 torch）。"
  exit 1
fi

TORCH_VER="$(python -c 'import torch;print(torch.__version__)')"
log "torch 版本：$TORCH_VER"

# ---------- 2. 安装 torch_geometric + pyg-lib（关键步骤） ----------
log "安装 torch_geometric（最新版，>=2.5 才能与 pyg-lib 配对）"
pip install -U torch_geometric

log "安装 pyg-lib（匹配 torch ${TORCH_VER}，走 pyg 官方源，清华镜像没有此包）"
if ! pip install pyg-lib -f "https://data.pyg.org/whl/torch-${TORCH_VER}.html"; then
  log "错误：pyg-lib 安装失败。可能该 torch 版本无对应 wheel。"
  log "      可到 https://data.pyg.org/whl/ 查看是否有 torch-${TORCH_VER} 的目录。"
  exit 1
fi

# ---------- 3. 通用依赖 ----------
log "安装通用依赖（numpy/scipy/einops/timm/tqdm/matplotlib）"
pip install numpy scipy einops timm tqdm matplotlib

# ---------- 4. 汽车任务依赖 ----------
log "安装汽车任务依赖（vtk/scikit-learn）"
pip install vtk scikit-learn

# ---------- 5. 翼型任务依赖 ----------
log "安装翼型任务依赖（seaborn/pyvista/pyyaml）"
pip install seaborn pyvista pyyaml

# ---------- 6. 校验关键依赖 ----------
log "校验关键依赖..."
python - <<'PY'
import sys
import torch
import torch_geometric
from torch_geometric.nn import radius_graph

print("  torch           ", torch.__version__)
print("  torch_geometric ", torch_geometric.__version__)
print("  CUDA available  ", torch.cuda.is_available())

# 触发 radius_graph 的真实调用（之前卡住的地方）
radius_graph(torch.randn(8, 2), r=0.5, max_num_neighbors=8)
print("  radius_graph    OK")
PY
[ $? -ne 0 ] && { log "依赖校验失败，请检查上方输出"; exit 1; }

log "依赖安装完成。可运行： bash run/train_car.sh 开始汽车任务训练。"
