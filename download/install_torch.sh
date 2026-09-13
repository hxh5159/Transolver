#!/usr/bin/env bash
# =============================================================================
# 安装 PyTorch（CUDA 12.8 / cu128）
#
# 用法：
#   bash install_torch.sh
# 可选环境变量：
#   CONDA_ENV=transolver   目标 conda 环境（默认 transolver）
#   CUDA_VER=cu128         torch 的 CUDA 标签（默认 cu128 = CUDA 12.8）
#
# 说明：
#   - CUDA 12.8 对应 PyTorch 索引 cu128。
#   - 会先卸载旧 torch，避免之前 cu 版本不匹配（如 cu130）导致 "driver too old"。
#   - 验证时做真实 GPU 矩阵运算，确保驱动/内核真正可用，而不是仅 import 成功。
# =============================================================================
set -uo pipefail

CONDA_ENV="${CONDA_ENV:-transolver}"
CUDA_VER="${CUDA_VER:-cu128}"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 1. 激活/创建 conda 环境 ----------
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

# ---------- 2. 卸载旧 torch（避免 cu 版本冲突） ----------
log "卸载旧 torch / torchvision / torchaudio（如有）"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# ---------- 3. 安装 torch（CUDA 12.8） ----------
log "安装 torch（CUDA 12.8，索引：${CUDA_VER}）"
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_VER}"

# ---------- 4. 验证（含真实 GPU 运算） ----------
log "验证 torch 与 CUDA..."
python - <<'PY'
import torch
print("  torch            ", torch.__version__)
print("  torch 编译的 CUDA ", torch.version.cuda)
print("  CUDA 可用         ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU 名称          ", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("  GPU 矩阵运算      OK（结果 %.2f）" % (x @ x).sum().item())
else:
    print("  警告：CUDA 不可用。可能驱动低于 CUDA 12.8。")
    print("        请更新驱动，或用更低 CUDA 版本重装：CUDA_VER=cu124 bash install_torch.sh")
PY

log "torch 安装完成。"
