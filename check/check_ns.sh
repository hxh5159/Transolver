#!/usr/bin/env bash
# =============================================================================
# 检查 DPOT-plus 数据目录里是否有本仓库 NS 任务需要的数据，若有则复制到目标 data 路径。
#
# 仓库 NS 任务需要的文件（exp_ns.py）：
#   NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat
#
# 用法（在远端执行）：
#   bash check_ns.sh
# 可选环境变量：
#   SOURCE_DIR  源目录（默认 DPOT-plus 的 fno 目录）
#   DATA        目标数据根目录
# =============================================================================
set -uo pipefail

SOURCE_DIR="${SOURCE_DIR:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/pdefoundationmodel/DPOT-plus/data/raw/fno}"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
TARGET_DIR="$DATA/fno"
REQUIRED="NavierStokes_V1e-5_N1200_T20.mat"
DST="$TARGET_DIR/NavierStokes_V1e-5_N1200_T20/$REQUIRED"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 1. 检查源目录 ----------
if [ ! -d "$SOURCE_DIR" ]; then
  echo "错误：源目录不存在：$SOURCE_DIR"
  exit 1
fi

log "=== 源目录内容 $SOURCE_DIR ==="
ls -la "$SOURCE_DIR" 2>/dev/null

log "=== 所有 NavierStokes 相关文件（含子目录，最多 4 层）==="
find "$SOURCE_DIR" -maxdepth 4 -iname '*navierstokes*' -o -maxdepth 4 -iname '*navier*' 2>/dev/null | sort

# ---------- 2. 定位所需文件 ----------
log "=== 定位所需文件 $REQUIRED ==="
MAT_FILES=$(find "$SOURCE_DIR" -type f -name "$REQUIRED" 2>/dev/null)

if [ -n "$MAT_FILES" ]; then
  # 直接找到 .mat
  SRC_MAT=$(echo "$MAT_FILES" | head -n1)
  log "找到 .mat：$SRC_MAT（共 $(echo "$MAT_FILES" | wc -l) 处）"
  mkdir -p "$(dirname "$DST")"
  if [ -s "$DST" ]; then
    log "目标已存在且非空，跳过复制：$DST"
  else
    cp -v "$SRC_MAT" "$DST"
  fi
else
  # 没找到 .mat，尝试找 zip
  ZIP=$(find "$SOURCE_DIR" -type f -name 'NavierStokes_V1e-5_N1200_T20*.zip' 2>/dev/null | head -n1)
  if [ -n "$ZIP" ]; then
    log "未找到 .mat，但找到 zip：$ZIP，解压到目标..."
    mkdir -p "$(dirname "$DST")"
    unzip -o "$ZIP" -d "$(dirname "$DST")" >/dev/null
    FOUND=$(find "$(dirname "$DST")" -type f -name "$REQUIRED" 2>/dev/null | head -n1)
    if [ -n "$FOUND" ] && [ "$FOUND" != "$DST" ]; then
      mv "$FOUND" "$DST"
      log "已归位：$FOUND -> $DST"
    fi
  else
    log "错误：源目录里既没有 $REQUIRED，也没有对应的 zip。"
    log "源目录实际文件（最多 50 个）如下，请人工确认是否有其它名字/分辨率的 NS 数据："
    find "$SOURCE_DIR" -maxdepth 4 -type f 2>/dev/null | head -50
    exit 1
  fi
fi

# ---------- 3. 校验 ----------
if [ ! -s "$DST" ]; then
  echo "错误：复制/解压后目标仍为空：$DST"
  exit 1
fi

SIZE=$(du -h "$DST" | cut -f1)
log "已就位：$DST（$SIZE）"

# 用 scipy.io.loadmat 校验可读，并打印内部键
python - "$DST" <<'PY'
import sys
try:
    import scipy.io as sio
    import numpy as np
except ImportError as e:
    print("缺少依赖，跳过内容校验：", e)
    sys.exit(0)

path = sys.argv[1]
data = sio.loadmat(path)
keys = [k for k in data.keys() if not k.startswith('__')]
print("matlab 内部键：", keys)
for k in keys:
    v = data[k]
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
print("校验通过：scipy.io.loadmat 可正常读取")
PY

log "完成。目标文件可直接用于 exp_ns.py（--data_path 指向 $TARGET_DIR）。"
