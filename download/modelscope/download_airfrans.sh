#!/usr/bin/env bash
# =============================================================================
# 用 ModelScope 下载 AirfRANS 翼型数据（OneScience/airfrans），
# 并自动归位到远端正确路径 $DATA/naca/Dataset/，带中断后 10s 自动重试。
#
# 用法： bash download_airfrans.sh
# 环境变量： DATA（默认远端数据根）
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

STAGE="$DATA/.staging"

ensure_modelscope() {
  if ! command -v modelscope >/dev/null 2>&1; then
    log "安装 modelscope..."
    pip install -U modelscope
  fi
}

# ---------- 下载 ----------
log "=== 下载 airfrans（AirfRANS 翼型设计，约 15GB）==="
ensure_modelscope
mkdir -p "$STAGE"
retry modelscope download --dataset OneScience/airfrans --local_dir "$STAGE/airfrans"

# ---------- 归位到 $DATA/naca/Dataset ----------
log "=== 归位到 $DATA/naca/Dataset ==="
if [ -s "$DATA/naca/Dataset/manifest.json" ]; then
  log "AirfRANS 已存在，跳过"
else
  MANIFEST=$(find "$STAGE/airfrans" -name manifest.json -print -quit 2>/dev/null)
  if [ -n "$MANIFEST" ]; then
    REAL_DIR="$(dirname "$MANIFEST")"
    mkdir -p "$DATA/naca/Dataset"
    cp -r "$REAL_DIR"/. "$DATA/naca/Dataset/"
    log "AirfRANS 已归位到 $DATA/naca/Dataset"
  else
    log "警告：暂存目录里没找到 manifest.json，请检查下载结果"
    exit 1
  fi
fi

# ---------- 校验 ----------
log "=== 校验 ==="
if [ -s "$DATA/naca/Dataset/manifest.json" ]; then
  log "OK        $DATA/naca/Dataset/manifest.json"
  log "✔ AirfRANS 就位。可删除暂存： rm -rf $STAGE/airfrans"
else
  log "MISSING   $DATA/naca/Dataset/manifest.json"
  exit 1
fi
