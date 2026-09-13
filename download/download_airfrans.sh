#!/usr/bin/env bash
# 下载 AirfRANS 翼型设计数据（Dataset.zip，约 9.3GB）
# 落地到 $DATA/naca/Dataset/（含 manifest.json），匹配 main.py 的 --my_path。
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

AIRFRANS_URL="https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip"
ZIP="$AIRFRANS_DIR/Dataset.zip"

log "下载 AirfRANS 翼型设计数据 → $AIRFRANS_DIR"
mkdir -p "$AIRFRANS_DIR"

# 优先用 aria2c 多连接并行下载（-x 16 -s 16），否则回退 wget -c 续传。
# 加 --connect-timeout 让连不上时 15s 快速失败（否则会一直卡在 TCP/TLS 握手），
# 失败后由 common.sh 的 retry 在 10s 后自动重试。
if command -v aria2c >/dev/null 2>&1; then
  retry aria2c -x 16 -s 16 -c --connect-timeout=15 --timeout=60 \
    -d "$AIRFRANS_DIR" -o Dataset.zip "$AIRFRANS_URL"
else
  retry wget -c --connect-timeout=15 --timeout=60 "$AIRFRANS_URL" -O "$ZIP"
fi

log "下载完成，开始解压..."
if ! unzip -o "$ZIP" -d "$AIRFRANS_DIR" >/dev/null; then
  log "解压失败，可能下载不完整。请重跑本脚本（会自动续传）后再次解压。"
  exit 1
fi

# 把 manifest.json 所在目录归位到 $AIRFRANS_DIR/Dataset（兼容 zip 内带/不带 Dataset 目录两种情况）
MANIFEST=$(find "$AIRFRANS_DIR" -maxdepth 3 -name manifest.json -print -quit)
if [ -z "$MANIFEST" ]; then
  log "错误：解压后未找到 manifest.json，请检查 Dataset.zip 内容。"
  exit 1
fi
REAL_DIR="$(dirname "$MANIFEST")"
if [ "$REAL_DIR" != "$AIRFRANS_DIR/Dataset" ]; then
  log "manifest.json 位于 $REAL_DIR，归位到 $AIRFRANS_DIR/Dataset"
  mkdir -p "$AIRFRANS_DIR/Dataset"
  mv "$REAL_DIR"/* "$AIRFRANS_DIR/Dataset"/
  rmdir "$REAL_DIR" 2>/dev/null || true
fi

log "AirfRANS 数据就绪：$AIRFRANS_DIR/Dataset"
