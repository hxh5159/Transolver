#!/usr/bin/env bash
# 并行启动 3 个下载源（利用多核），全部完成后统一校验。
# 也可单独运行某个下载脚本，例如： bash download_airfrans.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "并行启动 3 个下载任务（FNO / Geo-FNO / AirfRANS）..."
"$SCRIPT_DIR/download_fno.sh"     & PID1=$!
"$SCRIPT_DIR/download_geofno.sh"  & PID2=$!
"$SCRIPT_DIR/download_airfrans.sh" & PID3=$!

wait "$PID1"; R1=$?
wait "$PID2"; R2=$?
wait "$PID3"; R3=$?

echo "下载脚本退出码：fno=$R1 geofno=$R2 airfrans=$R3"

if [ "$R1" -eq 0 ] && [ "$R2" -eq 0 ] && [ "$R3" -eq 0 ]; then
  echo "所有下载完成，开始校验..."
  "$SCRIPT_DIR/verify.sh"
else
  echo "存在下载失败（退出码非 0），请查看上方日志。"
  exit 1
fi
