#!/usr/bin/env bash
# 公共配置与工具函数，被各下载脚本 source。
# 用法： source "$(dirname "$0")/common.sh"

# 数据根目录（与远端已有的 mlcfd 数据同级）。
export DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"

# 6 个标准 PDE 基准数据根（exp_*.py 里的 --data_path 都指向这里）
FNO_DIR="$DATA/fno"
# AirfRANS 翼型设计数据根（main.py 里的 --my_path 指向 $AIRFRANS_DIR/Dataset）
AIRFRANS_DIR="$DATA/naca"

# 普通失败（网络中断等）重试等待秒数
RETRY_DELAY=10
# Google Drive 配额限制（"Too many users..."）重试等待秒数，默认 30 分钟
QUOTA_DELAY="${QUOTA_DELAY:-1800}"
# 可选：浏览器导出的 Google cookies.txt，可绕过匿名下载配额
# 用法： export GDRIVE_COOKIE=/path/to/cookies.txt
GDRIVE_COOKIE="${GDRIVE_COOKIE:-}"

log() { echo "[$(date '+%F %T')] $*"; }

# 确保 gdown 可用
ensure_gdown() {
  if ! command -v gdown >/dev/null 2>&1; then
    log "未找到 gdown，正在 pip 安装..."
    pip install -q gdown
  fi
}

# 通用重试：命令失败 -> sleep $RETRY_DELAY 秒 -> 重试，直到成功（Ctrl+C 可中断）
retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    log "第 ${n} 次执行失败，${RETRY_DELAY}s 后自动重试：$*"
    sleep "$RETRY_DELAY"
  done
  log "执行成功：$*"
}

# 下载单个 Google Drive 文件（按文件 ID），配额感知的重试。
# 用法： download_file <file_id> <target_path>
#   - 目标已非空 -> 跳过（幂等）
#   - 普通失败 -> 10s 后重试
#   - 配额限制("Too many users") -> 等 QUOTA_DELAY 秒后重试
download_file() {
  local id="$1" target="$2"
  mkdir -p "$(dirname "$target")"
  local n=0 out cookie_args=()
  [ -n "$GDRIVE_COOKIE" ] && cookie_args=(--cookie "$GDRIVE_COOKIE")
  while :; do
    if [ -s "$target" ]; then
      log "已完成：$target"
      return 0
    fi
    out=$(gdown --id "$id" "${cookie_args[@]}" -O "$target" 2>&1)
    if [ -s "$target" ]; then
      log "完成：$target"
      return 0
    fi
    n=$((n + 1))
    if printf '%s' "$out" | grep -q "Too many users"; then
      log "[$(basename "$target")] 触发 Google Drive 配额限制（第 ${n} 次），${QUOTA_DELAY}s 后重试"
      sleep "$QUOTA_DELAY"
    else
      log "[$(basename "$target")] 下载失败（第 ${n} 次），${RETRY_DELAY}s 后重试"
      sleep "$RETRY_DELAY"
    fi
  done
}
