#!/usr/bin/env bash
# =============================================================================
# 用 ModelScope 替代源下载数据集，并自动归位到远端正确路径。
#   - 6 个标准基准：OneScience/cfd_benchmark  -> $DATA/fno/
#   - AirfRANS 翼型： OneScience/airfrans      -> $DATA/naca/Dataset/
#
# 说明：
#   - ModelScope 下载到 $DATA/.staging（暂存，与目标同盘），再按文件名归位到 fno/。
#   - 带 10s 自动重试（复用 common.sh 的 retry）；modelscope 下载幂等，重试会跳过已下文件。
#   - NS 数据你已从 DPOT-plus 拿到，脚本里对 NS 也会归位但若已存在则跳过。
#
# 用法： bash download_modelscope.sh
# 环境变量： DATA（默认远端数据根）
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

STAGE="$DATA/.staging"

ensure_modelscope() {
  if ! command -v modelscope >/dev/null 2>&1; then
    log "安装 modelscope..."
    pip install -U modelscope
  fi
}

# 按文件名在暂存目录里找，移动到目标路径（已有则跳过）
move_into() {
  local base="$1" target="$2"
  [ -s "$target" ] && return 0
  local found
  found=$(find "$STAGE" -type f -name "$base" 2>/dev/null | head -n1)
  if [ -n "$found" ]; then
    mkdir -p "$(dirname "$target")"
    mv "$found" "$target"
    log "已归位：$found -> $target"
  fi
}

# ---------- 1. 下载 cfd_benchmark（6 个标准基准） ----------
log "=== 下载 cfd_benchmark（6 个标准基准）==="
ensure_modelscope
mkdir -p "$STAGE"
retry modelscope download --dataset OneScience/cfd_benchmark --local_dir "$STAGE/cfd_benchmark"

# ---------- 2. 归位 12 个文件到 $FNO_DIR ----------
log "=== 归位到 $FNO_DIR ==="
move_into "piececonst_r421_N1024_smooth1.mat" "$FNO_DIR/piececonst_r421_N1024_smooth1.mat"
move_into "piececonst_r421_N1024_smooth2.mat" "$FNO_DIR/piececonst_r421_N1024_smooth2.mat"
move_into "NavierStokes_V1e-5_N1200_T20.mat" "$FNO_DIR/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"
move_into "plas_N987_T20.mat" "$FNO_DIR/plas_N987_T20.mat"
move_into "Random_UnitCell_sigma_10.npy" "$FNO_DIR/elasticity/Meshes/Random_UnitCell_sigma_10.npy"
move_into "Random_UnitCell_XY_10.npy" "$FNO_DIR/elasticity/Meshes/Random_UnitCell_XY_10.npy"
move_into "NACA_Cylinder_X.npy" "$FNO_DIR/airfoil/naca/NACA_Cylinder_X.npy"
move_into "NACA_Cylinder_Y.npy" "$FNO_DIR/airfoil/naca/NACA_Cylinder_Y.npy"
move_into "NACA_Cylinder_Q.npy" "$FNO_DIR/airfoil/naca/NACA_Cylinder_Q.npy"
move_into "Pipe_X.npy" "$FNO_DIR/pipe/Pipe_X.npy"
move_into "Pipe_Y.npy" "$FNO_DIR/pipe/Pipe_Y.npy"
move_into "Pipe_Q.npy" "$FNO_DIR/pipe/Pipe_Q.npy"

# ---------- 3. 下载 AirfRANS 并归位 ----------
log "=== 下载 AirfRANS（翼型设计）==="
retry modelscope download --dataset OneScience/airfrans --local_dir "$STAGE/airfrans"

if [ ! -s "$DATA/naca/Dataset/manifest.json" ]; then
  MANIFEST=$(find "$STAGE/airfrans" -name manifest.json -print -quit 2>/dev/null)
  if [ -n "$MANIFEST" ]; then
    REAL_DIR="$(dirname "$MANIFEST")"
    mkdir -p "$DATA/naca/Dataset"
    cp -r "$REAL_DIR"/. "$DATA/naca/Dataset/"
    log "AirfRANS 已归位到 $DATA/naca/Dataset"
  else
    log "警告：暂存目录里没找到 manifest.json，请检查 airfrans 下载结果"
  fi
else
  log "AirfRANS 已存在，跳过"
fi

# ---------- 4. 校验 ----------
log "=== 校验 ==="
fail=0
for f in \
  "$FNO_DIR/piececonst_r421_N1024_smooth1.mat" \
  "$FNO_DIR/piececonst_r421_N1024_smooth2.mat" \
  "$FNO_DIR/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat" \
  "$FNO_DIR/plas_N987_T20.mat" \
  "$FNO_DIR/elasticity/Meshes/Random_UnitCell_sigma_10.npy" \
  "$FNO_DIR/elasticity/Meshes/Random_UnitCell_XY_10.npy" \
  "$FNO_DIR/airfoil/naca/NACA_Cylinder_X.npy" \
  "$FNO_DIR/airfoil/naca/NACA_Cylinder_Y.npy" \
  "$FNO_DIR/airfoil/naca/NACA_Cylinder_Q.npy" \
  "$FNO_DIR/pipe/Pipe_X.npy" \
  "$FNO_DIR/pipe/Pipe_Y.npy" \
  "$FNO_DIR/pipe/Pipe_Q.npy" \
  "$DATA/naca/Dataset/manifest.json"; do
  if [ -s "$f" ]; then log "OK        $f"; else log "MISSING   $f"; fail=1; fi
done

if [ "$fail" -eq 0 ]; then
  log "✔ 全部数据就位。可删除暂存目录： rm -rf $STAGE"
else
  log "✘ 有缺失文件，请检查上方输出（可能 cfd_benchmark 内部文件名与预期不同，贴出 find 结果再排查）"
fi
