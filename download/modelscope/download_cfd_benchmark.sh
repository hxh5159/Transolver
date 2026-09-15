#!/usr/bin/env bash
# =============================================================================
# 用 ModelScope 下载 6 个标准 PDE benchmark（OneScience/cfd_benchmark），
# 并自动归位到远端正确路径 $DATA/fno/，带中断后 10s 自动重试。
#
# 用法： bash download_cfd_benchmark.sh
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

# ---------- 下载 ----------
log "=== 下载 cfd_benchmark（6 个标准基准）==="
ensure_modelscope
mkdir -p "$STAGE"
retry modelscope download --dataset OneScience/cfd_benchmark --local_dir "$STAGE/cfd_benchmark"

# ---------- 归位到 $FNO_DIR ----------
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

# ---------- 校验 ----------
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
  "$FNO_DIR/pipe/Pipe_Q.npy"; do
  if [ -s "$f" ]; then log "OK        $f"; else log "MISSING   $f"; fail=1; fi
done

if [ "$fail" -eq 0 ]; then
  log "✔ 标准基准 12 个文件全部就位。可删除暂存： rm -rf $STAGE/cfd_benchmark"
else
  log "✘ 有缺失文件，请贴出 find $STAGE/cfd_benchmark -type f 结果排查"
fi
