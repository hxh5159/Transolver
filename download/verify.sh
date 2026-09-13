#!/usr/bin/env bash
# 校验全部数据是否就位（缺失时尝试按文件名在数据根目录下找回并归位），
# 校验通过即代表可直接用于训练。
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# 训练所需的全部文件（相对 $DATA）
EXPECTED=(
  "$FNO_DIR/piececonst_r421_N1024_smooth1.mat"                     # Darcy
  "$FNO_DIR/piececonst_r421_N1024_smooth2.mat"                     # Darcy
  "$FNO_DIR/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"  # Navier-Stokes
  "$FNO_DIR/plas_N987_T20.mat"                                     # Plasticity
  "$FNO_DIR/elasticity/Meshes/Random_UnitCell_sigma_10.npy"        # Elasticity
  "$FNO_DIR/elasticity/Meshes/Random_UnitCell_XY_10.npy"           # Elasticity
  "$FNO_DIR/airfoil/naca/NACA_Cylinder_X.npy"                      # Airfoil(基准)
  "$FNO_DIR/airfoil/naca/NACA_Cylinder_Y.npy"
  "$FNO_DIR/airfoil/naca/NACA_Cylinder_Q.npy"
  "$FNO_DIR/pipe/Pipe_X.npy"                                       # Pipe
  "$FNO_DIR/pipe/Pipe_Y.npy"
  "$FNO_DIR/pipe/Pipe_Q.npy"
  "$AIRFRANS_DIR/Dataset/manifest.json"                            # Airfoil(设计)
)

# 目标缺失时，在对应根目录下按文件名找回并归位
repair() {
  local target="$1" base="${target##*/}" root
  case "$target" in
    "$FNO_DIR"/*)      root="$FNO_DIR" ;;
    "$AIRFRANS_DIR"/*) root="$AIRFRANS_DIR" ;;
    *)                 root="$DATA" ;;
  esac
  [ -s "$target" ] && return 0
  local found
  found=$(find "$root" -type f -name "$base" 2>/dev/null | head -n1)
  if [ -n "$found" ] && [ "$found" != "$target" ]; then
    mkdir -p "$(dirname "$target")"
    mv "$found" "$target"
    log "已归位：$found -> $target"
  fi
}

fail=0
for f in "${EXPECTED[@]}"; do
  repair "$f"
  if [ -s "$f" ]; then
    log "OK        $f"
  else
    log "MISSING   $f"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  log "✔ 全部数据就绪，可直接用于训练。"
else
  log "✘ 仍有缺失文件，请重跑对应下载脚本后再执行本脚本。"
  exit 1
fi
