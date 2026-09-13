#!/usr/bin/env bash
# 下载 FNO 数据：Darcy Flow + Navier-Stokes（标准基准）
# 只下载训练所需的 2 个 zip（不用 gdown --folder 整夹下载，避免下到 Burgers/KFvorticity/KS 等无关数据），
# 解压后把 .mat 归位到 exp_darcy.py / exp_ns.py 期望的路径。
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log "下载 FNO 数据（仅 Darcy + Navier-Stokes）→ $FNO_DIR"
ensure_gdown
mkdir -p "$FNO_DIR"

# 1) Darcy_421.zip -> piececonst_r421_N1024_smooth1.mat / smooth2.mat
download_file "1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf" "$FNO_DIR/Darcy_421.zip"

# 2) NavierStokes_V1e-5_N1200_T20.zip -> NavierStokes_V1e-5_N1200_T20.mat
download_file "1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5" "$FNO_DIR/NavierStokes_V1e-5_N1200_T20.zip"

# 解压
for z in Darcy_421.zip NavierStokes_V1e-5_N1200_T20.zip; do
  if [ -s "$FNO_DIR/$z" ]; then
    log "解压 $z ..."
    unzip -o "$FNO_DIR/$z" -d "$FNO_DIR" >/dev/null || log "解压 $z 失败，请检查 zip 是否完整"
  fi
done

# 把解压出的 .mat 归位到代码期望的路径（exp_darcy.py / exp_ns.py 里拼接的路径）
move_into() {
  local base="$1" target="$2"
  if [ -s "$target" ]; then return 0; fi
  local found
  found=$(find "$FNO_DIR" -type f -name "$base" -print -quit)
  if [ -n "$found" ] && [ "$found" != "$target" ]; then
    mkdir -p "$(dirname "$target")"
    mv "$found" "$target"
    log "已归位：$found -> $target"
  fi
}
move_into "piececonst_r421_N1024_smooth1.mat" "$FNO_DIR/piececonst_r421_N1024_smooth1.mat"
move_into "piececonst_r421_N1024_smooth2.mat" "$FNO_DIR/piececonst_r421_N1024_smooth2.mat"
move_into "NavierStokes_V1e-5_N1200_T20.mat" "$FNO_DIR/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"

log "FNO 数据下载完成（可执行 bash verify.sh 校验）。"
