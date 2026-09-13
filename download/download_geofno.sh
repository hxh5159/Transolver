#!/usr/bin/env bash
# 下载 Geo-FNO 数据（仅训练所需的 9 个文件，按文件 ID 逐个下载）：
#   Elasticity + Plasticity + Airfoil(基准) + Pipe
# 落地到 $DATA/fno/，直接匹配 exp_elas.py / exp_plas.py / exp_airfoil.py / exp_pipe.py 的路径。
# 说明：不再用 gdown --folder 整文件夹下（该 Drive 文件夹还含 car-cfd/channel-shocks 等无关数据，
#       且整夹下载容易撞 Google Drive 配额）。改为按 ID 并行下载所需文件。
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log "下载 Geo-FNO 数据（9 个文件）→ $FNO_DIR"
ensure_gdown

# 格式： <file_id>|<相对 $FNO_DIR 的目标路径>
FILES=(
  "16EY0obqsccypaDFVY0wlsXX73SlD5TMy|airfoil/naca/NACA_Cylinder_X.npy"
  "1rJUPtIhTAsG8TQnqV5mljjgQlyz0NJvJ|airfoil/naca/NACA_Cylinder_Y.npy"
  "1AjW0t0YolY680J6xTQJ_g5bqTSJAZDZc|airfoil/naca/NACA_Cylinder_Q.npy"
  "1Ia5izgUum-IQLdO6PW70HO8AdAqA_IVb|elasticity/Meshes/Random_UnitCell_sigma_10.npy"
  "1I-fO-RsFvD3nqBuFrg67R0yqTFdD_gpA|elasticity/Meshes/Random_UnitCell_XY_10.npy"
  "1_FS-9Z1eZ3BmlDvANB7Y4-auwxwNnhnY|pipe/Pipe_X.npy"
  "1ZjB4kEauqViMmz18ZfVaMpiuFhgZtRDX|pipe/Pipe_Y.npy"
  "1Yj1EOYwCYBhlURISNHqvE3atVLDS_Jmp|pipe/Pipe_Q.npy"
  "14CPGK_ljae5c6dm2nRraY2kIDt39JX3d|plas_N987_T20.mat"
)

# 并行下载（利用多核）；每个文件独立重试，已完成自动跳过。
pids=()
for entry in "${FILES[@]}"; do
  id="${entry%%|*}"
  rel="${entry#*|}"
  download_file "$id" "$FNO_DIR/$rel" &
  pids+=("$!")
done

for p in "${pids[@]}"; do
  wait "$p"
done

log "Geo-FNO 数据下载完成。"
