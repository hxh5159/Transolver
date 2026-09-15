#!/usr/bin/env bash
# =============================================================================
# 检查所有数据集是否可直接用于训练和评估。
# 覆盖：6 个标准基准（12 文件）+ 汽车（MLCFD）+ 翼型（AirfRANS）。
#
# 用法： bash check_other.sh
# 环境变量： DATA（默认远端数据根）
# =============================================================================
set -u

DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
FNO_DIR="$DATA/fno"

log() { echo "[$(date '+%F %T')] $*"; }

FAIL=0

# ===== 1. 标准基准 12 个文件存在性 =====
log "===== 1. 标准基准（12 个文件）存在性 ====="
BENCH_FILES=(
  "$FNO_DIR/piececonst_r421_N1024_smooth1.mat"
  "$FNO_DIR/piececonst_r421_N1024_smooth2.mat"
  "$FNO_DIR/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"
  "$FNO_DIR/plas_N987_T20.mat"
  "$FNO_DIR/elasticity/Meshes/Random_UnitCell_sigma_10.npy"
  "$FNO_DIR/elasticity/Meshes/Random_UnitCell_XY_10.npy"
  "$FNO_DIR/airfoil/naca/NACA_Cylinder_X.npy"
  "$FNO_DIR/airfoil/naca/NACA_Cylinder_Y.npy"
  "$FNO_DIR/airfoil/naca/NACA_Cylinder_Q.npy"
  "$FNO_DIR/pipe/Pipe_X.npy"
  "$FNO_DIR/pipe/Pipe_Y.npy"
  "$FNO_DIR/pipe/Pipe_Q.npy"
)
for f in "${BENCH_FILES[@]}"; do
  if [ -s "$f" ]; then log "OK        $f"; else log "MISSING   $f"; FAIL=1; fi
done

# ===== 2. 标准基准深度加载校验 =====
log "===== 2. 标准基准深度加载校验 ====="
python - "$FNO_DIR" <<'PY'
import sys, os
import numpy as np
import scipy.io as sio

fno_dir = sys.argv[1]
mat_files = [
    (os.path.join(fno_dir, 'piececonst_r421_N1024_smooth1.mat'), 'Darcy'),
    (os.path.join(fno_dir, 'piececonst_r421_N1024_smooth2.mat'), 'Darcy'),
    (os.path.join(fno_dir, 'NavierStokes_V1e-5_N1200_T20', 'NavierStokes_V1e-5_N1200_T20.mat'), 'Navier-Stokes'),
    (os.path.join(fno_dir, 'plas_N987_T20.mat'), 'Plasticity'),
]
npy_files = [
    (os.path.join(fno_dir, 'elasticity/Meshes/Random_UnitCell_sigma_10.npy'), 'Elasticity'),
    (os.path.join(fno_dir, 'elasticity/Meshes/Random_UnitCell_XY_10.npy'), 'Elasticity'),
    (os.path.join(fno_dir, 'airfoil/naca/NACA_Cylinder_X.npy'), 'Airfoil'),
    (os.path.join(fno_dir, 'airfoil/naca/NACA_Cylinder_Y.npy'), 'Airfoil'),
    (os.path.join(fno_dir, 'airfoil/naca/NACA_Cylinder_Q.npy'), 'Airfoil'),
    (os.path.join(fno_dir, 'pipe/Pipe_X.npy'), 'Pipe'),
    (os.path.join(fno_dir, 'pipe/Pipe_Y.npy'), 'Pipe'),
    (os.path.join(fno_dir, 'pipe/Pipe_Q.npy'), 'Pipe'),
]

errors = 0
for path, name in mat_files:
    if not os.path.isfile(path):
        print(f"ERROR  [{name}] 缺失: {path}"); errors += 1; continue
    try:
        d = sio.loadmat(path)
        keys = [k for k in d.keys() if not k.startswith('__')]
        shapes = {k: d[k].shape for k in keys if hasattr(d[k], 'shape')}
        print(f"OK     [{name}] {os.path.basename(path)} 键={keys} shape={shapes}")
    except Exception as e:
        print(f"ERROR  [{name}] loadmat 失败 {path}: {e}"); errors += 1

for path, name in npy_files:
    if not os.path.isfile(path):
        print(f"ERROR  [{name}] 缺失: {path}"); errors += 1; continue
    try:
        a = np.load(path, allow_pickle=True)
        print(f"OK     [{name}] {os.path.basename(path)} shape={a.shape} dtype={a.dtype}")
    except Exception as e:
        print(f"ERROR  [{name}] np.load 失败 {path}: {e}"); errors += 1

if errors:
    print(f"深度校验发现 {errors} 个问题")
    sys.exit(1)
print("标准基准深度校验通过")
PY
[ $? -ne 0 ] && FAIL=1

# ===== 3. 汽车数据（MLCFD）=====
log "===== 3. 汽车数据（MLCFD）====="
if [ -d "$DATA/mlcfd/training_data" ]; then
  n_press=$(find "$DATA/mlcfd/training_data" -name 'quadpress_smpl.vtk' 2>/dev/null | wc -l)
  n_velo=$(find "$DATA/mlcfd/training_data" -name 'hexvelo_smpl.vtk' 2>/dev/null | wc -l)
  log "training_data: quadpress_smpl.vtk=$n_press, hexvelo_smpl.vtk=$n_velo（期望各 889）"
  if [ "$n_press" -ne 889 ] || [ "$n_velo" -ne 889 ]; then FAIL=1; fi
else
  log "MISSING   $DATA/mlcfd/training_data"; FAIL=1
fi

if [ -d "$DATA/mlcfd/preprocessed_data" ]; then
  n_npy=$(find "$DATA/mlcfd/preprocessed_data" -name 'x.npy' 2>/dev/null | wc -l)
  log "preprocessed_data: x.npy=$n_npy（期望 889；若为 0 则训练时需 --preprocessed 0）"
else
  log "MISSING   $DATA/mlcfd/preprocessed_data"; FAIL=1
fi

# ===== 4. 翼型设计（AirfRANS）=====
log "===== 4. 翼型设计（AirfRANS）====="
if [ -s "$DATA/naca/Dataset/manifest.json" ]; then
  log "OK        $DATA/naca/Dataset/manifest.json"
  python - "$DATA/naca/Dataset/manifest.json" <<'PY'
import sys, json
m = json.load(open(sys.argv[1]))
keys = list(m.keys())
print(f"  manifest.json 顶层键: {keys[:8]}{'...' if len(keys) > 8 else ''}（共 {len(keys)} 个）")
PY
else
  log "MISSING   $DATA/naca/Dataset/manifest.json"; FAIL=1
fi

# ===== 汇总 =====
log "===== 汇总 ====="
if [ "$FAIL" -eq 0 ]; then
  log "✔ 全部数据集就位且可加载，可直接用于训练和评估。"
else
  log "✘ 存在缺失或不可加载的数据，请根据上方输出排查。"
fi
exit $FAIL
