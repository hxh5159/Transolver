#!/usr/bin/env bash
# =============================================================================
# 汽车设计(MLCFD/ShapeNetCar) 论文评测：fold 0-8 逐折评估 + 取平均
#
# 论文方式：9 折交叉验证，每折用 param{fold_id} 作为测试集，最终指标取 9 折平均。
#
# 用法：
#   bash evaluate_mlcfd.sh
# 可选环境变量：
#   CAR_DIR=.../Car-Design-ShapeNetCar   代码与 checkpoint 所在目录（默认 Transolver_sceen）
#   DATA=.../data                        数据根目录
#   GPU=1                                用哪张卡
#   START_FOLD=0  END_FOLD=8             评估折范围
# =============================================================================
set -uo pipefail

# ---------- 路径配置 ----------
CAR_DIR="${CAR_DIR:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/Transolver_sceen/Car-Design-ShapeNetCar}"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
DATA_DIR="$DATA/mlcfd/training_data"
SAVE_DIR="$DATA/mlcfd/preprocessed_data"

# ---------- 评估参数 ----------
CONDA_ENV="${CONDA_ENV:-transolver}"
GPU="${GPU:-1}"
NB_EPOCHS="${NB_EPOCHS:-200}"       # 对应 checkpoint 目录里的 200_0.5
WEIGHT="${WEIGHT:-0.5}"
CKPT_EPOCH="${CKPT_EPOCH:-0200}"    # 对应 checkpoint_epoch_0200.pth
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-8}"
RESULTS_DIR="${RESULTS_DIR:-$CAR_DIR/evaluation_results}"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 1. 激活 conda 环境 ----------
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || log "警告：无法激活 $CONDA_ENV，使用当前环境"
fi

# ---------- 2. 前置检查 ----------
if [ ! -f "$CAR_DIR/evaluate_checkpoint.py" ]; then
  echo "错误：找不到 $CAR_DIR/evaluate_checkpoint.py"; exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "错误：找不到数据目录 $DATA_DIR"; exit 1
fi

cd "$CAR_DIR"
mkdir -p "$RESULTS_DIR"
log "结果目录：$RESULTS_DIR"

# ---------- 3. 逐折评估 ----------
for fold in $(seq "$START_FOLD" "$END_FOLD"); do
  ckpt="$CAR_DIR/ckpt/Transolver/fold_${fold}/${NB_EPOCHS}_${WEIGHT}/checkpoint_epoch_${CKPT_EPOCH}.pth"
  out_json="$RESULTS_DIR/fold_${fold}.json"

  if [ ! -f "$ckpt" ]; then
    log "[fold $fold] 缺少 checkpoint，跳过：$ckpt"
    continue
  fi

  log "[fold $fold] 评估中（GPU $GPU）..."
  python evaluate_checkpoint.py \
    --checkpoint "$ckpt" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --fold_id "$fold" \
    --gpu "$GPU" \
    --out_json "$out_json" \
    || log "[fold $fold] 评估失败，继续下一折"
done

# ---------- 4. 汇总取平均 ----------
log "汇总各折结果..."
python - "$RESULTS_DIR" <<'PY'
import sys, os, json, glob
import numpy as np

results_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(results_dir, 'fold_*.json')))
if not files:
    print(f'未找到任何 fold_*.json 结果（目录 {results_dir}）')
    sys.exit(1)

scalar_keys = [
    ('volume_velocity_relative_l2',       '体速度 相对 L2'),
    ('surface_pressure_relative_l2',      '表面压力 相对 L2'),
    ('surface_pressure_rmse_pa',          '表面压力 RMSE (Pa)'),
    ('drag_coefficient_relative_error_mean', '阻力系数 Cd 相对误差'),
    ('drag_coefficient_spearman',         '阻力系数 Spearman (rho)'),
]

print('=' * 70)
print('逐折结果：')
rows = []
for f in files:
    d = json.load(open(f))
    rows.append(d)
    print(f"  fold {d['fold_id']}: "
          f"velo_relL2={d['volume_velocity_relative_l2']:.4f}  "
          f"press_relL2={d['surface_pressure_relative_l2']:.4f}  "
          f"Cd_relerr={d['drag_coefficient_relative_error_mean']:.4f}  "
          f"rho={d['drag_coefficient_spearman']:.4f}")

print('-' * 70)
print(f'平均（{len(rows)} 折）：')
summary = {'n_folds': len(rows)}
for key, label in scalar_keys:
    mean = float(np.mean([r[key] for r in rows]))
    summary[key] = mean
    print(f"  {label:28s}: {mean:.6f}")

# 速度 RMSE 是 3 分量(u/v/w)，逐分量平均
velo_rmse = np.array([r['volume_velocity_rmse_mps'] for r in rows])
mean_velo = velo_rmse.mean(axis=0)
summary['volume_velocity_rmse_mps'] = [float(x) for x in mean_velo]
print(f"  {'体速度 RMSE (m/s) u/v/w':28s}: {np.round(mean_velo, 6)}")
print('=' * 70)

out = os.path.join(results_dir, 'summary_mean.json')
json.dump(summary, open(out, 'w'), indent=2, ensure_ascii=False)
print('汇总已保存:', out)
PY

log "评估完成。"
