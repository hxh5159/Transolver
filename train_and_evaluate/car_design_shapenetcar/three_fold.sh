#!/usr/bin/env bash
# =============================================================================
# Car-Design-ShapeNetCar 简化「3 折」训练 + 评估 + 取平均（快速验证全流程）
#
# 说明：沿用 9 折的完整逻辑（同样的数据切分、训练、评估、取平均），
#       只是只训 fold 0,1,2（1/3），以简代繁快速跑通，验证训练与评估链路正确。
# 论文的完整结果仍需用 nine_fold.sh 跑满 9 折。
#
# 用法： bash three_fold.sh
# 环境变量： REPO / DATA / CONDA_ENV / GPU
# =============================================================================
set -uo pipefail

# ---------- 路径配置 ----------
# 从脚本自身位置推导仓库根：脚本在 <repo>/train_and_evaluate/car_design_shapenetcar/ 下。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
CAR_DIR="$REPO/Car-Design-ShapeNetCar"
DATA_DIR="$DATA/mlcfd/training_data"
SAVE_DIR="$DATA/mlcfd/preprocessed_data"

# ---------- 参数 ----------
CONDA_ENV="${CONDA_ENV:-transolver}"
GPU="${GPU:-0}"
FOLDS=(0 1 2)   # 只训 3 折（9 折逻辑的 1/3）

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 0. 激活 conda ----------
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || log "警告：无法激活 $CONDA_ENV，使用当前环境"
fi

# ---------- 1. 前置检查 ----------
if [ ! -f "$CAR_DIR/main.py" ]; then echo "错误：找不到 $CAR_DIR/main.py"; exit 1; fi
if [ ! -d "$DATA_DIR" ]; then echo "错误：找不到 $DATA_DIR"; exit 1; fi
cd "$CAR_DIR"

# ---------- 2. 训练 ----------
log "开始训练 ${#FOLDS[@]} 折（每折约 200 epoch）"
for fold in "${FOLDS[@]}"; do
  log "==== 训练 fold $fold ===="
  python main.py \
    --cfd_model=Transolver \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --fold_id "$fold" \
    --gpu "$GPU" \
    --seed 0 \
    --experiment_dir "$CAR_DIR/output/fold_${fold}" \
    || { log "fold $fold 训练失败，中止"; exit 1; }
done

# ---------- 3. 评估 ----------
log "开始评估 ${#FOLDS[@]} 折"
for fold in "${FOLDS[@]}"; do
  log "==== 评估 fold $fold ===="
  python main_evaluation.py \
    --cfd_model=Transolver \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --fold_id "$fold" \
    --gpu "$GPU" \
    --experiment_dir "$CAR_DIR/output/fold_${fold}" \
    || { log "fold $fold 评估失败，中止"; exit 1; }
done

# ---------- 4. 取平均 ----------
RESULT_FILES=()
for fold in "${FOLDS[@]}"; do
  RESULT_FILES+=("$CAR_DIR/output/fold_${fold}/evaluation/evaluation_metrics.json")
done

log "汇总取平均"
python - "$CAR_DIR/output" "${RESULT_FILES[@]}" <<'PY'
import sys, os, json
import numpy as np

out_root = sys.argv[1]
files = sys.argv[2:]
missing = [f for f in files if not os.path.isfile(f)]
if missing:
    print('缺少评估结果文件：')
    for f in missing:
        print('  ', f)
    sys.exit(1)

rows = [json.load(open(f)) for f in files]
print('=' * 70)
print('逐折结果：')
for i, r in enumerate(rows):
    print(f"  fold {i}: "
          f"velo_relL2={r['volume_velocity_relative_l2_mean']:.4f}  "
          f"press_relL2={r['surface_pressure_relative_l2_mean']:.4f}  "
          f"Cd_mape={r['drag_coefficient']['mape']:.4f}  "
          f"rho={r['drag_coefficient']['spearman']:.4f}")

print('-' * 70)
print(f'平均（{len(rows)} 折）：')
summary = {'n_folds': len(rows)}
summary['volume_velocity_relative_l2_mean'] = float(np.mean([r['volume_velocity_relative_l2_mean'] for r in rows]))
summary['surface_pressure_relative_l2_mean'] = float(np.mean([r['surface_pressure_relative_l2_mean'] for r in rows]))
summary['drag_mape'] = float(np.mean([r['drag_coefficient']['mape'] for r in rows]))
summary['drag_spearman'] = float(np.mean([r['drag_coefficient']['spearman'] for r in rows]))
print(f"  体速度 相对 L2 均值       : {summary['volume_velocity_relative_l2_mean']:.6f}")
print(f"  表面压力 相对 L2 均值     : {summary['surface_pressure_relative_l2_mean']:.6f}")
print(f"  阻力系数 Cd 相对误差均值  : {summary['drag_mape']:.6f}")
print(f"  阻力系数 Spearman 均值    : {summary['drag_spearman']:.6f}")
print('=' * 70)

out = os.path.join(out_root, 'three_fold_summary.json')
json.dump(summary, open(out, 'w'), indent=2, ensure_ascii=False)
print('汇总已保存:', out)
PY

log "完成。"
