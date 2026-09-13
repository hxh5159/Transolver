#!/usr/bin/env bash
# =============================================================================
# Car-Design-ShapeNetCar 双卡并行「4 折」训练 + 评估 + 取平均（快速验证）
#
# 概念与 3 折脚本一致：沿用 9 折的完整逻辑（同样的数据切分/训练/评估/取平均），
# 只跑 fold 0,1,2,3 这 4 折（不到 9 折的一半）快速跑通全流程。
# 双卡并行：每张卡同时跑一折，4 折分成 2 批（每批 2 折，GPU0/GPU1 各一折）。
#
# 用法： bash four_fold_2gpu.sh
# 环境变量： REPO / DATA / CONDA_ENV / GPUS（如 GPUS="0 1"）/ FOLDS（如 "0 1 2 3"）
# =============================================================================
set -uo pipefail

# ---------- 路径配置 ----------
REPO="${REPO:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/Transolver}"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
CAR_DIR="$REPO/Car-Design-ShapeNetCar"
DATA_DIR="$DATA/mlcfd/training_data"
SAVE_DIR="$DATA/mlcfd/preprocessed_data"

# ---------- 参数 ----------
CONDA_ENV="${CONDA_ENV:-transolver}"
GPUS=(${GPUS:-0 1})        # 两张卡
FOLDS=(${FOLDS:-0 1 2 3})  # 4 折

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
if [ "${#GPUS[@]}" -lt 1 ]; then echo "错误：GPUS 不能为空"; exit 1; fi
cd "$CAR_DIR"

# ---------- 2. 并行训练/评估（每批 NGPU 个 job，一卡一个） ----------
run_parallel() {
  local phase="$1"   # train / eval
  local n="${#FOLDS[@]}"
  local ngpu="${#GPUS[@]}"
  local i j fold gpu p ok
  for ((i = 0; i < n; i += ngpu)); do
    local pids=()
    for ((j = 0; j < ngpu && i + j < n; j++)); do
      fold="${FOLDS[$((i + j))]}"
      gpu="${GPUS[$j]}"
      if [ "$phase" = "train" ]; then
        log "训练 fold $fold (GPU $gpu)"
        python main.py \
          --cfd_model=Transolver \
          --data_dir "$DATA_DIR" \
          --save_dir "$SAVE_DIR" \
          --fold_id "$fold" \
          --gpu "$gpu" \
          --seed 0 \
          --experiment_dir "$CAR_DIR/output/fold_${fold}" &
      else
        log "评估 fold $fold (GPU $gpu)"
        python main_evaluation.py \
          --cfd_model=Transolver \
          --data_dir "$DATA_DIR" \
          --save_dir "$SAVE_DIR" \
          --fold_id "$fold" \
          --gpu "$gpu" \
          --experiment_dir "$CAR_DIR/output/fold_${fold}" &
      fi
      pids+=("$!")
    done
    ok=1
    for p in "${pids[@]}"; do
      wait "$p" || ok=0
    done
    if [ "$ok" -eq 0 ]; then
      log "本批有 job 失败，中止"
      return 1
    fi
  done
  return 0
}

log "==== 并行训练 ${#FOLDS[@]} 折（${#GPUS[@]} 卡）===="
run_parallel train || exit 1

log "==== 并行评估 ${#FOLDS[@]} 折（${#GPUS[@]} 卡）===="
run_parallel eval || exit 1

# ---------- 3. 取平均 ----------
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

out = os.path.join(out_root, 'four_fold_summary.json')
json.dump(summary, open(out, 'w'), indent=2, ensure_ascii=False)
print('汇总已保存:', out)
PY

log "完成。"
