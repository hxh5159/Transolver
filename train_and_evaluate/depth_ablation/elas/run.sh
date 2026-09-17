#!/usr/bin/env bash
# =============================================================================
# Elasticity 深度消融：依次训练 block=4,5,6,7 的 Transolver（单卡 GPU 0）
# 论文配置除 --n-layers 外固定；输出到 depth_ablation/output/elas/block_<N>/
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTH_ROOT="$(dirname "$SCRIPT_DIR")"
REPO="${REPO:-$(dirname "$(dirname "$DEPTH_ROOT")")}"
BENCH_DIR="$REPO/PDE-Solving-StandardBenchmark"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
OUTPUT_DIR="$DEPTH_ROOT/output"
DATASET="elas"

# 论文配置（Elasticity，除 n-layers 外固定；不规则网格模型）
MODEL=Transolver_Irregular_Mesh
N_HIDDEN=128; N_HEADS=8; LR=0.001; MAX_GRAD_NORM=0.1; BATCH=1
SLICE=64; UNIFIED=0; REF=8
DATA_PATH="$DATA/fno"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
BLOCKS=(4 5 6 7)

log() { echo "[$(date '+%F %T')] $*"; }

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-transolver}" 2>/dev/null || log "警告：无法激活 conda 环境"
fi

[ -f "$BENCH_DIR/exp_elas.py" ] || { echo "错误：找不到 $BENCH_DIR/exp_elas.py"; exit 1; }
cd "$BENCH_DIR"

run_block() {
  local N="$1"
  local EXP_DIR="$OUTPUT_DIR/$DATASET/seed_${SEED}/block_$N"
  log "===== $DATASET | block $N | GPU $GPU ====="
  python exp_elas.py \
    --model "$MODEL" --n-hidden "$N_HIDDEN" --n-heads "$N_HEADS" --n-layers "$N" \
    --lr "$LR" --max_grad_norm "$MAX_GRAD_NORM" --batch-size "$BATCH" \
    --slice_num "$SLICE" --unified_pos "$UNIFIED" --ref "$REF" \
    --gpu "$GPU" --data_path "$DATA_PATH" --save_name "block_${N}" \
    --experiment_dir "$EXP_DIR" --seed "$SEED" --eval 0 \
    || { log "block $N 训练失败"; return 1; }

  log "评估 block $N"
  python exp_elas.py \
    --model "$MODEL" --n-hidden "$N_HIDDEN" --n-heads "$N_HEADS" --n-layers "$N" \
    --lr "$LR" --max_grad_norm "$MAX_GRAD_NORM" --batch-size "$BATCH" \
    --slice_num "$SLICE" --unified_pos "$UNIFIED" --ref "$REF" \
    --gpu "$GPU" --data_path "$DATA_PATH" --save_name "block_${N}" \
    --experiment_dir "$EXP_DIR" --eval 1 \
    || { log "block $N 评估失败"; return 1; }

  python - "$EXP_DIR" "$DATASET" "$N" <<'PY'
import sys, os, json
d, ds, n = sys.argv[1], sys.argv[2], sys.argv[3]
ts = json.load(open(os.path.join(d, 'training_summary.json')))
line = f"[{ds} block_{n}] 参数量={ts.get('nb_parameters')}, 平均epoch耗时={ts.get('mean_epoch_seconds'):.2f}s"
evp = os.path.join(d, 'evaluation', 'evaluation_metrics.json')
if os.path.isfile(evp):
    ev = json.load(open(evp))
    for k in ('relative_l2', 'full_relative_l2', 'test_relative_l2'):
        if k in ev:
            line += f", {k}={ev[k]:.6f}"
            break
print(line)
PY
}

for N in "${BLOCKS[@]}"; do
  run_block "$N" || exit 1
done

log "完成。所有输出在 $OUTPUT_DIR/$DATASET/"
