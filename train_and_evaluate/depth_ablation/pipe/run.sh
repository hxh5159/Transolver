#!/usr/bin/env bash
# =============================================================================
# Pipe 深度消融：双卡并行训练 block=4,5,6,7 的 Transolver。
#   GPU 0 跑 block 4 和 block 7；GPU 1 跑 block 5 和 block 6。
# 论文配置除 --n-layers 外固定；输出到 depth_ablation/output/pipe/block_<N>/
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTH_ROOT="$(dirname "$SCRIPT_DIR")"
REPO="${REPO:-$(dirname "$(dirname "$DEPTH_ROOT")")}"
BENCH_DIR="$REPO/PDE-Solving-StandardBenchmark"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
OUTPUT_DIR="$DEPTH_ROOT/output"
DATASET="pipe"

# 论文配置（Pipe，除 n-layers 外固定；注意 mlp_ratio=2）
MODEL=Transolver_Structured_Mesh_2D
N_HIDDEN=128; N_HEADS=8; MLP_RATIO=2; LR=0.001; MAX_GRAD_NORM=0.1; BATCH=8
SLICE=64; UNIFIED=0; REF=8
DATA_PATH="$DATA/fno/pipe"
# (block, gpu) 对：GPU0 跑 4、7；GPU1 跑 5、6
TASKS=("4 0" "5 1" "7 0" "6 1")
SEED="${SEED:-0}"

log() { echo "[$(date '+%F %T')] $*"; }

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-transolver}" 2>/dev/null || log "警告：无法激活 conda 环境"
fi

[ -f "$BENCH_DIR/exp_pipe.py" ] || { echo "错误：找不到 $BENCH_DIR/exp_pipe.py"; exit 1; }
cd "$BENCH_DIR"

run_block() {
  local N="$1" gpu="$2"
  local EXP_DIR="$OUTPUT_DIR/$DATASET/seed_${SEED}/block_$N"
  log "===== $DATASET | block $N | GPU $gpu ====="
  python exp_pipe.py \
    --model "$MODEL" --n-hidden "$N_HIDDEN" --n-heads "$N_HEADS" --n-layers "$N" \
    --mlp_ratio "$MLP_RATIO" --lr "$LR" --max_grad_norm "$MAX_GRAD_NORM" --batch-size "$BATCH" \
    --slice_num "$SLICE" --unified_pos "$UNIFIED" --ref "$REF" \
    --gpu "$gpu" --data_path "$DATA_PATH" --save_name "block_${N}" \
    --experiment_dir "$EXP_DIR" --seed "$SEED" --eval 0 \
    || { log "block $N 训练失败"; return 1; }

  log "评估 block $N"
  python exp_pipe.py \
    --model "$MODEL" --n-hidden "$N_HIDDEN" --n-heads "$N_HEADS" --n-layers "$N" \
    --mlp_ratio "$MLP_RATIO" --lr "$LR" --max_grad_norm "$MAX_GRAD_NORM" --batch-size "$BATCH" \
    --slice_num "$SLICE" --unified_pos "$UNIFIED" --ref "$REF" \
    --gpu "$gpu" --data_path "$DATA_PATH" --save_name "block_${N}" \
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

# 双卡并行：每批 2 个任务（GPU0 一个、GPU1 一个）
run_batch() {
  local idx1="$1" idx2="$2"
  local n1 g1 n2 g2
  n1="${TASKS[$idx1]%% *}"; g1="${TASKS[$idx1]##* }"
  n2="${TASKS[$idx2]%% *}"; g2="${TASKS[$idx2]##* }"
  run_block "$n1" "$g1" & P1=$!
  run_block "$n2" "$g2" & P2=$!
  wait "$P1"; R1=$?
  wait "$P2"; R2=$?
  [ "$R1" -eq 0 ] && [ "$R2" -eq 0 ]
}

# 第 1 批：block4(GPU0) + block5(GPU1)；第 2 批：block7(GPU0) + block6(GPU1)
run_batch 0 1 || { log "第 1 批失败"; exit 1; }
run_batch 2 3 || { log "第 2 批失败"; exit 1; }

log "完成。所有输出在 $OUTPUT_DIR/$DATASET/"
