#!/usr/bin/env bash
# Train and evaluate the AirfRANS design Transolver experiment on one GPU.
# The model hyperparameters come from AirfRANS/params.yaml and the original
# Transolver.sh script.  The paper reports 400 epochs for this experiment;
# the legacy repository YAML says 398, so the wrapper overrides that value at
# runtime without modifying the repository YAML.  The legacy main.py rewrites
# CUDA_VISIBLE_DEVICES; the launcher initializes CUDA first so visible index 0
# remains the requested physical GPU (GPU_ID, default 1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AIR_ROOT="$REPO_ROOT/Airfoil-Design-AirfRANS"

DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
GPU_ID="${GPU_ID:-1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-400}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
VISUALIZATION_INTERVAL="${VISUALIZATION_INTERVAL:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AIR_ROOT/output}"
AIRFRANS_DATA="${AIRFRANS_DATA:-$DATA_ROOT/naca/Dataset}"
if [[ ! -f "$AIRFRANS_DATA/manifest.json" ]]; then
  echo "ERROR: missing AirfRANS manifest: $AIRFRANS_DATA/manifest.json" >&2
  exit 1
fi

RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
EXPERIMENT_DIR="$OUTPUT_ROOT/${RUN_STAMP}_full_Transolver"
SUFFIX=0
while [[ -e "$EXPERIMENT_DIR" ]]; do
  SUFFIX=$((SUFFIX + 1))
  EXPERIMENT_DIR="$OUTPUT_ROOT/${RUN_STAMP}_full_Transolver_${SUFFIX}"
done
mkdir -p "$EXPERIMENT_DIR/logs"

echo "AirfRANS experiment: $EXPERIMENT_DIR"
echo "GPU: $GPU_ID"
echo "Dataset: $AIRFRANS_DATA"

run_airfrans_main() {
  CUDA_VISIBLE_DEVICES="$GPU_ID" python - "$@" <<'PY'
import os
import runpy
import sys
import torch
import yaml

# AirfRANS stores its epoch count in params.yaml and exposes no epoch CLI
# argument. Keep the source YAML unchanged while selecting the paper setting.
_safe_load = yaml.safe_load
def _paper_safe_load(stream):
    config = _safe_load(stream)
    epochs = os.environ.get('TRANSOLVER_AIRFRANS_EPOCHS')
    if epochs and isinstance(config, dict) and isinstance(config.get('Transolver'), dict):
        config['Transolver']['nb_epochs'] = int(epochs)
    return config

yaml.safe_load = _paper_safe_load

# Initialize the CUDA runtime before the legacy entrypoint assigns its own
# visibility mask.  This maps visible cuda:0 to the physical GPU selected by
# the wrapper without changing the repository training code.
if torch.cuda.is_available():
    torch.cuda.init()
sys.argv = ['main.py'] + sys.argv[1:]
runpy.run_path(os.path.join(os.environ['TRANSOLVER_AIR_ROOT'], 'main.py'),
               run_name='__main__')
PY
}

export TRANSOLVER_AIR_ROOT="$AIR_ROOT"
export TRANSOLVER_AIRFRANS_EPOCHS="$EPOCHS"
cd "$AIR_ROOT"
run_airfrans_main \
  --model Transolver \
  --nmodel 1 \
  --task full \
  --score 1 \
  --my_path "$AIRFRANS_DATA" \
  --gpu 0 \
  --output_root "$OUTPUT_ROOT" \
  --experiment_dir "$EXPERIMENT_DIR" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --visualization_interval "$VISUALIZATION_INTERVAL" \
  --seed "$SEED" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/train_console.log"

# Run the repository's full evaluation entrypoint as well.  It writes the
# detailed score arrays and plots into this same experiment/evaluation folder.
CUDA_VISIBLE_DEVICES="$GPU_ID" python main_evaluation.py \
  --my_path "$DATA_ROOT/naca" \
  --task full \
  --experiment_dir "$EXPERIMENT_DIR" \
  --model_path "$EXPERIMENT_DIR/ensemble_full.pth" \
  --seed "$SEED" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/evaluation_console.log"

echo "AirfRANS training and evaluation completed: $EXPERIMENT_DIR"
