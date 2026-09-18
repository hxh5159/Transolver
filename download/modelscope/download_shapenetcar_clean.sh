#!/usr/bin/env bash
# Download a clean ShapeNetCar / MLCFD dataset from ModelScope.
#
# This script is designed for the remote Transolver_re layout used in this
# project. It downloads into an isolated directory and exposes a stable
# compatibility layout that can be passed directly to Car-Design-ShapeNetCar:
#
#   --data_dir <FINAL_ROOT>/training_data
#   --save_dir <FINAL_ROOT>/preprocessed_data
#
# It does not modify the Transolver source tree and it refuses to overwrite an
# existing non-symlink data directory.

set -Eeuo pipefail

DATASET_ID="${DATASET_ID:-OneScience/ShapeNetCar}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/Transolver_re}"

# Keep the ModelScope checkout separate from the training path. This avoids
# mixing a partial download with the existing data/mlcfd directory.
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${REMOTE_DATA_ROOT}/modelscope_downloads/ShapeNetCar}"
FINAL_ROOT="${FINAL_ROOT:-${REMOTE_DATA_ROOT}/mlcfd_modelscope_clean}"

RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-10}"
# 0 means retry forever. For a finite retry count, run with e.g. MAX_RETRIES=20.
MAX_RETRIES="${MAX_RETRIES:-0}"
AUTO_INSTALL_MODELSCOPE="${AUTO_INSTALL_MODELSCOPE:-0}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_modelscope_cli() {
  if command -v modelscope >/dev/null 2>&1; then
    return 0
  fi

  if [[ "$AUTO_INSTALL_MODELSCOPE" == "1" ]]; then
    log "modelscope CLI not found; installing modelscope==1.40.1"
    python -m pip install --upgrade "modelscope==1.40.1"
  fi

  if ! command -v modelscope >/dev/null 2>&1; then
    cat >&2 <<'EOF'
ERROR: modelscope CLI is not installed.

Install it in your active remote environment, then rerun this script:

  python -m pip install --upgrade "modelscope==1.40.1"

Or let this script install it automatically:

  AUTO_INSTALL_MODELSCOPE=1 bash download_shapenetcar_clean.sh
EOF
    exit 127
  fi
}

PREPARED_TRAINING_DIR=""
PREPARED_PREPROCESSED_DIR=""

download_and_prepare_with_retry() {
  local attempt=1
  local status=0
  local mlcfd_root=""

  while true; do
    log "download+validation attempt ${attempt}: modelscope download --dataset ${DATASET_ID} --local_dir ${DOWNLOAD_ROOT}"

    if modelscope download --dataset "$DATASET_ID" --local_dir "$DOWNLOAD_ROOT"; then
      if mlcfd_root="$(find_mlcfd_root)"; then
        if validate_training_data "${mlcfd_root}/training_data" && prepare_clean_views "$mlcfd_root" "$FINAL_ROOT"; then
          PREPARED_TRAINING_DIR="${FINAL_ROOT}/training_data"
          PREPARED_PREPROCESSED_DIR="${FINAL_ROOT}/preprocessed_data"
          log "download, validation, and clean-view preparation completed successfully"
          return 0
        fi
        status=$?
        log "download command finished, but validation or clean-view preparation failed with exit code ${status}"
      else
        status=$?
        log "download command finished, but could not find mlcfd_data/training_data under ${DOWNLOAD_ROOT}"
      fi
    else
      status=$?
      log "download interrupted or failed with exit code ${status}"
    fi

    if [[ "$MAX_RETRIES" != "0" && "$attempt" -ge "$MAX_RETRIES" ]]; then
      die "download/validation failed after ${attempt} attempts with exit code ${status}"
    fi

    log "retrying in ${RETRY_DELAY_SECONDS}s"
    sleep "$RETRY_DELAY_SECONDS"
    attempt=$((attempt + 1))
  done
}

find_mlcfd_root() {
  local candidate
  for candidate in \
    "${DOWNLOAD_ROOT}/data/mlcfd_data" \
    "${DOWNLOAD_ROOT}/mlcfd_data" \
    "${DOWNLOAD_ROOT}/data/mlcfd" \
    "${DOWNLOAD_ROOT}/mlcfd"; do
    if [[ -d "${candidate}/training_data" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

validate_training_data() {
  local training_dir="$1"
  python - "$training_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "param0": 100,
    "param1": 99,
    "param2": 97,
    "param3": 100,
    "param4": 100,
    "param5": 96,
    "param6": 100,
    "param7": 98,
    "param8": 99,
}
required = ("quadpress_smpl.vtk", "hexvelo_smpl.vtk")

errors = []
warnings = []

def valid_data_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    # A failed Git-LFS-style download may leave a tiny pointer file. Transolver
    # needs the actual VTK payload, not an LFS pointer.
    with path.open("rb") as handle:
        prefix = handle.read(128)
    return not prefix.startswith(b"version https://git-lfs.github.com/spec/v1")

valid_total = 0
for fold, expected_count in expected.items():
    fold_dir = root / fold
    if not fold_dir.is_dir():
        errors.append(f"missing fold directory: {fold_dir}")
        continue
    sample_dirs = sorted(p for p in fold_dir.iterdir() if p.is_dir())
    valid_dirs = []
    invalid_details = []
    for sample_dir in sample_dirs:
        missing = [name for name in required if not valid_data_file(sample_dir / name)]
        if missing:
            invalid_details.append(f"{sample_dir.relative_to(root)} missing/nonempty/LFS-pointer check failed: {', '.join(missing)}")
        else:
            valid_dirs.append(sample_dir)
    valid_total += len(valid_dirs)
    print(f"{fold}: valid {len(valid_dirs)}/{expected_count}, directories {len(sample_dirs)}")
    if len(valid_dirs) != expected_count:
        errors.append(f"{fold} has {len(valid_dirs)} valid samples; expected {expected_count}")
        errors.extend(invalid_details[:20])
    elif invalid_details:
        warnings.append(f"{fold} has {len(invalid_details)} extra invalid sample directories that will be ignored in the clean view")
        warnings.extend(invalid_details[:20])

if valid_total != sum(expected.values()):
    errors.append(f"valid total is {valid_total}; expected {sum(expected.values())}")

if errors:
    print("\nDATASET CHECK FAILED")
    for message in errors[:60]:
        print(f"ERROR: {message}")
    if len(errors) > 60:
        print(f"... {len(errors) - 60} more errors suppressed")
    sys.exit(2)

if warnings:
    print("\nDATASET CHECK WARNINGS")
    for message in warnings[:60]:
        print(f"WARNING: {message}")
    if len(warnings) > 60:
        print(f"... {len(warnings) - 60} more warnings suppressed")

print(f"DATASET CHECK PASSED: {valid_total} valid samples with both VTK files")
PY
}

validate_preprocessed_data() {
  local preprocessed_dir="$1"
  python - "$preprocessed_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "param0": 100,
    "param1": 99,
    "param2": 97,
    "param3": 100,
    "param4": 100,
    "param5": 96,
    "param6": 100,
    "param7": 98,
    "param8": 99,
}
required = ("x.npy", "y.npy", "pos.npy", "surf.npy", "edge_index.npy")

errors = []

def valid_data_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    with path.open("rb") as handle:
        prefix = handle.read(128)
    return not prefix.startswith(b"version https://git-lfs.github.com/spec/v1")

valid_total = 0
for fold, expected_count in expected.items():
    fold_dir = root / fold
    if not fold_dir.is_dir():
        errors.append(f"missing preprocessed fold directory: {fold_dir}")
        continue
    sample_dirs = sorted(p for p in fold_dir.iterdir() if p.is_dir())
    valid_dirs = []
    for sample_dir in sample_dirs:
        missing = [name for name in required if not valid_data_file(sample_dir / name)]
        if missing:
            errors.append(f"{sample_dir.relative_to(root)} missing/nonempty/LFS-pointer check failed: {', '.join(missing)}")
        else:
            valid_dirs.append(sample_dir)
    valid_total += len(valid_dirs)
    print(f"{fold}: valid preprocessed {len(valid_dirs)}/{expected_count}, directories {len(sample_dirs)}")
    if len(valid_dirs) != expected_count:
        errors.append(f"{fold} has {len(valid_dirs)} valid preprocessed samples; expected {expected_count}")

if valid_total != sum(expected.values()):
    errors.append(f"valid preprocessed total is {valid_total}; expected {sum(expected.values())}")

if errors:
    print("\nPREPROCESSED DATA CHECK FAILED")
    for message in errors[:80]:
        print(f"ERROR: {message}")
    if len(errors) > 80:
        print(f"... {len(errors) - 80} more errors suppressed")
    sys.exit(2)

print(f"PREPROCESSED DATA CHECK PASSED: {valid_total} valid samples")
PY
}

prepare_clean_views() {
  local mlcfd_root="$1"
  local final_root="$2"
  python - "$mlcfd_root" "$final_root" <<'PY'
from pathlib import Path
import os
import sys

source_root = Path(sys.argv[1]).resolve()
final_root = Path(sys.argv[2]).resolve()
source_training = source_root / "training_data"
source_preprocessed = source_root / "preprocessed_data"
target_training = final_root / "training_data"
target_preprocessed = final_root / "preprocessed_data"

expected = {
    "param0": 100,
    "param1": 99,
    "param2": 97,
    "param3": 100,
    "param4": 100,
    "param5": 96,
    "param6": 100,
    "param7": 98,
    "param8": 99,
}
required_raw = ("quadpress_smpl.vtk", "hexvelo_smpl.vtk")
required_preprocessed = ("x.npy", "y.npy", "pos.npy", "surf.npy", "edge_index.npy")

def valid_data_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    with path.open("rb") as handle:
        prefix = handle.read(128)
    return not prefix.startswith(b"version https://git-lfs.github.com/spec/v1")

def ensure_clean_root(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    if path.exists() and not path.is_dir():
        raise SystemExit(f"ERROR: clean view target exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)

def sync_symlink_dir(target_fold: Path, desired: dict[str, Path]) -> None:
    target_fold.mkdir(parents=True, exist_ok=True)
    for entry in target_fold.iterdir():
        if entry.name not in desired:
            if entry.is_symlink():
                entry.unlink()
            else:
                raise SystemExit(f"ERROR: unexpected non-symlink entry in clean view: {entry}")
    for name, source_path in desired.items():
        target = target_fold / name
        if target.is_symlink():
            if Path(os.path.realpath(target)) == source_path.resolve():
                continue
            target.unlink()
        elif target.exists():
            raise SystemExit(f"ERROR: refusing to overwrite non-symlink clean-view entry: {target}")
        os.symlink(source_path, target, target_is_directory=True)

ensure_clean_root(target_training)
ensure_clean_root(target_preprocessed)

ignored = []
valid_by_fold: dict[str, dict[str, Path]] = {}
preprocessed_by_fold: dict[str, dict[str, Path]] = {}

for fold, expected_count in expected.items():
    fold_dir = source_training / fold
    if not fold_dir.is_dir():
        raise SystemExit(f"ERROR: missing raw fold directory: {fold_dir}")
    valid_samples: dict[str, Path] = {}
    for sample_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
        if all(valid_data_file(sample_dir / filename) for filename in required_raw):
            valid_samples[sample_dir.name] = sample_dir.resolve()
        else:
            ignored.append(str(sample_dir.relative_to(source_training)))
    if len(valid_samples) != expected_count:
        raise SystemExit(f"ERROR: {fold} has {len(valid_samples)} valid raw samples; expected {expected_count}")
    valid_by_fold[fold] = valid_samples

    if source_preprocessed.is_dir():
        pre_fold = source_preprocessed / fold
        prepared_samples: dict[str, Path] = {}
        missing_preprocessed = []
        for sample_name in valid_samples:
            sample_pre = pre_fold / sample_name
            if sample_pre.is_dir() and all(valid_data_file(sample_pre / filename) for filename in required_preprocessed):
                prepared_samples[sample_name] = sample_pre.resolve()
            else:
                missing_preprocessed.append(f"{fold}/{sample_name}")
        if missing_preprocessed:
            print("ERROR: ModelScope preprocessed_data is incomplete for valid raw samples")
            for item in missing_preprocessed[:60]:
                print(f"ERROR: missing/incomplete preprocessed sample: {item}")
            if len(missing_preprocessed) > 60:
                print(f"... {len(missing_preprocessed) - 60} more missing samples suppressed")
            raise SystemExit(2)
        preprocessed_by_fold[fold] = prepared_samples

for fold, samples in valid_by_fold.items():
    sync_symlink_dir(target_training / fold, samples)

if preprocessed_by_fold:
    for fold, samples in preprocessed_by_fold.items():
        sync_symlink_dir(target_preprocessed / fold, samples)
else:
    print("WARNING: source preprocessed_data not found; clean preprocessed_data directory was created empty")

ignored_path = final_root / "ignored_invalid_raw_samples.txt"
ignored_path.write_text("\n".join(ignored) + ("\n" if ignored else ""), encoding="utf-8")

print(f"clean raw training view: {target_training}")
print(f"clean preprocessed view: {target_preprocessed}")
print(f"ignored invalid raw sample directories: {len(ignored)}")
PY
}

main() {
  require_modelscope_cli

  mkdir -p "$DOWNLOAD_ROOT"
  mkdir -p "$FINAL_ROOT"

  log "dataset id: ${DATASET_ID}"
  log "download root: ${DOWNLOAD_ROOT}"
  log "final training layout: ${FINAL_ROOT}"

  download_and_prepare_with_retry

  local training_dir="$PREPARED_TRAINING_DIR"
  local preprocessed_dir="$PREPARED_PREPROCESSED_DIR"
  if [[ -z "$training_dir" || -z "$preprocessed_dir" ]]; then
    die "could not prepare clean ShapeNetCar training/preprocessed views under ${FINAL_ROOT}"
  fi

  log "validating clean raw training view: ${training_dir}"
  validate_training_data "$training_dir"
  log "validating clean preprocessed view: ${preprocessed_dir}"
  validate_preprocessed_data "$preprocessed_dir"

  if [[ -f "${REMOTE_PROJECT_ROOT}/CODEX/check/check_shapenetcar_dataset.py" ]]; then
    log "running project checker in structure-only mode"
    python "${REMOTE_PROJECT_ROOT}/CODEX/check/check_shapenetcar_dataset.py" \
      --data-dir "$training_dir" \
      --skip-vtk-content
  else
    log "project checker not found at ${REMOTE_PROJECT_ROOT}/CODEX/check/check_shapenetcar_dataset.py; internal check already passed"
  fi

  cat > "${FINAL_ROOT}/paths.env" <<EOF
SHAPENETCAR_DATA_DIR=${training_dir}
SHAPENETCAR_SAVE_DIR=${preprocessed_dir}
SHAPENETCAR_DOWNLOAD_ROOT=${DOWNLOAD_ROOT}
SHAPENETCAR_DATASET_ID=${DATASET_ID}
EOF

  log "ready for Transolver Car training/evaluation"
  printf '\nUse these paths:\n'
  printf '  --data_dir %s\n' "${FINAL_ROOT}/training_data"
  printf '  --save_dir %s\n' "${FINAL_ROOT}/preprocessed_data"
  printf '\nA reusable env file was written to:\n  %s\n' "${FINAL_ROOT}/paths.env"
}

main "$@"
