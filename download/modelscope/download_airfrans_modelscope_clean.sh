#!/usr/bin/env bash
# Download a clean AirfRANS dataset from ModelScope for Transolver.
#
# The Transolver Airfoil-Design-AirfRANS code expects --my_path to point
# directly at a Dataset directory with this layout:
#
#   Dataset/manifest.json
#   Dataset/<case>/<case>_internal.vtu
#   Dataset/<case>/<case>_aerofoil.vtp
#   Dataset/<case>/<case>_freestream.vtp
#
# This script downloads OneScience/airfrans into an isolated directory, checks
# the split manifest and case files, then exposes a stable clean Dataset path.
# It retries failed downloads after 10 seconds and refuses to overwrite an
# existing non-symlink Dataset path.

set -Eeuo pipefail

DATASET_ID="${DATASET_ID:-OneScience/airfrans}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/Transolver_re}"

DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${REMOTE_DATA_ROOT}/modelscope_downloads/AirfRANS}"
FINAL_ROOT="${FINAL_ROOT:-${REMOTE_DATA_ROOT}/airfrans_modelscope_clean}"

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

  AUTO_INSTALL_MODELSCOPE=1 bash download_airfrans_modelscope_clean.sh
EOF
    exit 127
  fi
}

VALIDATED_DATASET_DIR=""

download_and_validate_with_retry() {
  local attempt=1
  local status=0
  local dataset_dir=""

  while true; do
    log "download+validation attempt ${attempt}: modelscope download --dataset ${DATASET_ID} --local_dir ${DOWNLOAD_ROOT}"

    if modelscope download --dataset "$DATASET_ID" --local_dir "$DOWNLOAD_ROOT"; then
      if dataset_dir="$(find_dataset_dir)" && validate_airfrans_dataset "$dataset_dir"; then
        VALIDATED_DATASET_DIR="$dataset_dir"
        log "download and validation completed successfully"
        return 0
      fi
      status=$?
      log "download command finished, but dataset validation failed with exit code ${status}"
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

find_dataset_dir() {
  local candidate
  for candidate in \
    "${DOWNLOAD_ROOT}/data/Dataset" \
    "${DOWNLOAD_ROOT}/Dataset" \
    "${DOWNLOAD_ROOT}/data/airfrans/Dataset" \
    "${DOWNLOAD_ROOT}/airfrans/Dataset"; do
    if [[ -f "${candidate}/manifest.json" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

validate_airfrans_dataset() {
  local dataset_dir="$1"
  python - "$dataset_dir" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
required_splits = {
    "full_train": 800,
    "full_test": 200,
    "scarce_train": 200,
    "reynolds_train": 504,
    "reynolds_test": 496,
    "aoa_train": 804,
    "aoa_test": 196,
}
required_suffixes = ("_internal.vtu", "_aerofoil.vtp", "_freestream.vtp")

errors = []

def valid_data_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    # A failed Git-LFS-style download may leave a tiny pointer file. Transolver
    # needs the actual XML VTK payload, not an LFS pointer.
    with path.open("rb") as handle:
        prefix = handle.read(128)
    return not prefix.startswith(b"version https://git-lfs.github.com/spec/v1")

if not manifest_path.is_file():
    print(f"ERROR: missing manifest.json: {manifest_path}")
    sys.exit(2)

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)

for split, expected_count in required_splits.items():
    values = manifest.get(split)
    if not isinstance(values, list):
        errors.append(f"manifest split {split} is missing or not a list")
        continue
    print(f"{split}: {len(values)}/{expected_count}")
    if len(values) != expected_count:
        errors.append(f"manifest split {split} has {len(values)} cases; expected {expected_count}")

cases = sorted({case for split in required_splits if isinstance(manifest.get(split), list) for case in manifest[split]})
print(f"unique cases referenced by required splits: {len(cases)}/1000")
if len(cases) != 1000:
    errors.append(f"required manifest splits reference {len(cases)} unique cases; expected 1000")

valid_cases = 0
for case in cases:
    case_dir = root / case
    if not case_dir.is_dir():
        errors.append(f"missing case directory: {case_dir}")
        continue
    missing = []
    for suffix in required_suffixes:
        path = case_dir / f"{case}{suffix}"
        if not valid_data_file(path):
            missing.append(path.name)
    if missing:
        errors.append(f"{case} missing/nonempty check failed: {', '.join(missing)}")
    else:
        valid_cases += 1

print(f"valid cases with all VTU/VTP files: {valid_cases}/1000")
if valid_cases != 1000:
    errors.append(f"valid case count is {valid_cases}; expected 1000")

if errors:
    print("\nDATASET CHECK FAILED")
    for message in errors[:80]:
        print(f"ERROR: {message}")
    if len(errors) > 80:
        print(f"... {len(errors) - 80} more errors suppressed")
    sys.exit(2)

print("DATASET CHECK PASSED: AirfRANS layout matches Transolver requirements")
PY
}

safe_symlink() {
  local source_path="$1"
  local link_path="$2"

  if [[ -e "$link_path" && ! -L "$link_path" ]]; then
    die "refusing to overwrite existing non-symlink path: ${link_path}"
  fi
  ln -sfn "$source_path" "$link_path"
}

main() {
  require_modelscope_cli

  mkdir -p "$DOWNLOAD_ROOT"
  mkdir -p "$FINAL_ROOT"

  log "dataset id: ${DATASET_ID}"
  log "download root: ${DOWNLOAD_ROOT}"
  log "final dataset layout: ${FINAL_ROOT}/Dataset"

  download_and_validate_with_retry

  local dataset_dir="$VALIDATED_DATASET_DIR"
  if [[ -z "$dataset_dir" ]]; then
    die "could not validate downloaded Dataset/manifest.json under ${DOWNLOAD_ROOT}"
  fi

  safe_symlink "$dataset_dir" "${FINAL_ROOT}/Dataset"


  cat > "${FINAL_ROOT}/paths.env" <<EOF
AIRFRANS_DATASET_DIR=${FINAL_ROOT}/Dataset
AIRFRANS_DOWNLOAD_ROOT=${DOWNLOAD_ROOT}
AIRFRANS_DATASET_ID=${DATASET_ID}
EOF

  if [[ -f "${REMOTE_PROJECT_ROOT}/Airfoil-Design-AirfRANS/main.py" ]]; then
    log "Transolver AirfRANS entry found at ${REMOTE_PROJECT_ROOT}/Airfoil-Design-AirfRANS/main.py"
  else
    log "warning: Transolver AirfRANS entry was not found at ${REMOTE_PROJECT_ROOT}/Airfoil-Design-AirfRANS/main.py"
  fi

  log "ready for Transolver AirfRANS training/evaluation"
  printf '\nUse this path:\n'
  printf '  --my_path %s\n' "${FINAL_ROOT}/Dataset"
  printf '\nA reusable env file was written to:\n  %s\n' "${FINAL_ROOT}/paths.env"
}

main "$@"
