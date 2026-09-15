#!/usr/bin/env bash
# =============================================================================
# 检查 DPOT-plus 数据目录里的 NS 数据，找出本仓库需要的 V1e-5 文件并归位到目标路径。
#
# 本仓库 NS 任务（exp_ns.py）需要：NavierStokes_V1e-5_N1200_T20.mat
#   - 粘度 1e-5（注意：1e-3 / 1e-4 是另一个 benchmark，不能用）
#   - u 键，shape 需满足 (N>=1200, 64, 64, T>=20)
#
# 用法（远端执行）： bash check_ns.sh
# 环境变量： SOURCE_DIR / DATA
# =============================================================================
set -uo pipefail

SOURCE_DIR="${SOURCE_DIR:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/pdefoundationmodel/DPOT-plus/data/raw/fno}"
DATA="${DATA:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
TARGET_DIR="$DATA/fno"
REQUIRED_MAT="NavierStokes_V1e-5_N1200_T20.mat"
REQUIRED_ZIP="NavierStokes_V1e-5_N1200_T20.zip"
DST="$TARGET_DIR/NavierStokes_V1e-5_N1200_T20/$REQUIRED_MAT"

log() { echo "[$(date '+%F %T')] $*"; }

if [ ! -d "$SOURCE_DIR" ]; then
  echo "错误：源目录不存在：$SOURCE_DIR"
  exit 1
fi

# ---------- 1. 列出所有 NavierStokes 文件，标注哪些是需要的 ----------
log "=== 源目录里所有 NavierStokes 相关文件 ==="
find "$SOURCE_DIR" -maxdepth 4 \( -iname '*navierstokes*' -o -iname 'ns_*' \) -type f 2>/dev/null | while read -r f; do
  base="$(basename "$f")"
  if [[ "$base" == *"V1e-5_N1200_T20"* ]]; then
    echo "  [需要]  $f"
  else
    echo "  [跳过]  $f  （粘度/分辨率不符，非本仓库基准）"
  fi
done

# ---------- 2. 定位 V1e-5 文件（.mat 或 .zip） ----------
log "=== 定位 $REQUIRED_MAT（或 $REQUIRED_ZIP）==="
MAT_SRC=$(find "$SOURCE_DIR" -type f -name "$REQUIRED_MAT" 2>/dev/null | head -n1)
ZIP_SRC=$(find "$SOURCE_DIR" -type f -name "$REQUIRED_ZIP" 2>/dev/null | head -n1)

mkdir -p "$(dirname "$DST")"

if [ -n "$MAT_SRC" ]; then
  log "找到 .mat：$MAT_SRC"
  if [ -s "$DST" ]; then
    log "目标已存在，跳过复制"
  else
    cp -v "$MAT_SRC" "$DST"
  fi
elif [ -n "$ZIP_SRC" ]; then
  log "找到 zip：$ZIP_SRC，解压到目标目录..."
  unzip -o "$ZIP_SRC" -d "$(dirname "$DST")" >/dev/null
  FOUND=$(find "$(dirname "$DST")" -type f -name "$REQUIRED_MAT" 2>/dev/null | head -n1)
  if [ -n "$FOUND" ] && [ "$FOUND" != "$DST" ]; then
    mv "$FOUND" "$DST"
    log "已归位：$FOUND -> $DST"
  fi
else
  echo "错误：源目录里没有 $REQUIRED_MAT，也没有 $REQUIRED_ZIP"
  exit 1
fi

# ---------- 3. 校验 u 键与 shape ----------
if [ ! -s "$DST" ]; then
  echo "错误：目标文件缺失/为空：$DST"
  exit 1
fi
log "目标文件：$DST（$(du -h "$DST" | cut -f1)）"

python - "$DST" <<'PY'
import sys
import scipy.io as sio

path = sys.argv[1]
data = sio.loadmat(path)
keys = [k for k in data.keys() if not k.startswith('__')]
print("matlab 键：", keys)

if 'u' not in data:
    print("错误：缺少 'u' 键，不能用于 exp_ns.py")
    sys.exit(1)

u = data['u']
print("u.shape =", u.shape)

ok = True
n, s1, s2, t = (u.shape[0], u.shape[1], u.shape[2], u.shape[3]) if u.ndim == 4 else (0, 0, 0, 0)
if u.ndim != 4:
    print("错误：u 应为 4 维 (N, H, W, T)，实际", u.ndim, "维")
    ok = False
else:
    if n < 1200:
        print(f"警告：样本数 N={n} < 1200（exp_ns.py 需前1000+后200）"); ok = False
    if s1 != 64 or s2 != 64:
        print(f"警告：空间分辨率 {s1}x{s2} != 64x64"); ok = False
    if t < 20:
        print(f"警告：时间步 T={t} < 20（需输入10+输出10）"); ok = False

if ok:
    print("✅ 校验通过：该文件可直接用于 exp_ns.py（--data_path 指向",
          path.rsplit('/', 2)[0] + "/", "）")
else:
    print("❌ 该文件与 exp_ns.py 的期望不完全一致，请根据上面警告判断")
PY
