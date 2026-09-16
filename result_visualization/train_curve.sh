#!/usr/bin/env bash
# =============================================================================
# 汇总各数据集的训练损失曲线，画到同一张图。
#
# 从每个数据集的实验目录里读 logs/training_history.json，提取"训练损失"曲线，
# 用 matplotlib 画在一张图上（log 轴）。每个数据集若有多个输出文件夹，只取一个
# （优先用默认实验名，找不到则取 output/ 下最近修改的那一个）。
#
# 用法： bash train_curve.sh
# 环境变量： REPO（仓库根）
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

python - "$REPO" "$SCRIPT_DIR" <<'PY'
import sys, os, json, glob
import matplotlib
matplotlib.use('Agg')  # 无显示环境也能出图
import matplotlib.pyplot as plt

repo = sys.argv[1]
out_dir = sys.argv[2]
bench = os.path.join(repo, 'PDE-Solving-StandardBenchmark')

# 每个数据集：label + 默认 history 路径 + 搜索根（找不到默认时在搜索根下找最近的）
datasets = [
    ('Darcy',         os.path.join(bench, 'output/darcy_Transolver/logs/training_history.json'), os.path.join(bench, 'output')),
    ('Navier-Stokes', os.path.join(bench, 'output/ns_Transolver/logs/training_history.json'),      os.path.join(bench, 'output')),
    ('Plasticity',    os.path.join(bench, 'output/plas_Transolver/logs/training_history.json'),     os.path.join(bench, 'output')),
    ('Elasticity',    os.path.join(bench, 'output/elas_Transolver/logs/training_history.json'),     os.path.join(bench, 'output')),
    ('Airfoil',       os.path.join(bench, 'output/airfoil_Transolver/logs/training_history.json'),  os.path.join(bench, 'output')),
    ('Pipe',          os.path.join(bench, 'output/pipe_Transolver/logs/training_history.json'),     os.path.join(bench, 'output')),
    ('ShapeNet-Car',  os.path.join(repo, 'Car-Design-ShapeNetCar/output/fold_0/logs/training_history.json'),
                      os.path.join(repo, 'Car-Design-ShapeNetCar/output')),
]

# 训练损失 key 优先级（覆盖 6 个标准基准 + 汽车）
TRAIN_KEYS = ['train_loss', 'train_relative_l2', 'train_full_loss', 'train_total', 'train_step_loss']

def find_history(default_path, search_root):
    if os.path.isfile(default_path):
        return default_path
    # 找不到默认，取搜索根下最近修改的 training_history.json
    candidates = glob.glob(os.path.join(search_root, '**', 'training_history.json'), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def pick_train(history):
    if not history:
        return None, None
    first = history[0]
    for k in TRAIN_KEYS:
        if k in first:
            return k, [r.get(k) for r in history]
    return None, None

plt.figure(figsize=(12, 7))
found = 0
for label, default_path, search_root in datasets:
    path = find_history(default_path, search_root)
    if path is None:
        print(f'[跳过] {label}: 未找到 training_history.json')
        continue
    try:
        history = json.load(open(path))
    except Exception as e:
        print(f'[跳过] {label}: 读取失败 {e}')
        continue
    key, loss = pick_train(history)
    if key is None or loss is None:
        print(f'[跳过] {label}: 无训练损失字段（{path}）')
        continue
    epochs = [r.get('epoch', i + 1) for i, r in enumerate(history)]
    xs, ys = [], []
    for e, v in zip(epochs, loss):
        if v is not None:
            xs.append(e); ys.append(v)
    if not ys:
        print(f'[跳过] {label}: 训练损失全为空')
        continue
    plt.plot(xs, ys, label=f'{label} ({key})')
    found += 1
    print(f'[绘制] {label}: {key}, {len(ys)} epochs, 来自 {path}')

if found == 0:
    print('未找到任何训练历史，请先运行训练。')
    sys.exit(1)

plt.xlabel('Epoch')
plt.ylabel('Training loss (log scale)')
plt.yscale('log')
plt.legend(fontsize=9)
plt.grid(True, which='both', alpha=0.3)
plt.title('Training loss curves across datasets')
plt.tight_layout()
out_png = os.path.join(out_dir, 'train_curves.png')
plt.savefig(out_png, dpi=150)
print('已保存:', out_png)
PY
