#!/usr/bin/env python
"""从 checkpoint_epoch_XXXX.pth 恢复模型，在留出折(测试集)上评估，输出论文指标。

用法（在 Car-Design-ShapeNetCar 目录下运行）：
  python evaluate_checkpoint.py --checkpoint <ckpt路径> \
      --data_dir <training_data路径> --save_dir <preprocessed_data路径> --fold_id 0

说明：
  - 汽车任务是 9 折交叉验证，没有独立测试集；fold_id 对应的被留出折(param{fold_id})
    即该折的"测试集"。
  - 本脚本直接从 checkpoint 取 coef_norm，只加载被留出折的数据，避免重载全部训练数据。
  - 输出论文中的指标：表面压力/体速度相对 L2、RMSE、阻力系数(Cd)相对误差与 Spearman 相关系数。
"""
import os
import argparse
import json
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import scipy as sc

from utils.drag_coefficient import cal_coefficient
from dataset.load_dataset import get_samples
from dataset.dataset import get_datalist, GraphDataset
from models.Transolver import Model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='checkpoint_epoch_0200.pth 的路径')
parser.add_argument('--data_dir', default='/data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir', default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--out_json', default=None, help='结果 JSON 输出路径（默认保存到 checkpoint 同目录）')
parser.add_argument('--experiment_dir', default=None,
                    help='实验目录；评估结果默认写入该目录下的 evaluation/')
parser.add_argument('--evaluation_dir', default=None,
                    help='显式指定评估结果目录')
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# ---------- 1. 加载 checkpoint ----------
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
print('checkpoint epoch:', ckpt.get('epoch'))
coef_norm = tuple(torch.as_tensor(x) for x in ckpt['coef_norm'])

# ---------- 2. 重建模型（与 main.py 完全一致的架构） ----------
model = Model(n_hidden=256, n_layers=8, space_dim=7, fun_dim=0,
              n_head=8, mlp_ratio=2, out_dim=4, slice_num=32, unified_pos=0)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device).eval()

# ---------- 3. 只加载被留出的折作为测试集 ----------
samples = get_samples(args.data_dir)
vallst = samples[args.fold_id]
val_data = get_datalist(args.data_dir, vallst, coef_norm=coef_norm,
                        savedir=args.save_dir, preprocessed=True)
val_ds = GraphDataset(val_data, use_cfd_mesh=False, r=args.r)
test_loader = DataLoader(val_ds, batch_size=1)

# ---------- 4. 评估 ----------
criterion = nn.MSELoss(reduction='none')
l2errs_press, l2errs_velo = [], []
mses_press, mses_velo = [], []
gt_coef_list, pred_coef_list = [], []
coef_error = 0.0
mean_out = coef_norm[2].to(device)
std_out = coef_norm[3].to(device)

with torch.no_grad():
    for i, (cfd_data, geom) in enumerate(test_loader):
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y

        out_denorm = out * std_out + mean_out
        y_denorm = targets * std_out + mean_out

        pred_press = out_denorm[cfd_data.surf, -1]
        gt_press = y_denorm[cfd_data.surf, -1]
        pred_surf_velo = out_denorm[cfd_data.surf, :-1]
        gt_surf_velo = y_denorm[cfd_data.surf, :-1]
        pred_velo = out_denorm[~cfd_data.surf, :-1]
        gt_velo = y_denorm[~cfd_data.surf, :-1]

        sample_path = vallst[i]
        pred_coef = cal_coefficient(sample_path, pred_press[:, None].detach().cpu().numpy(),
                                    pred_surf_velo.detach().cpu().numpy(), root=args.data_dir)
        gt_coef = cal_coefficient(sample_path, gt_press[:, None].detach().cpu().numpy(),
                                  gt_surf_velo.detach().cpu().numpy(), root=args.data_dir)
        gt_coef_list.append(gt_coef)
        pred_coef_list.append(pred_coef)
        coef_error += abs(pred_coef - gt_coef) / max(abs(gt_coef), 1e-12)

        l2errs_press.append(float((torch.norm(pred_press - gt_press)
                                   / torch.clamp(torch.norm(gt_press), min=1e-12)).cpu()))
        l2errs_velo.append(float((torch.norm(pred_velo - gt_velo)
                                  / torch.clamp(torch.norm(gt_velo), min=1e-12)).cpu()))
        mses_press.append(float(criterion(out[cfd_data.surf, -1],
                                          targets[cfd_data.surf, -1]).mean().cpu()))
        mses_velo.append(criterion(out[~cfd_data.surf, :-1],
                                   targets[~cfd_data.surf, :-1]).mean(dim=0).cpu().numpy())

n = len(gt_coef_list)
gt_coef_list = np.array(gt_coef_list)
pred_coef_list = np.array(pred_coef_list)
spear = sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]
l2err_press = np.mean(l2errs_press)
l2err_velo = np.mean(l2errs_velo)
rmse_press = np.sqrt(np.mean(mses_press)) * coef_norm[3][-1].item()
rmse_velo = np.sqrt(np.mean(mses_velo, axis=0)) * coef_norm[3][:-1].cpu().numpy()

print('=' * 62)
print(f'fold {args.fold_id} | 测试样本数 {n}')
print(f'表面压力 相对 L2 误差 (pressure rel L2) : {l2err_press:.6f}')
print(f'体速度   相对 L2 误差 (velocity rel L2): {l2err_velo:.6f}')
print(f'表面压力 RMSE (Pa)                     : {rmse_press:.6f}')
print(f'体速度   RMSE (m/s) u/v/w              : {np.round(rmse_velo, 6)}')
print(f'阻力系数 平均相对误差 (Cd rel err)      : {coef_error / n:.6f}')
print(f'阻力系数 Spearman 相关系数 (rho)        : {spear:.6f}')
print('=' * 62)

result = {
    'checkpoint': args.checkpoint,
    'fold_id': args.fold_id,
    'n_samples': n,
    'surface_pressure_relative_l2': float(l2err_press),
    'volume_velocity_relative_l2': float(l2err_velo),
    'surface_pressure_rmse_pa': float(rmse_press),
    'volume_velocity_rmse_mps': [float(x) for x in rmse_velo],
    'drag_coefficient_relative_error_mean': float(coef_error / n),
    'drag_coefficient_spearman': float(spear),
}
if args.evaluation_dir:
    evaluation_dir = os.path.abspath(args.evaluation_dir)
elif args.experiment_dir:
    evaluation_dir = os.path.join(os.path.abspath(args.experiment_dir), 'evaluation')
else:
    checkpoint_parent = os.path.dirname(os.path.abspath(args.checkpoint))
    if os.path.basename(checkpoint_parent) == 'checkpoints':
        evaluation_dir = os.path.join(os.path.dirname(checkpoint_parent), 'evaluation')
    else:
        evaluation_dir = checkpoint_parent
out_json = args.out_json or os.path.join(evaluation_dir, f'eval_fold_{args.fold_id}.json')
os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print('结果已保存:', out_json)
