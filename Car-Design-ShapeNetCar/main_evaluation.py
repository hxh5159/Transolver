import os
import torch
import argparse
import numpy as np
import time
import csv
import json
import warnings
from torch import nn
from torch_geometric.loader import DataLoader
from utils.drag_coefficient import cal_coefficient
from utils.visualization import (
    create_sample_visualizations,
    drag_coefficient_metrics,
    plot_cd_evaluation,
    plot_inference_time_distribution,
    plot_l2_error_distribution,
    plot_rmse_by_variable,
)
from dataset.load_dataset import load_train_val_fold_file
from dataset.dataset import GraphDataset
from models.Transolver import Model
import scipy as sc


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir', default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--cfd_model')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--nb_epochs', default=200, type=int)
parser.add_argument('--model_path', default=None,
                    help='模型文件路径；不提供时使用旧 metrics 路径')
parser.add_argument('--experiment_dir', default=None,
                    help='实验目录；评估输出写入该目录下的 evaluation/')
parser.add_argument('--evaluation_dir', default=None,
                    help='显式指定评估输出目录')
args = parser.parse_args()
print(args)


n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

train_data, val_data, coef_norm, vallst = load_train_val_fold_file(args, preprocessed=True)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

if args.experiment_dir:
    experiment_dir = os.path.abspath(args.experiment_dir)
elif args.model_path:
    model_parent = os.path.dirname(os.path.abspath(args.model_path))
    experiment_dir = (os.path.dirname(model_parent)
                      if os.path.basename(model_parent) == 'checkpoints'
                      else model_parent)
else:
    experiment_dir = None

if args.model_path:
    model_path = os.path.abspath(args.model_path)
elif experiment_dir:
    model_path = os.path.join(experiment_dir, 'model_final.pth')
else:
    path = f'metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'
    model_path = os.path.join(path, f'model_{args.nb_epochs}.pth')

try:
    loaded_model = torch.load(model_path, map_location='cpu', weights_only=False)
except TypeError:
    loaded_model = torch.load(model_path, map_location='cpu')
if isinstance(loaded_model, nn.Module):
    model = loaded_model
elif isinstance(loaded_model, dict) and 'model_state_dict' in loaded_model:
    model_kwargs = loaded_model.get('metadata', {}).get('model_kwargs', {
        'n_hidden': 256, 'n_layers': 8, 'space_dim': 7, 'fun_dim': 0,
        'n_head': 8, 'mlp_ratio': 2, 'out_dim': 4, 'slice_num': 32,
        'unified_pos': 0,
    })
    model = Model(**model_kwargs)
    model.load_state_dict(loaded_model['model_state_dict'])
elif isinstance(loaded_model, dict):
    model = Model(n_hidden=256, n_layers=8, space_dim=7, fun_dim=0,
                  n_head=8, mlp_ratio=2, out_dim=4, slice_num=32,
                  unified_pos=0)
    model.load_state_dict(loaded_model)
else:
    raise ValueError(f'无法从模型文件加载 nn.Module 或 checkpoint: {model_path}')
model = model.to(device)

test_loader = DataLoader(val_ds, batch_size=1)

if args.evaluation_dir:
    evaluation_dir = os.path.abspath(args.evaluation_dir)
elif experiment_dir:
    evaluation_dir = os.path.join(experiment_dir, 'evaluation')
else:
    evaluation_dir = os.path.join(
        'results', args.cfd_model, 'evaluation', f'fold_{args.fold_id}',
        f'epoch_{args.nb_epochs:04d}')
prediction_dir = os.path.join(evaluation_dir, 'predictions')
case_dir = os.path.join(evaluation_dir, 'cases')
os.makedirs(prediction_dir, exist_ok=True)
os.makedirs(case_dir, exist_ok=True)

with torch.no_grad():
    model.eval()
    criterion_func = nn.MSELoss(reduction='none')
    l2errs_press = []
    l2errs_velo = []
    mses_press = []
    mses_velo_var = []
    times = []
    gt_coef_list = []
    pred_coef_list = []
    coef_error = 0
    sample_metrics = []
    index = 0
    for cfd_data, geom in test_loader:
        sample_path = vallst[index]
        print(sample_path)
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        tic = time.perf_counter()
        out = model((cfd_data, geom))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        toc = time.perf_counter()
        targets = cfd_data.y

        if coef_norm is not None:
            mean = torch.as_tensor(coef_norm[2], device=device, dtype=out.dtype)
            std = torch.as_tensor(coef_norm[3], device=device, dtype=out.dtype)
            out_denorm = out * std + mean
            y_denorm = targets * std + mean
        else:
            out_denorm = out
            y_denorm = targets

        pred_press = out_denorm[cfd_data.surf, -1]
        gt_press = y_denorm[cfd_data.surf, -1]
        pred_surf_velo = out_denorm[cfd_data.surf, :-1]
        gt_surf_velo = y_denorm[cfd_data.surf, :-1]
        pred_velo = out_denorm[~cfd_data.surf, :-1]
        gt_velo = y_denorm[~cfd_data.surf, :-1]

        prediction_path = os.path.join(prediction_dir, f'{index}_pred.npy')
        target_path = os.path.join(prediction_dir, f'{index}_gt.npy')
        np.save(prediction_path, out_denorm.detach().cpu().numpy())
        np.save(target_path, y_denorm.detach().cpu().numpy())

        pred_coef = cal_coefficient(
            sample_path, pred_press[:, None].detach().cpu().numpy(),
            pred_surf_velo.detach().cpu().numpy(), root=args.data_dir)
        gt_coef = cal_coefficient(
            sample_path, gt_press[:, None].detach().cpu().numpy(),
            gt_surf_velo.detach().cpu().numpy(), root=args.data_dir)

        gt_coef_list.append(gt_coef)
        pred_coef_list.append(pred_coef)
        sample_coef_error = abs(pred_coef - gt_coef) / max(abs(gt_coef), 1e-12)
        coef_error += sample_coef_error
        print(coef_error / (index + 1))

        l2err_press = torch.norm(pred_press - gt_press) / torch.clamp(torch.norm(gt_press), min=1e-12)
        l2err_velo = torch.norm(pred_velo - gt_velo) / torch.clamp(torch.norm(gt_velo), min=1e-12)

        mse_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        mse_velo_var = criterion_func(out[~cfd_data.surf, :-1], targets[~cfd_data.surf, :-1]).mean(dim=0)

        physical_press_rmse = torch.sqrt(
            criterion_func(pred_press, gt_press).mean(dim=0))
        physical_velocity_rmse = torch.sqrt(
            criterion_func(pred_velo, gt_velo).mean(dim=0))
        latency_ms = (toc - tic) * 1000.0

        l2errs_press.append(float(l2err_press.cpu()))
        l2errs_velo.append(float(l2err_velo.cpu()))
        mses_press.append(float(mse_press.cpu()))
        mses_velo_var.append(mse_velo_var.cpu().numpy())
        times.append(toc - tic)
        velocity_rmse_values = physical_velocity_rmse.cpu().numpy()
        sample_metrics.append({
            'index': index,
            'sample_path': sample_path,
            'surface_pressure_relative_l2': float(l2err_press.cpu()),
            'volume_velocity_relative_l2': float(l2err_velo.cpu()),
            'surface_pressure_rmse_pa': float(physical_press_rmse.cpu()),
            'volume_velocity_u_rmse_mps': float(velocity_rmse_values[0]),
            'volume_velocity_v_rmse_mps': float(velocity_rmse_values[1]),
            'volume_velocity_w_rmse_mps': float(velocity_rmse_values[2]),
            'ground_truth_cd': float(gt_coef),
            'predicted_cd': float(pred_coef),
            'cd_relative_error': float(sample_coef_error),
            'model_forward_latency_ms': float(latency_ms),
        })
        index += 1

    gt_coef_list = np.array(gt_coef_list)
    pred_coef_list = np.array(pred_coef_list)
    spear = sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]
    print("rho_d: ", spear)
    print("c_d: ", coef_error / index)
    l2err_press = np.mean(l2errs_press)
    l2err_velo = np.mean(l2errs_velo)
    rmse_press = np.sqrt(np.mean(mses_press))
    rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
    if coef_norm is not None:
        rmse_press *= coef_norm[3][-1]
        rmse_velo_var *= coef_norm[3][:-1]
    print('relative l2 error press:', l2err_press)
    print('relative l2 error velo:', l2err_velo)
    print('press:', rmse_press)
    print('velo:', rmse_velo_var, np.sqrt(np.mean(np.square(rmse_velo_var))))
    print('time:', np.mean(times))

pressure_l2_values = np.asarray(l2errs_press, dtype=float)
velocity_l2_values = np.asarray(l2errs_velo, dtype=float)
pressure_rmse_values = np.asarray(
    [record['surface_pressure_rmse_pa'] for record in sample_metrics], dtype=float)
velocity_rmse_values = np.asarray([
    [record['volume_velocity_u_rmse_mps'], record['volume_velocity_v_rmse_mps'],
     record['volume_velocity_w_rmse_mps']]
    for record in sample_metrics
], dtype=float)
forward_latency_ms = np.asarray(
    [record['model_forward_latency_ms'] for record in sample_metrics], dtype=float)

plot_l2_error_distribution(
    os.path.join(evaluation_dir, 'l2_error_distribution.png'),
    pressure_l2_values, velocity_l2_values)
plot_rmse_by_variable(
    os.path.join(evaluation_dir, 'rmse_by_variable.png'),
    pressure_rmse_values, velocity_rmse_values)
cd_metrics = plot_cd_evaluation(
    os.path.join(evaluation_dir, 'cd_evaluation.png'), gt_coef_list, pred_coef_list)
latency_metrics = plot_inference_time_distribution(
    os.path.join(evaluation_dir, 'inference_time_distribution.png'), forward_latency_ms)

aggregate_metrics = {
    'model_path': model_path,
    'evaluation_dir': evaluation_dir,
    'evaluation_scope': {
        'pressure': 'car-surface nodes',
        'velocity': 'non-surface volume nodes',
        'drag_coefficient': 'postprocessed from surface pressure and surface velocity',
        'inference_time': 'model forward only; excludes data loading, device transfer, and Cd postprocessing',
    },
    'sample_count': index,
    'surface_pressure_relative_l2_mean': float(l2err_press),
    'volume_velocity_relative_l2_mean': float(l2err_velo),
    'surface_pressure_rmse_pa': float(rmse_press),
    'volume_velocity_component_rmse_mps': {
        'u': float(rmse_velo_var[0]),
        'v': float(rmse_velo_var[1]),
        'w': float(rmse_velo_var[2]),
    },
    'volume_velocity_combined_rmse_mps': float(np.sqrt(np.mean(np.square(rmse_velo_var)))),
    'drag_coefficient': {
        'mape': cd_metrics['mape'],
        'mae': cd_metrics['mae'],
        'r_squared': finite_or_none(cd_metrics['r_squared']),
        'spearman': finite_or_none(cd_metrics['spearman']),
    },
    'model_forward_latency_ms': latency_metrics,
}

with open(os.path.join(evaluation_dir, 'evaluation_metrics.json'), 'w') as file:
    json.dump(aggregate_metrics, file, indent=2, allow_nan=False)
with open(os.path.join(evaluation_dir, 'per_sample_metrics.csv'), 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=sample_metrics[0].keys())
    writer.writeheader()
    writer.writerows(sample_metrics)

cd_relative_errors = drag_coefficient_metrics(gt_coef_list, pred_coef_list)['relative_error']
cd_order = np.argsort(cd_relative_errors)
selected_cases = (
    ('best', int(cd_order[0])),
    ('median', int(cd_order[len(cd_order) // 2])),
    ('worst', int(cd_order[-1])),
)
for label, sample_index in selected_cases:
    sample_path = vallst[sample_index]
    output_dir = os.path.join(case_dir, f'cd_{label}_{sample_index}')
    try:
        sample_data, _ = val_ds[sample_index]
        prediction = np.load(os.path.join(prediction_dir, f'{sample_index}_pred.npy'))
        target = np.load(os.path.join(prediction_dir, f'{sample_index}_gt.npy'))
        create_sample_visualizations(
            output_dir, args.data_dir, sample_path,
            sample_data.pos.cpu().numpy(), sample_data.surf.cpu().numpy(),
            prediction, target,
            cd_values=(float(gt_coef_list[sample_index]), float(pred_coef_list[sample_index])))
    except Exception as exc:
        warnings.warn(f'Could not create {label} Cd-error case {sample_index}: {exc}')

print('evaluation outputs:', evaluation_dir)
