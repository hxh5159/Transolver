import argparse
import datetime
import json
import os
import platform
import random
import shlex
import subprocess
import sys
import traceback

import numpy as np
import torch
import torch_geometric

import train

from dataset.load_dataset import load_train_val_fold_file
from dataset.dataset import GraphDataset
from models.Transolver import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir', default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
parser.add_argument('--cfd_model')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs', default=200, type=int)
parser.add_argument('--preprocessed', default=1, type=int)
parser.add_argument('--seed', default=0, type=int,
                    help='随机种子，用于记录和复现实验')
parser.add_argument('--deterministic', action='store_true',
                    help='启用 cuDNN 确定性设置')
parser.add_argument('--checkpoint_interval', default=50, type=int)
parser.add_argument('--visualization_interval', default=50, type=int)
parser.add_argument('--visualization_sample', default=-1, type=int,
                    help='Validation sample index; -1 selects the median ground-truth Cd case')
parser.add_argument('--output_root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
parser.add_argument('--experiment_dir', default=None,
                    help='Existing experiment directory; normally used together with --resume')
parser.add_argument('--checkpoint_dir', default=None,
                    help=argparse.SUPPRESS)
parser.add_argument('--resume', default=None)
args = parser.parse_args()
print(args)


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path, payload):
    with open(path, 'w') as file:
        json.dump(_jsonable(payload), file, indent=2, ensure_ascii=False)


def _git_metadata():
    try:
        root = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, check=True).stdout.strip()
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=root,
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain'], cwd=root,
            capture_output=True, text=True, check=True).stdout.strip() != ''
        return {'root': root, 'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'root': None, 'commit': None, 'dirty': None}


def _new_experiment_dir(output_root):
    os.makedirs(output_root, exist_ok=True)
    stamp = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    candidate = os.path.join(output_root, stamp)
    suffix = 0
    while os.path.exists(candidate):
        suffix += 1
        candidate = os.path.join(output_root, f'{stamp}_{suffix:02d}')
    os.makedirs(candidate)
    return os.path.abspath(candidate)


def _resume_experiment_dir(resume_path):
    resume_path = os.path.abspath(resume_path)
    parent = os.path.dirname(resume_path)
    if os.path.basename(parent) == 'checkpoints':
        return os.path.dirname(parent)
    return parent


def _is_within(path, root):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


output_root = os.path.abspath(args.output_root)
if args.experiment_dir:
    experiment_dir = os.path.abspath(args.experiment_dir)
    if not _is_within(experiment_dir, output_root):
        raise ValueError(f'--experiment_dir 必须位于 --output_root 下: {experiment_dir}')
    os.makedirs(experiment_dir, exist_ok=True)
elif args.resume:
    inferred_experiment_dir = _resume_experiment_dir(args.resume)
    experiment_dir = (inferred_experiment_dir
                      if _is_within(inferred_experiment_dir, output_root)
                      else _new_experiment_dir(output_root))
    os.makedirs(experiment_dir, exist_ok=True)
else:
    experiment_dir = _new_experiment_dir(output_root)

experiment_id = os.path.basename(os.path.normpath(experiment_dir))
config_path = os.path.join(experiment_dir, 'config.json')
status_path = os.path.join(experiment_dir, 'run_status.json')
model_kwargs = {
    'n_hidden': 256,
    'n_layers': 8,
    'space_dim': 7,
    'fun_dim': 0,
    'n_head': 8,
    'mlp_ratio': 2,
    'out_dim': 4,
    'slice_num': 32,
    'unified_pos': 0,
}
started_at = datetime.datetime.now().astimezone().isoformat()
config = {
    'config_version': 1,
    'experiment_id': experiment_id,
    'started_at': started_at,
    'command': shlex.join([sys.executable] + sys.argv),
    'working_directory': os.getcwd(),
    'arguments': vars(args),
    'paths': {
        'experiment_dir': experiment_dir,
        'output_root': os.path.abspath(args.output_root),
        'data_dir': os.path.abspath(args.data_dir),
        'save_dir': os.path.abspath(args.save_dir),
    },
    'model': {'name': args.cfd_model, 'kwargs': model_kwargs},
    'environment': {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torch_geometric': getattr(torch_geometric, '__version__', None),
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count(),
        'gpu_name': (torch.cuda.get_device_name(args.gpu)
                     if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count()
                     else None),
        'git': _git_metadata(),
    },
}
_write_json(config_path, config)
_write_json(status_path, {
    'status': 'running',
    'experiment_id': experiment_id,
    'started_at': started_at,
    'config': config_path,
})
with open(os.path.join(experiment_dir, 'command.txt'), 'w') as command_file:
    command_file.write(config['command'] + '\n')


def _mark_failed(exc):
    _write_json(status_path, {
        'status': 'failed',
        'experiment_id': experiment_id,
        'started_at': started_at,
        'finished_at': datetime.datetime.now().astimezone().isoformat(),
        'error': repr(exc),
        'traceback': traceback.format_exc(),
        'config': config_path,
    })

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

hparams = {
    'lr': args.lr,
    'batch_size': args.batch_size,
    'nb_epochs': args.nb_epochs,
    'weight': args.weight,
    'val_iter': args.val_iter,
    'fold_id': args.fold_id,
    'r': args.r,
    'preprocessed': args.preprocessed,
    'seed': args.seed,
    'deterministic': args.deterministic,
}

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
config['device'] = {
    'requested_gpu': args.gpu,
    'selected': str(device),
    'cuda_enabled': use_cuda,
}
_write_json(config_path, config)

try:
    train_data, val_data, coef_norm, val_sample_paths = load_train_val_fold_file(
        args, preprocessed=args.preprocessed)
    train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
    val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

    config['dataset'] = {
        'train_samples': len(train_data),
        'validation_samples': len(val_data),
        'validation_sample_paths': val_sample_paths,
        'coef_norm': coef_norm,
    }
    _write_json(config_path, config)

    if args.cfd_model == 'Transolver':
        model = Model(**model_kwargs).to(device)
except Exception as exc:
    _mark_failed(exc)
    raise

run_metadata = {
    'experiment_id': experiment_id,
    'config_path': config_path,
    'command': config['command'],
    'started_at': started_at,
    'fold_id': args.fold_id,
    'model_kwargs': model_kwargs,
    'seed': args.seed,
    'deterministic': args.deterministic,
}

try:
    model = train.main(
        device, train_ds, val_ds, model, hparams, experiment_dir, val_iter=args.val_iter, reg=args.weight,
        coef_norm=coef_norm, checkpoint_dir=os.path.join(experiment_dir, 'checkpoints'),
        checkpoint_interval=args.checkpoint_interval, visualization_interval=args.visualization_interval,
        visualization_sample=args.visualization_sample, sample_paths=val_sample_paths,
        data_dir=args.data_dir, resume_path=args.resume, experiment_dir=experiment_dir,
        run_metadata=run_metadata)
except Exception as exc:
    _mark_failed(exc)
    raise
else:
    _write_json(status_path, {
        'status': 'completed',
        'experiment_id': experiment_id,
        'started_at': started_at,
        'finished_at': datetime.datetime.now().astimezone().isoformat(),
        'config': config_path,
        'summary': os.path.join(experiment_dir, 'training_summary.json'),
    })
