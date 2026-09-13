"""Experiment artifact helpers for the AirfRANS task.

The helpers in this module deliberately do not know anything about a model
architecture or a loss function.  They only make an experiment directory and
serialize the state needed to inspect or resume a run.
"""
import datetime
import json
import os
import platform
import random
import shlex
import subprocess
import sys

import numpy as np
import torch


class NumpyTorchEncoder(json.JSONEncoder):
    def default(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return json.JSONEncoder.default(self, value)


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as file:
        json.dump(_jsonable(payload), file, indent=2, ensure_ascii=False,
                  allow_nan=False)


def _git_metadata():
    try:
        root = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, check=True).stdout.strip()
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip() != ''
        return {'root': root, 'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'root': None, 'commit': None, 'dirty': None}


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _timestamp_dir(output_root, label):
    os.makedirs(output_root, exist_ok=True)
    stamp = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    safe_label = ''.join(char if char.isalnum() or char in '-_' else '_' for char in label)
    base = f'{stamp}_{safe_label}' if safe_label else stamp
    candidate = os.path.join(output_root, base)
    suffix = 0
    while os.path.exists(candidate):
        suffix += 1
        candidate = os.path.join(output_root, f'{base}_{suffix:02d}')
    os.makedirs(candidate)
    return os.path.abspath(candidate)


def prepare_experiment(output_root, label, experiment_dir=None):
    if experiment_dir is None:
        root = _timestamp_dir(os.path.abspath(output_root), label)
    else:
        root = os.path.abspath(experiment_dir)
        os.makedirs(root, exist_ok=True)
    paths = {
        'root': root,
        'checkpoints': os.path.join(root, 'checkpoints'),
        'logs': os.path.join(root, 'logs'),
        'visualizations': os.path.join(root, 'visualizations'),
        'evaluation': os.path.join(root, 'evaluation'),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def environment_metadata():
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count(),
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
        'git': _git_metadata(),
    }


def initialize_run(paths, args, extra=None):
    started_at = datetime.datetime.now().astimezone().isoformat()
    primary_config_path = os.path.join(paths['root'], 'config.json')
    session_suffix = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    config_path = (os.path.join(paths['logs'], f'config_session_{session_suffix}.json')
                   if os.path.exists(primary_config_path) else primary_config_path)
    command_path = (os.path.join(paths['logs'], f'command_session_{session_suffix}.txt')
                    if os.path.exists(primary_config_path)
                    else os.path.join(paths['root'], 'command.txt'))
    config = {
        'config_version': 1,
        'experiment_id': os.path.basename(paths['root']),
        'started_at': started_at,
        'command': shlex.join([sys.executable] + sys.argv),
        'working_directory': os.getcwd(),
        'arguments': vars(args),
        'paths': paths,
        'environment': environment_metadata(),
        '_config_path': config_path,
    }
    if extra:
        config.update(extra)
    write_json(config_path, config)
    write_json(os.path.join(paths['root'], 'run_status.json'), {
        'status': 'running', 'experiment_id': config['experiment_id'],
        'started_at': started_at,
    })
    with open(command_path, 'w') as file:
        file.write(config['command'] + '\n')
    return config, started_at


def update_config(paths, config):
    write_json(config.get('_config_path', os.path.join(paths['root'], 'config.json')), config)


def update_status(paths, status, started_at, **extra):
    write_json(os.path.join(paths['root'], 'run_status.json'), {
        'status': status,
        'experiment_id': os.path.basename(paths['root']),
        'started_at': started_at,
        'finished_at': datetime.datetime.now().astimezone().isoformat(),
        **extra,
    })


def _atomic_save(payload, path):
    temporary = path + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_save(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_save(payload, path)


def rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(paths, epoch, model, optimizer, scheduler, history, metadata=None,
                    filename=None, extra=None):
    path = os.path.join(paths['checkpoints'], filename or f'checkpoint_epoch_{epoch:04d}.pth')
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'metadata': metadata or {},
        'rng_state': rng_state(),
    }
    if extra:
        payload.update(extra)
    _atomic_save(payload, path)
    return path


def save_model_files(paths, model, prefix='model_final'):
    full_path = os.path.join(paths['root'], prefix + '.pth')
    state_path = os.path.join(paths['root'], prefix + '_state_dict.pth')
    _atomic_save(model, full_path)
    _atomic_save(model.state_dict(), state_path)
    return full_path, state_path


def save_history(paths, history, name='training_history'):
    write_json(os.path.join(paths['logs'], name + '.json'), history)
    if not history:
        return
    fields = sorted({key for row in history for key in row})
    import csv
    with open(os.path.join(paths['logs'], name + '.csv'), 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def summarize(paths, history, started_at, target_epochs, train_size, val_size,
              final_paths, extra=None, training_started=None):
    epoch_times = [row['epoch_seconds'] for row in history if row.get('epoch_seconds') is not None]
    summary = {
        'status': 'completed',
        'started_at': started_at,
        'training_started_at': training_started or started_at,
        'finished_at': datetime.datetime.now().astimezone().isoformat(),
        'epochs_completed': len(history),
        'target_epochs': target_epochs,
        'time_elapsed_seconds': float(sum(epoch_times)) if epoch_times else 0.0,
        'mean_epoch_seconds': float(np.mean(epoch_times)) if epoch_times else None,
        'median_epoch_seconds': float(np.median(epoch_times)) if epoch_times else None,
        'min_epoch_seconds': float(np.min(epoch_times)) if epoch_times else None,
        'max_epoch_seconds': float(np.max(epoch_times)) if epoch_times else None,
        'train_samples': train_size,
        'validation_samples': val_size,
        'final_model': final_paths[0],
        'final_state_dict': final_paths[1],
    }
    if extra:
        summary.update(extra)
    write_json(os.path.join(paths['root'], 'training_summary.json'), summary)
    return summary
