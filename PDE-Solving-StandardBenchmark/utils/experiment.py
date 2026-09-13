"""Shared experiment-output helpers for the standard PDE benchmarks."""
import csv
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


class Encoder(json.JSONEncoder):
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


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _git_metadata():
    try:
        root = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, check=True).stdout.strip()
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
        return {'root': root, 'commit': commit,
                'dirty': bool(subprocess.run(['git', 'status', '--porcelain'], cwd=root,
                                             capture_output=True, text=True,
                                             check=True).stdout.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {'root': None, 'commit': None, 'dirty': None}


def prepare_experiment(output_root, label, experiment_dir=None):
    if experiment_dir is None:
        stamp = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in label)
        root = os.path.join(os.path.abspath(output_root), f'{stamp}_{safe}')
        suffix = 0
        while os.path.exists(root):
            suffix += 1
            root = os.path.join(os.path.abspath(output_root), f'{stamp}_{safe}_{suffix:02d}')
    else:
        root = os.path.abspath(experiment_dir)
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


def infer_experiment_dir(checkpoint_path):
    if checkpoint_path is None:
        return None
    current = os.path.dirname(os.path.abspath(checkpoint_path))
    while current != os.path.dirname(current):
        if os.path.isfile(os.path.join(current, 'config.json')):
            return current
        current = os.path.dirname(current)
    return None


def initialize(paths, args, extra=None):
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
        'environment': {
            'python': sys.version,
            'platform': platform.platform(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
            'git': _git_metadata(),
        },
        '_config_path': config_path,
    }
    if extra:
        config.update(extra)
    write_json(config_path, config)
    write_json(os.path.join(paths['root'], 'run_status.json'),
               {'status': 'running', 'experiment_id': config['experiment_id'],
                'started_at': started_at})
    with open(command_path, 'w') as file:
        file.write(config['command'] + '\n')
    return config, started_at


def update_config(paths, config):
    write_json(config.get('_config_path', os.path.join(paths['root'], 'config.json')), config)


def initialize_evaluation(paths, args, extra=None):
    started_at = datetime.datetime.now().astimezone().isoformat()
    payload = {
        'started_at': started_at,
        'command': shlex.join([sys.executable] + sys.argv),
        'arguments': vars(args),
        'experiment_dir': paths['root'],
    }
    if extra:
        payload.update(extra)
    write_json(os.path.join(paths['evaluation'], 'evaluation_config.json'), payload)
    return started_at


def update_status(paths, status, started_at, **extra):
    write_json(os.path.join(paths['root'], 'run_status.json'), {
        'status': status, 'experiment_id': os.path.basename(paths['root']),
        'started_at': started_at,
        'finished_at': datetime.datetime.now().astimezone().isoformat(), **extra})


def _atomic_save(payload, path):
    temporary = path + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(paths, epoch, model, optimizer, scheduler, history,
                    metadata=None, filename=None, extra=None):
    path = os.path.join(paths['checkpoints'], filename or f'checkpoint_epoch_{epoch:04d}.pth')
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'metadata': metadata or {},
        'rng_state': {
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if extra:
        payload.update(extra)
    _atomic_save(payload, path)
    return path


def load_model_state(path, model, device='cpu'):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, torch.nn.Module):
        model.load_state_dict(payload.state_dict(), strict=False)
        return {'model_state_dict': payload.state_dict()}
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        model.load_state_dict(payload['model_state_dict'], strict=False)
        return payload
    model.load_state_dict(payload, strict=False)
    return {'model_state_dict': payload}


def restore_training(path, model, optimizer, scheduler, device='cpu'):
    if path is None:
        return 0, []
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    model.load_state_dict(payload['model_state_dict'])
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    scheduler.load_state_dict(payload['scheduler_state_dict'])
    state = payload.get('rng_state')
    if state:
        if state.get('python') is not None:
            random.setstate(state['python'])
        if state.get('numpy') is not None:
            np.random.set_state(state['numpy'])
        if state.get('torch') is not None:
            torch.set_rng_state(state['torch'])
        if torch.cuda.is_available() and state.get('cuda') is not None:
            torch.cuda.set_rng_state_all(state['cuda'])
    return int(payload.get('epoch', 0)), payload.get('history', [])


def save_model_files(paths, model, prefix='model_final'):
    full_path = os.path.join(paths['root'], prefix + '.pth')
    state_path = os.path.join(paths['root'], prefix + '_state_dict.pth')
    _atomic_save(model, full_path)
    _atomic_save(model.state_dict(), state_path)
    return full_path, state_path


def save_history(paths, history, filename='training_history'):
    write_json(os.path.join(paths['logs'], filename + '.json'), history)
    if history:
        fields = sorted({key for row in history for key in row})
        with open(os.path.join(paths['logs'], filename + '.csv'), 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)


def summarize(paths, history, started_at, epochs, train_size, test_size,
              final_paths, extra=None):
    durations = [row['epoch_seconds'] for row in history if row.get('epoch_seconds') is not None]
    summary = {
        'status': 'completed',
        'started_at': started_at,
        'finished_at': datetime.datetime.now().astimezone().isoformat(),
        'epochs_completed': len(history),
        'target_epochs': epochs,
        'time_elapsed_seconds': float(sum(durations)) if durations else 0.0,
        'mean_epoch_seconds': float(np.mean(durations)) if durations else None,
        'median_epoch_seconds': float(np.median(durations)) if durations else None,
        'min_epoch_seconds': float(np.min(durations)) if durations else None,
        'max_epoch_seconds': float(np.max(durations)) if durations else None,
        'train_samples': train_size, 'test_samples': test_size,
        'final_model': final_paths[0], 'final_state_dict': final_paths[1],
    }
    if extra:
        summary.update(extra)
    write_json(os.path.join(paths['root'], 'training_summary.json'), summary)
    return summary


def record_epoch(paths, history, epoch, epoch_started, **metrics):
    row = {'epoch': epoch, **metrics,
           'epoch_seconds': float(__import__('time').perf_counter() - epoch_started)}
    history.append(row)
    save_history(paths, history)
    return row


def plot_training_history(paths, history):
    if not history:
        return None
    import matplotlib.pyplot as plt
    excluded = {'epoch', 'epoch_seconds'}
    metric_names = [key for key in history[0]
                    if key not in excluded and any(row.get(key) is not None for row in history)]
    if not metric_names:
        return None
    epochs = [row['epoch'] for row in history]
    figure, axis = plt.subplots(figsize=(10, 6))
    for name in metric_names:
        values = [np.nan if row.get(name) is None else row[name] for row in history]
        axis.plot(epochs, values, label=name)
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Metric')
    axis.set_title('Training history')
    axis.legend(loc='best')
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = os.path.join(paths['visualizations'], 'training_curves.png')
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return path


def finalize_training(paths, epoch, model, optimizer, scheduler, history,
                      metadata, started_at, epochs, train_size, test_size,
                      extra=None, checkpoint_extra=None):
    final_checkpoint = save_checkpoint(
        paths, epoch, model, optimizer, scheduler, history,
        metadata=metadata, filename='checkpoint_final.pth', extra=checkpoint_extra)
    final_paths = save_model_files(paths, model)
    training_curve = plot_training_history(paths, history)
    details = {'final_checkpoint': final_checkpoint,
               'training_curve': training_curve,
               'nb_parameters': int(sum(parameter.numel() for parameter in model.parameters()
                                        if parameter.requires_grad))}
    if extra:
        details.update(extra)
    summary = summarize(paths, history, started_at, epochs, train_size,
                        test_size, final_paths, details)
    update_status(paths, 'completed', started_at,
                  summary=os.path.join(paths['root'], 'training_summary.json'))
    return summary
