import csv
import datetime
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.visualization import (
    compute_and_plot_validation_drag,
    create_sample_visualizations,
    plot_loss_history,
    predict_sample,
    write_metrics,
)


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, reg=1):
    model.train()

    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for cfd_data, geom in train_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        optimizer.zero_grad()
        out = model((cfd_data, geom))
        targets = cfd_data.y

        loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()
        total_loss = loss_velo + reg * loss_press

        total_loss.backward()

        optimizer.step()
        scheduler.step()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())

    return np.mean(losses_press), np.mean(losses_velo)


@torch.no_grad()
def test(device, model, test_loader, coef_norm=None, sample_paths=None, collect_outputs=False):
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    records = []
    for index, (cfd_data, geom) in enumerate(test_loader):
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y

        loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())

        if collect_outputs:
            if coef_norm is not None:
                mean = torch.as_tensor(coef_norm[2], device=device, dtype=out.dtype)
                std = torch.as_tensor(coef_norm[3], device=device, dtype=out.dtype)
                output_physical = out * std + mean
                target_physical = targets * std + mean
            else:
                output_physical = out
                target_physical = targets
            surface = cfd_data.surf
            records.append({
                'index': index,
                'sample_path': sample_paths[index] if sample_paths is not None else str(index),
                'surface_points': cfd_data.pos[surface].detach().cpu().numpy(),
                'pred_pressure': output_physical[surface, -1].detach().cpu().numpy(),
                'true_pressure': target_physical[surface, -1].detach().cpu().numpy(),
                'pred_velocity': output_physical[surface, :-1].detach().cpu().numpy(),
                'true_velocity': target_physical[surface, :-1].detach().cpu().numpy(),
            })

    result = (np.mean(losses_press), np.mean(losses_velo))
    if collect_outputs:
        return result + (records,)
    return result


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def _save_history(history, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, 'training_history.json')
    csv_path = os.path.join(log_dir, 'training_history.csv')
    with open(json_path, 'w') as file:
        json.dump(history, file, indent=2, cls=NumpyEncoder)
    fieldnames = (
        'epoch', 'train_velocity', 'train_pressure', 'train_total',
        'val_velocity', 'val_pressure', 'val_total', 'learning_rate',
        'train_seconds', 'validation_seconds', 'artifact_seconds', 'epoch_seconds',
    )
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _load_checkpoint(resume_path, device, model, optimizer, scheduler):
    try:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    rng_state = checkpoint.get('rng_state')
    if rng_state:
        if rng_state.get('python') is not None:
            random.setstate(rng_state['python'])
        if rng_state.get('numpy') is not None:
            np.random.set_state(rng_state['numpy'])
        if rng_state.get('torch') is not None:
            torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available() and rng_state.get('cuda') is not None:
            torch.cuda.set_rng_state_all(rng_state['cuda'])
    return checkpoint.get('epoch', 0), checkpoint.get('history', [])


def _atomic_torch_save(payload, path):
    temporary_path = path + '.tmp'
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _save_checkpoint(checkpoint_dir, epoch, model, optimizer, scheduler, hparams, history, coef_norm,
                     metadata=None, filename=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, filename or f'checkpoint_epoch_{epoch:04d}.pth')
    _atomic_torch_save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'hparams': hparams,
        'history': history,
        'coef_norm': coef_norm,
        'metadata': metadata or {},
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }, checkpoint_path)
    return checkpoint_path


def _write_json(path, payload):
    with open(path, 'w') as file:
        json.dump(payload, file, indent=2, allow_nan=False, cls=NumpyEncoder)


def main(device, train_dataset, val_dataset, Net, hparams, path, reg=1, val_iter=1, coef_norm=None,
         checkpoint_dir=None, checkpoint_interval=50, visualization_interval=50,
         visualization_sample=-1, sample_paths=None, data_dir=None, resume_path=None,
         experiment_dir=None, run_metadata=None):
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        final_div_factor=1000.,
    )
    training_started_at = datetime.datetime.now().astimezone().isoformat()
    start = time.perf_counter()

    if experiment_dir is not None:
        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        log_dir = os.path.join(experiment_dir, 'logs')
        visualization_root = os.path.join(experiment_dir, 'visualizations')
    else:
        checkpoint_dir = checkpoint_dir or os.path.join(path, 'checkpoints')
        log_dir = os.path.join(checkpoint_dir, 'logs')
        visualization_root = os.path.join(checkpoint_dir, 'visualizations')
    artifact_root = experiment_dir or path
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(visualization_root, exist_ok=True)

    start_epoch = 0
    history = []
    if resume_path is not None:
        start_epoch, history = _load_checkpoint(
            resume_path, device, model, optimizer, lr_scheduler)
        print(f'Resumed training from epoch {start_epoch}: {resume_path}')

    train_loss, val_loss = 1e5, 1e5
    pbar_train = tqdm(range(start_epoch, hparams['nb_epochs']), position=0)
    gt_cd_cache = {}
    for epoch in pbar_train:
        current_epoch = epoch + 1
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        loss_press, loss_velo = train(device, model, train_loader, optimizer, lr_scheduler, reg=reg)
        train_seconds = time.perf_counter() - train_start
        train_loss = loss_velo + reg * loss_press
        del (train_loader)

        checkpoint_due = (checkpoint_interval > 0 and current_epoch % checkpoint_interval == 0)
        visualization_due = visualization_interval > 0 and current_epoch % visualization_interval == 0
        scheduled_validation = val_iter is not None and current_epoch % val_iter == 0
        should_validate = val_iter is not None and (
            current_epoch == hparams['nb_epochs'] or scheduled_validation or visualization_due)
        val_loss_velo = val_loss_press = None
        validation_records = None
        validation_seconds = 0.0
        if should_validate:
            validation_start = time.perf_counter()
            val_loader = DataLoader(val_dataset, batch_size=1)
            if visualization_due:
                val_loss_press, val_loss_velo, validation_records = test(
                    device, model, val_loader, coef_norm=coef_norm,
                    sample_paths=sample_paths, collect_outputs=True)
            else:
                val_loss_press, val_loss_velo = test(device, model, val_loader)
            val_loss = val_loss_velo + reg * val_loss_press
            del (val_loader)
            validation_seconds = time.perf_counter() - validation_start

            pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)
        else:
            pbar_train.set_postfix(train_loss=train_loss)

        history.append({
            'epoch': current_epoch,
            'train_velocity': float(loss_velo),
            'train_pressure': float(loss_press),
            'train_total': float(train_loss),
            'val_velocity': None if val_loss_velo is None else float(val_loss_velo),
            'val_pressure': None if val_loss_press is None else float(val_loss_press),
            'val_total': None if val_loss_velo is None else float(val_loss),
            'learning_rate': float(optimizer.param_groups[0]['lr']),
        })

        artifact_start = time.perf_counter()
        if visualization_due and validation_records is not None and data_dir is not None:
            epoch_visualization_dir = os.path.join(visualization_root, f'epoch_{current_epoch:04d}')
            os.makedirs(epoch_visualization_dir, exist_ok=True)
            plot_loss_history(
                os.path.join(epoch_visualization_dir, 'learning_curves.png'),
                history, checkpoint_interval=checkpoint_interval, pressure_weight=reg)
            drag_metrics = {}
            try:
                drag_metrics = compute_and_plot_validation_drag(
                    os.path.join(epoch_visualization_dir, 'validation_cd.png'),
                    validation_records, data_dir, gt_cache=gt_cd_cache)
            except Exception as exc:
                warnings.warn(f'Cd visualization at epoch {current_epoch} failed: {exc}')

            selected_index = visualization_sample
            if selected_index < 0:
                selected_index = drag_metrics.get('median_cd_record_index', 0)
            field_metrics = {}
            sample_path = None
            try:
                if not 0 <= selected_index < len(val_dataset):
                    raise IndexError(f'Visualization sample index {selected_index} is out of range')
                sample_path = sample_paths[selected_index] if sample_paths is not None else str(selected_index)
                sample_prediction = predict_sample(
                    device, model, val_dataset, selected_index, coef_norm)
                cd_values = drag_metrics.get('coefficients_by_index', {}).get(selected_index)
                field_metrics = create_sample_visualizations(
                    epoch_visualization_dir, data_dir, sample_path,
                    sample_prediction['points'], sample_prediction['surface_mask'],
                    sample_prediction['prediction'], sample_prediction['target'], cd_values=cd_values)
            except Exception as exc:
                warnings.warn(f'Field visualization at epoch {current_epoch} failed: {exc}')
            write_metrics(
                os.path.join(epoch_visualization_dir, 'metrics.json'),
                {'epoch': current_epoch, 'sample_path': sample_path, **drag_metrics, **field_metrics})

        epoch_seconds = time.perf_counter() - epoch_start
        history[-1].update({
            'train_seconds': float(train_seconds),
            'validation_seconds': float(validation_seconds),
            'artifact_seconds': float(time.perf_counter() - artifact_start),
            'epoch_seconds': float(epoch_seconds),
        })
        _save_history(history, log_dir)
        if checkpoint_due:
            _save_checkpoint(
                checkpoint_dir, current_epoch, model, optimizer, lr_scheduler,
                hparams, history, coef_norm, metadata=run_metadata)

    training_loop_seconds = time.perf_counter() - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(training_loop_seconds))
    final_checkpoint_path = _save_checkpoint(
        checkpoint_dir, hparams['nb_epochs'], model, optimizer, lr_scheduler,
        hparams, history, coef_norm, metadata=run_metadata, filename='checkpoint_final.pth')
    final_model_path = os.path.join(artifact_root, 'model_final.pth')
    final_state_dict_path = os.path.join(artifact_root, 'model_final_state_dict.pth')
    _atomic_torch_save(model, final_model_path)
    _atomic_torch_save(model.state_dict(), final_state_dict_path)
    plot_loss_history(
        os.path.join(visualization_root, 'final_training_loss.png'),
        history, checkpoint_interval=checkpoint_interval, pressure_weight=reg)

    time_elapsed = time.perf_counter() - start
    finished_at = datetime.datetime.now().astimezone().isoformat()
    epoch_times = [row.get('epoch_seconds') for row in history
                   if row.get('epoch_seconds') is not None]
    summary = {
        'status': 'completed',
        'started_at': (run_metadata or {}).get('started_at'),
        'training_started_at': training_started_at,
        'finished_at': finished_at,
        'epochs_completed': len(history),
        'target_epochs': hparams['nb_epochs'],
        'time_elapsed_seconds': float(time_elapsed),
        'training_loop_seconds': float(training_loop_seconds),
        'mean_epoch_seconds': float(np.mean(epoch_times)) if epoch_times else None,
        'median_epoch_seconds': float(np.median(epoch_times)) if epoch_times else None,
        'min_epoch_seconds': float(np.min(epoch_times)) if epoch_times else None,
        'max_epoch_seconds': float(np.max(epoch_times)) if epoch_times else None,
        'nb_parameters': float(params_model),
        'final_train_loss': float(train_loss),
        'final_validation_loss': float(val_loss) if val_iter is not None else None,
        'hparams': hparams,
        'coef_norm': coef_norm,
        'train_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'final_checkpoint': final_checkpoint_path,
        'final_model': final_model_path,
        'final_state_dict': final_state_dict_path,
        'checkpoints_dir': checkpoint_dir,
        'logs_dir': log_dir,
        'visualizations_dir': visualization_root,
        'resume_from': resume_path,
        'resume_start_epoch': start_epoch,
        'metadata': run_metadata or {},
    }
    _write_json(os.path.join(artifact_root, 'training_summary.json'), summary)

    return model
