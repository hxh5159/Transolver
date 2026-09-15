import random
import warnings
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import time, json

import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from pathlib import Path
import os.path as osp

from utils.experiment import (
    save_checkpoint,
    save_history,
    save_model_files,
    summarize,
)
from utils.visualization import (
    plot_fields, plot_streamlines, plot_training_history, write_metrics,
)


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, criterion='MSE', reg=1):
    model.train()
    avg_loss_per_var = torch.zeros(4, device=device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device=device)
    avg_loss_vol_var = torch.zeros(4, device=device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0

    for data in train_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        optimizer.zero_grad()
        out = model(data_clone)
        targets = data_clone.y

        if criterion == 'MSE' or criterion == 'MSE_weighted':
            loss_criterion = nn.MSELoss(reduction='none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction='none')
        loss_per_var = loss_criterion(out, targets).mean(dim=0)
        total_loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim=0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim=0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        if criterion == 'MSE_weighted':
            (loss_vol + reg * loss_surf).backward()
        else:
            total_loss.backward()

        optimizer.step()
        scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol
        iter += 1

    return avg_loss.cpu().data.numpy() / iter, avg_loss_per_var.cpu().data.numpy() / iter, avg_loss_surf_var.cpu().data.numpy() / iter, avg_loss_vol_var.cpu().data.numpy() / iter, \
           avg_loss_surf.cpu().data.numpy() / iter, avg_loss_vol.cpu().data.numpy() / iter


@torch.no_grad()
def test(device, model, test_loader, criterion='MSE', collect_sample=False):
    model.eval()
    avg_loss_per_var = np.zeros(4)
    avg_loss = 0
    avg_loss_surf_var = np.zeros(4)
    avg_loss_vol_var = np.zeros(4)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0
    sample = None

    for data in test_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        out = model(data_clone)

        if collect_sample and sample is None:
            sample = {
                'points': data_clone.pos.detach().cpu().numpy(),
                'prediction': out.detach().cpu().numpy(),
                'target': data_clone.y.detach().cpu().numpy(),
                'surface_mask': data_clone.surf.detach().cpu().numpy(),
            }

        targets = data_clone.y
        if criterion == 'MSE' or 'MSE_weighted':
            loss_criterion = nn.MSELoss(reduction='none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction='none')

        loss_per_var = loss_criterion(out, targets).mean(dim=0)
        loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim=0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim=0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        avg_loss_per_var += loss_per_var.cpu().numpy()
        avg_loss += loss.cpu().numpy()
        avg_loss_surf_var += loss_surf_var.cpu().numpy()
        avg_loss_vol_var += loss_vol_var.cpu().numpy()
        avg_loss_surf += loss_surf.cpu().numpy()
        avg_loss_vol += loss_vol.cpu().numpy()
        iter += 1

    result = (avg_loss / iter, avg_loss_per_var / iter,
              avg_loss_surf_var / iter, avg_loss_vol_var / iter,
              avg_loss_surf / iter, avg_loss_vol / iter)
    return (*result, sample) if collect_sample else result


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path, criterion='MSE', reg=1, val_iter=10,
         name_mod='GraphSAGE', val_sample=True, artifact_paths=None,
         checkpoint_interval=100, visualization_interval=100,
         resume_path=None, run_metadata=None,
         model_index=0):
    '''
        Args:
        device (str): device on which you want to do the computation.
        train_dataset (list): list of the data in the training set.
        val_dataset (list): list of the data in the validation set.
        Net (class): network to train.
        hparams (dict): hyper parameters of the network.
        path (str): where to save the trained model and the figures.
        criterion (str, optional): chose between 'MSE', 'MAE', and 'MSE_weigthed'. The latter is the volumetric MSE plus the surface MSE computed independently. Default: 'MSE'.
        reg (float, optional): weigth for the surface loss when criterion is 'MSE_weighted'. Default: 1.
        val_iter (int, optional): number of epochs between each validation step. Default: 10.
        name_mod (str, optional): type of model. Default: 'GraphSAGE'.
    '''
    Path(path).mkdir(parents=True, exist_ok=True)
    if artifact_paths is None:
        artifact_paths = {
            'root': osp.abspath(path),
            'checkpoints': osp.join(path, 'checkpoints'),
            'logs': osp.join(path, 'logs'),
            'visualizations': osp.join(path, 'visualizations'),
            'evaluation': osp.join(path, 'evaluation'),
        }
        for artifact_path in artifact_paths.values():
            Path(artifact_path).mkdir(parents=True, exist_ok=True)

    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
    )
    val_loader = DataLoader(val_dataset, batch_size=1)
    start = time.perf_counter()
    training_started_at = (run_metadata or {}).get('started_at')

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []
    val_surf_list = []
    val_vol_list = []
    val_surf_var_list = []
    val_vol_var_list = []
    history = []
    start_epoch = 0
    val_loss = None
    val_surf = None
    val_vol = None

    if resume_path is not None:
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint.get('epoch', 0))
        history = checkpoint.get('history', [])
        train_loss_surf_list = checkpoint.get('train_loss_surf_list', [])
        train_loss_vol_list = checkpoint.get('train_loss_vol_list', [])
        loss_surf_var_list = checkpoint.get('loss_surf_var_list', [])
        loss_vol_var_list = checkpoint.get('loss_vol_var_list', [])
        val_surf_list = checkpoint.get('val_surf_list', [])
        val_vol_list = checkpoint.get('val_vol_list', [])
        val_surf_var_list = checkpoint.get('val_surf_var_list', [])
        val_vol_var_list = checkpoint.get('val_vol_var_list', [])
        rng = checkpoint.get('rng_state')
        if rng:
            if rng.get('python') is not None:
                random.setstate(rng['python'])
            if rng.get('numpy') is not None:
                np.random.set_state(rng['numpy'])
            if rng.get('torch') is not None:
                torch.set_rng_state(rng['torch'])
            if torch.cuda.is_available() and rng.get('cuda') is not None:
                torch.cuda.set_rng_state_all(rng['cuda'])

    pbar_train = tqdm(range(start_epoch, hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        epoch_start = time.perf_counter()
        visualization_due = (visualization_interval > 0
                             and (epoch + 1) % visualization_interval == 0)
        visualization_sample_data = None
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
            idx = torch.tensor(idx)

            data_sampled.pos = data_sampled.pos[idx]
            data_sampled.x = data_sampled.x[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]

            if name_mod != 'PointNet' and name_mod != 'MLP':
                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device), r=hparams['r'], loop=True,
                                                           max_num_neighbors=int(hparams['max_neighbors'])).cpu()

            train_dataset_sampled.append(data_sampled)
        train_loader = DataLoader(train_dataset_sampled, batch_size=hparams['batch_size'], shuffle=True)
        del (train_dataset_sampled)

        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train(device, model, train_loader, optimizer,
                                                                                lr_scheduler, criterion, reg=reg)
        print('epoch: ' + str(epoch))
        print('train_loss： ' + str(train_loss))
        print('loss_vol： ' + str(loss_vol))
        print('loss_surf： ' + str(loss_surf))

        if criterion == 'MSE_weighted':
            train_loss = reg * loss_surf + loss_vol
        del (train_loader)

        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)

        val_loss_value = None
        val_surface_value = None
        val_volume_value = None
        validation_start = time.perf_counter()
        if val_iter is not None:
            if epoch % val_iter == val_iter - 1 or epoch == 0:
                if val_sample:
                    val_surf_vars, val_vol_vars, val_surfs, val_vols = [], [], [], []
                    for i in range(20):
                        val_dataset_sampled = []
                        for data in val_dataset:
                            data_sampled = data.clone()
                            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
                            idx = torch.tensor(idx)

                            data_sampled.pos = data_sampled.pos[idx]
                            data_sampled.x = data_sampled.x[idx]
                            data_sampled.y = data_sampled.y[idx]
                            data_sampled.surf = data_sampled.surf[idx]

                            if name_mod != 'PointNet' and name_mod != 'MLP':
                                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device),
                                                                           r=hparams['r'], loop=True,
                                                                           max_num_neighbors=int(
                                                                               hparams['max_neighbors'])).cpu()

                                # if name_mod == 'GNO' or name_mod == 'MGNO':
                                #     x, edge_index = data_sampled.x, data_sampled.edge_index
                                #     x_i, x_j = x[edge_index[0], 0:2], x[edge_index[1], 0:2]
                                #     v_i, v_j = x[edge_index[0], 2:4], x[edge_index[1], 2:4]
                                #     p_i, p_j = x[edge_index[0], 4:5], x[edge_index[1], 4:5]
                                #     v_inf = torch.linalg.norm(v_i, dim = 1, keepdim = True)
                                #     sdf_i, sdf_j = x[edge_index[0], 5:6], x[edge_index[1], 5:6]
                                #     normal_i, normal_j = x[edge_index[0], 6:8], x[edge_index[1], 6:8]

                                #     data_sampled.edge_attr = torch.cat([x_i - x_j, v_i - v_j, p_i - p_j, sdf_i, sdf_j, v_inf, normal_i, normal_j], dim = 1)

                            val_dataset_sampled.append(data_sampled)
                        val_loader = DataLoader(val_dataset_sampled, batch_size=1, shuffle=True)
                        del (val_dataset_sampled)

                        if visualization_due and i == 0:
                            (val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol,
                             visualization_sample_data) = test(
                                device, model, val_loader, criterion, collect_sample=True)
                        else:
                            val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(
                                device, model, val_loader, criterion)
                        del (val_loader)
                        val_surf_vars.append(val_surf_var)
                        val_vol_vars.append(val_vol_var)
                        val_surfs.append(val_surf)
                        val_vols.append(val_vol)
                    val_surf_var = np.array(val_surf_vars).mean(axis=0)
                    val_vol_var = np.array(val_vol_vars).mean(axis=0)
                    val_surf = np.array(val_surfs).mean(axis=0)
                    val_vol = np.array(val_vols).mean(axis=0)
                else:
                    val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device, model, val_loader,
                                                                                     criterion)
                print("=====validation=====")
                print('epoch: ' + str(epoch))
                print('val_vol： ' + str(val_vol))
                print('val_surf： ' + str(val_surf))
                if criterion == 'MSE_weigthed':
                    val_loss = reg * val_surf + val_vol
                val_surf_list.append(val_surf)
                val_vol_list.append(val_vol)
                val_surf_var_list.append(val_surf_var)
                val_vol_var_list.append(val_vol_var)
                val_loss_value = float(val_loss)
                val_surface_value = float(val_surf)
                val_volume_value = float(val_vol)
                pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf, val_loss=val_loss, val_surf=val_surf)
            else:
                pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf, val_loss=val_loss, val_surf=val_surf)
        else:
            pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf)

        epoch_seconds = time.perf_counter() - epoch_start
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_surface_loss': float(loss_surf),
            'train_volume_loss': float(loss_vol),
            'validation_loss': val_loss_value,
            'validation_surface_loss': val_surface_value,
            'validation_volume_loss': val_volume_value,
            'epoch_seconds': float(epoch_seconds),
        })
        save_history(artifact_paths, history)
        if visualization_due and visualization_sample_data is not None:
            epoch_dir = osp.join(
                artifact_paths['visualizations'], f'epoch_{epoch + 1:04d}')
            Path(epoch_dir).mkdir(parents=True, exist_ok=True)
            plot_training_history(
                osp.join(epoch_dir, 'learning_curves.png'), history,
                checkpoint_interval=checkpoint_interval)
            output_norm = (run_metadata or {}).get('coef_norm')
            prediction = visualization_sample_data['prediction']
            target = visualization_sample_data['target']
            if output_norm is not None:
                prediction = prediction * (np.asarray(output_norm[3]) + 1e-8) + np.asarray(output_norm[2])
                target = target * (np.asarray(output_norm[3]) + 1e-8) + np.asarray(output_norm[2])
            visual_metrics = {
                'epoch': epoch + 1,
                'sample_index': 0,
                'field_relative_l2': float(
                    np.linalg.norm(prediction - target)
                    / max(np.linalg.norm(target), 1e-12)),
            }
            try:
                plot_fields(
                    epoch_dir, visualization_sample_data['points'], prediction, target,
                    visualization_sample_data['surface_mask'], sample_name='validation sample 0',
                    metrics=visual_metrics)
                plot_streamlines(
                    epoch_dir, visualization_sample_data['points'], prediction, target,
                    visualization_sample_data['surface_mask'])
            except Exception as exc:
                visual_metrics['visualization_error'] = repr(exc)
                warnings.warn(f'AirfRANS visualization at epoch {epoch + 1} failed: {exc}')
            write_metrics(epoch_dir, visual_metrics)
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                artifact_paths, epoch + 1, model, optimizer, lr_scheduler, history,
                metadata=run_metadata, extra={
                    'train_loss_surf_list': train_loss_surf_list,
                    'train_loss_vol_list': train_loss_vol_list,
                    'loss_surf_var_list': loss_surf_var_list,
                    'loss_vol_var_list': loss_vol_var_list,
                    'val_surf_list': val_surf_list,
                    'val_vol_list': val_vol_list,
                    'val_surf_var_list': val_surf_var_list,
                    'val_vol_var_list': val_vol_var_list,
                })

    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)
    val_surf_var_list = np.array(val_surf_var_list)
    val_vol_var_list = np.array(val_vol_var_list)

    time_elapsed = time.perf_counter() - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    final_model_path, final_state_path = save_model_files(artifact_paths, model, prefix=f'model_{model_index:03d}_final')
    # Keep a simple, stable name in each model's subdirectory for evaluation.
    torch.save(model, osp.join(path, 'model'))

    try:
        plot_training_history(
            osp.join(artifact_paths['visualizations'], 'final_training_loss.png'), history,
            checkpoint_interval=checkpoint_interval)
    except Exception as exc:
        warnings.warn(f'Final AirfRANS training-curve visualization failed: {exc}')

    sns.set()
    fig_train_surf, ax_train_surf = plt.subplots(figsize=(20, 5))
    ax_train_surf.plot(train_loss_surf_list, label='Mean loss')
    ax_train_surf.plot(loss_surf_var_list[:, 0], label=r'$v_x$ loss')
    ax_train_surf.plot(loss_surf_var_list[:, 1], label=r'$v_y$ loss')
    ax_train_surf.plot(loss_surf_var_list[:, 2], label=r'$p$ loss')
    ax_train_surf.plot(loss_surf_var_list[:, 3], label=r'$\nu_t$ loss')
    ax_train_surf.set_xlabel('epochs')
    ax_train_surf.set_yscale('log')
    ax_train_surf.set_title('Train losses over the surface')
    ax_train_surf.legend(loc='best')
    fig_train_surf.savefig(osp.join(artifact_paths['visualizations'], 'train_loss_surf.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_train_surf)

    fig_train_vol, ax_train_vol = plt.subplots(figsize=(20, 5))
    ax_train_vol.plot(train_loss_vol_list, label='Mean loss')
    ax_train_vol.plot(loss_vol_var_list[:, 0], label=r'$v_x$ loss')
    ax_train_vol.plot(loss_vol_var_list[:, 1], label=r'$v_y$ loss')
    ax_train_vol.plot(loss_vol_var_list[:, 2], label=r'$p$ loss')
    ax_train_vol.plot(loss_vol_var_list[:, 3], label=r'$\nu_t$ loss')
    ax_train_vol.set_xlabel('epochs')
    ax_train_vol.set_yscale('log')
    ax_train_vol.set_title('Train losses over the volume')
    ax_train_vol.legend(loc='best')
    fig_train_vol.savefig(osp.join(artifact_paths['visualizations'], 'train_loss_vol.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_train_vol)

    if val_iter is not None:
        fig_val_surf, ax_val_surf = plt.subplots(figsize=(20, 5))
        ax_val_surf.plot(val_surf_list, label='Mean loss')
        ax_val_surf.plot(val_surf_var_list[:, 0], label=r'$v_x$ loss')
        ax_val_surf.plot(val_surf_var_list[:, 1], label=r'$v_y$ loss')
        ax_val_surf.plot(val_surf_var_list[:, 2], label=r'$p$ loss')
        ax_val_surf.plot(val_surf_var_list[:, 3], label=r'$\nu_t$ loss')
        ax_val_surf.set_xlabel('epochs')
        ax_val_surf.set_yscale('log')
        ax_val_surf.set_title('Validation losses over the surface')
        ax_val_surf.legend(loc='best')
        fig_val_surf.savefig(osp.join(artifact_paths['visualizations'], 'val_loss_surf.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_val_surf)

        fig_val_vol, ax_val_vol = plt.subplots(figsize=(20, 5))
        ax_val_vol.plot(val_vol_list, label='Mean loss')
        ax_val_vol.plot(val_vol_var_list[:, 0], label=r'$v_x$ loss')
        ax_val_vol.plot(val_vol_var_list[:, 1], label=r'$v_y$ loss')
        ax_val_vol.plot(val_vol_var_list[:, 2], label=r'$p$ loss')
        ax_val_vol.plot(val_vol_var_list[:, 3], label=r'$\nu_t$ loss')
        ax_val_vol.set_xlabel('epochs')
        ax_val_vol.set_yscale('log')
        ax_val_vol.set_title('Validation losses over the volume')
        ax_val_vol.legend(loc='best')
        fig_val_vol.savefig(osp.join(artifact_paths['visualizations'], 'val_loss_vol.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_val_vol)

        if val_iter is not None:
            with open(osp.join(artifact_paths['logs'], name_mod + '_log.json'), 'w') as f:
                json.dump(
                    {
                        'regression': 'Total',
                        'loss': 'MSE',
                        'nb_parameters': params_model,
                        'time_elapsed': time_elapsed,
                        'hparams': hparams,
                        'train_loss_surf': str(train_loss_surf_list[-1]),
                        'train_loss_surf_var': str(loss_surf_var_list[-1]),
                        'train_loss_vol': str(train_loss_vol_list[-1]),
                        'train_loss_vol_var': str(loss_vol_var_list[-1]),
                        'val_loss_surf': str(val_surf_list[-1]),
                        'val_loss_surf_var': str(val_surf_var_list[-1]),
                        'val_loss_vol': str(val_vol_list[-1]),
                        'val_loss_vol_var': str(val_vol_var_list[-1]),
                    }, f, indent=12, cls=NumpyEncoder
                )

    final_checkpoint = save_checkpoint(
        artifact_paths, hparams['nb_epochs'], model, optimizer, lr_scheduler, history,
        metadata=run_metadata, filename='checkpoint_final.pth', extra={
            'train_loss_surf_list': train_loss_surf_list,
            'train_loss_vol_list': train_loss_vol_list,
            'loss_surf_var_list': loss_surf_var_list,
            'loss_vol_var_list': loss_vol_var_list,
            'val_surf_list': val_surf_list,
            'val_vol_list': val_vol_list,
            'val_surf_var_list': val_surf_var_list,
            'val_vol_var_list': val_vol_var_list,
        })
    summary = summarize(
        artifact_paths, history, (run_metadata or {}).get('started_at'), hparams['nb_epochs'],
        len(train_dataset), len(val_dataset), (final_model_path, final_state_path),
        extra={
            'model_index': model_index,
            'model_name': name_mod,
            'criterion': criterion,
            'surface_weight': reg,
            'nb_parameters': float(params_model),
            'final_checkpoint': final_checkpoint,
            'final_train_loss': float(train_loss),
            'final_validation_loss': None if val_loss is None else float(val_loss),
            'hparams': hparams,
            'resume_from': resume_path,
            'resume_start_epoch': start_epoch,
        }, training_started=training_started_at)
    return model
