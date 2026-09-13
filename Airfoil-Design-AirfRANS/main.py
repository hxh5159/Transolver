import argparse
import datetime
import json
import os
import os.path as osp
import time
import traceback

import numpy as np
import torch
import yaml

import train
import utils.metrics as metrics
from dataset.dataset import Dataset
from utils.experiment import (
    atomic_save,
    initialize_run,
    prepare_experiment,
    set_seed,
    update_config,
    update_status,
    write_json,
)


SCRIPT_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model: MLP, GraphSAGE, PointNet, GUNet or Transolver', type=str)
parser.add_argument('-n', '--nmodel', default=1, type=int)
parser.add_argument('-w', '--weight', default=1, type=float)
parser.add_argument('-t', '--task', default='full', type=str)
parser.add_argument('-s', '--score', default=0, type=int)
parser.add_argument('--my_path', default='/data/path', type=str)
parser.add_argument('--save_path', default=None, type=str, help=argparse.SUPPRESS)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--output_root', default=osp.join(SCRIPT_DIR, 'output'))
parser.add_argument('--experiment_dir', default=None)
parser.add_argument('--checkpoint_interval', default=100, type=int)
parser.add_argument('--resume', default=None)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--deterministic', action='store_true')
args = parser.parse_args()


def _infer_experiment_dir(checkpoint_path):
    current = osp.dirname(osp.abspath(checkpoint_path))
    while current != osp.dirname(current):
        if osp.isfile(osp.join(current, 'config.json')):
            return current
        current = osp.dirname(current)
    return None


if args.resume and args.nmodel != 1:
    raise ValueError('--resume currently requires --nmodel=1')

experiment_dir = args.experiment_dir
if experiment_dir is None and args.resume:
    experiment_dir = _infer_experiment_dir(args.resume)
paths = prepare_experiment(args.save_path or args.output_root, f'{args.task}_{args.model}', experiment_dir)
config, started_at = initialize_run(paths, args)
run_start = time.perf_counter()
set_seed(args.seed, deterministic=args.deterministic)

try:
    with open(osp.join(args.my_path, 'manifest.json'), 'r') as file:
        manifest = json.load(file)

    manifest_train = manifest[args.task + '_train']
    test_dataset = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
    n_validation = int(.1 * len(manifest_train))
    train_names = manifest_train[:-n_validation]
    val_names = manifest_train[-n_validation:]
    print('start load data')
    train_dataset, coef_norm = Dataset(train_names, norm=True, sample=None, my_path=args.my_path)
    val_dataset = Dataset(val_names, sample=None, coef_norm=coef_norm, my_path=args.my_path)
    print('load data finish')

    use_cuda = torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    print('Using GPU' if use_cuda else 'Using CPU')

    with open(osp.join(SCRIPT_DIR, 'params.yaml'), 'r') as file:
        hparams = yaml.safe_load(file)[args.model]
    model_kwargs = ({
        'n_hidden': 256, 'n_layers': 8, 'space_dim': 7, 'fun_dim': 0,
        'n_head': 8, 'mlp_ratio': 2, 'out_dim': 4, 'slice_num': 32,
        'unified_pos': 1,
    } if args.model == 'Transolver' else None)

    config.update({
        'dataset': {
            'train_samples': len(train_dataset),
            'validation_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'normalization': coef_norm,
        },
        'hparams': hparams,
        'model': {'name': args.model, 'kwargs': model_kwargs},
        'device': str(device),
    })
    update_config(paths, config)

    from models.MLP import MLP

    models = []
    member_summaries = []
    for index in range(args.nmodel):
        member_seed = args.seed + index
        set_seed(member_seed, deterministic=args.deterministic)

        if args.model == 'Transolver':
            from models.Transolver import Transolver
            model = Transolver(**model_kwargs).to(device)
        else:
            encoder = MLP(hparams['encoder'], batch_norm=False)
            decoder = MLP(hparams['decoder'], batch_norm=False)
            if args.model == 'GraphSAGE':
                from models.GraphSAGE import GraphSAGE
                model = GraphSAGE(hparams, encoder, decoder)
            elif args.model == 'PointNet':
                from models.PointNet import PointNet
                model = PointNet(hparams, encoder, decoder)
            elif args.model == 'MLP':
                from models.NN import NN
                model = NN(hparams, encoder, decoder)
            elif args.model == 'GUNet':
                from models.GUNet import GUNet
                model = GUNet(hparams, encoder, decoder)

        member_name = f'model_{index:03d}'
        member_paths = {
            'root': osp.join(paths['root'], 'models', member_name),
            'checkpoints': osp.join(paths['checkpoints'], member_name),
            'logs': osp.join(paths['logs'], member_name),
            'visualizations': osp.join(paths['visualizations'], member_name),
            'evaluation': paths['evaluation'],
        }
        for member_path in member_paths.values():
            os.makedirs(member_path, exist_ok=True)
        member_started_at = datetime.datetime.now().astimezone().isoformat()
        run_metadata = {
            'experiment_id': osp.basename(paths['root']),
            'model_index': index,
            'member_seed': member_seed,
            'started_at': member_started_at,
            'hparams': hparams,
            'model_kwargs': model_kwargs,
            'coef_norm': coef_norm,
        }
        print(f'start training model {index + 1}/{args.nmodel}')
        model = train.main(
            device, train_dataset, val_dataset, model, hparams, member_paths['root'],
            criterion='MSE_weighted', val_iter=10, reg=args.weight,
            name_mod=args.model, val_sample=True, artifact_paths=member_paths,
            checkpoint_interval=args.checkpoint_interval,
            resume_path=args.resume if index == 0 else None,
            run_metadata=run_metadata, model_index=index)
        print(f'end training model {index + 1}/{args.nmodel}')
        models.append(model)
        with open(osp.join(member_paths['root'], 'training_summary.json'), 'r') as file:
            member_summaries.append(json.load(file))

    ensemble_full_path = osp.join(paths['root'], 'ensemble_full.pth')
    ensemble_state_path = osp.join(paths['root'], 'ensemble_state_dict.pth')
    atomic_save(models, ensemble_full_path)
    atomic_save([model.state_dict() for model in models], ensemble_state_path)

    # Member summaries contain epoch-level timing.  Aggregate those values at
    # the experiment level as well, so the single top-level summary is useful
    # for both single-model and ensemble runs.
    member_epoch_counts = [int(summary.get('epochs_completed', 0))
                           for summary in member_summaries]
    member_training_seconds = [float(summary.get('time_elapsed_seconds', 0.0) or 0.0)
                               for summary in member_summaries]
    total_epoch_count = sum(member_epoch_counts)
    total_training_seconds = sum(member_training_seconds)
    mean_epoch_seconds = (total_training_seconds / total_epoch_count
                          if total_epoch_count else None)
    wall_clock_seconds = float(time.perf_counter() - run_start)

    evaluation_outputs = None
    if bool(args.score):
        print('start score')
        split = args.task + '_test' if args.task != 'scarce' else 'full_test'
        coefs = metrics.Results_test(
            device, [models], [hparams], coef_norm, args.my_path,
            path_out=paths['evaluation'], n_test=3, criterion='MSE', s=split)
        evaluation_outputs = {
            'true_coefs': osp.join(paths['evaluation'], 'true_coefs.npy'),
            'pred_coefs_mean': osp.join(paths['evaluation'], 'pred_coefs_mean.npy'),
            'pred_coefs_std': osp.join(paths['evaluation'], 'pred_coefs_std.npy'),
            'true_bls': osp.join(paths['evaluation'], 'true_bls.npy'),
            'bls': osp.join(paths['evaluation'], 'bls.npy'),
        }
        np.save(evaluation_outputs['true_coefs'], coefs[0])
        np.save(evaluation_outputs['pred_coefs_mean'], coefs[1])
        np.save(evaluation_outputs['pred_coefs_std'], coefs[2])
        for n, file in enumerate(coefs[3]):
            np.save(osp.join(paths['evaluation'], 'true_surf_coefs_' + str(n)), file)
        for n, file in enumerate(coefs[4]):
            np.save(osp.join(paths['evaluation'], 'surf_coefs_' + str(n)), file)
        np.save(evaluation_outputs['true_bls'], coefs[5])
        np.save(evaluation_outputs['bls'], coefs[6])
        print('end score')

    summary = {
        'status': 'completed',
        'started_at': started_at,
        'finished_at': datetime.datetime.now().astimezone().isoformat(),
        'wall_clock_seconds': wall_clock_seconds,
        'time_elapsed_seconds': wall_clock_seconds,
        'model_name': args.model,
        'task': args.task,
        'model_count': args.nmodel,
        'members': member_summaries,
        'training_seconds': total_training_seconds,
        'mean_epoch_seconds': mean_epoch_seconds,
        'ensemble_full': ensemble_full_path,
        'ensemble_state_dict': ensemble_state_path,
        'evaluation_outputs': evaluation_outputs,
    }
    write_json(osp.join(paths['root'], 'training_summary.json'), summary)
    update_status(paths, 'completed', started_at,
                  summary=osp.join(paths['root'], 'training_summary.json'))
    print('experiment outputs:', paths['root'])
except Exception as exc:
    update_status(paths, 'failed', started_at, error=repr(exc),
                  traceback=traceback.format_exc())
    raise
