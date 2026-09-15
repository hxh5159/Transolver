import yaml, json
import datetime
import shlex
import sys
import random
import torch
import utils.metrics as metrics
from dataset.dataset import Dataset
import os.path as osp
import argparse
import numpy as np
import os
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--my_path', default='/data/path', type=str)  # data save path
parser.add_argument('--save_path', default='./', type=str)  # model save path
parser.add_argument('--experiment_dir', default=None, type=str)
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--task', default='full', type=str)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Compute the normalization used for the training

use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

data_root_dir = args.my_path
ckpt_root_dir = args.save_path

tasks = [args.task]
script_dir = osp.dirname(osp.abspath(__file__))

for task in tasks:
    print('Generating results for task ' + task + '...')
    # task = 'full' # Choose between 'full', 'scarce', 'reynolds', and 'aoa'
    s = task + '_test' if task != 'scarce' else 'full_test'
    s_train = task + '_train'

    data_dir = osp.join(data_root_dir, 'Dataset')
    with open(osp.join(data_dir, 'manifest.json'), 'r') as f:
        manifest = json.load(f)

    manifest_train = manifest[s_train]
    n = int(.1 * len(manifest_train))
    train_dataset = manifest_train[:-n]

    _, coef_norm = Dataset(train_dataset, norm=True, sample=None, my_path=data_dir)

    # Compute the scores on the test set

    model_names = ['Transolver']
    models = []
    hparams = []

    for model in model_names:
        if args.model_path:
            model_path = osp.abspath(args.model_path)
        elif args.experiment_dir:
            model_path = osp.join(osp.abspath(args.experiment_dir), 'ensemble_full.pth')
        else:
            model_path = osp.join(ckpt_root_dir, 'metrics', task, model, model)
        mod = torch.load(model_path, map_location='cpu', weights_only=False)
        print(mod)
        if not isinstance(mod, (list, tuple)):
            mod = [mod]
        mod = [m.to(device) for m in mod]
        models.append(mod)

        with open(osp.join(script_dir, 'params.yaml'), 'r') as f:
            hparam = yaml.safe_load(f)[model]
            hparams.append(hparam)

    results_dir = (osp.join(osp.abspath(args.experiment_dir), 'evaluation')
                   if args.experiment_dir else osp.join(ckpt_root_dir, 'scores', task))
    os.makedirs(results_dir, exist_ok=True)
    with open(osp.join(results_dir, 'evaluation_config.json'), 'w') as file:
        json.dump({
            'started_at': datetime.datetime.now().astimezone().isoformat(),
            'command': shlex.join([sys.executable] + sys.argv),
            'arguments': vars(args),
            'model_path': model_path,
            'task': task,
        }, file, indent=2)
    coefs = metrics.Results_test(device, models, hparams, coef_norm, data_dir, results_dir, n_test=3, criterion='MSE',
                                 s=s)
    # models can be a stack of the same model (for example MLP) on the task s, if you have another stack of another model (for example GraphSAGE)
    # you can put in model argument [models_MLP, models_GraphSAGE] and it will output the results for both models (mean and std) in an ordered array.

    np.save(osp.join(results_dir, 'true_coefs'), coefs[0])
    np.save(osp.join(results_dir, 'pred_coefs_mean'), coefs[1])
    np.save(osp.join(results_dir, 'pred_coefs_std'), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(osp.join(results_dir, 'true_surf_coefs_' + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join(results_dir, 'surf_coefs_' + str(n)), file)
    np.save(osp.join(results_dir, 'true_bls'), coefs[5])
    np.save(osp.join(results_dir, 'bls'), coefs[6])
    with open(osp.join(results_dir, 'score.json'), 'r') as file:
        score = json.load(file)
    with open(osp.join(results_dir, 'evaluation_metrics.json'), 'w') as file:
        json.dump({
            'status': 'completed',
            'finished_at': datetime.datetime.now().astimezone().isoformat(),
            'task': task,
            'model_path': model_path,
            'test_split': s,
            'score': score,
        }, file, indent=2)
