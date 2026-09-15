import os
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import traceback
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer
from utils.experiment import (
    finalize_training, infer_experiment_dir, initialize, initialize_evaluation, load_model_state,
    prepare_experiment, record_epoch, restore_training, save_checkpoint, set_seed,
    update_config, update_status, write_json,
)
from utils.visualization import (
    create_scalar_training_visualization, finalize_scalar_evaluation,
    relative_l2_per_sample,
)

parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_1D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='elas_Transolver')
parser.add_argument('--data_path', type=str, default='/data/fno')
parser.add_argument('--output_root', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
parser.add_argument('--experiment_dir', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--checkpoint_interval', type=int, default=100)
parser.add_argument('--visualization_interval', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--deterministic', action='store_true')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name
artifact_paths = prepare_experiment(
    args.output_root, 'elas_' + save_name,
    args.experiment_dir or infer_experiment_dir(args.resume))
if eval:
    started_at = initialize_evaluation(artifact_paths, args, {'benchmark': 'elasticity'})
else:
    run_config, started_at = initialize(artifact_paths, args, {'benchmark': 'elasticity'})
set_seed(args.seed, args.deterministic)
result_dir = artifact_paths['visualizations']

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    ntrain = args.ntrain
    ntest = 200

    PATH_Sigma = args.data_path + '/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
    PATH_XY = args.data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'

    input_s = np.load(PATH_Sigma)
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0)
    input_xy = np.load(PATH_XY)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]

    print(input_s.shape, input_xy.shape)

    y_normalizer = UnitTransformer(train_s)

    train_s = y_normalizer.encode(train_s)
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_xy, train_xy, train_s),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_xy, test_xy, test_s),
                                              batch_size=args.batch_size,
                                              shuffle=False)

    print("Dataloading is over.")
    if not eval:
        run_config['dataset'] = {
            'files': [PATH_Sigma, PATH_XY],
            'train_samples': ntrain, 'test_samples': ntest,
            'points_per_sample': int(train_xy.shape[1]),
        }
        update_config(artifact_paths, run_config)

    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=0,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos).cuda()

    if not eval:
        run_config['model_runtime'] = {
            'space_dim': 2, 'n_layers': args.n_layers, 'n_hidden': args.n_hidden,
            'dropout': args.dropout, 'n_head': args.n_heads, 'Time_Input': False,
            'mlp_ratio': args.mlp_ratio, 'fun_dim': 0, 'out_dim': 1,
            'slice_num': args.slice_num, 'ref': args.ref,
            'unified_pos': args.unified_pos,
        }
        update_config(artifact_paths, run_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    myloss = TestLoss(size_average=False)
    start_epoch, history = restore_training(
        args.resume, model, optimizer, scheduler, device='cpu') if not eval else (0, [])

    if eval:
        checkpoint_path = (args.model_path or
                           (os.path.join(artifact_paths['checkpoints'], 'checkpoint_final.pth')
                            if args.experiment_dir else os.path.join('./checkpoints', save_name + '.pt')))
        load_model_state(checkpoint_path, model)
        model.eval()
        os.makedirs(result_dir, exist_ok=True)
        rel_err = 0.0
        sample_errors = []
        latencies_ms = []
        evaluation_cases = []
        showcase = 10
        id = 0

        with torch.no_grad():
            for pos, fx, y in test_loader:
                id += 1
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if x.is_cuda:
                    torch.cuda.synchronize(x.device)
                forward_started = time.perf_counter()
                out = model(x, None).squeeze(-1)
                if x.is_cuda:
                    torch.cuda.synchronize(x.device)
                batch_latency_ms = (time.perf_counter() - forward_started) * 1000.0 / x.shape[0]
                out = y_normalizer.decode(out)
                tl = myloss(out, y).item()
                rel_err += tl
                batch_errors = relative_l2_per_sample(out, y)
                sample_errors.extend(batch_errors.tolist())
                latencies_ms.extend([batch_latency_ms] * x.shape[0])
                for sample in range(x.shape[0]):
                    evaluation_cases.append({
                        'coordinates': x[sample].detach().cpu().numpy(),
                        'prediction': out[sample].detach().cpu().numpy(),
                        'target': y[sample].detach().cpu().numpy(),
                    })
                if id < showcase:
                    print(id)
                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(), y=fx[0, :, 1].detach().cpu().numpy(),
                                c=y[0, :].detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1000)
                    plt.savefig(
                        os.path.join(result_dir,
                                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(), y=fx[0, :, 1].detach().cpu().numpy(),
                                c=out[0, :].detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1000)
                    plt.savefig(
                        os.path.join(result_dir,
                                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(), y=fx[0, :, 1].detach().cpu().numpy(),
                                c=((y[0, :] - out[0, :])).detach().cpu().numpy(), cmap='coolwarm')
                    plt.clim(-8, 8)
                    plt.colorbar()
                    plt.savefig(
                        os.path.join(result_dir,
                                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

        rel_err /= ntest
        print("rel_err : {}".format(rel_err))
        visualization_metrics = finalize_scalar_evaluation(
            artifact_paths['evaluation'], sample_errors, latencies_ms, evaluation_cases,
            field_name='Material stress')
        write_json(os.path.join(artifact_paths['evaluation'], 'evaluation_metrics.json'), {
            'status': 'completed', 'benchmark': 'elasticity',
            'checkpoint': checkpoint_path, 'relative_l2': float(rel_err),
            'test_samples': ntest, **visualization_metrics,
        })
    else:
        for ep in range(start_epoch, args.epochs):
            epoch_started = time.perf_counter()

            model.train()
            train_loss = 0

            for pos, fx, y in train_loader:

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()  # x:B,N,2  fx:B,N,2  y:B,N,
                optimizer.zero_grad()
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                loss = myloss(out, y)
                loss.backward()

                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            train_loss = train_loss / ntrain
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

            model.eval()
            rel_err = 0.0
            periodic_case = None
            with torch.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                    out = model(x, None).squeeze(-1)
                    out = y_normalizer.decode(out)
                    if periodic_case is None:
                        periodic_case = (
                            x[0].detach().cpu().numpy(),
                            out[0].detach().cpu().numpy(),
                            y[0].detach().cpu().numpy())
                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest
            print("rel_err : {}".format(rel_err))

            record_epoch(artifact_paths, history, ep + 1, epoch_started,
                         train_relative_l2=float(train_loss), test_relative_l2=float(rel_err))

            if (args.visualization_interval > 0
                    and (ep + 1) % args.visualization_interval == 0
                    and periodic_case is not None):
                create_scalar_training_visualization(
                    artifact_paths, history, ep + 1, args.checkpoint_interval,
                    periodic_case[0], periodic_case[1], periodic_case[2],
                    field_name='Material stress')

            if args.checkpoint_interval > 0 and (ep + 1) % args.checkpoint_interval == 0:
                print('save model')
                save_checkpoint(
                    artifact_paths, ep + 1, model, optimizer, scheduler, history,
                    metadata=vars(args), extra={'normalizer': {
                        'y_mean': y_normalizer.mean, 'y_std': y_normalizer.std}})

        print('save model')
        finalize_training(
            artifact_paths, args.epochs, model, optimizer, scheduler, history,
            vars(args), started_at, args.epochs, ntrain, ntest,
            extra={'final_train_relative_l2': float(train_loss),
                   'final_test_relative_l2': float(rel_err)},
            checkpoint_extra={'normalizer': {
                'y_mean': y_normalizer.mean, 'y_std': y_normalizer.std}})
        print('experiment outputs:', artifact_paths['root'])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if not eval:
            update_status(artifact_paths, 'failed', started_at, error=repr(exc),
                          traceback=traceback.format_exc())
        raise
