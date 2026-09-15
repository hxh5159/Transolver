import os
import argparse
import numpy as np
import scipy.io as scio
import torch
import time
import traceback
import torch.nn.functional as F
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from model_dict import get_model
from utils.normalizer import UnitTransformer
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.experiment import (
    finalize_training, infer_experiment_dir, initialize, initialize_evaluation, load_model_state,
    prepare_experiment, record_epoch, restore_training, save_checkpoint, set_seed,
    update_config, update_status, write_json,
)
from utils.visualization import (
    create_scalar_training_visualization, finalize_scalar_evaluation,
    relative_l2_per_sample,
)

parser = argparse.ArgumentParser('Training Transolver')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_2D')
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
parser.add_argument('--save_name', type=str, default='darcy_Transolver')
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

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
ntrain = args.ntrain
ntest = 200
epochs = args.epochs
eval = args.eval
save_name = args.save_name
artifact_paths = prepare_experiment(
    args.output_root, 'darcy_' + save_name,
    args.experiment_dir or infer_experiment_dir(args.resume))
if eval:
    started_at = initialize_evaluation(artifact_paths, args, {'benchmark': 'darcy'})
else:
    run_config, started_at = initialize(artifact_paths, args, {'benchmark': 'darcy'})
set_seed(args.seed, args.deterministic)
result_dir = artifact_paths['visualizations']


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def central_diff(x: torch.Tensor, h, resolution):
    # assuming PBC
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x,
              (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y


def main():
    r = args.downsample
    h = int(((421 - 1) / r) + 1)
    s = h
    dx = 1.0 / s

    train_data = scio.loadmat(train_path)
    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
    x_train = x_train.reshape(ntrain, -1)
    x_train = torch.from_numpy(x_train).float()
    y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
    y_train = y_train.reshape(ntrain, -1)
    y_train = torch.from_numpy(y_train)

    test_data = scio.loadmat(test_path)
    x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
    x_test = x_test.reshape(ntest, -1)
    x_test = torch.from_numpy(x_test).float()
    y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
    y_test = y_test.reshape(ntest, -1)
    y_test = torch.from_numpy(y_test)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    y_normalizer.cuda()

    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    print("Dataloading is over.")
    if not eval:
        run_config['dataset'] = {
            'train_file': train_path, 'test_file': test_path,
            'train_samples': ntrain, 'test_samples': ntest,
            'resolution': [s, s],
        }
        update_config(artifact_paths, run_config)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)

    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=1,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  H=s, W=s).cuda()

    if not eval:
        run_config['model_runtime'] = {
            'space_dim': 2, 'n_layers': args.n_layers, 'n_hidden': args.n_hidden,
            'dropout': args.dropout, 'n_head': args.n_heads, 'Time_Input': False,
            'mlp_ratio': args.mlp_ratio, 'fun_dim': 1, 'out_dim': 1,
            'slice_num': args.slice_num, 'ref': args.ref,
            'unified_pos': args.unified_pos, 'H': s, 'W': s,
        }
        update_config(artifact_paths, run_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)
    de_x = TestLoss(size_average=False)
    de_y = TestLoss(size_average=False)
    start_epoch, history = restore_training(
        args.resume, model, optimizer, scheduler, device='cpu') if not eval else (0, [])

    if eval:
        print("model evaluation")
        print(s, s)
        checkpoint_path = (args.model_path or
                           (os.path.join(artifact_paths['checkpoints'], 'checkpoint_final.pth')
                            if args.experiment_dir else os.path.join('./checkpoints', save_name + '.pt')))
        load_model_state(checkpoint_path, model)
        model.eval()
        showcase = 10
        id = 0
        os.makedirs(result_dir, exist_ok=True)
        sample_errors = []
        latencies_ms = []
        evaluation_cases = []

        with torch.no_grad():
            rel_err = 0.0
            with torch.no_grad():
                for x, fx, y in test_loader:
                    id += 1
                    x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                    if x.is_cuda:
                        torch.cuda.synchronize(x.device)
                    forward_started = time.perf_counter()
                    out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                    if x.is_cuda:
                        torch.cuda.synchronize(x.device)
                    batch_latency_ms = (time.perf_counter() - forward_started) * 1000.0 / x.shape[0]
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()

                    rel_err += tl
                    batch_errors = relative_l2_per_sample(out, y)
                    sample_errors.extend(batch_errors.tolist())
                    latencies_ms.extend([batch_latency_ms] * x.shape[0])
                    physical_input = x_normalizer.decode(fx)
                    for sample in range(x.shape[0]):
                        evaluation_cases.append({
                            'coordinates': x[sample].detach().cpu().numpy(),
                            'input_field': physical_input[sample].detach().cpu().numpy(),
                            'prediction': out[sample].detach().cpu().numpy(),
                            'target': y[sample].detach().cpu().numpy(),
                        })

                    if id < showcase:
                        print(id)
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(out[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(
                            os.path.join(result_dir,
                                         "case_" + str(id) + "_pred.pdf"))
                        plt.close()
                        # ============ #
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(y[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(
                            os.path.join(result_dir, "case_" + str(id) + "_gt.pdf"))
                        plt.close()
                        # ============ #
                        plt.figure()
                        plt.axis('off')
                        plt.imshow((y[0, :] - out[0, :]).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.clim(-0.0005, 0.0005)
                        plt.savefig(
                            os.path.join(result_dir, "case_" + str(id) + "_error.pdf"))
                        plt.close()
                        # ============ #
                        plt.figure()
                        plt.axis('off')
                        plt.imshow((fx[0, :].unsqueeze(-1)).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(
                            os.path.join(result_dir, "case_" + str(id) + "_input.pdf"))
                        plt.close()

            rel_err /= ntest
            print("rel_err:{}".format(rel_err))
            visualization_metrics = finalize_scalar_evaluation(
                artifact_paths['evaluation'], sample_errors, latencies_ms, evaluation_cases,
                grid_shape=(s, s), field_name='Darcy pressure', input_name='Permeability')
            write_json(os.path.join(artifact_paths['evaluation'], 'evaluation_metrics.json'), {
                'status': 'completed', 'benchmark': 'darcy',
                'checkpoint': checkpoint_path, 'relative_l2': float(rel_err),
                'test_samples': ntest, **visualization_metrics,
            })
    else:
        for ep in range(start_epoch, args.epochs):
            epoch_started = time.perf_counter()
            model.train()
            train_loss = 0
            reg = 0
            for x, fx, y in train_loader:
                x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                optimizer.zero_grad()

                out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)  # B, N , 2, fx: B, N, y: B, N
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                l2loss = myloss(out, y)

                out = rearrange(out.unsqueeze(-1), 'b (h w) c -> b c h w', h=s)
                out = out[..., 1:-1, 1:-1].contiguous()
                out = F.pad(out, (1, 1, 1, 1), "constant", 0)
                out = rearrange(out, 'b c h w -> b (h w) c')
                gt_grad_x, gt_grad_y = central_diff(y.unsqueeze(-1), dx, s)
                pred_grad_x, pred_grad_y = central_diff(out, dx, s)
                deriv_loss = de_x(pred_grad_x, gt_grad_x) + de_y(pred_grad_y, gt_grad_y)
                loss = 0.1 * deriv_loss + l2loss
                loss.backward()

                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_loss += l2loss.item()
                reg += deriv_loss.item()
                scheduler.step()

            train_loss /= ntrain
            reg /= ntrain
            print("Epoch {} Reg : {:.5f} Train loss : {:.5f}".format(ep, reg, train_loss))

            model.eval()
            rel_err = 0.0
            id = 0
            periodic_case = None
            with torch.no_grad():
                for x, fx, y in test_loader:
                    id += 1
                    if id == 2:
                        vis = True
                    else:
                        vis = False
                    x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                    out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                    out = y_normalizer.decode(out)
                    if periodic_case is None:
                        periodic_case = (
                            x[0].detach().cpu().numpy(),
                            x_normalizer.decode(fx)[0].detach().cpu().numpy(),
                            out[0].detach().cpu().numpy(),
                            y[0].detach().cpu().numpy())
                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest
            print("rel_err:{}".format(rel_err))

            record_epoch(artifact_paths, history, ep + 1, epoch_started,
                         train_relative_l2=float(train_loss),
                         derivative_loss=float(reg), test_relative_l2=float(rel_err))

            if (args.visualization_interval > 0
                    and (ep + 1) % args.visualization_interval == 0
                    and periodic_case is not None):
                create_scalar_training_visualization(
                    artifact_paths, history, ep + 1, args.checkpoint_interval,
                    periodic_case[0], periodic_case[2], periodic_case[3],
                    input_field=periodic_case[1], grid_shape=(s, s),
                    field_name='Darcy pressure', input_name='Permeability')

            if args.checkpoint_interval > 0 and (ep + 1) % args.checkpoint_interval == 0:
                print('save model')
                save_checkpoint(
                    artifact_paths, ep + 1, model, optimizer, scheduler, history,
                    metadata=vars(args), extra={'normalizers': {
                        'x_mean': x_normalizer.mean, 'x_std': x_normalizer.std,
                        'y_mean': y_normalizer.mean, 'y_std': y_normalizer.std}})

        print('save model')
        finalize_training(
            artifact_paths, args.epochs, model, optimizer, scheduler, history,
            vars(args), started_at, args.epochs, ntrain, ntest,
            extra={'final_train_relative_l2': float(train_loss),
                   'final_derivative_loss': float(reg),
                   'final_test_relative_l2': float(rel_err)},
            checkpoint_extra={'normalizers': {
                'x_mean': x_normalizer.mean, 'x_std': x_normalizer.std,
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
