import os
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import traceback

parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_2D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='pipe_UniPDE')
parser.add_argument('--data_path', type=str, default='/data/fno/pipe')
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

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
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

artifact_paths = prepare_experiment(
    args.output_root, 'pipe_' + save_name,
    args.experiment_dir or infer_experiment_dir(args.resume))
if eval:
    started_at = initialize_evaluation(artifact_paths, args, {'benchmark': 'pipe'})
else:
    run_config, started_at = initialize(artifact_paths, args, {'benchmark': 'pipe'})
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


def main():
    INPUT_X = args.data_path + '/Pipe_X.npy'
    INPUT_Y = args.data_path + '/Pipe_Y.npy'
    OUTPUT_Sigma = args.data_path + '/Pipe_Q.npy'

    ntrain = 1000
    ntest = 200
    N = 1200

    r1 = args.downsamplex
    r2 = args.downsampley
    s1 = int(((129 - 1) / r1) + 1)
    s2 = int(((129 - 1) / r2) + 1)

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 0]
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)
    x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    x_train = x_train.reshape(ntrain, -1, 2)
    x_test = x_test.reshape(ntest, -1, 2)
    y_train = y_train.reshape(ntrain, -1)
    y_test = y_test.reshape(ntest, -1)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, x_train, y_train),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, y_test),
                                              batch_size=args.batch_size,
                                              shuffle=False)

    print("Dataloading is over.")
    if not eval:
        run_config['dataset'] = {
            'files': [INPUT_X, INPUT_Y, OUTPUT_Sigma],
            'train_samples': ntrain, 'test_samples': ntest,
            'resolution': [s1, s2],
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
                                  unified_pos=args.unified_pos,
                                  H=s1, W=s2).cuda()

    if not eval:
        run_config['model_runtime'] = {
            'space_dim': 2, 'n_layers': args.n_layers, 'n_hidden': args.n_hidden,
            'dropout': args.dropout, 'n_head': args.n_heads, 'Time_Input': False,
            'mlp_ratio': args.mlp_ratio, 'fun_dim': 0, 'out_dim': 1,
            'slice_num': args.slice_num, 'ref': args.ref,
            'unified_pos': args.unified_pos, 'H': s1, 'W': s2,
        }
        update_config(artifact_paths, run_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)
    start_epoch, history = restore_training(
        args.resume, model, optimizer, scheduler, device='cpu') if not eval else (0, [])

    if eval:
        checkpoint_path = (args.model_path or
                           (os.path.join(artifact_paths['checkpoints'], 'checkpoint_final.pth')
                            if args.experiment_dir else os.path.join('./checkpoints', save_name + '.pt')))
        load_model_state(checkpoint_path, model)
        torch.save(model.state_dict(), os.path.join(artifact_paths['evaluation'], 'loaded_state_dict.pth'))
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
                physical_coordinates = x_normalizer.decode(x)
                for sample in range(x.shape[0]):
                    evaluation_cases.append({
                        'coordinates': physical_coordinates[sample].detach().cpu().numpy(),
                        'prediction': out[sample].detach().cpu().numpy(),
                        'target': y[sample].detach().cpu().numpy(),
                    })

                if id < showcase:
                    print(id)
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(129, 129).detach().cpu().numpy(),
                                   x[0, :, 1].reshape(129, 129).detach().cpu().numpy(),
                                   np.zeros([129, 129]),
                                   shading='auto',
                                   edgecolors='black', linewidths=0.1)
                    plt.colorbar()
                    plt.savefig(
                        os.path.join(result_dir,
                                     "input_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(129, 129).detach().cpu().numpy(),
                                   x[0, :, 1].reshape(129, 129).detach().cpu().numpy(),
                                   out[0, :].reshape(129, 129).detach().cpu().numpy(),
                                   shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 0.3)
                    plt.savefig(
                        os.path.join(result_dir,
                                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(129, 129).detach().cpu().numpy(),
                                   x[0, :, 1].reshape(129, 129).detach().cpu().numpy(),
                                   y[0, :].reshape(129, 129).detach().cpu().numpy(),
                                   shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 0.3)
                    plt.savefig(
                        os.path.join(result_dir,
                                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(129, 129).detach().cpu().numpy(),
                                   x[0, :, 1].reshape(129, 129).detach().cpu().numpy(),
                                   out[0, :].reshape(129, 129).detach().cpu().numpy() - \
                                   y[0, :].reshape(129, 129).detach().cpu().numpy(),
                                   shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-0.02, 0.02)
                    plt.savefig(
                        os.path.join(result_dir,
                                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))
        visualization_metrics = finalize_scalar_evaluation(
            artifact_paths['evaluation'], sample_errors, latencies_ms, evaluation_cases,
            grid_shape=(s1, s2), field_name='Pipe-flow velocity', show_mesh=True)
        write_json(os.path.join(artifact_paths['evaluation'], 'evaluation_metrics.json'), {
            'status': 'completed', 'benchmark': 'pipe',
            'checkpoint': checkpoint_path, 'relative_l2': float(rel_err),
            'test_samples': ntest, **visualization_metrics,
        })
    else:
        for ep in range(start_epoch, args.epochs):
            epoch_started = time.perf_counter()

            model.train()
            train_loss = 0

            for pos, fx, y in train_loader:

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()  # x:B,N,2  fx:B,N,2  y:B,N
                optimizer.zero_grad()
                out = model(x, None).squeeze(-1)

                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                loss = myloss(out, y)
                loss.backward()

                # print("loss:{}".format(loss.item()/batch_size))
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
                            x_normalizer.decode(x)[0].detach().cpu().numpy(),
                            out[0].detach().cpu().numpy(),
                            y[0].detach().cpu().numpy())

                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest
            print("rel_err:{}".format(rel_err))

            record_epoch(artifact_paths, history, ep + 1, epoch_started,
                         train_relative_l2=float(train_loss), test_relative_l2=float(rel_err))

            if (args.visualization_interval > 0
                    and (ep + 1) % args.visualization_interval == 0
                    and periodic_case is not None):
                create_scalar_training_visualization(
                    artifact_paths, history, ep + 1, args.checkpoint_interval,
                    periodic_case[0], periodic_case[1], periodic_case[2],
                    grid_shape=(s1, s2), field_name='Pipe-flow velocity', show_mesh=True)

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
