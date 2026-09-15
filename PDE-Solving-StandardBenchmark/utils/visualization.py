"""Headless, task-aware visualizations for the standard PDE benchmarks."""
import csv
from contextlib import contextmanager
import json
import os
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite(values):
    values = _array(values).astype(float, copy=False).reshape(-1)
    return values[np.isfinite(values)]


def _range(values, symmetric=False, include_zero=False):
    values = _finite(values)
    if not values.size:
        return (-1.0, 1.0)
    if symmetric:
        bound = float(np.percentile(np.abs(values), 99.0))
        bound = max(bound, float(np.max(np.abs(values))) * 1e-6, 1e-12)
        return (-bound, bound)
    low, high = np.percentile(values, (1.0, 99.0))
    if include_zero:
        low, high = min(float(low), 0.0), max(float(high), 0.0)
    if not high > low:
        margin = max(abs(float(low)) * 0.05, 1e-12)
        low, high = float(low) - margin, float(high) + margin
    return float(low), float(high)


def _draw_field(axis, coordinates, values, limits, grid_shape=None, cmap='viridis'):
    coordinates = _array(coordinates).reshape(-1, 2)
    values = _array(values).reshape(-1)
    norm = colors.Normalize(vmin=limits[0], vmax=limits[1])
    if grid_shape is not None and int(np.prod(grid_shape)) == values.size:
        rows, columns = grid_shape
        artist = axis.pcolormesh(
            coordinates[:, 0].reshape(rows, columns),
            coordinates[:, 1].reshape(rows, columns),
            values.reshape(rows, columns), shading='auto', cmap=cmap, norm=norm)
    else:
        marker_size = max(2.0, min(14.0, 18000.0 / max(values.size, 1)))
        artist = axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=values, s=marker_size,
            linewidths=0, cmap=cmap, norm=norm)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    return artist


def relative_l2_per_sample(prediction, target):
    prediction = _array(prediction)
    target = _array(target)
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    numerator = np.linalg.norm(prediction - target, axis=1)
    denominator = np.maximum(np.linalg.norm(target, axis=1), 1e-12)
    return numerator / denominator


def representative_indices(errors):
    errors = _array(errors).astype(float, copy=False).reshape(-1)
    if not errors.size:
        return {}
    order = np.argsort(errors)
    selected = {
        'best': int(order[0]),
        'median': int(order[len(order) // 2]),
        'worst': int(order[-1]),
    }
    # Tiny evaluation sets can map two labels to the same sample; keep the
    # labels because they still describe the selection rule unambiguously.
    return selected


def _summary(values):
    values = _finite(values)
    if not values.size:
        return {'mean': None, 'median': None, 'p95': None, 'max': None}
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
        'max': float(np.max(values)),
    }


def plot_scalar_case(output_dir, coordinates, prediction, target, *,
                     input_field=None, grid_shape=None, field_name='Solution',
                     input_name='Input coefficient', sample_index=None,
                     relative_l2=None, show_mesh=False):
    os.makedirs(output_dir, exist_ok=True)
    coordinates = _array(coordinates).reshape(-1, 2)
    prediction = _array(prediction).reshape(-1)
    target = _array(target).reshape(-1)
    error = prediction - target
    panels = []
    if show_mesh:
        panels.append(('Computational mesh', None, None, None))
    if input_field is not None:
        panels.append((input_name, _array(input_field).reshape(-1), 'viridis',
                       _range(input_field)))
    shared = _range(np.concatenate((target, prediction)))
    panels.extend([
        (f'Ground truth: {field_name}', target, 'viridis', shared),
        (f'Prediction: {field_name}', prediction, 'viridis', shared),
        ('Prediction - ground truth', error, 'coolwarm', _range(error, symmetric=True)),
    ])
    figure, axes = plt.subplots(
        1, len(panels), figsize=(4.6 * len(panels), 4.2),
        constrained_layout=True, squeeze=False)
    for axis, (title, values, cmap, limits) in zip(axes[0], panels):
        if values is None:
            if grid_shape is not None and int(np.prod(grid_shape)) == coordinates.shape[0]:
                rows, columns = grid_shape
                axis.pcolormesh(
                    coordinates[:, 0].reshape(rows, columns),
                    coordinates[:, 1].reshape(rows, columns),
                    np.zeros(grid_shape), shading='auto', facecolor='none',
                    edgecolors='#555555', linewidth=0.12)
            else:
                axis.scatter(coordinates[:, 0], coordinates[:, 1], s=2,
                             color='#333333', linewidths=0)
            axis.set_aspect('equal', adjustable='box')
            axis.set_xlabel('x')
            axis.set_ylabel('y')
            artist = None
        else:
            artist = _draw_field(axis, coordinates, values, limits, grid_shape, cmap)
        axis.set_title(title)
        if artist is not None:
            figure.colorbar(artist, ax=axis, shrink=0.82)
    suffix = '' if relative_l2 is None else f' | relative L2={relative_l2:.5f}'
    label = 'PDE field comparison' if sample_index is None else f'Test sample {sample_index}'
    figure.suptitle(label + suffix)
    output_path = os.path.join(output_dir, 'sample_fields.png')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_temporal_scalar_case(output_dir, coordinates, prediction, target, *,
                              grid_shape, field_name, sample_index=None,
                              relative_l2=None):
    os.makedirs(output_dir, exist_ok=True)
    coordinates = _array(coordinates).reshape(-1, 2)
    prediction = _array(prediction)
    target = _array(target)
    steps = sorted(set((0, target.shape[-1] // 2, target.shape[-1] - 1)))
    figure, axes = plt.subplots(
        len(steps), 3, figsize=(13.5, 4.0 * len(steps)), constrained_layout=True,
        squeeze=False)
    for row, step in enumerate(steps):
        truth = target[..., step].reshape(-1)
        estimate = prediction[..., step].reshape(-1)
        residual = estimate - truth
        shared = _range(np.concatenate((truth, estimate)))
        error_range = _range(residual, symmetric=True)
        for column, (title, values, cmap, limits) in enumerate((
                (f'Ground truth: t={step + 1}', truth, 'viridis', shared),
                (f'Prediction: t={step + 1}', estimate, 'viridis', shared),
                (f'Error: t={step + 1}', residual, 'coolwarm', error_range))):
            artist = _draw_field(
                axes[row, column], coordinates, values, limits, grid_shape, cmap)
            axes[row, column].set_title(title)
            figure.colorbar(artist, ax=axes[row, column], shrink=0.82)
    suffix = '' if relative_l2 is None else f' | full relative L2={relative_l2:.5f}'
    label = field_name if sample_index is None else f'{field_name}, test sample {sample_index}'
    figure.suptitle(label + suffix)
    output_path = os.path.join(output_dir, 'temporal_fields.png')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_deformation_case(output_dir, prediction, target, *, sample_index=None,
                          relative_l2=None):
    os.makedirs(output_dir, exist_ok=True)
    prediction = _array(prediction)
    target = _array(target)
    steps = sorted(set((0, target.shape[-1] // 2, target.shape[-1] - 1)))
    figure, axes = plt.subplots(
        len(steps), 3, figsize=(13.5, 4.0 * len(steps)), constrained_layout=True,
        squeeze=False)
    for row, step in enumerate(steps):
        truth_xy = target[:, :2, step]
        pred_xy = prediction[:, :2, step]
        truth_mag = np.linalg.norm(target[:, 2:, step], axis=1)
        pred_mag = np.linalg.norm(prediction[:, 2:, step], axis=1)
        residual = pred_mag - truth_mag
        shared = _range(np.concatenate((truth_mag, pred_mag)), include_zero=True)
        for column, (title, xy, values, cmap, limits) in enumerate((
                (f'Ground truth deformation: t={step + 1}', truth_xy, truth_mag,
                 'viridis', shared),
                (f'Predicted deformation: t={step + 1}', pred_xy, pred_mag,
                 'viridis', shared),
                (f'Magnitude error: t={step + 1}', truth_xy, residual,
                 'coolwarm', _range(residual, symmetric=True)))):
            artist = _draw_field(axes[row, column], xy, values, limits, None, cmap)
            axes[row, column].set_title(title)
            figure.colorbar(artist, ax=axes[row, column], shrink=0.82)
    suffix = '' if relative_l2 is None else f' | full relative L2={relative_l2:.5f}'
    label = 'Plasticity deformation' if sample_index is None else f'Plasticity test sample {sample_index}'
    figure.suptitle(label + suffix)
    output_path = os.path.join(output_dir, 'deformation_sequence.png')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_error_distribution(output_path, errors, title='Test relative L2 errors'):
    errors = _finite(errors)
    if not errors.size:
        return None
    ordered = np.sort(errors)
    cumulative = np.arange(1, errors.size + 1) / errors.size
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    bins = min(25, max(5, int(np.sqrt(errors.size))))
    axes[0].hist(errors, bins=bins, color='#356aa0', edgecolor='white', alpha=0.85)
    axes[0].axvline(np.median(errors), color='#333333', linestyle='--',
                    label=f'median={np.median(errors):.5f}')
    axes[0].set_xlabel('Per-sample relative L2')
    axes[0].set_ylabel('Number of test samples')
    axes[0].legend()
    axes[1].step(ordered, cumulative, where='post', color='#207567', linewidth=1.8)
    axes[1].set_xlabel('Per-sample relative L2')
    axes[1].set_ylabel('Empirical cumulative probability')
    axes[1].text(
        0.98, 0.04,
        f'mean={np.mean(errors):.5f}\nmedian={np.median(errors):.5f}\n'
        f'P95={np.percentile(errors, 95):.5f}\nmax={np.max(errors):.5f}',
        transform=axes[1].transAxes, ha='right', va='bottom', fontsize=9,
        bbox={'facecolor': 'white', 'edgecolor': '#aaaaaa', 'alpha': 0.9})
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_latency_distribution(output_path, latencies_ms):
    latencies_ms = _finite(latencies_ms)
    if not latencies_ms.size:
        return None
    ordered = np.sort(latencies_ms)
    cumulative = np.arange(1, latencies_ms.size + 1) / latencies_ms.size
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    bins = min(25, max(5, int(np.sqrt(latencies_ms.size))))
    axes[0].hist(latencies_ms, bins=bins, color='#b45f4c', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('Model inference latency [ms/sample]')
    axes[0].set_ylabel('Number of test samples')
    axes[1].step(ordered, cumulative, where='post', color='#207567', linewidth=1.8)
    axes[1].set_xlabel('Model inference latency [ms/sample]')
    axes[1].set_ylabel('Empirical cumulative probability')
    axes[1].text(
        0.98, 0.04,
        f'mean={np.mean(latencies_ms):.3f} ms\nmedian={np.median(latencies_ms):.3f} ms\n'
        f'P95={np.percentile(latencies_ms, 95):.3f} ms\nmax={np.max(latencies_ms):.3f} ms',
        transform=axes[1].transAxes, ha='right', va='bottom', fontsize=9,
        bbox={'facecolor': 'white', 'edgecolor': '#aaaaaa', 'alpha': 0.9})
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle('Model inference latency (data transfer and plotting excluded)')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def write_per_sample_metrics(path, errors, extra_columns=None):
    errors = _array(errors).astype(float, copy=False).reshape(-1)
    extra_columns = extra_columns or {}
    fields = ['sample_index', 'relative_l2', *extra_columns]
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, error in enumerate(errors):
            row = {'sample_index': index, 'relative_l2': float(error)}
            for name, values in extra_columns.items():
                row[name] = float(_array(values).reshape(-1)[index])
            writer.writerow(row)


def write_case_metrics(output_dir, payload):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as file:
        json.dump(payload, file, indent=2, allow_nan=False)


@contextmanager
def visualization_guard(output_dir, payload):
    """Keep optional plotting failures from interrupting model training."""
    try:
        yield payload
    except Exception as exc:
        payload['visualization_error'] = repr(exc)
        warnings.warn(f'Visualization failed for {output_dir}: {exc}')
    finally:
        write_case_metrics(output_dir, payload)


def create_scalar_training_visualization(paths, history, epoch, checkpoint_interval,
                                         coordinates, prediction, target, **plot_kwargs):
    from utils.experiment import plot_training_history

    output_dir = os.path.join(paths['visualizations'], f'epoch_{epoch:04d}')
    payload = {'epoch': epoch, 'sample_index': 0}
    with visualization_guard(output_dir, payload):
        plot_training_history(
            paths, history, checkpoint_interval=checkpoint_interval,
            output_path=os.path.join(output_dir, 'learning_curves.png'))
        relative_l2 = float(relative_l2_per_sample(
            _array(prediction)[None], _array(target)[None])[0])
        payload['relative_l2'] = relative_l2
        plot_scalar_case(
            output_dir, coordinates, prediction, target, sample_index=0,
            relative_l2=relative_l2, **plot_kwargs)


def create_temporal_training_visualization(paths, history, epoch, checkpoint_interval,
                                           coordinates, prediction, target, **plot_kwargs):
    from utils.experiment import plot_training_history

    output_dir = os.path.join(paths['visualizations'], f'epoch_{epoch:04d}')
    payload = {'epoch': epoch, 'sample_index': 0}
    with visualization_guard(output_dir, payload):
        plot_training_history(
            paths, history, checkpoint_interval=checkpoint_interval,
            output_path=os.path.join(output_dir, 'learning_curves.png'))
        relative_l2 = float(relative_l2_per_sample(
            _array(prediction)[None], _array(target)[None])[0])
        payload['full_relative_l2'] = relative_l2
        plot_temporal_scalar_case(
            output_dir, coordinates, prediction, target, sample_index=0,
            relative_l2=relative_l2, **plot_kwargs)


def create_deformation_training_visualization(paths, history, epoch,
                                              checkpoint_interval, prediction, target):
    from utils.experiment import plot_training_history

    output_dir = os.path.join(paths['visualizations'], f'epoch_{epoch:04d}')
    payload = {'epoch': epoch, 'sample_index': 0}
    with visualization_guard(output_dir, payload):
        plot_training_history(
            paths, history, checkpoint_interval=checkpoint_interval,
            output_path=os.path.join(output_dir, 'learning_curves.png'))
        relative_l2 = float(relative_l2_per_sample(
            _array(prediction)[None], _array(target)[None])[0])
        payload['full_relative_l2'] = relative_l2
        plot_deformation_case(
            output_dir, prediction, target, sample_index=0,
            relative_l2=relative_l2)


def finalize_scalar_evaluation(evaluation_dir, errors, latencies_ms, cases, *,
                               grid_shape=None, field_name='Solution',
                               input_name='Input coefficient', show_mesh=False):
    os.makedirs(evaluation_dir, exist_ok=True)
    cases_dir = os.path.join(evaluation_dir, 'cases')
    os.makedirs(cases_dir, exist_ok=True)
    errors = _array(errors).astype(float, copy=False).reshape(-1)
    plot_error_distribution(os.path.join(evaluation_dir, 'error_distribution.png'), errors)
    plot_latency_distribution(
        os.path.join(evaluation_dir, 'inference_time_distribution.png'), latencies_ms)
    write_per_sample_metrics(
        os.path.join(evaluation_dir, 'per_sample_metrics.csv'), errors)
    selected = representative_indices(errors)
    for label, index in selected.items():
        case = cases[index]
        output_dir = os.path.join(cases_dir, f'{label}_{index}')
        plot_scalar_case(
            output_dir, case['coordinates'], case['prediction'], case['target'],
            input_field=case.get('input_field'), grid_shape=grid_shape,
            field_name=field_name, input_name=input_name, sample_index=index,
            relative_l2=float(errors[index]), show_mesh=show_mesh)
        write_case_metrics(output_dir, {
            'selection': label, 'sample_index': index,
            'relative_l2': float(errors[index]),
        })
    return {
        'representative_cases': selected,
        'relative_l2_distribution': _summary(errors),
        'model_forward_latency_ms': _summary(latencies_ms),
    }


def finalize_temporal_evaluation(evaluation_dir, errors, latencies_ms, cases, *,
                                 grid_shape, field_name):
    os.makedirs(evaluation_dir, exist_ok=True)
    cases_dir = os.path.join(evaluation_dir, 'cases')
    os.makedirs(cases_dir, exist_ok=True)
    errors = _array(errors).astype(float, copy=False).reshape(-1)
    plot_error_distribution(os.path.join(evaluation_dir, 'error_distribution.png'), errors)
    plot_latency_distribution(
        os.path.join(evaluation_dir, 'inference_time_distribution.png'), latencies_ms)
    write_per_sample_metrics(
        os.path.join(evaluation_dir, 'per_sample_metrics.csv'), errors)
    selected = representative_indices(errors)
    for label, index in selected.items():
        case = cases[index]
        output_dir = os.path.join(cases_dir, f'{label}_{index}')
        plot_temporal_scalar_case(
            output_dir, case['coordinates'], case['prediction'], case['target'],
            grid_shape=grid_shape, field_name=field_name, sample_index=index,
            relative_l2=float(errors[index]))
        write_case_metrics(output_dir, {
            'selection': label, 'sample_index': index,
            'relative_l2': float(errors[index]),
        })
    return {
        'representative_cases': selected,
        'relative_l2_distribution': _summary(errors),
        'rollout_latency_ms': _summary(latencies_ms),
    }


def finalize_deformation_evaluation(evaluation_dir, errors, latencies_ms, cases):
    os.makedirs(evaluation_dir, exist_ok=True)
    cases_dir = os.path.join(evaluation_dir, 'cases')
    os.makedirs(cases_dir, exist_ok=True)
    errors = _array(errors).astype(float, copy=False).reshape(-1)
    plot_error_distribution(os.path.join(evaluation_dir, 'error_distribution.png'), errors)
    plot_latency_distribution(
        os.path.join(evaluation_dir, 'inference_time_distribution.png'), latencies_ms)
    write_per_sample_metrics(
        os.path.join(evaluation_dir, 'per_sample_metrics.csv'), errors)
    selected = representative_indices(errors)
    for label, index in selected.items():
        case = cases[index]
        output_dir = os.path.join(cases_dir, f'{label}_{index}')
        plot_deformation_case(
            output_dir, case['prediction'], case['target'], sample_index=index,
            relative_l2=float(errors[index]))
        write_case_metrics(output_dir, {
            'selection': label, 'sample_index': index,
            'relative_l2': float(errors[index]),
        })
    return {
        'representative_cases': selected,
        'relative_l2_distribution': _summary(errors),
        'rollout_latency_ms': _summary(latencies_ms),
    }
