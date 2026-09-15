"""Headless visualizations for AirfRANS training and evaluation."""
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


VARIABLES = (('u_x', 'Velocity x'), ('u_y', 'Velocity y'), ('p', 'Pressure'),
             ('nut', 'Turbulent viscosity'))


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _range(values, symmetric=False):
    values = _array(values).astype(float, copy=False).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return (-1.0, 1.0)
    if symmetric:
        bound = max(float(np.percentile(np.abs(values), 99)), 1e-12)
        return (-bound, bound)
    low, high = np.percentile(values, (1, 99))
    if not high > low:
        margin = max(abs(float(low)) * 0.05, 1e-12)
        low, high = low - margin, high + margin
    return float(low), float(high)


def _scatter(axis, points, values, limits, cmap):
    marker_size = max(1.0, min(10.0, 24000.0 / max(len(points), 1)))
    artist = axis.scatter(
        points[:, 0], points[:, 1], c=values, s=marker_size, linewidths=0,
        cmap=cmap, norm=colors.Normalize(vmin=limits[0], vmax=limits[1]),
        rasterized=True)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('x / chord')
    axis.set_ylabel('y / chord')
    return artist


def plot_fields(output_dir, points, prediction, target, surface_mask, *,
                sample_name=None, metrics=None):
    os.makedirs(output_dir, exist_ok=True)
    points = _array(points).reshape(-1, 2)
    prediction = _array(prediction)
    target = _array(target)
    surface_mask = _array(surface_mask).astype(bool).reshape(-1)
    figure, axes = plt.subplots(4, 3, figsize=(13.8, 15.0), constrained_layout=True)
    for row, (_, label) in enumerate(VARIABLES):
        truth = target[:, row]
        estimate = prediction[:, row]
        residual = estimate - truth
        shared = _range(np.concatenate((truth, estimate)))
        error_limits = _range(residual, symmetric=True)
        for column, (title, values, cmap, limits) in enumerate((
                (f'Ground truth: {label}', truth, 'viridis', shared),
                (f'Prediction: {label}', estimate, 'viridis', shared),
                (f'Error: {label}', residual, 'coolwarm', error_limits))):
            artist = _scatter(axes[row, column], points, values, limits, cmap)
            if surface_mask.any():
                axes[row, column].scatter(
                    points[surface_mask, 0], points[surface_mask, 1], s=0.4,
                    color='black', alpha=0.5, rasterized=True)
            axes[row, column].set_title(title)
            figure.colorbar(artist, ax=axes[row, column], shrink=0.8)
    title = 'AirfRANS physical fields' if sample_name is None else f'AirfRANS: {sample_name}'
    if metrics and metrics.get('field_relative_l2') is not None:
        title += f" | field relative L2={metrics['field_relative_l2']:.5f}"
    figure.suptitle(title)
    path = os.path.join(output_dir, 'field_comparison.png')
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_streamlines(output_dir, points, prediction, target, surface_mask):
    from scipy.interpolate import griddata

    os.makedirs(output_dir, exist_ok=True)
    points = _array(points).reshape(-1, 2)
    prediction = _array(prediction)
    target = _array(target)
    surface_mask = _array(surface_mask).astype(bool).reshape(-1)
    low = np.percentile(points, 1, axis=0)
    high = np.percentile(points, 99, axis=0)
    grid_x, grid_y = np.meshgrid(
        np.linspace(low[0], high[0], 220), np.linspace(low[1], high[1], 120))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True,
                               sharex=True, sharey=True)
    speed_limits = _range(np.concatenate((
        np.linalg.norm(target[:, :2], axis=1),
        np.linalg.norm(prediction[:, :2], axis=1))))
    for axis, velocity, title in zip(
            axes, (target[:, :2], prediction[:, :2]), ('Ground truth', 'Prediction')):
        u = griddata(points, velocity[:, 0], (grid_x, grid_y), method='linear')
        v = griddata(points, velocity[:, 1], (grid_x, grid_y), method='linear')
        speed = np.sqrt(u ** 2 + v ** 2)
        stream = axis.streamplot(
            grid_x[0], grid_y[:, 0], u, v, color=speed, cmap='viridis',
            norm=colors.Normalize(vmin=speed_limits[0], vmax=speed_limits[1]),
            density=1.25, linewidth=0.75, arrowsize=0.8)
        if surface_mask.any():
            axis.scatter(points[surface_mask, 0], points[surface_mask, 1],
                         s=2, color='black', zorder=5)
        axis.set_aspect('equal', adjustable='box')
        axis.set_title(title)
        axis.set_xlabel('x / chord')
        figure.colorbar(stream.lines, ax=axis, label='Velocity magnitude')
    axes[0].set_ylabel('y / chord')
    figure.suptitle('AirfRANS velocity streamlines')
    path = os.path.join(output_dir, 'velocity_streamlines.png')
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_surface_quantities(output_dir, true_airfoil, predicted_airfoil):
    os.makedirs(output_dir, exist_ok=True)
    true_points = _array(true_airfoil.points)
    predicted_points = _array(predicted_airfoil.points)
    true_pressure = _array(true_airfoil.point_data['p']).reshape(-1)
    predicted_pressure = _array(predicted_airfoil.point_data['p']).reshape(-1)
    true_wss = np.linalg.norm(_array(true_airfoil.point_data['wallShearStress']), axis=1)
    predicted_wss = np.linalg.norm(
        _array(predicted_airfoil.point_data['wallShearStress']), axis=1)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    for row, (name, truth, estimate) in enumerate((
            ('Surface pressure', true_pressure, predicted_pressure),
            ('Wall-shear-stress magnitude', true_wss, predicted_wss))):
        axes[row, 0].scatter(true_points[:, 0], truth, s=9, color='#222222',
                             label='Ground truth', alpha=0.8)
        axes[row, 0].scatter(predicted_points[:, 0], estimate, s=7, color='#d17c2f',
                             label='Prediction', alpha=0.7)
        axes[row, 0].set_xlabel('x / chord')
        axes[row, 0].set_ylabel(name)
        axes[row, 0].set_title(f'{name} along airfoil surface')
        axes[row, 0].legend()
        residual = estimate - truth
        axes[row, 1].axhline(0, color='#333333', linestyle='--', linewidth=1)
        axes[row, 1].scatter(true_points[:, 0], residual, s=8, color='#b45f4c', alpha=0.75)
        axes[row, 1].set_xlabel('x / chord')
        axes[row, 1].set_ylabel('Prediction - ground truth')
        axes[row, 1].set_title(f'{name} residual')
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle('AirfRANS airfoil-surface quantities')
    path = os.path.join(output_dir, 'surface_quantities.png')
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_boundary_layers(output_dir, true_profiles, predicted_profiles, x_locations):
    os.makedirs(output_dir, exist_ok=True)
    figure, axes = plt.subplots(
        len(x_locations), 3, figsize=(12.5, 3.4 * len(x_locations)),
        constrained_layout=True, squeeze=False)
    quantities = (('u / U_inf', 1), ('v / U_inf', 2), ('nu_t / nu', 3))
    for row, (x_location, truth, estimate) in enumerate(
            zip(x_locations, true_profiles, predicted_profiles)):
        for column, (label, value_index) in enumerate(quantities):
            axes[row, column].plot(
                truth[value_index], truth[0], color='#222222', linewidth=1.5,
                label='Ground truth')
            axes[row, column].plot(
                estimate[value_index], estimate[0], color='#d17c2f', linewidth=1.3,
                label='Prediction')
            axes[row, column].set_xlabel(label)
            axes[row, column].set_ylabel('(y - y0) / chord')
            axes[row, column].set_title(f'Boundary layer at x/c={x_location:g}')
            axes[row, column].grid(alpha=0.2)
            axes[row, column].legend()
    figure.suptitle('AirfRANS boundary-layer profiles')
    path = os.path.join(output_dir, 'boundary_layers.png')
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_training_history(output_path, history, checkpoint_interval=None):
    if not history:
        return None
    epochs = np.asarray([row['epoch'] for row in history])
    groups = (
        ('Total loss', 'train_loss', 'validation_loss'),
        ('Surface loss', 'train_surface_loss', 'validation_surface_loss'),
        ('Volume loss', 'train_volume_loss', 'validation_volume_loss'),
    )
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True,
                               constrained_layout=True)
    for axis, (title, train_key, validation_key) in zip(axes, groups):
        train = np.asarray([row.get(train_key, np.nan) for row in history], dtype=float)
        valid_entries = [(row['epoch'], row.get(validation_key)) for row in history
                         if row.get(validation_key) is not None]
        axis.plot(epochs, train, color='#1769aa', linewidth=1.4, label='Train')
        if valid_entries:
            val_epochs, val_values = zip(*valid_entries)
            axis.plot(val_epochs, val_values, color='#c43c35', marker='o',
                      markersize=3, linewidth=1.1, label='Validation')
        if np.any(np.isfinite(train) & (train > 0)):
            axis.set_yscale('log')
        if checkpoint_interval:
            for epoch in range(checkpoint_interval, int(epochs[-1]) + 1,
                               checkpoint_interval):
                axis.axvline(epoch, color='#777777', alpha=0.22, linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel('MSE')
        axis.grid(alpha=0.2)
        axis.legend()
    axes[-1].set_xlabel('Epoch')
    figure.suptitle('AirfRANS training history')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_error_distribution(output_path, records):
    field = np.asarray([row['field_relative_l2'] for row in records], dtype=float)
    force = np.asarray([row['force_relative_error_mean'] for row in records], dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for axis, values, label, color in (
            (axes[0], field, 'Full-field relative L2', '#356aa0'),
            (axes[1], force, 'Mean Cd/Cl relative error', '#b45f4c')):
        bins = min(25, max(5, int(np.sqrt(values.size))))
        axis.hist(values, bins=bins, color=color, edgecolor='white', alpha=0.85)
        axis.axvline(np.median(values), color='#333333', linestyle='--',
                     label=f'median={np.median(values):.4f}')
        axis.set_xlabel(label)
        axis.set_ylabel('Number of test samples')
        axis.legend()
        axis.grid(alpha=0.2)
    axes[2].scatter(field, force, s=22, color='#207567', alpha=0.75,
                    edgecolor='white', linewidth=0.3)
    axes[2].set_xlabel('Full-field relative L2')
    axes[2].set_ylabel('Mean Cd/Cl relative error')
    axes[2].grid(alpha=0.2)
    figure.suptitle('AirfRANS test error distributions')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_force_evaluation(output_path, true_coefs, predicted_coefs):
    true_coefs = _array(true_coefs).reshape(-1, 2)
    predicted_coefs = _array(predicted_coefs).reshape(-1, 2)
    figure, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    for column, (index, name) in enumerate(((0, 'Cd'), (1, 'Cl'))):
        truth = true_coefs[:, index]
        estimate = predicted_coefs[:, index]
        residual = estimate - truth
        limits = _range(np.concatenate((truth, estimate)))
        margin = max((limits[1] - limits[0]) * 0.05, 1e-8)
        plot_limits = (limits[0] - margin, limits[1] + margin)
        axes[0, column].scatter(truth, estimate, s=24, color='#356aa0', alpha=0.75)
        axes[0, column].plot(plot_limits, plot_limits, 'k--', linewidth=1)
        axes[0, column].set_xlim(plot_limits)
        axes[0, column].set_ylim(plot_limits)
        axes[0, column].set_aspect('equal', adjustable='box')
        axes[0, column].set_xlabel(f'Ground-truth {name}')
        axes[0, column].set_ylabel(f'Predicted {name}')
        axes[0, column].set_title(f'{name} agreement')
        order = np.argsort(truth)
        rank = np.arange(1, truth.size + 1)
        axes[1, column].plot(rank, truth[order], color='#222222', linewidth=1.5,
                             label='Ground truth')
        axes[1, column].plot(rank, estimate[order], color='#d17c2f', linewidth=1.2,
                             marker='o', markersize=2.3, label='Prediction')
        axes[1, column].set_xlabel(f'Test-sample rank by ground-truth {name}')
        axes[1, column].set_ylabel(name)
        axes[1, column].set_title(f'{name} ranking response')
        axes[1, column].legend()
        axes[2, column].axhline(0, color='#333333', linestyle='--', linewidth=1)
        axes[2, column].scatter(truth, residual, s=24, color='#b45f4c', alpha=0.75)
        axes[2, column].set_xlabel(f'Ground-truth {name}')
        axes[2, column].set_ylabel('Prediction - ground truth')
        axes[2, column].set_title(f'{name} residuals')
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle('AirfRANS aerodynamic-coefficient evaluation')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_latency_distribution(output_path, values_ms):
    values = _array(values_ms).astype(float, copy=False).reshape(-1)
    ordered = np.sort(values)
    cumulative = np.arange(1, values.size + 1) / values.size
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].hist(values, bins=min(25, max(5, int(np.sqrt(values.size)))),
                 color='#75507b', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('Inference time [ms/sample]')
    axes[0].set_ylabel('Number of test samples')
    axes[1].step(ordered, cumulative, where='post', color='#207567', linewidth=1.8)
    axes[1].set_xlabel('Inference time [ms/sample]')
    axes[1].set_ylabel('Empirical cumulative probability')
    axes[1].text(
        0.98, 0.04, f'mean={values.mean():.3f} ms\nmedian={np.median(values):.3f} ms\n'
        f'P95={np.percentile(values, 95):.3f} ms', transform=axes[1].transAxes,
        ha='right', va='bottom', fontsize=9,
        bbox={'facecolor': 'white', 'edgecolor': '#aaaaaa', 'alpha': 0.9})
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle('AirfRANS model inference time')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_per_sample_metrics(path, records):
    if not records:
        return
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_metrics(output_dir, payload):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as file:
        json.dump(payload, file, indent=2, allow_nan=False)
