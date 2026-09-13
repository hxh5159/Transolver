import json
import os
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/transolver-matplotlib-cache')

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import vtk
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from utils.drag_coefficient import cal_coefficient


FIELD_NAMES = ('u', 'v', 'w', 'p')
COORDINATE_NAMES = ('x', 'y', 'z')


def _load_grid(path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    grid = reader.GetOutput()
    if grid.GetNumberOfPoints() == 0:
        raise ValueError(f'VTK file contains no points: {path}')
    return grid


def _sample_files(data_dir, sample_path):
    sample_dir = os.path.join(data_dir, sample_path)
    return (
        os.path.join(sample_dir, 'quadpress_smpl.vtk'),
        os.path.join(sample_dir, 'hexvelo_smpl.vtk'),
    )


def _map_values(source_points, source_values, query_points):
    distances, indices = cKDTree(np.asarray(source_points)).query(np.asarray(query_points), k=1)
    scale = max(np.ptp(query_points, axis=0).max(), 1.0)
    if distances.max(initial=0.0) > scale * 1e-5:
        raise ValueError(
            f'Could not align graph nodes with VTK mesh (max distance={distances.max():.3e})'
        )
    return np.asarray(source_values)[indices]


def _iter_cells(grid):
    cell_data = vtk_to_numpy(grid.GetCells().GetData())
    offset = 0
    while offset < len(cell_data):
        size = int(cell_data[offset])
        yield cell_data[offset + 1:offset + size + 1]
        offset += size + 1


def _robust_range(values, lower=1.0, upper=99.0, include_zero=False):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [lower, upper])
    if include_zero:
        vmin, vmax = min(vmin, 0.0), max(vmax, 0.0)
    if np.isclose(vmin, vmax):
        delta = max(abs(float(vmin)) * 0.05, 1e-8)
        vmin, vmax = vmin - delta, vmax + delta
    return float(vmin), float(vmax)


def _value_norm(vmin, vmax):
    if vmin < 0 < vmax:
        return colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _set_3d_limits(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-8)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(spans)
    ax.set_axis_off()


def plot_surface_pressure(
        output_path, data_dir, sample_path, graph_points, surface_mask, prediction, target, cd_values=None):
    pressure_file, _ = _sample_files(data_dir, sample_path)
    grid = _load_grid(pressure_file)
    mesh_points = vtk_to_numpy(grid.GetPoints().GetData())

    graph_surface_points = graph_points[surface_mask]
    pred_pressure = _map_values(graph_surface_points, prediction[surface_mask, 3], mesh_points)
    true_pressure = _map_values(graph_surface_points, target[surface_mask, 3], mesh_points)
    cells = list(_iter_cells(grid))
    display_points = mesh_points[:, [2, 0, 1]]
    polygons = [display_points[cell] for cell in cells]
    pred_cells = np.array([pred_pressure[cell].mean() for cell in cells])
    true_cells = np.array([true_pressure[cell].mean() for cell in cells])
    error_cells = np.abs(pred_cells - true_cells)

    value_min, value_max = _robust_range(true_cells, include_zero=True)
    value_norm = _value_norm(value_min, value_max)
    error_max = max(0.25 * (value_max - value_min), 1e-8)
    error_norm = colors.Normalize(vmin=0.0, vmax=error_max)

    fig = plt.figure(figsize=(13.5, 7.2), constrained_layout=True)
    axes = np.array([[fig.add_subplot(2, 3, row * 3 + col + 1, projection='3d')
                      for col in range(3)] for row in range(2)])
    fields = (true_cells, pred_cells, error_cells)
    cmaps = ('coolwarm', 'coolwarm', 'magma')
    norms = (value_norm, value_norm, error_norm)
    titles = ('Ground truth p', 'Predicted p', 'Absolute error |dp|')
    views = ((20, -55), (20, 125))

    for row, (elev, azim) in enumerate(views):
        for col in range(3):
            collection = Poly3DCollection(polygons, linewidth=0.0, rasterized=True)
            collection.set_array(fields[col])
            collection.set_cmap(cmaps[col])
            collection.set_norm(norms[col])
            axes[row, col].add_collection3d(collection)
            _set_3d_limits(axes[row, col], display_points)
            axes[row, col].view_init(elev=elev, azim=azim)
            if row == 0:
                axes[row, col].set_title(titles[col])
        view_label = 'Front oblique view' if row == 0 else 'Rear oblique view'
        axes[row, 0].text2D(0.01, 0.95, view_label, transform=axes[row, 0].transAxes)

    fig.colorbar(ScalarMappable(norm=value_norm, cmap='coolwarm'), ax=axes[:, :2].ravel().tolist(),
                 shrink=0.72, pad=0.01, label='Surface pressure p [Pa]')
    fig.colorbar(ScalarMappable(norm=error_norm, cmap='magma'), ax=axes[:, 2].ravel().tolist(),
                 shrink=0.72, pad=0.01, label='Absolute pressure error [Pa]')

    relative_l2 = np.linalg.norm(pred_pressure - true_pressure) / max(np.linalg.norm(true_pressure), 1e-12)
    mae = np.mean(np.abs(pred_pressure - true_pressure))
    summary = f'Sample: {sample_path} | pressure relative L2={relative_l2:.4f} | MAE={mae:.3f} Pa'
    if cd_values is not None:
        true_cd, pred_cd = cd_values
        cd_error = abs(pred_cd - true_cd) / max(abs(true_cd), 1e-12)
        summary += f' | Cd true/pred={true_cd:.4f}/{pred_cd:.4f} | Cd error={cd_error:.2%}'
    fig.suptitle(summary, fontsize=10)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        'pressure_relative_l2': float(relative_l2),
        'pressure_mae_pa': float(mae),
        'pressure_error_clipped_fraction': float(np.mean(error_cells > error_max)),
    }


def _add_point_array(grid, name, values):
    vtk_array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    vtk_array.SetName(name)
    grid.GetPointData().AddArray(vtk_array)


def _prepare_velocity_grid(volume_file, graph_points, prediction, target):
    grid = _load_grid(volume_file)
    mesh_points = vtk_to_numpy(grid.GetPoints().GetData())
    pred_velocity = _map_values(graph_points, prediction[:, :3], mesh_points)
    true_velocity = _map_values(graph_points, target[:, :3], mesh_points)
    pred_speed = np.linalg.norm(pred_velocity, axis=1)
    true_speed = np.linalg.norm(true_velocity, axis=1)

    median_velocity = np.median(true_velocity, axis=0)
    flow_axis = int(np.argmax(np.abs(median_velocity)))
    flow_sign = 1.0 if median_velocity[flow_axis] >= 0 else -1.0
    flow_coordinate = mesh_points[:, flow_axis]
    cutoff = np.percentile(flow_coordinate, 10 if flow_sign > 0 else 90)
    upstream = flow_coordinate <= cutoff if flow_sign > 0 else flow_coordinate >= cutoff
    inlet_speed = float(np.median(np.abs(true_velocity[upstream, flow_axis])))
    if not np.isfinite(inlet_speed) or inlet_speed < 1e-8:
        inlet_speed = float(np.percentile(true_speed, 90))

    arrays = {
        'true_velocity': true_velocity,
        'pred_velocity': pred_velocity,
        'true_speed_norm': true_speed / inlet_speed,
        'pred_speed_norm': pred_speed / inlet_speed,
        'vector_error_norm': np.linalg.norm(pred_velocity - true_velocity, axis=1) / inlet_speed,
        'true_stream_norm': true_velocity[:, flow_axis] / inlet_speed,
        'pred_stream_norm': pred_velocity[:, flow_axis] / inlet_speed,
        'stream_error_norm': np.abs(pred_velocity[:, flow_axis] - true_velocity[:, flow_axis]) / inlet_speed,
    }
    for name, values in arrays.items():
        _add_point_array(grid, name, values)
    return grid, mesh_points, arrays, flow_axis, flow_sign, inlet_speed


def _cut_grid(grid, normal_axis, value, horizontal_axis, vertical_axis):
    origin = [0.0, 0.0, 0.0]
    origin[normal_axis] = float(value)
    normal = [0.0, 0.0, 0.0]
    normal[normal_axis] = 1.0
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(grid)
    cutter.Update()
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputConnection(cutter.GetOutputPort())
    triangulator.Update()
    cut = triangulator.GetOutput()
    if cut.GetNumberOfPoints() < 3 or cut.GetNumberOfCells() == 0:
        raise ValueError('The requested velocity slice does not intersect the volume mesh')

    points = vtk_to_numpy(cut.GetPoints().GetData())
    polygon_data = vtk_to_numpy(cut.GetPolys().GetData())
    triangles = []
    offset = 0
    while offset < len(polygon_data):
        size = int(polygon_data[offset])
        if size == 3:
            triangles.append(polygon_data[offset + 1:offset + 4])
        offset += size + 1
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.size == 0:
        raise ValueError('VTK slice contains no triangles')
    values = {
        cut.GetPointData().GetArrayName(i): vtk_to_numpy(cut.GetPointData().GetArray(i))
        for i in range(cut.GetPointData().GetNumberOfArrays())
    }
    return points[:, horizontal_axis], points[:, vertical_axis], triangles, values


def _plot_tri(ax, slice_data, field, cmap, norm, title):
    horizontal, vertical, triangles, values = slice_data
    triangulation = mtri.Triangulation(horizontal, vertical, triangles)
    image = ax.tripcolor(triangulation, values[field], shading='gouraud', cmap=cmap, norm=norm,
                         rasterized=True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=10)
    return image


def plot_volume_velocity(output_path, volume_grid, mesh_points, surface_points, flow_axis, inlet_speed):
    axes_available = [axis for axis in range(3) if axis != flow_axis]
    vertical_axis = 1 if flow_axis != 1 else axes_available[-1]
    transverse_axis = next(axis for axis in axes_available if axis != vertical_axis)
    center_value = float(np.median(surface_points[:, transverse_axis]))
    height_value = float(np.median(surface_points[:, vertical_axis]))

    center_slice = _cut_grid(volume_grid, transverse_axis, center_value, flow_axis, vertical_axis)
    horizontal_slice = _cut_grid(volume_grid, vertical_axis, height_value, flow_axis, transverse_axis)
    speed_min, speed_max = _robust_range(center_slice[3]['true_speed_norm'])
    stream_values = np.concatenate((center_slice[3]['true_stream_norm'],
                                    horizontal_slice[3]['true_stream_norm']))
    stream_min, stream_max = _robust_range(stream_values, include_zero=True)
    speed_norm = colors.Normalize(vmin=max(0.0, speed_min), vmax=speed_max)
    stream_norm = _value_norm(stream_min, stream_max)
    error_norm = colors.Normalize(vmin=0.0, vmax=0.25)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    row_specs = (
        (center_slice, 'true_speed_norm', 'pred_speed_norm', 'vector_error_norm', speed_norm,
         '|V|/Uinf, center longitudinal slice'),
        (center_slice, 'true_stream_norm', 'pred_stream_norm', 'stream_error_norm', stream_norm,
         f'{FIELD_NAMES[flow_axis]}/Uinf, center longitudinal slice'),
        (horizontal_slice, 'true_stream_norm', 'pred_stream_norm', 'stream_error_norm', stream_norm,
         f'{FIELD_NAMES[flow_axis]}/Uinf, horizontal slice'),
    )
    for row, (slice_data, true_name, pred_name, error_name, value_norm, label) in enumerate(row_specs):
        cmap = 'viridis' if row == 0 else 'coolwarm'
        _plot_tri(axes[row, 0], slice_data, true_name, cmap, value_norm, f'Ground truth: {label}')
        _plot_tri(axes[row, 1], slice_data, pred_name, cmap, value_norm, f'Prediction: {label}')
        _plot_tri(axes[row, 2], slice_data, error_name, 'magma', error_norm,
                  'Vector error/Uinf' if row == 0 else f'|d{FIELD_NAMES[flow_axis]}|/Uinf')
        fig.colorbar(ScalarMappable(norm=value_norm, cmap=cmap), ax=axes[row, :2].tolist(),
                     shrink=0.82, pad=0.01)
        fig.colorbar(ScalarMappable(norm=error_norm, cmap='magma'), ax=axes[row, 2],
                     shrink=0.82, pad=0.01)
        axes[row, 0].set_ylabel('y' if row < 2 else 'transverse coordinate')
        for col in range(3):
            axes[row, col].set_xlabel(COORDINATE_NAMES[flow_axis] + ' coordinate')

    fig.suptitle(f'Volume velocity field | inferred Uinf={inlet_speed:.3f} m/s')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _probe_plane(grid, fixed_axis, fixed_value, horizontal_axis, vertical_axis, resolution=180):
    mesh_points = vtk_to_numpy(grid.GetPoints().GetData())
    horizontal = np.linspace(float(mesh_points[:, horizontal_axis].min()),
                             float(mesh_points[:, horizontal_axis].max()), resolution, dtype=np.float64)
    vertical = np.linspace(float(mesh_points[:, vertical_axis].min()),
                           float(mesh_points[:, vertical_axis].max()), resolution, dtype=np.float64)
    query_points = vtk.vtkPoints()
    for vertical_value in vertical:
        for horizontal_value in horizontal:
            point = [0.0, 0.0, 0.0]
            point[fixed_axis] = fixed_value
            point[horizontal_axis] = horizontal_value
            point[vertical_axis] = vertical_value
            query_points.InsertNextPoint(point)
    query = vtk.vtkPolyData()
    query.SetPoints(query_points)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(query)
    probe.SetSourceData(grid)
    probe.Update()
    result = probe.GetOutput()
    valid = vtk_to_numpy(result.GetPointData().GetArray('vtkValidPointMask')).reshape(resolution, resolution)
    arrays = {}
    for name in ('true_velocity', 'pred_velocity'):
        array = vtk_to_numpy(result.GetPointData().GetArray(name)).reshape(resolution, resolution, 3)
        arrays[name] = np.ma.array(array, mask=np.repeat((valid == 0)[:, :, None], 3, axis=2))
    return horizontal, vertical, arrays


def plot_velocity_streamlines(
        output_path, volume_grid, surface_points, flow_axis, flow_sign, inlet_speed):
    remaining_axes = [axis for axis in range(3) if axis != flow_axis]
    vertical_axis = 1 if flow_axis != 1 else remaining_axes[-1]
    transverse_axis = next(axis for axis in remaining_axes if axis != vertical_axis)
    fixed_value = float(np.median(surface_points[:, transverse_axis]))
    horizontal, vertical, arrays = _probe_plane(
        volume_grid, transverse_axis, fixed_value, flow_axis, vertical_axis)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, sharex=True, sharey=True)
    norm = colors.Normalize(vmin=0.0, vmax=1.2)
    for ax, (name, title) in zip(axes, (('true_velocity', 'Ground truth streamlines'),
                                       ('pred_velocity', 'Predicted streamlines'))):
        velocity = arrays[name]
        stream_component = velocity[:, :, flow_axis] * flow_sign
        vertical_component = velocity[:, :, vertical_axis]
        speed = np.ma.sqrt(np.ma.sum(velocity ** 2, axis=2)) / inlet_speed
        stream = ax.streamplot(horizontal, vertical, stream_component, vertical_component,
                               color=speed, cmap='viridis', norm=norm, density=1.5,
                               linewidth=0.8, arrowsize=0.7, integration_direction='both')
        tolerance = max(np.ptp(surface_points[:, transverse_axis]) * 0.012, 1e-6)
        near_plane = np.abs(surface_points[:, transverse_axis] - fixed_value) <= tolerance
        ax.scatter(surface_points[near_plane, flow_axis], surface_points[near_plane, vertical_axis],
                   s=1.0, c='black', alpha=0.7, zorder=3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.set_xlabel(COORDINATE_NAMES[flow_axis] + ' coordinate')
        ax.set_ylabel('vertical coordinate')
    fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), ax=axes.tolist(),
                 shrink=0.85, pad=0.01, label='|V|/Uinf')
    fig.suptitle('Center-plane velocity streamlines (derived from supervised velocity components)')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_sample_visualizations(
        output_dir, data_dir, sample_path, graph_points, surface_mask, prediction, target, cd_values=None):
    os.makedirs(output_dir, exist_ok=True)
    graph_points = np.asarray(graph_points)
    surface_mask = np.asarray(surface_mask, dtype=bool)
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    pressure_file, volume_file = _sample_files(data_dir, sample_path)

    metrics = plot_surface_pressure(
        os.path.join(output_dir, 'surface_pressure.png'), data_dir, sample_path, graph_points,
        surface_mask, prediction, target, cd_values=cd_values)
    volume_grid, mesh_points, arrays, flow_axis, flow_sign, inlet_speed = _prepare_velocity_grid(
        volume_file, graph_points, prediction, target)
    plot_volume_velocity(
        os.path.join(output_dir, 'volume_velocity.png'), volume_grid, mesh_points,
        graph_points[surface_mask], flow_axis, inlet_speed)
    plot_velocity_streamlines(
        os.path.join(output_dir, 'velocity_streamlines.png'), volume_grid,
        graph_points[surface_mask], flow_axis, flow_sign, inlet_speed)

    velocity_error = prediction[:, :3] - target[:, :3]
    velocity_relative_l2 = np.linalg.norm(velocity_error) / max(np.linalg.norm(target[:, :3]), 1e-12)
    metrics.update({
        'velocity_relative_l2_all_nodes': float(velocity_relative_l2),
        'velocity_vector_mae': float(np.mean(np.linalg.norm(velocity_error, axis=1))),
        'inlet_speed': inlet_speed,
        'flow_axis': FIELD_NAMES[flow_axis],
    })
    return metrics


def compute_and_plot_validation_drag(output_path, records, data_dir, gt_cache=None):
    gt_cache = {} if gt_cache is None else gt_cache
    true_coefficients = []
    pred_coefficients = []
    valid_indices = []
    for record in records:
        sample_path = record['sample_path']
        try:
            pressure_file, _ = _sample_files(data_dir, sample_path)
            pressure_grid = _load_grid(pressure_file)
            raw_points = vtk_to_numpy(pressure_grid.GetPoints().GetData())
            pred_pressure = _map_values(record['surface_points'], record['pred_pressure'], raw_points)
            true_pressure = _map_values(record['surface_points'], record['true_pressure'], raw_points)
            pred_velocity = _map_values(record['surface_points'], record['pred_velocity'], raw_points)
            true_velocity = _map_values(record['surface_points'], record['true_velocity'], raw_points)
            pred_cd = cal_coefficient(sample_path, pred_pressure[:, None], pred_velocity, root=data_dir)
            if sample_path not in gt_cache:
                gt_cache[sample_path] = cal_coefficient(
                    sample_path, true_pressure[:, None], true_velocity, root=data_dir)
            true_cd = gt_cache[sample_path]
        except Exception as exc:
            warnings.warn(f'Skipping Cd visualization for {sample_path}: {exc}')
            continue
        if np.isfinite(true_cd) and np.isfinite(pred_cd):
            true_coefficients.append(float(true_cd))
            pred_coefficients.append(float(pred_cd))
            valid_indices.append(record['index'])

    if not true_coefficients:
        raise RuntimeError('No valid drag coefficients were available for visualization')
    true_coefficients = np.asarray(true_coefficients)
    pred_coefficients = np.asarray(pred_coefficients)
    residual = pred_coefficients - true_coefficients
    relative_error = np.abs(residual) / np.maximum(np.abs(true_coefficients), 1e-12)
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((true_coefficients - true_coefficients.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    if len(true_coefficients) > 1:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(true_coefficients, pred_coefficients).statistic)
    else:
        spearman = float('nan')

    limits = _robust_range(np.concatenate((true_coefficients, pred_coefficients)), lower=0, upper=100)
    margin = max((limits[1] - limits[0]) * 0.05, 1e-4)
    plot_limits = (limits[0] - margin, limits[1] + margin)
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    scatter = ax.scatter(true_coefficients, pred_coefficients, c=relative_error * 100,
                         cmap='viridis', s=30, edgecolor='black', linewidth=0.25)
    ax.plot(plot_limits, plot_limits, color='black', linestyle='--', linewidth=1, label='Ideal: prediction = truth')
    ax.set_xlim(plot_limits)
    ax.set_ylim(plot_limits)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Ground-truth Cd')
    ax.set_ylabel('Predicted Cd')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.2)
    ax.set_title(
        'Validation drag coefficient\n'
        f'MAPE={relative_error.mean():.2%}, MAE={np.mean(np.abs(residual)):.4f}, '
        f'R2={r_squared:.3f}, Spearman={spearman:.3f}', fontsize=10)
    fig.colorbar(scatter, ax=ax, label='Absolute relative Cd error [%]')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    median_order = np.argsort(true_coefficients)
    median_record_index = valid_indices[int(median_order[len(median_order) // 2])]
    coefficients_by_index = {
        index: (true_cd, pred_cd)
        for index, true_cd, pred_cd in zip(valid_indices, true_coefficients, pred_coefficients)
    }
    return {
        'cd_mape': float(relative_error.mean()),
        'cd_mae': float(np.mean(np.abs(residual))),
        'cd_r_squared': float(r_squared),
        'cd_spearman': spearman,
        'cd_valid_samples': len(valid_indices),
        'median_cd_record_index': median_record_index,
        'coefficients_by_index': coefficients_by_index,
    }


def plot_loss_history(output_path, history, checkpoint_interval=None, pressure_weight=None):
    if not history:
        return
    epochs = np.asarray([entry['epoch'] for entry in history])
    fields = (
        ('total', 'Weighted total loss'),
        ('velocity', 'All-node velocity loss'),
        ('pressure', 'Surface pressure loss'),
    )
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True, constrained_layout=True)
    for ax, (field, title) in zip(axes, fields):
        train_values = np.asarray([entry[f'train_{field}'] for entry in history], dtype=float)
        val_epochs = np.asarray([entry['epoch'] for entry in history if entry.get(f'val_{field}') is not None])
        val_values = np.asarray([entry[f'val_{field}'] for entry in history
                                 if entry.get(f'val_{field}') is not None], dtype=float)
        ax.plot(epochs, train_values, label='Train', color='#1769aa', linewidth=1.4)
        if val_values.size:
            ax.plot(val_epochs, val_values, label='Validation', color='#c43c35', marker='o',
                    markersize=3, linewidth=1.1)
        if checkpoint_interval:
            for checkpoint_epoch in range(checkpoint_interval, int(epochs.max()) + 1, checkpoint_interval):
                ax.axvline(checkpoint_epoch, color='#777777', alpha=0.25, linewidth=0.8)
        positive = np.concatenate((train_values[train_values > 0], val_values[val_values > 0]))
        if positive.size:
            ax.set_yscale('log')
        ax.set_ylabel('MSE')
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel('Epoch')
    weight_label = 'weight' if pressure_weight is None else f'{pressure_weight:g}'
    fig.suptitle(f'Training history: Ltotal = Lvelocity + {weight_label} * Lpressure')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_metrics(path, metrics):
    serializable = {}
    for key, value in metrics.items():
        if key == 'coefficients_by_index':
            continue
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        serializable[key] = value
    with open(path, 'w') as file:
        json.dump(serializable, file, indent=2)


@torch.no_grad()
def predict_sample(device, model, dataset, index, coef_norm):
    cfd_data, geom = dataset[index]
    cfd_data = cfd_data.to(device)
    geom = geom.to(device)
    model.eval()
    prediction = model((cfd_data, geom))
    target = cfd_data.y
    if coef_norm is not None:
        mean = torch.as_tensor(coef_norm[2], device=device, dtype=prediction.dtype)
        std = torch.as_tensor(coef_norm[3], device=device, dtype=prediction.dtype)
        prediction = prediction * std + mean
        target = target * std + mean
    return {
        'points': cfd_data.pos.detach().cpu().numpy(),
        'surface_mask': cfd_data.surf.detach().cpu().numpy(),
        'prediction': prediction.detach().cpu().numpy(),
        'target': target.detach().cpu().numpy(),
    }


def _ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, values.size + 1, dtype=float) / values.size


def plot_l2_error_distribution(output_path, pressure_l2, velocity_l2):
    pressure_l2 = np.asarray(pressure_l2, dtype=float)
    velocity_l2 = np.asarray(velocity_l2, dtype=float)
    pressure_x, pressure_y = _ecdf(pressure_l2)
    velocity_x, velocity_y = _ecdf(velocity_l2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].step(pressure_x, pressure_y, where='post', color='#b33f40', linewidth=1.8)
    axes[0].axvline(np.median(pressure_l2), color='#555555', linestyle='--', linewidth=1,
                    label=f'median={np.median(pressure_l2):.4f}')
    axes[0].set_title('Surface pressure relative L2')
    axes[0].set_xlabel('Relative L2 error')
    axes[0].set_ylabel('Empirical cumulative probability')
    axes[0].legend()

    axes[1].step(velocity_x, velocity_y, where='post', color='#207567', linewidth=1.8)
    axes[1].axvline(np.median(velocity_l2), color='#555555', linestyle='--', linewidth=1,
                    label=f'median={np.median(velocity_l2):.4f}')
    axes[1].set_title('Non-surface volume velocity relative L2')
    axes[1].set_xlabel('Relative L2 error')
    axes[1].set_ylabel('Empirical cumulative probability')
    axes[1].legend()

    axes[2].scatter(pressure_l2, velocity_l2, s=24, alpha=0.75, color='#356aa0',
                    edgecolor='white', linewidth=0.3)
    axes[2].set_title('Per-sample supervised field errors')
    axes[2].set_xlabel('Surface pressure relative L2')
    axes[2].set_ylabel('Non-surface volume velocity relative L2')
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle('Validation relative L2 error distributions')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rmse_by_variable(output_path, pressure_rmse_pa, velocity_rmse_mps):
    pressure_rmse_pa = np.asarray(pressure_rmse_pa, dtype=float)
    velocity_rmse_mps = np.asarray(velocity_rmse_mps, dtype=float)
    if velocity_rmse_mps.ndim != 2 or velocity_rmse_mps.shape[1] != 3:
        raise ValueError('velocity_rmse_mps must have shape [samples, 3]')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    velocity_parts = axes[0].violinplot(
        [velocity_rmse_mps[:, component] for component in range(3)],
        positions=np.arange(1, 4), showmeans=True, showmedians=True, widths=0.75)
    for body, color in zip(velocity_parts['bodies'], ('#2878b5', '#4c956c', '#d17c2f')):
        body.set_facecolor(color)
        body.set_edgecolor('#333333')
        body.set_alpha(0.75)
    axes[0].set_xticks((1, 2, 3), ('u', 'v', 'w'))
    axes[0].set_ylabel('Per-sample RMSE [m/s]')
    axes[0].set_title('Non-surface volume velocity components')
    axes[0].grid(axis='y', alpha=0.2)

    bins = min(20, max(5, int(np.sqrt(pressure_rmse_pa.size))))
    axes[1].hist(pressure_rmse_pa, bins=bins, color='#b33f40', alpha=0.8,
                 edgecolor='white')
    axes[1].axvline(np.median(pressure_rmse_pa), color='#333333', linestyle='--', linewidth=1.2,
                    label=f'median={np.median(pressure_rmse_pa):.3f} Pa')
    axes[1].set_xlabel('Per-sample RMSE [Pa]')
    axes[1].set_ylabel('Number of validation samples')
    axes[1].set_title('Car-surface pressure')
    axes[1].grid(axis='y', alpha=0.2)
    axes[1].legend()
    fig.suptitle('Validation RMSE in physical units')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def drag_coefficient_metrics(true_cd, predicted_cd):
    true_cd = np.asarray(true_cd, dtype=float)
    predicted_cd = np.asarray(predicted_cd, dtype=float)
    residual = predicted_cd - true_cd
    relative_error = np.abs(residual) / np.maximum(np.abs(true_cd), 1e-12)
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((true_cd - true_cd.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    if true_cd.size > 1:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(true_cd, predicted_cd)[0])
    else:
        spearman = float('nan')
    return {
        'relative_error': relative_error,
        'residual': residual,
        'mape': float(relative_error.mean()),
        'mae': float(np.mean(np.abs(residual))),
        'r_squared': float(r_squared),
        'spearman': spearman,
    }


def plot_cd_evaluation(output_path, true_cd, predicted_cd):
    true_cd = np.asarray(true_cd, dtype=float)
    predicted_cd = np.asarray(predicted_cd, dtype=float)
    metrics = drag_coefficient_metrics(true_cd, predicted_cd)
    residual = metrics['residual']
    relative_error = metrics['relative_error']
    limits = _robust_range(np.concatenate((true_cd, predicted_cd)), lower=0, upper=100)
    margin = max((limits[1] - limits[0]) * 0.06, 1e-4)
    plot_limits = (limits[0] - margin, limits[1] + margin)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    scatter = axes[0, 0].scatter(
        true_cd, predicted_cd, c=relative_error * 100, cmap='viridis', s=30,
        edgecolor='black', linewidth=0.25)
    axes[0, 0].plot(plot_limits, plot_limits, color='#333333', linestyle='--', linewidth=1,
                    label='Ideal: prediction = truth')
    axes[0, 0].set_xlim(plot_limits)
    axes[0, 0].set_ylim(plot_limits)
    axes[0, 0].set_aspect('equal', adjustable='box')
    axes[0, 0].set_xlabel('Ground-truth Cd')
    axes[0, 0].set_ylabel('Predicted Cd')
    axes[0, 0].set_title('Drag-coefficient agreement')
    axes[0, 0].legend(loc='upper left')
    summary = (
        f'MAPE={metrics["mape"]:.2%}\nMAE={metrics["mae"]:.4f}\n'
        f'R2={metrics["r_squared"]:.3f}\nSpearman={metrics["spearman"]:.3f}')
    axes[0, 0].text(
        0.98, 0.03, summary, transform=axes[0, 0].transAxes, ha='right', va='bottom',
        fontsize=9, bbox={'facecolor': 'white', 'edgecolor': '#aaaaaa', 'alpha': 0.9})
    fig.colorbar(scatter, ax=axes[0, 0], label='Absolute relative Cd error [%]')

    order = np.argsort(true_cd)
    rank = np.arange(1, true_cd.size + 1)
    axes[0, 1].plot(rank, true_cd[order], color='#222222', linewidth=1.6,
                    label='Ground truth')
    axes[0, 1].plot(rank, predicted_cd[order], color='#d17c2f', linewidth=1.3,
                    marker='o', markersize=2.5, label='Prediction')
    axes[0, 1].set_xlabel('Validation sample rank (sorted by ground-truth Cd)')
    axes[0, 1].set_ylabel('Cd')
    axes[0, 1].set_title('Design ranking response')
    axes[0, 1].legend()

    bins = min(20, max(5, int(np.sqrt(relative_error.size))))
    axes[1, 0].hist(relative_error * 100, bins=bins, color='#4c956c', alpha=0.85,
                    edgecolor='white')
    axes[1, 0].axvline(np.median(relative_error) * 100, color='#333333', linestyle='--',
                       linewidth=1.2, label=f'median={np.median(relative_error):.2%}')
    axes[1, 0].set_xlabel('Absolute relative Cd error [%]')
    axes[1, 0].set_ylabel('Number of validation samples')
    axes[1, 0].set_title('Drag-coefficient relative error')
    axes[1, 0].legend()

    axes[1, 1].axhline(0.0, color='#333333', linestyle='--', linewidth=1)
    axes[1, 1].scatter(true_cd, residual, s=26, color='#b33f40', alpha=0.75,
                       edgecolor='white', linewidth=0.3)
    axes[1, 1].set_xlabel('Ground-truth Cd')
    axes[1, 1].set_ylabel('Residual: predicted Cd - ground-truth Cd')
    axes[1, 1].set_title('Drag-coefficient residuals')
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.suptitle('Validation drag-coefficient evaluation')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return metrics


def plot_inference_time_distribution(output_path, forward_latency_ms):
    forward_latency_ms = np.asarray(forward_latency_ms, dtype=float)
    latency_x, latency_y = _ecdf(forward_latency_ms)
    mean = float(np.mean(forward_latency_ms))
    median = float(np.median(forward_latency_ms))
    p95 = float(np.percentile(forward_latency_ms, 95))
    maximum = float(np.max(forward_latency_ms))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    bins = min(20, max(5, int(np.sqrt(forward_latency_ms.size))))
    axes[0].hist(forward_latency_ms, bins=bins, color='#356aa0', alpha=0.85,
                 edgecolor='white')
    axes[0].axvline(median, color='#333333', linestyle='--', linewidth=1.2,
                    label=f'median={median:.3f} ms')
    axes[0].set_xlabel('Model forward latency [ms/sample]')
    axes[0].set_ylabel('Number of validation samples')
    axes[0].set_title('Forward-latency distribution')
    axes[0].legend()

    axes[1].step(latency_x, latency_y, where='post', color='#207567', linewidth=1.8)
    axes[1].set_xlabel('Model forward latency [ms/sample]')
    axes[1].set_ylabel('Empirical cumulative probability')
    axes[1].set_title('Forward-latency empirical CDF')
    axes[1].text(
        0.98, 0.04, f'mean={mean:.3f} ms\nmedian={median:.3f} ms\nP95={p95:.3f} ms\nmax={maximum:.3f} ms',
        transform=axes[1].transAxes, ha='right', va='bottom', fontsize=9,
        bbox={'facecolor': 'white', 'edgecolor': '#aaaaaa', 'alpha': 0.9})
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle('Model forward latency only (batch size 1)')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {'mean_ms': mean, 'median_ms': median, 'p95_ms': p95, 'max_ms': maximum}
