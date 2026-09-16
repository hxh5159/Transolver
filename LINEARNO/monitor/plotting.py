"""Paper-oriented heatmaps for propagation-kernel similarity."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _labels(count):
    return [f"Block {index + 1}" for index in range(count)]


def _draw_matrix(axis, matrix, title, value_range=(0.0, 1.0), annotate=True):
    image = axis.imshow(
        matrix, cmap="viridis", vmin=value_range[0], vmax=value_range[1],
        interpolation="nearest", aspect="equal")
    labels = _labels(matrix.shape[0])
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_xlabel("Compared block")
    axis.set_ylabel("Reference block")
    axis.set_title(title, pad=8)
    if annotate and matrix.shape[0] <= 10:
        midpoint = (value_range[0] + value_range[1]) / 2.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isfinite(value):
                    color = "white" if value < midpoint else "black"
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                              fontsize=6.5, color=color)
    return image


def save_heatmap(matrix, title, stem, colorbar_label="Frobenius cosine similarity",
                 value_range=(0.0, 1.0)):
    plt = _setup_matplotlib()
    stem = Path(stem)
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        size = max(3.8, min(6.5, 3.0 + 0.35 * matrix.shape[0]))
        figure, axis = plt.subplots(figsize=(size, size), constrained_layout=True)
        image = _draw_matrix(axis, matrix, title, value_range=value_range)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def save_combined(base, full, stem):
    plt = _setup_matplotlib()
    stem = Path(stem)
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        width = max(7.4, min(12.0, 6.0 + 0.55 * base.shape[0]))
        figure, axes = plt.subplots(1, 2, figsize=(width, width * 0.46),
                                    constrained_layout=True)
        image = _draw_matrix(axes[0], base, "Base slice/deslice kernel")
        _draw_matrix(axes[1], full, "Full propagation kernel")
        colorbar = figure.colorbar(image, ax=axes, fraction=0.028, pad=0.025)
        colorbar.set_label("Frobenius cosine similarity")
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def plot_snapshot(snapshot_dir, base_mean, full_mean, base_std=None, full_std=None):
    snapshot_dir = Path(snapshot_dir)
    save_heatmap(
        full_mean, "Full propagation-kernel similarity",
        snapshot_dir / "full_kernel_similarity")
    save_heatmap(
        base_mean, "Base slice/deslice-kernel similarity",
        snapshot_dir / "base_kernel_similarity")
    save_combined(base_mean, full_mean, snapshot_dir / "base_vs_full")
    if full_std is not None:
        upper = max(0.01, float(np.nanmax(full_std)))
        save_heatmap(
            full_std, "Full-kernel similarity standard deviation",
            snapshot_dir / "full_kernel_similarity_std",
            colorbar_label="Standard deviation", value_range=(0.0, upper))
    if base_std is not None:
        upper = max(0.01, float(np.nanmax(base_std)))
        save_heatmap(
            base_std, "Base-kernel similarity standard deviation",
            snapshot_dir / "base_kernel_similarity_std",
            colorbar_label="Standard deviation", value_range=(0.0, upper))
