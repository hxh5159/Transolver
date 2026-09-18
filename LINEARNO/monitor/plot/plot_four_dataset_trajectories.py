#!/usr/bin/env python3
"""Plot cross-layer kernel-similarity trajectories for four complete benchmarks.

The primary curve is the permutation-invariant, optimally head-matched full
propagation kernel. The base slice/deslice kernel is included as a diagnostic.
Only completed training runs are selected; evaluation and failed runs are
ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DATASETS = (
    ("standard_airfoil", "Airfoil"),
    ("darcy", "Darcy"),
    ("elasticity", "Elasticity"),
    ("pipe", "Pipe"),
)
SNAPSHOT_PATTERN = re.compile(r"validation_(\d+)$")


@dataclass(frozen=True)
class TrajectoryPoint:
    validation_index: int
    sample_count: int
    base_mean: float
    base_std: float
    full_mean: float
    full_std: float


@dataclass(frozen=True)
class PlotProfile:
    width: float
    height: float
    figure_environment: str
    font_size: float
    label_size: float
    title_size: float
    tick_size: float
    legend_size: float


PLOT_PROFILES = {
    # ICLR 2026 has a 5.5-inch single-column text block and 10-point body text.
    "iclr": PlotProfile(
        width=5.5,
        height=4.25,
        figure_environment="figure",
        font_size=8.0,
        label_size=8.5,
        title_size=9.0,
        tick_size=8.0,
        legend_size=7.5,
    ),
    # Full-width figure for two-column venues such as ICML.
    "two-column": PlotProfile(
        width=7.0,
        height=4.85,
        figure_environment="figure*",
        font_size=8.0,
        label_size=8.5,
        title_size=9.0,
        tick_size=8.0,
        legend_size=8.0,
    ),
}


def parse_args() -> argparse.Namespace:
    monitor_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Plot P_base and P_full mean off-diagonal cross-layer similarity "
            "for Airfoil, Darcy, Elasticity, and Pipe."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=monitor_dir / "output",
        help="Monitor output root containing one directory per dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for PNG, PDF, and CSV outputs.",
    )
    parser.add_argument(
        "--stem",
        default="four_dataset_similarity_trajectories",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster output resolution (default: 600 DPI).",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PLOT_PROFILES),
        default="iclr",
        help=(
            "Publication layout: native 5.5-inch ICLR width (default) or a "
            "7-inch full-width figure for two-column venues."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help=(
            "Optional title embedded above the figure. The publication default "
            "omits it because the generated LaTeX caption supplies the title."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help=(
            "Use an explicit run directory for a dataset. May be repeated; "
            "otherwise the latest completed training run is selected."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def parse_run_overrides(values: Iterable[str]) -> dict[str, Path]:
    valid_names = {name for name, _ in DATASETS}
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --run value {value!r}; expected DATASET=PATH")
        dataset, raw_path = value.split("=", 1)
        if dataset not in valid_names:
            choices = ", ".join(sorted(valid_names))
            raise ValueError(f"Unknown dataset {dataset!r}; choose one of: {choices}")
        if not raw_path:
            raise ValueError(f"Missing path in --run value {value!r}")
        overrides[dataset] = Path(raw_path).expanduser().resolve()
    return overrides


def is_complete_training_run(run_dir: Path) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
    except RuntimeError:
        return False
    if summary.get("status") != "completed":
        return False
    if summary.get("target_status", "completed") != "completed":
        return False
    snapshots = summary.get("snapshots", [])
    if not isinstance(snapshots, list) or len(snapshots) < 2:
        return False

    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        try:
            config = read_json(config_path)
            parsed = config.get("target", {}).get(
                "parsed_arguments_best_effort", {}
            )
            eval_value = str(parsed.get("eval", "0")).strip().lower()
            if eval_value not in {"0", "false", "none", ""}:
                return False
        except (AttributeError, RuntimeError):
            pass
    return True


def choose_run(input_root: Path, dataset: str, override: Path | None) -> Path:
    if override is not None:
        if not is_complete_training_run(override):
            raise RuntimeError(
                f"Explicit {dataset} run is not a completed training run: {override}"
            )
        return override

    dataset_dir = input_root / dataset
    if not dataset_dir.is_dir():
        raise RuntimeError(f"Dataset output directory does not exist: {dataset_dir}")
    candidates = sorted(
        (path for path in dataset_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    complete = [path for path in candidates if is_complete_training_run(path)]
    if not complete:
        raise RuntimeError(
            f"No completed training run with at least two snapshots: {dataset_dir}"
        )
    return complete[-1]


def snapshot_index(snapshot_dir: Path) -> int:
    match = SNAPSHOT_PATTERN.fullmatch(snapshot_dir.name)
    if match is None:
        raise RuntimeError(f"Invalid snapshot directory name: {snapshot_dir}")
    return int(match.group(1))


def mean_off_diagonal_per_sample(values: np.ndarray, source: Path) -> np.ndarray:
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise RuntimeError(
            f"Expected [sample, layer, layer] data in {source}, got {values.shape}"
        )
    layer_count = values.shape[1]
    if layer_count < 2:
        raise RuntimeError(f"At least two layers are required in {source}")
    upper_triangle = np.triu_indices(layer_count, k=1)
    per_sample = np.nanmean(values[:, upper_triangle[0], upper_triangle[1]], axis=1)
    if not np.all(np.isfinite(per_sample)):
        raise RuntimeError(f"Non-finite off-diagonal similarity in {source}")
    return per_sample


def load_snapshot(snapshot_dir: Path) -> TrajectoryPoint:
    values_path = snapshot_dir / "similarity_values.npz"
    if not values_path.is_file():
        raise RuntimeError(f"Missing similarity data: {values_path}")
    try:
        with np.load(values_path, allow_pickle=False) as values:
            base = mean_off_diagonal_per_sample(values["base_matched"], values_path)
            full = mean_off_diagonal_per_sample(values["full_matched"], values_path)
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Cannot load similarity data {values_path}: {exc}") from exc
    if base.shape != full.shape:
        raise RuntimeError(f"Base/full sample counts differ in {values_path}")
    return TrajectoryPoint(
        validation_index=snapshot_index(snapshot_dir),
        sample_count=int(base.size),
        base_mean=float(np.mean(base)),
        base_std=float(np.std(base, ddof=0)),
        full_mean=float(np.mean(full)),
        full_std=float(np.std(full, ddof=0)),
    )


def load_trajectory(run_dir: Path) -> list[TrajectoryPoint]:
    snapshots_dir = run_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise RuntimeError(f"Missing snapshots directory: {snapshots_dir}")
    snapshot_dirs = [
        path
        for path in snapshots_dir.iterdir()
        if path.is_dir() and SNAPSHOT_PATTERN.fullmatch(path.name)
    ]
    points = [load_snapshot(path) for path in snapshot_dirs]
    points.sort(key=lambda point: point.validation_index)
    if len(points) < 2:
        raise RuntimeError(f"Need at least two valid snapshots in {snapshots_dir}")
    return points


def save_source_data(
    path: Path,
    selected_runs: dict[str, Path],
    trajectories: dict[str, list[TrajectoryPoint]],
) -> None:
    fieldnames = [
        "dataset",
        "run_directory",
        "validation_index",
        "sample_count",
        "base_mean",
        "base_std_across_samples",
        "full_mean",
        "full_std_across_samples",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, _ in DATASETS:
            for point in trajectories[dataset]:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "run_directory": str(selected_runs[dataset]),
                        "validation_index": point.validation_index,
                        "sample_count": point.sample_count,
                        "base_mean": f"{point.base_mean:.10g}",
                        "base_std_across_samples": f"{point.base_std:.10g}",
                        "full_mean": f"{point.full_mean:.10g}",
                        "full_std_across_samples": f"{point.full_std:.10g}",
                    }
                )


def save_latex_caption(
    path: Path, figure_name: str, profile: PlotProfile
) -> None:
    environment = profile.figure_environment
    caption = rf"""\begin{{{environment}}}[t]
  \centering
  \includegraphics[width=\linewidth]{{{figure_name}.pdf}}
  \caption{{\textbf{{Evolution of cross-layer propagation-kernel similarity.}}
  Mean off-diagonal Frobenius cosine similarity $\rho_{{ij}}$ between the
  propagation kernels of different Transolver blocks on four PDE benchmarks.
  The base kernel $P^{{\mathrm{{base}}}}=WD^{{-1}}W^\top$ measures the
  slice/deslice routing, whereas the full kernel
  $P^{{\mathrm{{full}}}}=WAD^{{-1}}W^\top$ additionally includes attention
  between slice tokens. Higher similarity indicates less differentiated
  spatial propagation across blocks. Curves report the mean over four
  monitored samples and shaded regions denote one standard deviation.
  Cross-layer heads are compared by optimal bipartite matching, making the
  metric invariant to head permutations. Each validation index corresponds
  to one training epoch in these experiments.}}
  \label{{fig:cross-layer-kernel-similarity}}
\end{{{environment}}}
"""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(caption)


def plot_trajectories(
    trajectories: dict[str, list[TrajectoryPoint]],
    output_stem: Path,
    dpi: int,
    title: str | None,
    profile: PlotProfile,
) -> None:
    # Okabe-Ito colors remain distinguishable for common color-vision deficits.
    # Solid/dashed lines and circle/square markers preserve meaning in grayscale.
    base_color = "#D55E00"
    full_color = "#0072B2"
    panel_labels = "abcd"

    with plt.rc_context(
        {
            "font.family": "STIXGeneral",
            "font.size": profile.font_size,
            "mathtext.fontset": "stix",
            "axes.titlesize": profile.title_size,
            "axes.labelsize": profile.label_size,
            "xtick.labelsize": profile.tick_size,
            "ytick.labelsize": profile.tick_size,
            "legend.fontsize": profile.legend_size,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(profile.width, profile.height),
            sharex=True,
            sharey=True,
        )
        axes_top = 0.79 if title else 0.845
        figure.subplots_adjust(
            left=0.115,
            right=0.985,
            bottom=0.13,
            top=axes_top,
            wspace=0.08,
            hspace=0.24,
        )
        legend_handles = None
        for panel_index, (axis, (dataset, display_name)) in enumerate(
            zip(axes.flat, DATASETS)
        ):
            points = trajectories[dataset]
            x = np.asarray([point.validation_index for point in points])
            base_mean = np.asarray([point.base_mean for point in points])
            base_std = np.asarray([point.base_std for point in points])
            full_mean = np.asarray([point.full_mean for point in points])
            full_std = np.asarray([point.full_std for point in points])

            full_line = axis.plot(
                x,
                full_mean,
                color=full_color,
                linewidth=1.55,
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=r"Full kernel: $WAD^{-1}W^\top$",
                zorder=3,
            )[0]
            base_line = axis.plot(
                x,
                base_mean,
                color=base_color,
                linewidth=1.35,
                linestyle="--",
                marker="s",
                markersize=3.4,
                markerfacecolor="white",
                markeredgewidth=0.85,
                label=r"Base routing: $WD^{-1}W^\top$",
                zorder=3,
            )[0]
            axis.fill_between(
                x,
                np.clip(full_mean - full_std, 0.0, 1.0),
                np.clip(full_mean + full_std, 0.0, 1.0),
                color=full_color,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
            axis.fill_between(
                x,
                np.clip(base_mean - base_std, 0.0, 1.0),
                np.clip(base_mean + base_std, 0.0, 1.0),
                color=base_color,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            axis.set_title(
                f"({panel_labels[panel_index]}) {display_name}", loc="left", pad=5
            )
            axis.set_xlim(0, max(point.validation_index for point in points) + 10)
            axis.set_ylim(0.0, 1.025)
            axis.set_yticks(np.linspace(0.0, 1.0, 6))
            axis.grid(
                axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.42, zorder=0
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if legend_handles is None:
                legend_handles = (full_line, base_line)

        figure.supxlabel(
            "Validation index", x=0.55, y=0.035, fontsize=profile.label_size
        )
        figure.supylabel(
            r"Mean cross-layer similarity, $\bar{\rho}$",
            x=0.018,
            y=0.49,
            fontsize=profile.label_size,
        )

        if title:
            figure.suptitle(title, fontsize=10, fontweight="semibold", y=0.99)
        figure.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.54, 0.885 if title else 0.96),
            ncol=2,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.6,
            handletextpad=0.55,
        )
        figure.savefig(
            output_stem.with_suffix(".png"),
            dpi=dpi,
            facecolor="white",
        )
        figure.savefig(
            output_stem.with_suffix(".pdf"),
            facecolor="white",
            metadata={
                "Title": "Cross-layer propagation-kernel similarity",
                "Subject": "Transolver propagation-kernel monitoring",
            },
        )
        figure.savefig(output_stem.with_suffix(".svg"), facecolor="white")
        plt.close(figure)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        overrides = parse_run_overrides(args.run)
        selected_runs = {
            dataset: choose_run(input_root, dataset, overrides.get(dataset))
            for dataset, _ in DATASETS
        }
        trajectories = {
            dataset: load_trajectory(selected_runs[dataset])
            for dataset, _ in DATASETS
        }
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / args.stem
    if args.dpi <= 0:
        raise SystemExit("error: --dpi must be a positive integer")
    profile = PLOT_PROFILES[args.profile]
    plot_trajectories(
        trajectories,
        output_stem,
        dpi=args.dpi,
        title=args.title,
        profile=profile,
    )
    save_source_data(
        output_stem.with_name(f"{output_stem.name}_data.csv"),
        selected_runs,
        trajectories,
    )
    save_latex_caption(
        output_stem.with_name(f"{output_stem.name}_caption.tex"),
        output_stem.name,
        profile,
    )

    for dataset, display_name in DATASETS:
        print(f"{display_name}: {selected_runs[dataset]}")
    print(f"Profile: {args.profile} ({profile.width:g} x {profile.height:g} inches)")
    print(f"PNG: {output_stem.with_suffix('.png')}")
    print(f"PDF: {output_stem.with_suffix('.pdf')}")
    print(f"SVG: {output_stem.with_suffix('.svg')}")
    print(f"CSV: {output_stem.with_name(output_stem.name + '_data.csv')}")
    print(f"TeX: {output_stem.with_name(output_stem.name + '_caption.tex')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
