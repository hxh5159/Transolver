#!/usr/bin/env python3
"""Plot cross-dataset heatmap trajectories at 100-epoch intervals.

The script writes two new figures under monitor/plot/output:
one for the matched full propagation kernel and one for the matched base
routing kernel. Rows are benchmarks and columns are epochs 1, 100, ..., 500.
The monitor's validation_000001 snapshot is the validation after epoch 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

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
EPOCHS = (1, 100, 200, 300, 400, 500)
SNAPSHOT_RE = re.compile(r"validation_(\d+)$")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def default_input_root(monitor_dir: Path) -> Path:
    candidates = sorted(
        p / "LINEARNO" / "monitor" / "output"
        for p in (monitor_dir / "analysis").glob("transolver_monitor_output_*")
        if (p / "LINEARNO" / "monitor" / "output").is_dir()
    )
    if candidates:
        return candidates[-1]
    direct = monitor_dir / "output"
    if direct.is_dir():
        return direct
    raise RuntimeError("Monitor output root not found; pass --input-root")


def complete_training_run(path: Path) -> bool:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False
    if summary.get("status") != "completed":
        return False
    if summary.get("target_status", "completed") != "completed":
        return False
    config_path = path / "run_config.json"
    if config_path.is_file():
        try:
            config = read_json(config_path)
            parsed = config.get("target", {}).get("parsed_arguments_best_effort", {})
            if str(parsed.get("eval", "0")).lower() not in {"0", "false", "none", ""}:
                return False
        except (OSError, json.JSONDecodeError, AttributeError, RuntimeError):
            pass
    return True


def parse_overrides(values: list[str]) -> dict[str, Path]:
    valid = {name for name, _ in DATASETS}
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --run {value!r}; expected DATASET=PATH")
        dataset, raw_path = value.split("=", 1)
        if dataset not in valid or not raw_path:
            raise ValueError(f"Invalid dataset override: {value!r}")
        result[dataset] = Path(raw_path).expanduser().resolve()
    return result


def choose_run(root: Path, dataset: str, override: Path | None) -> Path:
    if override is not None:
        if not complete_training_run(override):
            raise RuntimeError(f"Not a completed training run: {override}")
        return override
    dataset_root = root / dataset
    if not dataset_root.is_dir():
        raise RuntimeError(f"Missing dataset output directory: {dataset_root}")
    runs = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir() and complete_training_run(path)
    ]
    if not runs:
        raise RuntimeError(f"No completed training run found in {dataset_root}")
    return runs[-1]


def snapshot_index(run: Path, epoch: int) -> int:
    index = 1 if epoch == 0 else epoch
    path = run / "snapshots" / f"validation_{index:06d}"
    if not path.is_dir():
        available = sorted(
            int(match.group(1))
            for item in (run / "snapshots").iterdir()
            if (match := SNAPSHOT_RE.fullmatch(item.name))
        )
        raise RuntimeError(
            f"Missing {path}; available validation indices: {available}"
        )
    return index


def load_matrix(run: Path, index: int, key: str) -> np.ndarray:
    source = run / "snapshots" / f"validation_{index:06d}" / "similarity_values.npz"
    if not source.is_file():
        raise RuntimeError(f"Missing similarity data: {source}")
    with np.load(source, allow_pickle=False) as values:
        matrix = np.asarray(values[key], dtype=float)
        layers = np.asarray(values["layer_indices"])
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(f"Expected a square matrix in {source}; got {matrix.shape}")
    if layers.size != matrix.shape[0] or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"Invalid layer metadata or non-finite values in {source}")
    return np.clip(matrix, 0.0, 1.0)


def save_matrix_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset", "run_directory", "kernel", "requested_epoch",
                "source_validation_index", "layer_i", "layer_j", "similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def plot_kernel(
    kernel: str,
    kernel_label: str,
    matrices: dict[str, list[np.ndarray]],
    runs: dict[str, Path],
    output_dir: Path,
    profile: str,
    dpi: int,
) -> dict:
    width, height = ((7.0, 5.8) if profile == "two-column" else (7.0, 5.8))
    with plt.rc_context(
        {
            "font.family": "STIXGeneral",
            "font.size": 7.4,
            "mathtext.fontset": "stix",
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(
            len(DATASETS), len(EPOCHS), figsize=(width, height), squeeze=False
        )
        figure.subplots_adjust(
            left=0.095, right=0.895, bottom=0.105, top=0.875,
            wspace=0.08, hspace=0.20,
        )
        images = []
        for row, (dataset, display_name) in enumerate(DATASETS):
            for column, epoch in enumerate(EPOCHS):
                axis = axes[row, column]
                matrix = matrices[dataset][column]
                image = axis.imshow(
                    matrix,
                    origin="lower",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                    aspect="equal",
                )
                images.append(image)
                if row == 0:
                    axis.set_title(f"Epoch {epoch}", pad=2.5)
                axis.set_xticks(range(matrix.shape[0]))
                axis.set_yticks(range(matrix.shape[0]))
                axis.set_xticklabels(
                    [str(i + 1) for i in range(matrix.shape[0])] if row == len(DATASETS) - 1 else []
                )
                axis.set_yticklabels(
                    [str(i + 1) for i in range(matrix.shape[0])] if column == 0 else []
                )
                axis.tick_params(length=1.8, width=0.45, pad=1.0)
                if row == len(DATASETS) - 1:
                    axis.set_xlabel("Block index", labelpad=1.5)
                if column == 0:
                    axis.set_ylabel(display_name, labelpad=7)
                for spine in axis.spines.values():
                    spine.set_linewidth(0.45)
                    spine.set_color("#777777")

        figure.suptitle(
            f"Cross-dataset evolution of {kernel_label}",
            fontsize=9.2,
            fontweight="semibold",
            y=0.955,
        )
        colorbar = figure.colorbar(
            images[-1], ax=axes.ravel().tolist(), fraction=0.026, pad=0.018, aspect=34
        )
        colorbar.set_label(r"Frobenius cosine similarity, $\rho_{ij}$", fontsize=7.4, labelpad=4)
        colorbar.ax.tick_params(labelsize=6.4, width=0.45, length=1.8)
        colorbar.outline.set_linewidth(0.45)

        stem = output_dir / f"four_dataset_{kernel}_kernel_heatmaps"
        figure.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white", bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
        figure.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
        plt.close(figure)

    records: list[dict] = []
    for dataset, _ in DATASETS:
        for epoch, matrix in zip(EPOCHS, matrices[dataset]):
            source_index = 1 if epoch == 0 else epoch
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    records.append(
                        {
                            "dataset": dataset,
                            "run_directory": str(runs[dataset]),
                            "kernel": kernel,
                            "requested_epoch": epoch,
                            "source_validation_index": source_index,
                            "layer_i": i + 1,
                            "layer_j": j + 1,
                            "similarity": f"{matrix[i, j]:.10g}",
                        }
                    )
    csv_path = output_dir / f"four_dataset_{kernel}_kernel_heatmaps_data.csv"
    save_matrix_csv(csv_path, records)
    return {
        "kernel": kernel,
        "label": kernel_label,
        "png": str((stem.with_suffix(".png")).resolve()),
        "pdf": str((stem.with_suffix(".pdf")).resolve()),
        "svg": str((stem.with_suffix(".svg")).resolve()),
        "csv": str(csv_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    plot_dir = Path(__file__).resolve().parent
    monitor_dir = plot_dir.parent
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=plot_dir / "output")
    parser.add_argument("--profile", choices=("iclr", "two-column"), default="two-column")
    parser.add_argument("--run", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi <= 0:
        raise SystemExit("error: --dpi must be positive")

    input_root = (
        args.input_root.expanduser().resolve()
        if args.input_root is not None
        else default_input_root(monitor_dir)
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = parse_overrides(args.run)
    runs: dict[str, Path] = {}
    matrices = {"full": {}, "base": {}}
    snapshot_manifest = {}
    for dataset, _ in DATASETS:
        run = choose_run(input_root, dataset, overrides.get(dataset))
        runs[dataset] = run
        snapshot_manifest[dataset] = []
        for epoch in EPOCHS:
            source_index = snapshot_index(run, epoch)
            snapshot_manifest[dataset].append(
                {"requested_epoch": epoch, "source_validation_index": source_index}
            )
            matrices["full"].setdefault(dataset, []).append(
                load_matrix(run, source_index, "full_matched_mean")
            )
            matrices["base"].setdefault(dataset, []).append(
                load_matrix(run, source_index, "base_matched_mean")
            )

    outputs = [
        plot_kernel(
            "full",
            r"matched full kernel $WAD^{-1}W^\top$",
            matrices["full"], runs, output_dir, args.profile, args.dpi,
        ),
        plot_kernel(
            "base",
            r"matched base routing kernel $WD^{-1}W^\top$",
            matrices["base"], runs, output_dir, args.profile, args.dpi,
        ),
    ]
    manifest = {
        "input_root": str(input_root),
        "requested_epochs": list(EPOCHS),
        "epoch_one_mapping": "validation_000001",
        "runs": {dataset: str(path) for dataset, path in runs.items()},
        "snapshots": snapshot_manifest,
        "outputs": outputs,
    }
    manifest_path = output_dir / "four_dataset_kernel_heatmaps_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    caption_path = output_dir / "four_dataset_kernel_heatmaps_caption.tex"
    caption_path.write_text(
        r"""\begin{figure*}[t]
  \centering
  \includegraphics[width=\linewidth]{four_dataset_full_kernel_heatmaps.pdf}
  \caption{\textbf{Evolution of cross-layer propagation-kernel similarity across four PDE benchmarks.}
  Rows correspond to Airfoil, Darcy, Elasticity, and Pipe; columns show epoch 1,
  followed by epochs 100, 200, 300, 400, and 500. Each cell is the
  sample-mean Frobenius cosine similarity between two blocks, after optimal head matching.
  The displayed full propagation kernel is $P^{\mathrm{full}}=WAD^{-1}W^\top$ and all panels use
  one common 0--1 color scale. The epoch-1 panel is sourced from the monitor's validation\_000001
  snapshot, recorded after the first training epoch.}
  \label{fig:four-dataset-full-kernel-heatmaps}
\end{figure*}
""",
        encoding="utf-8",
    )
    print(f"Input root: {input_root}")
    for output in outputs:
        print(f"{output['kernel']}: {output['png']}")
    print(f"Manifest: {manifest_path}")
    print(f"Caption: {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
