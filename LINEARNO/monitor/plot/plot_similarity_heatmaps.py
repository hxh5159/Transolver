#!/usr/bin/env python3
"""Publication-ready cross-layer kernel-similarity heatmaps.

Creates one 2x3 figure per benchmark. Columns are requested epochs 1, 100,
and 200; rows are the permutation-invariant matched full kernel and base
routing kernel. The monitor stores the first validation in validation_000001,
which is used as the epoch-1 panel.
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
SNAPSHOT_RE = re.compile(r"validation_(\d+)$")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
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
    summary = path / "summary.json"
    if not summary.is_file():
        return False
    try:
        data = read_json(summary)
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False
    if data.get("status") != "completed" or data.get("target_status", "completed") != "completed":
        return False
    config = path / "run_config.json"
    if config.is_file():
        try:
            args = read_json(config).get("target", {}).get("parsed_arguments_best_effort", {})
            if str(args.get("eval", "0")).lower() not in {"0", "false", "none", ""}:
                return False
        except (OSError, json.JSONDecodeError, AttributeError, RuntimeError):
            pass
    return True


def choose_run(root: Path, dataset: str, override: str | None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if not complete_training_run(path):
            raise RuntimeError(f"Not a completed training run: {path}")
        return path
    dataset_root = root / dataset
    runs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir() and complete_training_run(p)]
    if not runs:
        raise RuntimeError(f"No completed training run in {dataset_root}")
    return runs[-1]


def parse_overrides(values: list[str]) -> dict[str, str]:
    result = {}
    valid = {name for name, _ in DATASETS}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --run {value!r}; expected DATASET=PATH")
        dataset, path = value.split("=", 1)
        if dataset not in valid or not path:
            raise ValueError(f"Invalid dataset override: {value!r}")
        result[dataset] = path
    return result


def parse_epochs(raw: str) -> tuple[int, ...]:
    try:
        epochs = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --epochs: {raw!r}") from exc
    if not epochs or any(x < 0 for x in epochs) or len(set(epochs)) != len(epochs):
        raise ValueError("--epochs must contain unique non-negative integers")
    return epochs


def source_index(run: Path, requested_epoch: int) -> int:
    index = 1 if requested_epoch == 0 else requested_epoch
    snapshot = run / "snapshots" / f"validation_{index:06d}"
    if not snapshot.is_dir():
        available = sorted(
            int(m.group(1))
            for p in (run / "snapshots").iterdir()
            if (m := SNAPSHOT_RE.fullmatch(p.name))
        )
        raise RuntimeError(f"Missing {snapshot}; available snapshots: {available}")
    return index


def load_matrix(run: Path, index: int, key: str) -> np.ndarray:
    path = run / "snapshots" / f"validation_{index:06d}" / "similarity_values.npz"
    if not path.is_file():
        raise RuntimeError(f"Missing similarity file: {path}")
    with np.load(path, allow_pickle=False) as data:
        matrix = np.asarray(data[key], dtype=float)
        layers = np.asarray(data["layer_indices"])
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or layers.size != matrix.shape[0]:
        raise RuntimeError(f"Invalid matrix shape in {path}: {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"Non-finite values in {path}::{key}")
    return np.clip(matrix, 0.0, 1.0)


def annotate(axis, matrix: np.ndarray) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.2,
                      color="black" if value > 0.58 else "white")


def plot_one(dataset: str, display: str, run: Path, epochs: tuple[int, ...], out: Path,
             dpi: int, annotate_cells: bool, width: float, height: float) -> dict:
    matrices = {"full": [], "base": []}
    indices = []
    for epoch in epochs:
        index = source_index(run, epoch)
        indices.append(index)
        matrices["full"].append(load_matrix(run, index, "full_matched_mean"))
        matrices["base"].append(load_matrix(run, index, "base_matched_mean"))
    n = matrices["full"][0].shape[0]
    with plt.rc_context({
        "font.family": "STIXGeneral",
        "font.size": 8,
        "mathtext.fontset": "stix",
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }):
        fig, axes = plt.subplots(2, len(epochs), figsize=(width, height), squeeze=False)
        fig.subplots_adjust(left=0.105, right=0.86, bottom=0.14, top=0.83,
                            wspace=0.13, hspace=0.24)
        images = []
        row_labels = [r"Full kernel $WAD^{-1}W^\top$", r"Base routing $WD^{-1}W^\top$"]
        panel = 0
        for row, kernel in enumerate(("full", "base")):
            for col, (epoch, matrix) in enumerate(zip(epochs, matrices[kernel])):
                axis = axes[row, col]
                image = axis.imshow(matrix, origin="lower", cmap="viridis",
                                    vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
                images.append(image)
                title = "Epoch 1" if epoch == 0 else f"Epoch {epoch}"
                axis.set_title(f"({chr(97 + panel)}) {title}", loc="left", pad=3)
                axis.set_xticks(range(n))
                axis.set_yticks(range(n))
                axis.set_xticklabels([str(i + 1) for i in range(n)] if row == 1 else [])
                axis.set_yticklabels([str(i + 1) for i in range(n)] if col == 0 else [])
                axis.tick_params(length=2.0, width=0.5, pad=1.2)
                if row == 1:
                    axis.set_xlabel("Block index", labelpad=2)
                if col == 0:
                    axis.set_ylabel(row_labels[row], labelpad=9)
                for spine in axis.spines.values():
                    spine.set_linewidth(0.55)
                    spine.set_color("#777777")
                if annotate_cells:
                    annotate(axis, matrix)
                panel += 1
        fig.suptitle(f"{display}: cross-layer propagation-kernel similarity",
                     fontsize=9.2, fontweight="semibold", y=0.965)
        cbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), fraction=0.035,
                            pad=0.025, aspect=28)
        cbar.set_label(r"Frobenius cosine similarity, $\rho_{ij}$", fontsize=7.5, labelpad=5)
        cbar.ax.tick_params(labelsize=7.1, width=0.5, length=2)
        cbar.outline.set_linewidth(0.55)
        stem = out / f"{dataset}_similarity_heatmaps"
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white", bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight",
                    metadata={"Title": f"{display} cross-layer kernel similarity"})
        fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
        plt.close(fig)

    csv_path = out / f"{dataset}_similarity_heatmaps_data.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "run_directory", "kernel", "requested_epoch",
                         "source_validation_index", "layer_i", "layer_j", "similarity"])
        for kernel in ("full", "base"):
            for epoch, index, matrix in zip(epochs, indices, matrices[kernel]):
                for i in range(n):
                    for j in range(n):
                        writer.writerow([dataset, run, kernel, epoch, index, i + 1, j + 1,
                                         f"{matrix[i, j]:.10g}"])
    caption_path = out / f"{dataset}_similarity_heatmaps_caption.tex"
    caption_path.write_text(
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{{dataset}_similarity_heatmaps.pdf}}\n"
        f"  \\caption{{\\textbf{{Evolution of cross-layer propagation-kernel similarity on {display}.}} "
        "Columns show epoch 1, epoch 100, and epoch 200. "
        "The upper row is the matched full kernel $WAD^{-1}W^\\top$ and the lower row is the matched "
        "base routing kernel $WD^{-1}W^\\top$. The common 0--1 color scale makes temporal changes "
        "directly comparable; epoch 1 is sourced from validation_000001.}}\n"
        f"  \\label{{fig:{dataset}-kernel-similarity-heatmaps}}\n"
        "\\end{figure}\n",
        encoding="utf-8",
    )
    return {"dataset": dataset, "display_name": display, "run_directory": str(run),
            "epochs": [{"requested_epoch": e, "source_validation_index": i} for e, i in zip(epochs, indices)],
            "png": str((out / f"{dataset}_similarity_heatmaps.png").resolve()),
            "pdf": str((out / f"{dataset}_similarity_heatmaps.pdf").resolve()),
            "svg": str((out / f"{dataset}_similarity_heatmaps.svg").resolve()),
            "csv": str(csv_path.resolve()), "caption": str(caption_path.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    plot_dir = Path(__file__).resolve().parent
    monitor_dir = plot_dir.parent
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=plot_dir / "output")
    parser.add_argument("--epochs", default="1,100,200")
    parser.add_argument("--run", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--profile", choices=("iclr", "two-column"), default="iclr")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    epochs = parse_epochs(args.epochs)
    root = args.input_root.expanduser().resolve() if args.input_root else default_input_root(monitor_dir)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    overrides = parse_overrides(args.run)
    width, height = ((5.5, 4.7) if args.profile == "iclr" else (7.0, 5.1))
    manifest = {"input_root": str(root), "profile": args.profile,
                "requested_epochs": list(epochs),
                "epoch_one_mapping": "validation_000001", "figures": []}
    for dataset, display in DATASETS:
        run = choose_run(root, dataset, overrides.get(dataset))
        manifest["figures"].append(plot_one(dataset, display, run, epochs, out,
                                              args.dpi, args.annotate, width, height))
    manifest_path = out / "four_dataset_similarity_heatmaps_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Input root: {root}")
    print(f"Output directory: {out}")
    for item in manifest["figures"]:
        print(f"{item['display_name']}: {item['png']}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
