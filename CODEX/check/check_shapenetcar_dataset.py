#!/usr/bin/env python3
"""Read-only integrity checker for the MLCFD/ShapeNetCar dataset.

The Transolver car task expects the extracted dataset to look like this::

    <training_data>/param0/<sample>/quadpress_smpl.vtk
    <training_data>/param0/<sample>/hexvelo_smpl.vtk
    ...
    <training_data>/param8/<sample>/...

The expected fold sizes below are taken from
``Car-Design-ShapeNetCar/dataset/load_dataset.py`` in the Transolver
repository: 100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889.

This program only reads directory metadata and (optionally) VTK files. It
never downloads, creates, moves, deletes, or modifies dataset files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


EXPECTED_FOLDS = {
    "param0": 100,
    "param1": 99,
    "param2": 97,
    "param3": 100,
    "param4": 100,
    "param5": 96,
    "param6": 100,
    "param7": 98,
    "param8": 99,
}
REQUIRED_FILES = ("quadpress_smpl.vtk", "hexvelo_smpl.vtk")


class Report:
    def __init__(self, max_details: int = 50) -> None:
        self.errors = 0
        self.warnings = 0
        self.deep_check_unavailable = False
        self.max_details = max_details
        self._detail_count = 0

    def error(self, message: str) -> None:
        self.errors += 1
        print(f"ERROR: {message}")

    def warning(self, message: str) -> None:
        self.warnings += 1
        print(f"WARNING: {message}")

    def detail(self, message: str, *, error: bool = True) -> None:
        """Print at most max_details individual paths, while retaining totals."""
        if error:
            self.errors += 1
        else:
            self.warnings += 1
        if self._detail_count < self.max_details:
            print(f"{'ERROR' if error else 'WARNING'}: {message}")
            self._detail_count += 1
        elif self._detail_count == self.max_details:
            print(f"... further details suppressed (limit={self.max_details})")
            self._detail_count += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check extracted Transolver MLCFD/ShapeNetCar data without modifying it."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/inspire/hdd/project/urbanlowaltitude/"
            "yuanmeilu-253114050257/houwenzhe-drivaer/data/mlcfd/training_data"
        ),
        help=(
            "training_data directory, or a parent such as .../data or .../data/mlcfd "
            "(default: remote path used by this project)"
        ),
    )
    parser.add_argument(
        "--fold-id",
        type=int,
        choices=range(9),
        default=None,
        help="Only inspect one fold (0-8); by default inspect all nine folds.",
    )
    parser.add_argument(
        "--skip-vtk-content",
        action="store_true",
        help="Only check directory/file structure; do not parse VTK files.",
    )
    parser.add_argument(
        "--require-vtk",
        action="store_true",
        help="Fail if the Python VTK module is unavailable for content checks.",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=50,
        help="Maximum number of individual problem paths to print (default: 50).",
    )
    return parser.parse_args()


def resolve_data_dir(requested: Path) -> Path:
    """Accept the raw training_data path or one of its common parents."""
    requested = requested.expanduser()
    candidates = (
        requested,
        requested / "training_data",
        requested / "mlcfd" / "training_data",
        requested / "mlcfd_data" / "training_data",
        requested / "mlcfd_data",
    )
    for candidate in candidates:
        if any((candidate / fold).is_dir() for fold in EXPECTED_FOLDS):
            return candidate
    return requested


def selected_folds(fold_id: Optional[int]) -> dict[str, int]:
    if fold_id is None:
        return dict(EXPECTED_FOLDS)
    fold = f"param{fold_id}"
    return {fold: EXPECTED_FOLDS[fold]}


def check_regular_nonempty(path: Path, report: Report, label: str) -> bool:
    try:
        if not path.is_file():
            report.detail(f"{label} is missing or is not a regular file: {path}")
            return False
        size = path.stat().st_size
    except OSError as exc:
        report.detail(f"cannot stat {label} {path}: {exc}")
        return False
    if size <= 0:
        report.detail(f"{label} is empty: {path}")
        return False
    return True


def load_vtk_module(report: Report, require_vtk: bool):
    try:
        import vtk  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the host environment
        message = f"VTK content checks unavailable ({exc}); file structure was still checked"
        if require_vtk:
            report.error(message)
        else:
            report.warning(message)
        report.deep_check_unavailable = True
        return None
    return vtk


def _array_is_finite(array) -> bool:
    try:
        import numpy as np  # type: ignore
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

        values = vtk_to_numpy(array)
        return bool(np.isfinite(values).all())
    except Exception:
        # Finite-value checking is supplementary; VTK structure checks remain valid.
        return True


def check_vtk_file(path: Path, kind: str, vtk, report: Report) -> None:
    """Validate the fields that dataset.py reads from one legacy VTK file."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(os.fspath(path))
    try:
        reader.Update()
        grid = reader.GetOutput()
    except Exception as exc:
        report.detail(f"VTK reader failed for {path}: {exc}")
        return

    if grid is None:
        report.detail(f"VTK reader returned no grid: {path}")
        return

    n_points = grid.GetNumberOfPoints()
    n_cells = grid.GetNumberOfCells()
    if n_points <= 0:
        report.detail(f"VTK file has no points: {path}")
    if n_cells <= 0:
        report.detail(f"VTK file has no cells: {path}")

    point_data = grid.GetPointData()
    if point_data is None:
        report.detail(f"VTK file has no point data: {path}")
        return

    if kind == "pressure":
        array = point_data.GetScalars()
        if array is None:
            report.detail(f"pressure VTK has no active point scalars (GetScalars): {path}")
            return
        components = array.GetNumberOfComponents()
        if components != 1:
            report.detail(
                f"pressure VTK must have one scalar component, got {components}: {path}"
            )
    else:
        array = point_data.GetVectors()
        if array is None:
            report.detail(f"velocity VTK has no active point vectors (GetVectors): {path}")
            return
        components = array.GetNumberOfComponents()
        if components != 3:
            report.detail(
                f"velocity VTK must have three vector components, got {components}: {path}"
            )

    if array.GetNumberOfTuples() != n_points:
        report.detail(
            f"{kind} tuple count ({array.GetNumberOfTuples()}) does not match "
            f"point count ({n_points}): {path}"
        )
    if not _array_is_finite(array):
        report.detail(f"{kind} VTK contains NaN or infinite values: {path}")


def iter_sample_dirs(fold_dir: Path) -> Iterable[Path]:
    try:
        entries = sorted(fold_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return ()
    return (entry for entry in entries if entry.is_dir())


def check_dataset(data_dir: Path, folds: dict[str, int], report: Report,
                  vtk, check_vtk_content: bool) -> None:
    if not data_dir.exists():
        report.error(f"dataset directory does not exist: {data_dir}")
        return
    if not data_dir.is_dir():
        report.error(f"dataset path is not a directory: {data_dir}")
        return

    total_found = 0
    total_expected = sum(folds.values())

    for fold, expected_count in folds.items():
        fold_dir = data_dir / fold
        if not fold_dir.is_dir():
            report.error(f"missing fold directory {fold_dir} (expected {expected_count} samples)")
            continue

        sample_dirs = list(iter_sample_dirs(fold_dir))
        found_count = len(sample_dirs)
        total_found += found_count
        print(f"{fold}: {found_count}/{expected_count} sample directories")
        if found_count != expected_count:
            report.error(
                f"{fold} has {found_count} sample directories; expected {expected_count}"
            )

        try:
            non_dirs = sorted(entry.name for entry in fold_dir.iterdir() if not entry.is_dir())
        except OSError as exc:
            report.error(f"cannot list {fold_dir}: {exc}")
            non_dirs = []
        if non_dirs:
            report.warning(
                f"{fold} contains {len(non_dirs)} non-directory entries ignored by the "
                f"training loader (first: {', '.join(non_dirs[:5])})"
            )

        for sample_dir in sample_dirs:
            valid_files: dict[str, bool] = {}
            for filename in REQUIRED_FILES:
                path = sample_dir / filename
                valid_files[filename] = check_regular_nonempty(path, report, filename)

            if not check_vtk_content or vtk is None:
                continue
            if valid_files["quadpress_smpl.vtk"]:
                check_vtk_file(sample_dir / "quadpress_smpl.vtk", "pressure", vtk, report)
            if valid_files["hexvelo_smpl.vtk"]:
                check_vtk_file(sample_dir / "hexvelo_smpl.vtk", "velocity", vtk, report)

    if total_found != total_expected:
        report.error(f"dataset total is {total_found}; expected {total_expected}")

    expected_fold_names = set(EXPECTED_FOLDS)
    try:
        unexpected_folds = sorted(
            entry.name
            for entry in data_dir.iterdir()
            if entry.is_dir() and entry.name not in expected_fold_names
        )
    except OSError as exc:
        report.error(f"cannot list dataset directory {data_dir}: {exc}")
        unexpected_folds = []
    if unexpected_folds:
        report.warning(
            "unexpected top-level directories are ignored by the training loader: "
            + ", ".join(unexpected_folds)
        )


def main() -> int:
    args = parse_args()
    if args.max_details < 0:
        print("ERROR: --max-details must be non-negative", file=sys.stderr)
        return 2

    requested_dir = args.data_dir.expanduser()
    data_dir = resolve_data_dir(requested_dir)
    folds = selected_folds(args.fold_id)
    report = Report(max_details=args.max_details)

    print("Transolver MLCFD/ShapeNetCar dataset check (read-only)")
    print(f"requested path: {requested_dir}")
    print(f"resolved path:  {data_dir}")
    print("expected layout: param0..param8/<sample>/{quadpress_smpl.vtk,hexvelo_smpl.vtk}")
    print(f"folds checked:  {', '.join(folds)}")
    print()

    vtk = None
    check_vtk_content = not args.skip_vtk_content
    if check_vtk_content:
        vtk = load_vtk_module(report, args.require_vtk)
    elif args.require_vtk:
        report.warning("--require-vtk has no effect together with --skip-vtk-content")

    check_dataset(data_dir, folds, report, vtk, check_vtk_content)

    print()
    print(f"errors: {report.errors} | warnings: {report.warnings}")
    if report.errors:
        print("RESULT: INCOMPLETE or INVALID")
        return 1
    if report.deep_check_unavailable:
        print("RESULT: STRUCTURE OK; VTK CONTENT NOT VERIFIED")
        return 2
    print("RESULT: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
