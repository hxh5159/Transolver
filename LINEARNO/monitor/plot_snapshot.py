#!/usr/bin/env python3
"""Regenerate heatmaps from a monitor snapshot's numeric archive."""

import argparse
import sys
from pathlib import Path

MONITOR_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MONITOR_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402

from LINEARNO.monitor.plotting import plot_snapshot  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", help="directory containing similarity_values.npz")
    args = parser.parse_args()
    snapshot = Path(args.snapshot).expanduser().resolve()
    with np.load(snapshot / "similarity_values.npz") as values:
        plot_snapshot(
            snapshot,
            values["base_matched_mean"], values["full_matched_mean"],
            values["base_matched_std"], values["full_matched_std"])
    print(f"plots written to {snapshot}")


if __name__ == "__main__":
    main()
