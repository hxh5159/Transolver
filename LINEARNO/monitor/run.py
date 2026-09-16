#!/usr/bin/env python3
"""Run an existing Transolver training script with optional kernel monitoring."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import runpy
import sys
import traceback
from pathlib import Path

MONITOR_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MONITOR_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from LINEARNO.monitor.metadata import (  # noqa: E402
    create_run_directory, environment_metadata, parse_target_arguments,
    shell_command, write_json,
)
from LINEARNO.monitor.runtime import PropagationMonitor  # noqa: E402
from LINEARNO.monitor.task_registry import ENTRYPOINT_HINTS, canonical_dataset  # noqa: E402

# Direct script execution prepends this directory. It is not needed after the
# package imports and could otherwise shadow a target module with a common name.
while str(MONITOR_DIR) in sys.path:
    sys.path.remove(str(MONITOR_DIR))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Execute an unchanged Transolver training entrypoint while observing "
            "cross-layer spatial propagation-kernel similarity during validation."
        ))
    parser.add_argument("--dataset", required=True, help="dataset/task name")
    parser.add_argument("--run-name", default=None, help="optional suffix after the timestamp")
    parser.add_argument(
        "--workdir", default=None,
        help="target working directory; defaults to the target script directory")
    parser.add_argument(
        "--capture-every-validations", type=int, default=1,
        help="capture each kth validation pass (default: 1)")
    parser.add_argument(
        "--no-capture-first", action="store_true",
        help="do not additionally capture validation pass 1 when cadence is greater than 1")
    parser.add_argument(
        "--max-samples-per-snapshot", type=int, default=8,
        help="maximum evaluated cases/batch items aggregated in one snapshot")
    parser.add_argument(
        "--max-points", type=int, default=4096,
        help="maximum independently sampled rows per point axis; <=0 means exact")
    parser.add_argument("--monitor-seed", type=int, default=1729)
    parser.add_argument(
        "--no-plots", action="store_true",
        help="save numeric results only (use plot_snapshot.py later)")
    parser.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="after --: target Python script followed by its unchanged arguments")
    return parser


def normalize_target_command(command, invocation_cwd):
    command = list(command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("missing target script after --")
    executable_names = {"python", "python3", Path(sys.executable).name, sys.executable}
    if command[0] in executable_names:
        command.pop(0)
        if command and command[0] == "-u":
            command.pop(0)
    if not command or command[0].startswith("-"):
        raise ValueError("the target must be a Python script path, not a module or option")
    script = Path(command[0]).expanduser()
    if not script.is_absolute():
        script = (invocation_cwd / script).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"target script does not exist: {script}")
    return script, command[1:]


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    dataset = canonical_dataset(arguments.dataset)
    invocation_cwd = Path.cwd().resolve()
    target_script, target_arguments = normalize_target_command(
        arguments.command, invocation_cwd)
    workdir = (
        Path(arguments.workdir).expanduser().resolve()
        if arguments.workdir else target_script.parent
    )
    if not workdir.is_dir():
        raise NotADirectoryError(f"target working directory does not exist: {workdir}")

    run_dir = create_run_directory(MONITOR_DIR / "output", dataset, arguments.run_name)
    started_at = dt.datetime.now().astimezone().isoformat()
    repository_root = REPOSITORY_ROOT
    target_argv = [str(target_script)] + target_arguments
    monitor_config = {
        "capture_every_validations": arguments.capture_every_validations,
        "capture_first": not arguments.no_capture_first,
        "max_samples_per_snapshot": arguments.max_samples_per_snapshot,
        "max_points": arguments.max_points,
        "seed": arguments.monitor_seed,
        "plots": not arguments.no_plots,
        "capture_phase": "model.eval() forwards only",
        "factor_compute": "detached float32 factors on the model tensor device",
        "large_mesh_sampling": "independent deterministic rows for the two point axes",
    }
    config = {
        "config_version": 1,
        "dataset": dataset,
        "started_at": started_at,
        "monitor_run_directory": str(run_dir),
        "wrapper_command": shell_command([sys.executable] + sys.argv),
        "target": {
            "script": str(target_script),
            "working_directory": str(workdir),
            "argv": target_argv,
            "command": shell_command([sys.executable] + target_argv),
            "parsed_arguments_best_effort": parse_target_arguments(target_arguments),
            "expected_entrypoint_hint": ENTRYPOINT_HINTS[dataset],
        },
        "monitor": monitor_config,
        # Do not initialize CUDA before the target has parsed and applied --gpu.
        "environment": environment_metadata(repository_root, query_cuda=False),
        "kernel_definitions": {
            "base": "P_base = W D^{-1} W^T",
            "full": "P_full = W A D^{-1} W^T",
            "D": "diag(W^T 1 + 1e-5)",
            "primary_similarity": (
                "Frobenius cosine, averaged after optimal cross-layer head matching"
            ),
        },
    }
    write_json(run_dir / "run_config.json", config)

    monitor = PropagationMonitor(
        run_dir=run_dir,
        capture_every=arguments.capture_every_validations,
        capture_first=not arguments.no_capture_first,
        max_samples=arguments.max_samples_per_snapshot,
        max_points=arguments.max_points,
        seed=arguments.monitor_seed,
        make_plots=not arguments.no_plots,
    ).install()

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    inserted_paths = []
    resolved_argument_namespaces = []
    original_parse_args = argparse.ArgumentParser.parse_args

    def record_parse_args(parser, *parse_args, **parse_kwargs):
        namespace = original_parse_args(parser, *parse_args, **parse_kwargs)
        try:
            resolved_argument_namespaces.append({
                "program": parser.prog,
                "arguments": vars(namespace).copy(),
            })
        except Exception:
            pass
        return namespace

    target_error = None
    exit_code = 0
    try:
        argparse.ArgumentParser.parse_args = record_parse_args
        for path in (str(target_script.parent), str(workdir)):
            if path not in sys.path:
                sys.path.insert(0, path)
                inserted_paths.append(path)
        os.chdir(workdir)
        sys.argv = target_argv
        runpy.run_path(str(target_script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            target_error = {
                "type": "SystemExit",
                "message": repr(exc.code),
                "traceback": traceback.format_exc(),
            }
            exit_code = exc.code if isinstance(exc.code, int) else 1
    except BaseException as exc:
        target_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
        monitor_summary = monitor.close()
        os.chdir(old_cwd)
        sys.argv = old_argv
        for path in inserted_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass

    finished_at = dt.datetime.now().astimezone().isoformat()
    config["target"]["resolved_argument_namespaces"] = resolved_argument_namespaces
    config["environment"] = environment_metadata(repository_root, query_cuda=True)
    config["finished_at"] = finished_at
    write_json(run_dir / "run_config.json", config)
    combined_summary = dict(monitor_summary)
    combined_summary.update({
        "status": "failed" if target_error else "completed",
        "target_status": "failed" if target_error else "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "target_error": target_error,
    })
    write_json(run_dir / "summary.json", combined_summary)
    print(f"[Transolver monitor] artifacts: {run_dir}")
    if target_error:
        sys.stderr.write(target_error["traceback"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
