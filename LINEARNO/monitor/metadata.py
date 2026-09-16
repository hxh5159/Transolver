"""Run-directory and reproducibility metadata helpers."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path


def jsonable(value):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def create_run_directory(output_root, dataset, run_name=None):
    root = Path(output_root).expanduser().resolve() / dataset
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if run_name is not None and (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_name)
            or ".." in run_name):
        raise ValueError(
            "run_name must be 1-80 ASCII letters/digits/._- characters, "
            "start with a letter or digit, and not contain '..'")
    base = f"{stamp}_{run_name}" if run_name else stamp
    candidate = root / base
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{base}_{suffix:02d}"
    candidate.mkdir()
    return candidate


def git_metadata(repository_root):
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=repository_root,
            check=True, capture_output=True, text=True).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            check=True, capture_output=True, text=True).stdout.strip()
        dirty_text = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            check=True, capture_output=True, text=True).stdout
        return {"root": root, "commit": commit, "dirty": bool(dirty_text.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"root": None, "commit": None, "dirty": None}


def parse_target_arguments(arguments):
    """Best-effort representation; the unmodified argv remains authoritative."""
    parsed = {}
    positionals = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token.startswith("--"):
            if "=" in token:
                key, value = token[2:].split("=", 1)
                parsed[key] = value
            elif index + 1 < len(arguments) and not arguments[index + 1].startswith("-"):
                parsed[token[2:]] = arguments[index + 1]
                index += 1
            else:
                parsed[token[2:]] = True
        else:
            positionals.append(token)
        index += 1
    if positionals:
        parsed["_positionals"] = positionals
    return parsed


def environment_metadata(repository_root, query_cuda=False):
    import torch

    gpu_names = []
    cuda_available = None
    gpu_count = None
    if query_cuda:
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            try:
                gpu_names.append(torch.cuda.get_device_name(index))
            except Exception:
                gpu_names.append(None)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "device_query_deferred": not query_cuda,
        "git": git_metadata(repository_root),
    }


def shell_command(parts):
    return shlex.join([str(part) for part in parts])
