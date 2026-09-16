"""Runtime-only hooks and artifact writing for Transolver monitoring."""

from __future__ import annotations

import csv
import datetime as dt
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook

from .kernels import build_factors, compare_forward, make_row_samples
from .metadata import write_json


PHYSICS_ATTENTION_CLASSES = {
    "Physics_Attention_Irregular_Mesh",
    "Physics_Attention_Structured_Mesh_2D",
    "Physics_Attention_Structured_Mesh_3D",
}


def is_physics_attention(module):
    return (
        module.__class__.__name__ in PHYSICS_ATTENTION_CLASSES
        and hasattr(module, "softmax")
        and hasattr(module, "in_project_slice")
        and hasattr(module, "heads")
    )


class MonitorLog:
    def __init__(self, path):
        self.path = Path(path)

    def write(self, level, message):
        now = dt.datetime.now().astimezone().isoformat()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{now} [{level}] {message}\n")
        if level in {"WARNING", "ERROR"}:
            print(f"[Transolver monitor] {level.lower()}: {message}")

    def info(self, message):
        self.write("INFO", message)

    def warning(self, message):
        self.write("WARNING", message)

    def error(self, message):
        self.write("ERROR", message)


class PropagationMonitor:
    """Observe validation forwards without changing model source or outputs."""

    def __init__(self, run_dir, capture_every=1, capture_first=True,
                 max_samples=8, max_points=4096, seed=1729, make_plots=True):
        self.run_dir = Path(run_dir).resolve()
        self.snapshot_root = self.run_dir / "snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.capture_every = int(capture_every)
        self.capture_first = bool(capture_first)
        self.max_samples = int(max_samples)
        self.max_points = int(max_points)
        self.seed = int(seed)
        self.make_plots = bool(make_plots)
        if self.capture_every <= 0:
            raise ValueError("capture_every must be positive")
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")

        self.log = MonitorLog(self.run_dir / "monitor.log")
        self._handles = []
        self._thread = threading.local()
        self._phase = None
        self._validation_index = 0
        self._capture_active = False
        self._forward_serial = 0
        self._current_layers = []
        self._current_ids = set()
        self._row_cache = {}
        self._samples = []
        self._sampling = []
        self._layer_registry = {}
        self._errors = []
        self._snapshots = []
        self._closed = False

    def install(self):
        self._handles = [
            register_module_forward_pre_hook(self._pre_hook),
            register_module_forward_hook(self._post_hook),
        ]
        self.log.info("global forward hooks installed; only evaluation forwards are captured")
        return self

    def _context_stack(self):
        if not hasattr(self._thread, "stack"):
            self._thread.stack = []
        return self._thread.stack

    def _record_error(self, context, exception):
        message = f"{context}: {type(exception).__name__}: {exception}"
        self._errors.append({
            "time": dt.datetime.now().astimezone().isoformat(),
            "context": context,
            "error": repr(exception),
            "traceback": traceback.format_exc(),
        })
        self.log.error(message + "; training continues")

    def _register_layer(self, module):
        key = id(module)
        if key not in self._layer_registry:
            index = len(self._layer_registry)
            slice_projection = getattr(module, "in_project_slice", None)
            input_projection = getattr(module, "in_project_x", None)
            self._layer_registry[key] = {
                "index": index,
                "label": f"Block {index + 1}",
                "class": module.__class__.__name__,
                "module": f"{module.__class__.__module__}.{module.__class__.__name__}",
                "heads": int(getattr(module, "heads", 0)),
                "dim_head": int(getattr(module, "dim_head", 0)),
                "hidden_dim": int(getattr(module, "heads", 0)) * int(
                    getattr(module, "dim_head", 0)),
                "slices": int(getattr(slice_projection, "out_features", 0)),
                "input_projection": (
                    input_projection.__class__.__name__ if input_projection is not None else None),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in module.parameters()
                    if parameter.requires_grad),
                "temperature_shape": list(getattr(module, "temperature").shape),
                "temperature_at_discovery": getattr(module, "temperature").detach().cpu().tolist(),
            }
        return self._layer_registry[key]

    def _should_capture(self, validation_index):
        return (
            validation_index % self.capture_every == 0
            or (self.capture_first and validation_index == 1)
        )

    def _change_phase(self, phase):
        if phase == self._phase:
            return
        if self._phase == "eval":
            self._finish_forward()
            self._finish_validation()
        self._phase = phase
        if phase == "eval":
            self._validation_index += 1
            self._capture_active = self._should_capture(self._validation_index)
            self._forward_serial = 0
            self._samples = []
            self._sampling = []
            state = "capturing" if self._capture_active else "observing cadence only"
            self.log.info(f"validation {self._validation_index}: {state}")

    def _start_new_forward_if_needed(self, module):
        module_id = id(module)
        if module_id in self._current_ids:
            self._finish_forward()
        self._current_ids.add(module_id)

    def _pre_hook(self, module, inputs):
        if not is_physics_attention(module):
            return
        try:
            phase = "train" if module.training else "eval"
            self._change_phase(phase)
            layer = self._register_layer(module)
            capture = (
                phase == "eval" and self._capture_active
                and len(self._samples) < self.max_samples
            )
            if capture:
                self._start_new_forward_if_needed(module)
            self._context_stack().append({
                "module": module,
                "layer": layer,
                "capture": capture,
                "softmax_outputs": [],
            })
        except Exception as exc:
            self._record_error("physics-attention pre-hook", exc)

    def _sample_rows(self, n_points):
        if n_points not in self._row_cache:
            derived_seed = (
                self.seed + self._validation_index * 1_000_003
                + self._forward_serial * 9_176 + int(n_points) * 17
            )
            self._row_cache[n_points] = make_row_samples(
                n_points, self.max_points, derived_seed)
        return self._row_cache[n_points]

    def _post_hook(self, module, inputs, output):
        stack = self._context_stack()
        if stack and module is getattr(stack[-1]["module"], "softmax", None):
            if stack[-1]["capture"] and torch.is_tensor(output):
                stack[-1]["softmax_outputs"].append(output.detach())
            return
        if not is_physics_attention(module):
            return
        try:
            context = None
            for position in range(len(stack) - 1, -1, -1):
                if stack[position]["module"] is module:
                    context = stack.pop(position)
                    break
            if context is None or not context["capture"]:
                return
            outputs = context["softmax_outputs"]
            if len(outputs) != 2:
                raise RuntimeError(
                    f"expected two shared-Softmax outputs (W and A), observed {len(outputs)}")
            weights, attention = outputs
            if weights.ndim != 4 or attention.ndim != 4:
                raise RuntimeError(
                    f"unexpected Softmax shapes: W={tuple(weights.shape)}, A={tuple(attention.shape)}")
            row_u, row_v, _ = self._sample_rows(int(weights.shape[-2]))
            layer = context["layer"]
            factors = build_factors(
                weights, attention, row_u, row_v,
                layer_index=layer["index"], module_name=layer["label"],
                storage_dtype=torch.float32)
            self._current_layers.append(factors)
        except Exception as exc:
            self._record_error("physics-attention post-hook", exc)

    def _finish_forward(self):
        if not self._current_layers:
            self._current_ids.clear()
            self._row_cache.clear()
            return
        try:
            layers = sorted(self._current_layers, key=lambda item: item.layer_index)
            batch_size = min(layer.batch_size for layer in layers)
            remaining = self.max_samples - len(self._samples)
            for batch_index in range(min(batch_size, remaining)):
                result = compare_forward(layers, batch_index)
                result["forward_index"] = self._forward_serial
                result["batch_index"] = batch_index
                self._samples.append(result)
            first = layers[0]
            self._sampling.append({
                "forward_index": self._forward_serial,
                "n_points": first.n_points,
                "sampled_rows_u": first.row_u.tolist(),
                "sampled_rows_v": first.row_v.tolist(),
                "sample_count_u": int(first.row_u.numel()),
                "sample_count_v": int(first.row_v.numel()),
                "exact": first.exact,
            })
            self._forward_serial += 1
        except Exception as exc:
            self._record_error("finalize monitored forward", exc)
        finally:
            self._current_layers = []
            self._current_ids.clear()
            self._row_cache.clear()

    @staticmethod
    def _write_matrix_csv(path, matrix):
        labels = [f"Block {index + 1}" for index in range(matrix.shape[0])]
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([""] + labels)
            for label, row in zip(labels, matrix):
                writer.writerow([label] + [f"{value:.10g}" for value in row])

    @staticmethod
    def _stack_metric(samples, kernel, field):
        return np.stack([sample[kernel][field] for sample in samples], axis=0)

    def _write_snapshot(self):
        snapshot_name = f"validation_{self._validation_index:06d}"
        snapshot_dir = self.snapshot_root / snapshot_name
        snapshot_dir.mkdir(parents=True, exist_ok=False)

        reference_layers = self._samples[0]["layer_indices"]
        valid_samples = [
            sample for sample in self._samples
            if np.array_equal(sample["layer_indices"], reference_layers)
        ]
        if len(valid_samples) != len(self._samples):
            self.log.warning("discarded samples with a different executed layer set")

        arrays = {}
        for kernel in ("base", "full"):
            for field in ("matched", "same_head", "pairwise", "assignments"):
                arrays[f"{kernel}_{field}"] = self._stack_metric(valid_samples, kernel, field)
            arrays[f"{kernel}_matched_mean"] = np.mean(arrays[f"{kernel}_matched"], axis=0)
            arrays[f"{kernel}_matched_std"] = np.std(arrays[f"{kernel}_matched"], axis=0)
            arrays[f"{kernel}_same_head_mean"] = np.mean(
                arrays[f"{kernel}_same_head"], axis=0)
            arrays[f"{kernel}_same_head_std"] = np.std(
                arrays[f"{kernel}_same_head"], axis=0)
        arrays["layer_indices"] = reference_layers
        np.savez_compressed(snapshot_dir / "similarity_values.npz", **arrays)
        np.savez_compressed(
            snapshot_dir / "head_matching.npz",
            base_assignments=arrays["base_assignments"],
            full_assignments=arrays["full_assignments"],
            base_pairwise=arrays["base_pairwise"],
            full_pairwise=arrays["full_pairwise"],
        )

        self._write_matrix_csv(
            snapshot_dir / "full_kernel_similarity.csv", arrays["full_matched_mean"])
        self._write_matrix_csv(
            snapshot_dir / "base_kernel_similarity.csv", arrays["base_matched_mean"])

        with (snapshot_dir / "per_sample_metrics.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "sample", "kernel", "layer_i", "layer_j",
                "matched_head_similarity", "same_head_similarity",
            ])
            for sample_index, sample in enumerate(valid_samples):
                layer_count = len(sample["layer_indices"])
                for kernel in ("base", "full"):
                    for layer_i in range(layer_count):
                        for layer_j in range(layer_count):
                            writer.writerow([
                                sample_index, kernel, layer_i + 1, layer_j + 1,
                                sample[kernel]["matched"][layer_i, layer_j],
                                sample[kernel]["same_head"][layer_i, layer_j],
                            ])

        write_json(snapshot_dir / "sampling.json", {
            "strategy": (
                "exact" if all(item["exact"] for item in self._sampling)
                else "two independent deterministic point-row samples"
            ),
            "seed": self.seed,
            "normalization_uses_all_points": True,
            "forwards": self._sampling,
        })
        write_json(snapshot_dir / "snapshot_summary.json", {
            "validation_index": self._validation_index,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "sample_count": len(valid_samples),
            "layer_count": int(len(reference_layers)),
            "primary_metric": "permutation-invariant optimal-head matched full-kernel similarity",
            "base_kernel": "W D^{-1} W^T",
            "full_kernel": "W A D^{-1} W^T",
            "full_mean_off_diagonal": self._off_diagonal_mean(arrays["full_matched_mean"]),
            "base_mean_off_diagonal": self._off_diagonal_mean(arrays["base_matched_mean"]),
        })

        if self.make_plots:
            from .plotting import plot_snapshot

            plot_snapshot(
                snapshot_dir,
                arrays["base_matched_mean"], arrays["full_matched_mean"],
                arrays["base_matched_std"], arrays["full_matched_std"])
        self._snapshots.append(str(snapshot_dir.relative_to(self.run_dir)))
        self.log.info(
            f"wrote {snapshot_name} using {len(valid_samples)} sample(s) and "
            f"{len(reference_layers)} block(s)")

    @staticmethod
    def _off_diagonal_mean(matrix):
        if matrix.shape[0] < 2:
            return None
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        return float(np.nanmean(matrix[mask]))

    def _finish_validation(self):
        if self._capture_active and self._samples:
            try:
                self._write_snapshot()
            except Exception as exc:
                self._record_error(f"write validation {self._validation_index}", exc)
        elif self._capture_active:
            self.log.warning(
                f"validation {self._validation_index} was selected but no complete sample was captured")
        self._capture_active = False
        self._samples = []
        self._sampling = []

    def _model_structure(self):
        layers = []
        for record in sorted(self._layer_registry.values(), key=lambda item: item["index"]):
            copied = dict(record)
            copied.pop("index", None)
            layers.append(copied)
        return {
            "captured_at": dt.datetime.now().astimezone().isoformat(),
            "physics_attention_blocks": len(layers),
            "layers": layers,
            "scope": (
                "spatial propagation kernels before value/output channel projections; "
                "this is not the full channel-valued block operator"
            ),
        }

    def summary(self):
        return {
            "status": "closed" if self._closed else "running",
            "validation_passes_observed": self._validation_index,
            "snapshots": list(self._snapshots),
            "monitor_error_count": len(self._errors),
            "monitor_errors": list(self._errors),
        }

    def close(self):
        if self._closed:
            return self.summary()
        try:
            if self._phase == "eval":
                self._finish_forward()
                self._finish_validation()
        except Exception as exc:
            self._record_error("close monitor", exc)
        finally:
            for handle in self._handles:
                try:
                    handle.remove()
                except Exception as exc:
                    self._record_error("remove hook", exc)
            self._handles = []
            self._closed = True
            write_json(self.run_dir / "model_structure.json", self._model_structure())
            write_json(self.run_dir / "summary.json", self.summary())
            self.log.info("hooks removed; monitor closed")
        return self.summary()
