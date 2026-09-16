import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

MONITOR_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MONITOR_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from LINEARNO.monitor.runtime import PropagationMonitor  # noqa: E402


class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dimension=6, heads=2, slices=3):
        super().__init__()
        self.heads = heads
        self.dim_head = dimension // heads
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.in_project_slice = nn.Linear(self.dim_head, slices)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, points, _ = x.shape
        split = x.reshape(batch, points, self.heads, self.dim_head).permute(0, 2, 1, 3)
        weights = self.softmax(self.in_project_slice(split) / self.temperature)
        norm = weights.sum(dim=2) + 1e-5
        tokens = torch.einsum("bhnc,bhnm->bhmc", split, weights) / norm.unsqueeze(-1)
        query, key, value = self.to_q(tokens), self.to_k(tokens), self.to_v(tokens)
        attention = self.softmax(query @ key.transpose(-1, -2) * self.dim_head ** -0.5)
        output = torch.einsum("bhmc,bhnm->bhnc", attention @ value, weights)
        return output.permute(0, 2, 1, 3).reshape(batch, points, -1)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            Physics_Attention_Irregular_Mesh(),
            Physics_Attention_Irregular_Mesh(),
        ])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class RuntimeIsolationTests(unittest.TestCase):
    def test_repository_attention_variants_are_observed(self):
        source = MONITOR_DIR.parents[1] / "Physics_Attention.py"
        specification = importlib.util.spec_from_file_location(
            "transolver_monitor_test_attention", source)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        cases = [
            (module.Physics_Attention_Irregular_Mesh(
                dim=8, heads=2, dim_head=4, slice_num=3), torch.randn(1, 6, 8)),
            (module.Physics_Attention_Structured_Mesh_2D(
                dim=8, heads=2, dim_head=4, slice_num=3, H=2, W=3),
             torch.randn(1, 6, 8)),
            (module.Physics_Attention_Structured_Mesh_3D(
                dim=8, heads=2, dim_head=4, slice_num=3, H=2, W=2, D=2),
             torch.randn(1, 8, 8)),
        ]
        for attention, inputs in cases:
            with self.subTest(attention=attention.__class__.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    monitor = PropagationMonitor(
                        directory, max_samples=1, max_points=4,
                        seed=123, make_plots=False).install()
                    attention.eval()
                    expected = attention(inputs).detach()
                    summary = monitor.close()
                    self.assertEqual(tuple(expected.shape), tuple(inputs.shape))
                    self.assertEqual(summary["monitor_error_count"], 0)
                    self.assertEqual(len(summary["snapshots"]), 1)

    def test_monitor_does_not_change_rng_outputs_or_training_step(self):
        torch.manual_seed(13)
        baseline = TinyModel()
        monitored = copy.deepcopy(baseline)
        input_tensor = torch.randn(2, 9, 6)

        baseline.eval()
        baseline_eval = baseline(input_tensor).detach()
        baseline.train()
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01)
        baseline_optimizer.zero_grad()
        baseline_loss = baseline(input_tensor).square().mean()
        baseline_loss.backward()
        baseline_gradients = [parameter.grad.clone() for parameter in baseline.parameters()]
        baseline_optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            monitor = PropagationMonitor(
                directory, max_samples=2, max_points=5, seed=123, make_plots=False).install()
            rng_before = torch.get_rng_state().clone()
            monitored.eval()
            monitored_eval = monitored(input_tensor).detach()
            rng_after = torch.get_rng_state().clone()
            monitored.train()
            monitored_optimizer = torch.optim.SGD(monitored.parameters(), lr=0.01)
            monitored_optimizer.zero_grad()
            monitored_loss = monitored(input_tensor).square().mean()
            monitored_loss.backward()
            monitored_gradients = [parameter.grad.clone() for parameter in monitored.parameters()]
            monitored_optimizer.step()
            summary = monitor.close()

            self.assertEqual(summary["monitor_error_count"], 0)
            self.assertEqual(len(summary["snapshots"]), 1)
            self.assertTrue(torch.equal(rng_before, rng_after))

        self.assertTrue(torch.equal(baseline_eval, monitored_eval))
        self.assertEqual(baseline_loss.item(), monitored_loss.item())
        for expected, actual in zip(baseline_gradients, monitored_gradients):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(baseline.parameters(), monitored.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
