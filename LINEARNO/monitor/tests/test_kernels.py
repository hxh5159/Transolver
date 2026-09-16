import sys
import unittest
from pathlib import Path

import numpy as np
import torch

MONITOR_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MONITOR_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from LINEARNO.monitor.kernels import (  # noqa: E402
    build_factors, factor_cosine, optimal_head_matching, pairwise_head_similarity,
)


class KernelFormulaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        logits = torch.randn(1, 2, 7, 3, dtype=torch.float64)
        attention_logits = torch.randn(1, 2, 3, 3, dtype=torch.float64)
        self.weights = torch.softmax(logits, dim=-1)
        self.attention = torch.softmax(attention_logits, dim=-1)
        rows = torch.arange(7)
        self.factors = build_factors(
            self.weights, self.attention, rows, rows, 0, "Block 1")

    def _explicit(self, head, full):
        weights = self.weights[0, head]
        norm = weights.sum(dim=0) + 1e-5
        if full:
            return weights @ self.attention[0, head] @ torch.diag(1.0 / norm) @ weights.T
        return weights @ torch.diag(1.0 / norm) @ weights.T

    def test_trace_formula_matches_explicit_kernel(self):
        for full in (False, True):
            u = self.factors.full_u if full else self.factors.base_u
            measured = factor_cosine(
                u[0, 0], self.factors.v[0, 0],
                u[0, 1], self.factors.v[0, 1])
            left = self._explicit(0, full)
            right = self._explicit(1, full)
            expected = torch.sum(left * right) / (
                torch.linalg.norm(left) * torch.linalg.norm(right))
            self.assertAlmostEqual(measured, expected.item(), places=11)

    def test_slice_permutation_invariance(self):
        permutation = torch.tensor([2, 0, 1])
        permuted_w = self.weights[..., permutation]
        permuted_a = self.attention[..., permutation, :][..., permutation]
        rows = torch.arange(7)
        permuted = build_factors(
            permuted_w, permuted_a, rows, rows, 1, "Block 2")
        for full in (False, True):
            original_u = self.factors.full_u if full else self.factors.base_u
            permuted_u = permuted.full_u if full else permuted.base_u
            value = factor_cosine(
                original_u[0, 0], self.factors.v[0, 0],
                permuted_u[0, 0], permuted.v[0, 0])
            self.assertAlmostEqual(value, 1.0, places=11)

    def test_vectorized_head_matrix_matches_scalar_formula(self):
        measured = pairwise_head_similarity(
            self.factors, self.factors, "full", batch_index=0)
        for head_i in range(self.factors.heads):
            for head_j in range(self.factors.heads):
                expected = factor_cosine(
                    self.factors.full_u[0, head_i], self.factors.v[0, head_i],
                    self.factors.full_u[0, head_j], self.factors.v[0, head_j])
                self.assertAlmostEqual(measured[head_i, head_j], expected, places=11)

    def test_optimal_head_matching_is_permutation_invariant(self):
        matrix = np.asarray([
            [0.1, 0.2, 0.99],
            [0.98, 0.3, 0.2],
            [0.2, 0.97, 0.1],
        ])
        score, rows, columns = optimal_head_matching(matrix)
        self.assertAlmostEqual(score, (0.99 + 0.98 + 0.97) / 3)
        self.assertEqual(dict(zip(rows.tolist(), columns.tolist())), {0: 2, 1: 0, 2: 1})


if __name__ == "__main__":
    unittest.main()
