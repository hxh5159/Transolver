"""Low-rank Transolver propagation-kernel construction and comparison."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class LayerFactors:
    layer_index: int
    module_name: str
    base_u: torch.Tensor
    full_u: torch.Tensor
    v: torch.Tensor
    n_points: int
    row_u: torch.Tensor
    row_v: torch.Tensor
    exact: bool

    @property
    def batch_size(self):
        return int(self.base_u.shape[0])

    @property
    def heads(self):
        return int(self.base_u.shape[1])

    @property
    def slices(self):
        return int(self.base_u.shape[-1])


def make_row_samples(n_points, max_points, seed):
    """Return independent row samples for the two point axes of P."""
    if max_points <= 0 or n_points <= max_points:
        rows = torch.arange(n_points, dtype=torch.long)
        return rows, rows.clone(), True
    generator_u = torch.Generator(device="cpu")
    generator_v = torch.Generator(device="cpu")
    generator_u.manual_seed(int(seed))
    generator_v.manual_seed(int(seed) + 1)
    row_u = torch.randperm(n_points, generator=generator_u)[:max_points].sort().values
    row_v = torch.randperm(n_points, generator=generator_v)[:max_points].sort().values
    return row_u, row_v, False


@torch.no_grad()
def build_factors(slice_weights, attention, row_u, row_v, layer_index, module_name,
                  epsilon=1e-5, storage_dtype=None, storage_device=None):
    """Build factors for W D^-1 W^T and W A D^-1 W^T.

    Inputs have shapes W=[B,H,N,M] and A=[B,H,M,M]. Only sampled factor rows
    are retained. D is nevertheless computed from all N points.
    """
    if slice_weights.ndim != 4 or attention.ndim != 4:
        raise ValueError("expected W=[B,H,N,M] and A=[B,H,M,M]")
    batch, heads, n_points, slices = slice_weights.shape
    if tuple(attention.shape[:2]) != (batch, heads):
        raise ValueError("W and A batch/head dimensions differ")
    if tuple(attention.shape[-2:]) != (slices, slices):
        raise ValueError("attention slice dimensions do not match W")

    device = slice_weights.device
    rows_u_device = row_u.to(device=device)
    rows_v_device = row_v.to(device=device)
    w_u = slice_weights.detach().index_select(-2, rows_u_device)
    w_v = slice_weights.detach().index_select(-2, rows_v_device)
    norm = slice_weights.detach().sum(dim=-2).add(float(epsilon))
    base_u = w_u
    full_u = torch.matmul(w_u, attention.detach())
    v = w_v / norm.unsqueeze(-2)

    def store(tensor):
        target_device = tensor.device if storage_device is None else storage_device
        target_dtype = tensor.dtype if storage_dtype is None else storage_dtype
        return tensor.to(device=target_device, dtype=target_dtype).contiguous()

    return LayerFactors(
        layer_index=layer_index,
        module_name=module_name,
        base_u=store(base_u),
        full_u=store(full_u),
        v=store(v),
        n_points=int(n_points),
        row_u=row_u.clone(),
        row_v=row_v.clone(),
        exact=len(row_u) == n_points and len(row_v) == n_points,
    )


def factor_cosine(u_i, v_i, u_j, v_j, epsilon=1e-12):
    """Frobenius cosine of (U_i V_i^T) and (U_j V_j^T)."""
    if u_i.shape[0] != u_j.shape[0] or v_i.shape[0] != v_j.shape[0]:
        raise ValueError("kernel factors must use common point-row samples")
    gram_u = u_i.transpose(0, 1).matmul(u_j)
    gram_v = v_i.transpose(0, 1).matmul(v_j)
    inner = torch.sum(gram_u * gram_v)
    norm_i_sq = torch.sum(
        u_i.transpose(0, 1).matmul(u_i) * v_i.transpose(0, 1).matmul(v_i))
    norm_j_sq = torch.sum(
        u_j.transpose(0, 1).matmul(u_j) * v_j.transpose(0, 1).matmul(v_j))
    denominator = torch.sqrt(torch.clamp(norm_i_sq, min=0.0)) * torch.sqrt(
        torch.clamp(norm_j_sq, min=0.0))
    value = inner / (denominator + epsilon)
    return float(torch.clamp(value, min=-1.0, max=1.0).item())


def pairwise_head_similarity(left, right, kernel, batch_index):
    if left.n_points != right.n_points:
        raise ValueError("layers act on different point counts")
    left_u = left.full_u if kernel == "full" else left.base_u
    right_u = right.full_u if kernel == "full" else right.base_u
    left_u = left_u[batch_index]
    left_v = left.v[batch_index]
    right_u = right_u[batch_index]
    right_v = right.v[batch_index]

    # Compute every H_i x H_j comparison together. This is the same trace
    # identity as factor_cosine, but avoids thousands of Python-level matmuls.
    gram_u = torch.einsum("hnr,kns->hkrs", left_u, right_u)
    gram_v = torch.einsum("hnr,kns->hkrs", left_v, right_v)
    inner = torch.sum(gram_u * gram_v, dim=(-2, -1))

    left_uu = torch.einsum("hnr,hns->hrs", left_u, left_u)
    left_vv = torch.einsum("hnr,hns->hrs", left_v, left_v)
    right_uu = torch.einsum("hnr,hns->hrs", right_u, right_u)
    right_vv = torch.einsum("hnr,hns->hrs", right_v, right_v)
    left_norm = torch.sqrt(torch.clamp(torch.sum(left_uu * left_vv, dim=(-2, -1)), min=0.0))
    right_norm = torch.sqrt(torch.clamp(torch.sum(right_uu * right_vv, dim=(-2, -1)), min=0.0))
    similarity = inner / (left_norm[:, None] * right_norm[None, :] + 1e-12)
    return similarity.clamp(-1.0, 1.0).detach().cpu().double().numpy()


def _scipy_assignment(similarity):
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(-similarity)
        return rows.astype(np.int64), columns.astype(np.int64)
    except ImportError:
        return None


def _dp_assignment(similarity):
    """Exact dependency-free fallback for the repository's usual <=8 heads."""
    transposed = False
    matrix = similarity
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T
        transposed = True
    rows, columns = matrix.shape
    if rows > 10:
        available = set(range(columns))
        choices = []
        for row in range(rows):
            column = max(available, key=lambda item: matrix[row, item])
            available.remove(column)
            choices.append(column)
    else:
        states = {0: (0.0, ())}
        for row in range(rows):
            next_states = {}
            for mask, (score, chosen) in states.items():
                for column in range(columns):
                    if mask & (1 << column):
                        continue
                    new_mask = mask | (1 << column)
                    candidate = (score + float(matrix[row, column]), chosen + (column,))
                    if new_mask not in next_states or candidate[0] > next_states[new_mask][0]:
                        next_states[new_mask] = candidate
            states = next_states
        _, choices = max(states.values(), key=lambda item: item[0])
    row_ids = np.arange(rows, dtype=np.int64)
    column_ids = np.asarray(choices, dtype=np.int64)
    if transposed:
        return column_ids, row_ids
    return row_ids, column_ids


def optimal_head_matching(similarity):
    assignment = _scipy_assignment(similarity)
    rows, columns = assignment if assignment is not None else _dp_assignment(similarity)
    score = float(np.mean(similarity[rows, columns])) if len(rows) else float("nan")
    return score, rows, columns


def compare_layers(layers, kernel, batch_index):
    layer_count = len(layers)
    max_heads = max(layer.heads for layer in layers)
    matched = np.full((layer_count, layer_count), np.nan, dtype=np.float64)
    same_head = np.full_like(matched, np.nan)
    pairwise = np.full(
        (layer_count, layer_count, max_heads, max_heads), np.nan, dtype=np.float64)
    assignments = np.full((layer_count, layer_count, max_heads), -1, dtype=np.int64)

    for index_i, left in enumerate(layers):
        for index_j in range(index_i, layer_count):
            right = layers[index_j]
            similarity = pairwise_head_similarity(left, right, kernel, batch_index)
            pairwise[index_i, index_j, :left.heads, :right.heads] = similarity
            diagonal = min(left.heads, right.heads)
            diagonal_score = float(
                np.mean(similarity[np.arange(diagonal), np.arange(diagonal)]))
            same_head[index_i, index_j] = diagonal_score
            score, rows, columns = optimal_head_matching(similarity)
            matched[index_i, index_j] = score
            for row, column in zip(rows, columns):
                assignments[index_i, index_j, row] = column
            if index_i != index_j:
                pairwise[index_j, index_i, :right.heads, :left.heads] = similarity.T
                same_head[index_j, index_i] = diagonal_score
                matched[index_j, index_i] = score
                for row, column in zip(rows, columns):
                    assignments[index_j, index_i, column] = row
    return {
        "matched": matched,
        "same_head": same_head,
        "pairwise": pairwise,
        "assignments": assignments,
    }


def compare_forward(layers, batch_index):
    ordered = sorted(layers, key=lambda item: item.layer_index)
    return {
        "layer_indices": np.asarray([item.layer_index for item in ordered], dtype=np.int64),
        "layer_names": [item.module_name for item in ordered],
        "base": compare_layers(ordered, "base", batch_index),
        "full": compare_layers(ordered, "full", batch_index),
    }
