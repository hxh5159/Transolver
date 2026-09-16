"""Tiny standalone target used to exercise the monitor CLI end to end."""

import argparse
import torch
import torch.nn as nn


class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dimension=8, heads=2, slices=4):
        super().__init__()
        self.heads = heads
        self.dim_head = dimension // heads
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.in_project_slice = nn.Linear(self.dim_head, slices)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch, points, _ = inputs.shape
        values = inputs.reshape(batch, points, self.heads, self.dim_head).permute(0, 2, 1, 3)
        weights = self.softmax(self.in_project_slice(values) / self.temperature)
        norm = weights.sum(dim=2) + 1e-5
        tokens = torch.einsum("bhnc,bhnm->bhmc", values, weights) / norm.unsqueeze(-1)
        attention = self.softmax(
            self.to_q(tokens) @ self.to_k(tokens).transpose(-1, -2)
            * self.dim_head ** -0.5)
        output = torch.einsum("bhmc,bhnm->bhnc", attention @ self.to_v(tokens), weights)
        return output.permute(0, 2, 1, 3).reshape(batch, points, -1)


class TinyTransolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            Physics_Attention_Irregular_Mesh(),
            Physics_Attention_Irregular_Mesh(),
            Physics_Attention_Irregular_Mesh(),
        ])

    def forward(self, inputs):
        for block in self.blocks:
            inputs = inputs + block(inputs)
        return inputs


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1)
arguments = parser.parse_args()

torch.manual_seed(41)
model = TinyTransolver()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
inputs = torch.randn(2, 12, 8)

for _ in range(arguments.steps):
    model.train()
    optimizer.zero_grad()
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    model(inputs[:1])
    model(inputs[1:])
