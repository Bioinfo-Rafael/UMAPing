"""B_phi(y, t): the only *learned* part of the low-dimensional UMAP force
(docs/method.md, Algorithm 4). Distills the global mean repulsive field
B_X(y, t) = E_{c ~ X}[g_minus(y, y_c(t))] into a residual MLP with a
sinusoidal time embedding, so a single query point can evaluate an
approximate expected repulsion without ever touching another query point or
the O(N) reference set at inference time.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from umaping.models.mlp import get_activation


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 1000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time_embed_dim must be even")
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(max_period), half))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation: str = "gelu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            get_activation(activation),
            nn.Linear(dim, dim),
        )
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class RepulsionField(nn.Module):
    """input = (y: embedding_dim, Fourier(t): time_embed_dim) -> output: embedding_dim"""

    def __init__(
        self,
        embedding_dim: int = 2,
        hidden_dim: int = 256,
        n_residual_blocks: int = 4,
        time_embed_dim: int = 32,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.time_embedding = FourierTimeEmbedding(time_embed_dim)
        self.input_proj = nn.Linear(embedding_dim + time_embed_dim, hidden_dim)
        self.input_act = get_activation(activation)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, activation) for _ in range(n_residual_blocks)])
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        # Start near a benign zero-repulsion prior; the field is learned from there.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.embedding_dim = embedding_dim

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_embedding(t)
        h = self.input_act(self.input_proj(torch.cat([y, te], dim=-1)))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)
