"""DPR-style dual encoder (Neighborhood Retriever) and its candidate index.

Two independent towers ``f_theta`` (query) and ``g_psi`` (key) map the fixed
preprocessed feature space to a shared L2-normalized retrieval space, scored
by inner product / temperature. The retriever is trained against the
*directed* fuzzy membership matrix Mu (see training/retriever.py) and, at
inference, its only job is to produce a high-recall *candidate* set; final
edge weights are always recomputed analytically from exact distances
(inference.py), so the retriever never needs to regress fuzzy weights.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from umaping.models.mlp import MLP


class DualEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        retrieval_dim: int,
        activation: str = "gelu",
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_encoder = MLP(input_dim, hidden_dims, retrieval_dim, activation)
        self.key_encoder = MLP(input_dim, hidden_dims, retrieval_dim, activation)
        self.temperature = temperature

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(x), dim=-1)

    def encode_key(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_encoder(x), dim=-1)

    def score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return (q @ k.transpose(-2, -1)) / self.temperature


class NeighborIndex(Protocol):
    """Common contract for candidate retrieval backends: numpy in, numpy out,
    so callers (inference.py, evaluation/retrieval.py) never need to know
    which backend is active."""

    def search(self, queries: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]: ...


class ExactChunkedIndex:
    """Deterministic exact inner-product search, chunked over the reference
    key dimension so a full (n_queries x N) score matrix is never
    materialized at once. This is the default backend (N <= ~16k here); an
    ANN backend can be swapped in via :func:`build_neighbor_index` without
    touching any calling code."""

    def __init__(self, keys: np.ndarray, temperature: float = 1.0, chunk_size: int = 4096, device: str = "cpu"):
        self.device = torch.device(device)
        self.keys = torch.as_tensor(keys, dtype=torch.float32, device=self.device)
        self.temperature = temperature
        self.chunk_size = chunk_size

    @torch.no_grad()
    def search(self, queries: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        q = torch.as_tensor(queries, dtype=torch.float32, device=self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        n = self.keys.shape[0]
        topk = min(topk, n)
        batch = q.shape[0]
        best_scores = torch.full((batch, topk), float("-inf"), device=self.device)
        best_idx = torch.full((batch, topk), -1, dtype=torch.long, device=self.device)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            chunk = self.keys[start:end]
            scores = (q @ chunk.T) / self.temperature
            k_here = min(topk, scores.shape[1])
            chunk_scores, chunk_idx = torch.topk(scores, k_here, dim=1)
            chunk_idx = chunk_idx + start
            merged_scores = torch.cat([best_scores, chunk_scores], dim=1)
            merged_idx = torch.cat([best_idx, chunk_idx], dim=1)
            best_scores, sel = torch.topk(merged_scores, topk, dim=1)
            best_idx = torch.gather(merged_idx, 1, sel)
        return best_idx.cpu().numpy(), best_scores.cpu().numpy()


class PyNNDescentIndex:
    """Optional approximate backend, illustrating how an ANN index can be
    plugged in behind the same :class:`NeighborIndex` contract. Not the
    default: kept for future scalability beyond the ~16k-point regime used
    here, and deliberately not required (FAISS/HNSW are never mandatory)."""

    def __init__(self, keys: np.ndarray, n_neighbors: int = 30, seed: int = 0):
        import pynndescent  # local import: optional-backend dependency boundary

        self._index = pynndescent.NNDescent(
            keys.astype(np.float32),
            n_neighbors=max(n_neighbors, 10),
            metric="cosine",
            random_state=seed,
        )
        self._index.prepare()

    def search(self, queries: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.atleast_2d(queries).astype(np.float32)
        idx, cosine_dist = self._index.query(queries, k=topk)
        similarity = 1.0 - cosine_dist
        return idx, similarity


def build_neighbor_index(
    backend: str, keys: np.ndarray, temperature: float = 1.0, device: str = "cpu", seed: int = 0
) -> NeighborIndex:
    if backend == "exact":
        return ExactChunkedIndex(keys, temperature=temperature, device=device)
    if backend == "pynndescent":
        return PyNNDescentIndex(keys, seed=seed)
    raise ValueError(f"Unknown ann_backend '{backend}'")
