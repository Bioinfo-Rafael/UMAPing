"""Algorithm 2: Neighborhood Retriever training (docs/method.md).

Trains the DPR-style dual encoder against the *directed* fuzzy membership
matrix Mu (never the symmetrized graph W) with weighted positive sampling,
in-batch + random negatives, and an InfoNCE/DPR loss. Never constructs an
N x N score matrix: every step scores a (batch_size, batch_size +
n_random_negatives) matrix only.

This module is a pure function of (model, data, config, seed): it never
touches the filesystem. Checkpointing/resumability is the caller's job
(see pipeline.py), so this stays trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import trange

from umaping.config import RetrieverConfig
from umaping.models.retriever import DualEncoder


@dataclass
class RetrieverTrainState:
    step: int = 0
    losses: list[float] = field(default_factory=list)
    batch_top1_acc: list[float] = field(default_factory=list)


def _sample_positive_per_row(mu: sp.csr_matrix, query_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each query i, sample one positive j with probability proportional
    to mu_{i,j} (the directed fuzzy membership row)."""
    indptr, indices, data = mu.indptr, mu.indices, mu.data
    positives = np.empty(len(query_idx), dtype=np.int64)
    for out_pos, i in enumerate(query_idx):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            raise ValueError(f"Reference point {i} has no outgoing Mu edges; cannot sample a positive neighbor.")
        row_idx = indices[start:end]
        row_w = data[start:end]
        probs = row_w / row_w.sum()
        positives[out_pos] = row_idx[rng.choice(len(row_idx), p=probs)]
    return positives


def build_negative_mask(
    query_idx: np.ndarray,
    candidate_idx: np.ndarray,
    mu: sp.csr_matrix,
    exclude_known_neighbors: bool,
) -> np.ndarray:
    """(B, C) boolean mask of candidate positions that must NOT be treated as
    negatives for each query: candidates that are literally the query point
    itself, and -- when `exclude_known_neighbors` is set -- candidates that
    are already a true (if unsampled) Mu-neighbor of that query. The sampled
    positive (assumed to sit at `candidate_idx[i]` for query `i`, i.e. the
    diagonal) is always explicitly unmasked, since it is itself a known
    Mu-neighbor by construction and must remain the label."""
    b = len(query_idx)
    mask = candidate_idx[None, :] == query_idx[:, None]
    if exclude_known_neighbors:
        known = np.asarray(mu[query_idx][:, candidate_idx].todense()) > 0.0
        mask = mask | known
    diag = np.arange(b)
    mask[diag, diag] = False
    return mask


def train_retriever(
    model: DualEncoder,
    features: np.ndarray,
    mu: sp.csr_matrix,
    cfg: RetrieverConfig,
    device: torch.device,
    seed: int = 0,
    resume_state: RetrieverTrainState | None = None,
    progress: bool = True,
) -> RetrieverTrainState:
    n = features.shape[0]
    features_t = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)

    rng = np.random.default_rng(seed + (resume_state.step if resume_state else 0))
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    state = resume_state or RetrieverTrainState()
    iterator = trange(state.step, cfg.steps, disable=not progress, desc="retriever", initial=state.step, total=cfg.steps)

    for step in iterator:
        query_idx = rng.integers(0, n, size=cfg.batch_size)
        pos_idx = _sample_positive_per_row(mu, query_idx, rng)
        rand_neg_idx = (
            rng.integers(0, n, size=cfg.n_random_negatives)
            if cfg.n_random_negatives > 0
            else np.empty(0, dtype=np.int64)
        )
        candidate_idx = np.concatenate([pos_idx, rand_neg_idx])

        q = model.encode_query(features_t[query_idx])
        k = model.encode_key(features_t[candidate_idx])
        logits = model.score(q, k)  # (B, B + R)

        b = len(query_idx)
        mask = build_negative_mask(query_idx, candidate_idx, mu, cfg.exclude_known_neighbors_from_negatives)

        logits = logits.masked_fill(torch.as_tensor(mask, device=device), float("-inf"))
        target = torch.arange(b, device=device)
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == target).float().mean().item()
        state.step = step + 1
        state.losses.append(float(loss.item()))
        state.batch_top1_acc.append(acc)
        if progress and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            iterator.set_postfix(loss=float(loss.item()), acc=acc)

    return state
