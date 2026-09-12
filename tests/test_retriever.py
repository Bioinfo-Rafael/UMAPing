"""Tests for training/retriever.py and models/retriever.py.

The core invariant under test: the retriever must never treat a query point
as a valid negative for itself (whether it appears at its own diagonal
position or anywhere else in the candidate pool), and known-but-unsampled
Mu-neighbors are excluded from the negative pool when configured to be.

Written to be run remotely (see RUNTIME_CHECKS.md); not executed here.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from umaping.config import RetrieverConfig
from umaping.models.retriever import DualEncoder, ExactChunkedIndex
from umaping.training.retriever import _sample_positive_per_row, build_negative_mask, train_retriever


def _toy_mu(n: int = 6, seed: int = 0, degree: int = 2) -> sp.csr_matrix:
    """A directed fuzzy membership matrix with no self-loops and `degree`
    outgoing edges per row."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        targets = rng.choice(others, size=degree, replace=False)
        for j in targets:
            rows.append(i)
            cols.append(j)
            vals.append(rng.uniform(0.2, 1.0))
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def test_sample_positive_never_returns_self():
    mu = _toy_mu(n=10)
    rng = np.random.default_rng(42)
    query_idx = np.arange(10)
    for _ in range(50):
        positives = _sample_positive_per_row(mu, query_idx, rng)
        assert np.all(positives != query_idx)


def test_sample_positive_only_returns_true_mu_neighbors():
    mu = _toy_mu(n=8)
    rng = np.random.default_rng(7)
    query_idx = np.arange(8)
    for _ in range(50):
        positives = _sample_positive_per_row(mu, query_idx, rng)
        for i, j in zip(query_idx, positives):
            assert mu[i, j] > 0.0


def test_negative_mask_excludes_query_itself_wherever_it_appears():
    mu = _toy_mu(n=6)
    query_idx = np.array([0, 1, 2])
    # candidate_idx[i] is meant to be query i's own sampled positive (the
    # real call pattern), but candidate_idx[1] here happens to equal query 0
    # itself -- point 0 must never be usable as a negative for query 0, even
    # though it sits off the diagonal (at query 1's position, not query 0's).
    candidate_idx = np.array([3, 0, 5])
    mask = build_negative_mask(query_idx, candidate_idx, mu, exclude_known_neighbors=False)

    assert mask[0, 1] == True  # noqa: E712 -- query 0 must mask candidate "0" wherever it appears
    assert mask[1, 1] == False  # noqa: E712 -- diagonal (query 1's own sampled positive) stays unmasked


def test_negative_mask_never_masks_the_sampled_positive_diagonal():
    mu = _toy_mu(n=5)
    rng = np.random.default_rng(3)
    query_idx = np.arange(5)
    positives = _sample_positive_per_row(mu, query_idx, rng)
    candidate_idx = positives  # diagonal candidates ARE the sampled positives
    mask = build_negative_mask(query_idx, candidate_idx, mu, exclude_known_neighbors=True)
    assert not np.any(np.diag(mask))


def test_negative_mask_excludes_known_neighbors_when_enabled():
    mu = _toy_mu(n=6, degree=2)
    query_idx = np.array([0])
    known_js = mu[0].indices
    assert len(known_js) >= 2
    sampled_positive, other_known_neighbor = known_js[0], known_js[1]
    candidate_idx = np.array([sampled_positive, other_known_neighbor])

    mask = build_negative_mask(query_idx, candidate_idx, mu, exclude_known_neighbors=True)
    assert mask[0, 0] == False  # noqa: E712 -- diagonal (the sampled positive) stays unmasked
    assert mask[0, 1] == True  # noqa: E712 -- a genuinely-known-but-unsampled neighbor IS masked out

    mask_disabled = build_negative_mask(query_idx, candidate_idx, mu, exclude_known_neighbors=False)
    assert mask_disabled[0, 1] == False  # noqa: E712 -- without the flag it's an ordinary negative


def test_train_retriever_runs_and_produces_finite_losses():
    """Smoke test: a handful of steps on a tiny model/dataset should run
    without shape errors and produce finite losses."""
    torch.manual_seed(0)
    n, d = 20, 6
    features = np.random.default_rng(0).normal(size=(n, d)).astype(np.float32)
    mu = _toy_mu(n=n)

    model = DualEncoder(input_dim=d, hidden_dims=[16], retrieval_dim=8, activation="gelu", temperature=0.1)
    cfg = RetrieverConfig(hidden_dims=[16], retrieval_dim=8, batch_size=6, n_random_negatives=4, steps=5, log_every=1)
    state = train_retriever(model, features, mu, cfg, device=torch.device("cpu"), seed=0, progress=False)

    assert state.step == 5
    assert len(state.losses) == 5
    assert all(np.isfinite(loss) for loss in state.losses)


def test_exact_chunked_index_recovers_true_top_k_by_inner_product():
    rng = np.random.default_rng(1)
    keys = rng.normal(size=(50, 4)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    index = ExactChunkedIndex(keys, temperature=1.0, chunk_size=7)  # deliberately smaller than N

    query = keys[[3]]  # the query IS one of the keys -> should retrieve itself with the top score
    ids, scores = index.search(query, topk=5)

    assert ids[0, 0] == 3
    brute_force_scores = (query @ keys.T).ravel()
    top5_expected = np.argsort(-brute_force_scores)[:5]
    assert set(ids[0].tolist()) == set(top5_expected.tolist())
