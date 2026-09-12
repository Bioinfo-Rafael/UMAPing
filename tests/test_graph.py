"""Tests for graph.py: smooth-kNN semantics cross-checked against the actual
umap-learn implementation, and the symmetric fuzzy union formula.

Written to be run remotely (see RUNTIME_CHECKS.md); not executed here.
"""

from __future__ import annotations

import numpy as np
import pytest

from umaping.graph import (
    chunked_exact_knn,
    compute_directed_membership,
    row_degree,
    smooth_knn_dist,
    symmetrize_fuzzy_union,
)


def _toy_dataset(n: int = 30, d: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_smooth_knn_dist_matches_umap_learn_accounting_for_self_column():
    """umap-learn's private `smooth_knn_dist` assumes column 0 of its input
    is a self-distance of 0 and always skips it in the sigma binary search
    (verified against the current source, see RUNTIME_CHECKS.md); our K_i
    has no self column at all. Prepending a synthetic zero self-column
    recovers matching rho/sigma from both implementations on identical
    underlying neighbor distances."""
    umap_umap_ = pytest.importorskip("umap.umap_")

    x = _toy_dataset()
    k = 5
    _, knn_dists = chunked_exact_knn(x, x, k=k, exclude_self_indices=np.arange(x.shape[0]))

    sigma_ours, rho_ours = smooth_knn_dist(knn_dists, local_connectivity=1.0, n_iter=64, bandwidth=1.0)

    distances_with_self = np.concatenate(
        [np.zeros((x.shape[0], 1), dtype=np.float64), knn_dists.astype(np.float64)], axis=1
    )
    sigma_umap, rho_umap = umap_umap_.smooth_knn_dist(distances_with_self, float(k), n_iter=64, local_connectivity=1.0)

    # rho is a direct/interpolated distance value (no iterative solve), so it
    # should match tightly; sigma is float32-internal in umap-learn's numba
    # locals vs. our float64 throughout, hence the looser tolerance.
    np.testing.assert_allclose(rho_ours, rho_umap, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(sigma_ours, sigma_umap, rtol=3e-2, atol=1e-3)


def test_directed_membership_matches_umap_learn_compute_membership_strengths():
    umap_umap_ = pytest.importorskip("umap.umap_")

    x = _toy_dataset(seed=1)
    k = 4
    knn_idx, knn_dists = chunked_exact_knn(x, x, k=k, exclude_self_indices=np.arange(x.shape[0]))
    sigma, rho = smooth_knn_dist(knn_dists, local_connectivity=1.0)

    mu = compute_directed_membership(knn_idx, knn_dists, sigma, rho, n_reference=x.shape[0])

    _, _, vals_umap = umap_umap_.compute_membership_strengths(knn_idx, knn_dists, sigma, rho)
    rows = np.repeat(np.arange(x.shape[0]), k)
    cols = knn_idx.reshape(-1)
    vals_ours = np.asarray(mu[rows, cols]).ravel()

    np.testing.assert_allclose(vals_ours, vals_umap, rtol=1e-4, atol=1e-6)


def test_symmetric_fuzzy_union_matches_closed_form_and_is_symmetric():
    x = _toy_dataset(seed=2)
    k = 5
    knn_idx, knn_dists = chunked_exact_knn(x, x, k=k, exclude_self_indices=np.arange(x.shape[0]))
    sigma, rho = smooth_knn_dist(knn_dists)
    mu = compute_directed_membership(knn_idx, knn_dists, sigma, rho, n_reference=x.shape[0])

    w = symmetrize_fuzzy_union(mu, set_op_mix_ratio=1.0)
    expected = (mu + mu.transpose() - mu.multiply(mu.transpose())).tocsr()

    np.testing.assert_allclose(w.toarray(), expected.toarray(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(w.toarray(), w.toarray().T, rtol=1e-6, atol=1e-8)


def test_no_self_loops_in_directed_or_symmetric_graph():
    x = _toy_dataset(seed=3)
    k = 5
    knn_idx, knn_dists = chunked_exact_knn(x, x, k=k, exclude_self_indices=np.arange(x.shape[0]))
    sigma, rho = smooth_knn_dist(knn_dists)
    mu = compute_directed_membership(knn_idx, knn_dists, sigma, rho, n_reference=x.shape[0])
    w = symmetrize_fuzzy_union(mu)

    assert np.all(mu.diagonal() == 0.0)
    assert np.all(w.diagonal() == 0.0)
    # K_i must never contain i itself.
    assert not np.any(knn_idx == np.arange(x.shape[0])[:, None])


def test_row_degree_matches_sparse_row_sum():
    x = _toy_dataset(seed=4)
    knn_idx, knn_dists = chunked_exact_knn(x, x, k=5, exclude_self_indices=np.arange(x.shape[0]))
    sigma, rho = smooth_knn_dist(knn_dists)
    mu = compute_directed_membership(knn_idx, knn_dists, sigma, rho, n_reference=x.shape[0])
    w = symmetrize_fuzzy_union(mu)

    degree = row_degree(w)
    np.testing.assert_allclose(degree, np.asarray(w.sum(axis=1)).ravel())


def test_chunked_exact_knn_excludes_self_and_matches_brute_force():
    x = _toy_dataset(n=25, seed=5)
    k = 6
    idx, dist = chunked_exact_knn(x, x, k=k, chunk_size=7, exclude_self_indices=np.arange(x.shape[0]))

    for i in range(x.shape[0]):
        assert i not in idx[i]
        all_dists = np.linalg.norm(x - x[i], axis=1)
        all_dists[i] = np.inf
        expected_idx = np.argsort(all_dists, kind="stable")[:k]
        assert set(idx[i].tolist()) == set(expected_idx.tolist())
        np.testing.assert_allclose(sorted(dist[i]), sorted(all_dists[expected_idx]), rtol=1e-5, atol=1e-6)
