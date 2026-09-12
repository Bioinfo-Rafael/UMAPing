"""UMAP kNN graph construction.

Produces two distinct objects from a reference set X (see docs/method.md,
Algorithm 1):

* **directed** fuzzy membership ``Mu`` (``mu_{i->j}``) -- the retrieval
  teacher, never symmetrized;
* **symmetric** fuzzy union ``W`` -- used for the spectral encoder and the
  reference mean dynamics.

The smooth-kNN math (``smooth_knn_dist`` / membership-strength formula) is a
deliberate, from-scratch reimplementation rather than an import of
umap-learn's private ``umap.umap_.smooth_knn_dist``. That function was
verified (2026-09, umap-learn v0.5.12, see RUNTIME_CHECKS.md) to hard-code
the assumption that column 0 of its input distance matrix is a self-distance
of 0 -- its sigma binary search always sums ``distances[i, 1:]``, skipping
column 0 unconditionally. That assumption holds when umap-learn builds the
*fit-time* graph (nearest neighbor search on the reference set returns each
point as its own closest "neighbor"), but not for *transform*-time queries,
which upstream patches over with a ``local_connectivity - 1`` fudge factor
applied only to query rows.

Here, `K_i` is defined (per the project spec) to *exclude the query/reference
point itself from the neighbor search itself* -- there is never a self
column, for the reference graph or for a query. So one consistent formula
(summing over all k columns, using the same `local_connectivity` value in
both cases) is used everywhere; see `test_graph.py` for a numerical
cross-check against actual umap-learn output that accounts for this
column-count difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

SMOOTH_K_TOLERANCE = 1e-5


def chunked_exact_knn(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    chunk_size: int = 1024,
    exclude_self_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic exact Euclidean kNN, chunked over the reference set so a
    full (n_query x n_reference) distance matrix is never fully materialized.

    Parameters
    ----------
    query, reference: (M, d) / (N, d) float arrays in the same feature space.
    exclude_self_indices: optional (M,) int array. If given, row i additionally
        excludes reference row ``exclude_self_indices[i]`` from its candidate
        neighbors (used when the reference set is queried against itself).

    Returns
    -------
    knn_idx: (M, k) int64, knn_dist: (M, k) float32, both sorted ascending by
    distance.
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    m = query.shape[0]
    n = reference.shape[0]
    if exclude_self_indices is not None and n <= k:
        raise ValueError("Not enough reference points to exclude self and still return k neighbors")

    ref_sq_norm = np.einsum("ij,ij->i", reference, reference)
    knn_idx = np.empty((m, k), dtype=np.int64)
    knn_dist = np.empty((m, k), dtype=np.float32)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        q_chunk = query[start:end]
        q_sq_norm = np.einsum("ij,ij->i", q_chunk, q_chunk)
        # ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2, one BLAS matmul per chunk.
        sq_dist = q_sq_norm[:, None] - 2.0 * (q_chunk @ reference.T) + ref_sq_norm[None, :]
        np.maximum(sq_dist, 0.0, out=sq_dist)

        if exclude_self_indices is not None:
            rows = np.arange(end - start)
            sq_dist[rows, exclude_self_indices[start:end]] = np.inf

        part = np.argpartition(sq_dist, kth=k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(sq_dist, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        knn_idx[start:end] = np.take_along_axis(part, order, axis=1)
        knn_dist[start:end] = np.sqrt(np.take_along_axis(part_dist, order, axis=1))

    return knn_idx, knn_dist


def smooth_knn_dist(
    knn_dists: np.ndarray,
    local_connectivity: float = 1.0,
    n_iter: int = 64,
    bandwidth: float = 1.0,
    min_k_dist_scale: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (rho_i, sigma_i) via the standard UMAP smooth-kNN binary search,
    adapted for a `knn_dists` matrix with *no* self column (see module
    docstring). ``target = log2(k) * bandwidth``, exactly as upstream.
    """
    n_samples, k = knn_dists.shape
    target = np.log2(k) * bandwidth
    rho = np.zeros(n_samples, dtype=np.float64)
    sigma = np.zeros(n_samples, dtype=np.float64)
    mean_all = float(np.mean(knn_dists)) if knn_dists.size else 0.0

    for i in range(n_samples):
        row = knn_dists[i].astype(np.float64)
        non_zero = row[row > 0.0]

        if non_zero.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho_i = float(non_zero[index - 1])
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho_i += interpolation * (non_zero[index] - non_zero[index - 1])
            else:
                rho_i = interpolation * float(non_zero[0])
        elif non_zero.shape[0] > 0:
            rho_i = float(np.max(non_zero))
        else:
            rho_i = 0.0
        rho[i] = rho_i

        lo, hi, mid = 0.0, np.inf, 1.0
        for _ in range(n_iter):
            diff = row - rho_i
            psum = float(np.sum(np.where(diff <= 0.0, 1.0, np.exp(-diff / mid))))
            if abs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                mid = mid * 2.0 if np.isinf(hi) else (lo + hi) / 2.0
        sigma_i = mid

        if rho_i > 0.0:
            mean_row = float(np.mean(row))
            if sigma_i < min_k_dist_scale * mean_row:
                sigma_i = min_k_dist_scale * mean_row
        else:
            if sigma_i < min_k_dist_scale * mean_all:
                sigma_i = min_k_dist_scale * mean_all
        sigma[i] = sigma_i

    return sigma.astype(np.float32), rho.astype(np.float32)


def _membership_values(dists: np.ndarray, rho: np.ndarray, sigma: np.ndarray, eps: float) -> np.ndarray:
    """exp(-max(0, d - rho) / sigma), broadcasting dists/rho/sigma together."""
    diff = dists - rho
    sigma_safe = np.maximum(sigma, eps)
    return np.where(diff <= 0.0, 1.0, np.exp(-diff / sigma_safe)).astype(np.float32)


def compute_directed_membership(
    knn_idx: np.ndarray,
    knn_dists: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
    n_reference: int,
    eps: float = 1e-8,
) -> sp.csr_matrix:
    """mu_{i->j} = exp(-max(0, d(x_i,x_j) - rho_i) / sigma_i) for j in K_i, 0
    otherwise. Returns a (n_query, n_reference) sparse directed matrix."""
    n_query, k = knn_idx.shape
    rows = np.repeat(np.arange(n_query), k)
    cols = knn_idx.reshape(-1)
    vals = _membership_values(knn_dists.reshape(-1), np.repeat(rho, k), np.repeat(sigma, k), eps)
    mu = sp.coo_matrix((vals, (rows, cols)), shape=(n_query, n_reference)).tocsr()
    mu.eliminate_zeros()
    return mu


def symmetrize_fuzzy_union(mu: sp.csr_matrix, set_op_mix_ratio: float = 1.0) -> sp.csr_matrix:
    """W = Mu + Mu.T - Mu (elementwise*) Mu.T, i.e. the probabilistic t-conorm
    fuzzy union umap-learn calls `fuzzy_simplicial_set`'s symmetrization
    (verified current source, RUNTIME_CHECKS.md)."""
    transpose = mu.transpose().tocsr()
    prod = mu.multiply(transpose).tocsr()
    if set_op_mix_ratio == 1.0:
        result = mu + transpose - prod
    elif set_op_mix_ratio == 0.0:
        result = prod
    else:
        result = set_op_mix_ratio * (mu + transpose - prod) + (1.0 - set_op_mix_ratio) * prod
    result = result.tocsr()
    result.eliminate_zeros()
    return result


@dataclass
class ReferenceGraph:
    knn_idx: np.ndarray
    knn_dists: np.ndarray
    sigma: np.ndarray
    rho: np.ndarray
    mu: sp.csr_matrix
    w: sp.csr_matrix


def build_reference_graph(
    x: np.ndarray,
    n_neighbors: int,
    local_connectivity: float = 1.0,
    n_iter: int = 64,
    bandwidth: float = 1.0,
    min_k_dist_scale: float = 1e-3,
    set_op_mix_ratio: float = 1.0,
    eps: float = 1e-8,
    chunk_size: int = 1024,
) -> ReferenceGraph:
    """Build the reference kNN graph: directed Mu (retrieval teacher) and
    symmetric W (spectral + dynamics). K_i excludes point i itself."""
    n = x.shape[0]
    self_idx = np.arange(n)
    knn_idx, knn_dists = chunked_exact_knn(
        x, x, k=n_neighbors, chunk_size=chunk_size, exclude_self_indices=self_idx
    )
    sigma, rho = smooth_knn_dist(
        knn_dists,
        local_connectivity=local_connectivity,
        n_iter=n_iter,
        bandwidth=bandwidth,
        min_k_dist_scale=min_k_dist_scale,
    )
    mu = compute_directed_membership(knn_idx, knn_dists, sigma, rho, n_reference=n, eps=eps)
    w = symmetrize_fuzzy_union(mu, set_op_mix_ratio=set_op_mix_ratio)
    return ReferenceGraph(knn_idx=knn_idx, knn_dists=knn_dists, sigma=sigma, rho=rho, mu=mu, w=w)


def query_fuzzy_weights(
    neighbor_dists: np.ndarray,
    local_connectivity: float = 1.0,
    n_iter: int = 64,
    bandwidth: float = 1.0,
    min_k_dist_scale: float = 1e-3,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query-side mu_{*->j}: same smooth-kNN formula as the reference graph
    (see module docstring for why no `local_connectivity - 1` adjustment is
    needed here), applied to a query's exact top-k reranked neighbor
    distances. `neighbor_dists` may be a single row (1, k) or a batch (B, k);
    no query ever contributes to another query's graph.

    Returns a *dense* (n_query, k) weight array in the same column order as
    `neighbor_dists` (unlike `compute_directed_membership`, which returns a
    sparse (n_query, n_reference) matrix suited to the reference graph) --
    at query time we only ever need the k weights for the retrieved/reranked
    neighbor ids we already have in hand, in that same order.
    """
    neighbor_dists = np.atleast_2d(neighbor_dists)
    sigma, rho = smooth_knn_dist(
        neighbor_dists,
        local_connectivity=local_connectivity,
        n_iter=n_iter,
        bandwidth=bandwidth,
        min_k_dist_scale=min_k_dist_scale,
    )
    weights = _membership_values(neighbor_dists, rho[:, None], sigma[:, None], eps)
    return weights, sigma, rho


def row_degree(w: sp.csr_matrix) -> np.ndarray:
    return np.asarray(w.sum(axis=1)).ravel().astype(np.float32)


def normalized_laplacian(w: sp.csr_matrix, degree: np.ndarray | None = None, eps: float = 1e-12) -> sp.csr_matrix:
    """L_sym = I - D^{-1/2} W D^{-1/2}."""
    if degree is None:
        degree = row_degree(w)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, eps))
    d_mat = sp.diags(d_inv_sqrt)
    n = w.shape[0]
    l_sym = sp.identity(n, format="csr", dtype=np.float32) - (d_mat @ w @ d_mat)
    return l_sym.tocsr()


def upper_triangular_edges(w: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, col, weight) for i < j only -- each undirected edge once, used
    for the spectral encoder's Dirichlet-energy minibatches."""
    coo = sp.triu(w, k=1).tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float32)


def full_edges(w: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, col, weight) over every stored entry (both directions), used for
    the reference-dynamics scatter-add attraction sum A_i = sum_j W_ij g_plus(...)."""
    coo = w.tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float32)
