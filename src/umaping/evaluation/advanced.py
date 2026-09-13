"""Analysis-only diagnostics for already-trained runs (Experiment A).

Everything in this module is a pure function of already-computed arrays: a
frozen `ReferenceTrajectory`, a trained `RepulsionField`, and already-embedded
query/reference coordinates. Nothing here trains or retrains anything --
`pipeline.py::run_advanced_analysis` is the only caller, and it loads
existing checkpoints/memory rather than building them (see its module
docstring for the "fail loudly if artifacts are missing" contract).

Two families of diagnostics, matching the analysis spec:

1. Repulsion-field smoothness + Monte-Carlo denoising (``field_smoothness_report``,
   ``monte_carlo_denoising_test``) -- is B_phi smoother than a finite-M MC
   teacher, and does it approximate the population mean field B_X better than
   one ordinary finite-sample MC estimate does?
2. Per-query "did this query land somewhere weird" diagnostics
   (``compute_per_query_diagnostics``) -- multi-k neighbor recall, NDCG,
   fuzzy-neighborhood consistency, local displacement, tail statistics,
   class-conditional periphery percentile, and an accumulation/periphery
   score inspired by (but not a verified reproduction of) Islam and
   Fleischer's "On Out-of-sample Embedding in UMAP".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from umaping.dynamics import ReferenceTrajectory, exact_all_reference_mean_field, mean_negative_field
from umaping.graph import chunked_exact_knn
from umaping.models.repulsion import RepulsionField
from umaping.umap_forces import g_minus

# ---------------------------------------------------------------------------
# 2.1 / 2.2: repulsion-field smoothness + Monte-Carlo denoising
# ---------------------------------------------------------------------------


def make_reference_grid(
    reference_embedding: np.ndarray, resolution: int, pad_fraction: float = 0.1
) -> tuple[np.ndarray, float]:
    """A square ``(resolution, resolution, 2)`` grid covering the occupied
    reference-embedding region (padded by ``pad_fraction`` of its span).
    Returns ``(grid_points, delta)``, ``delta`` the (equal, both axes) grid
    spacing -- the same construction `evaluation/plotting.py::plot_repulsion_field`
    uses for its own visualization grid, reused here so smoothness metrics
    are computed over the same region a human would look at."""
    span = reference_embedding.max(axis=0) - reference_embedding.min(axis=0)
    pad = pad_fraction * float(np.max(span)) + 1e-6
    xmin, xmax = float(reference_embedding[:, 0].min() - pad), float(reference_embedding[:, 0].max() + pad)
    ymin, ymax = float(reference_embedding[:, 1].min() - pad), float(reference_embedding[:, 1].max() + pad)
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    delta = float(xs[1] - xs[0]) if resolution > 1 else 1.0
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    grid_points = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    return grid_points, delta


@dataclass
class FieldRoughnessMetrics:
    finite_diff_roughness: float
    second_order_roughness: float
    angular_variation_degrees: float
    magnitude_variation: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_field_roughness(field_grid: np.ndarray, delta: float) -> FieldRoughnessMetrics:
    """Metrics A-D of the smoothness spec, computed over every
    horizontally/vertically adjacent pair of grid points in a ``(H, W, d)``
    vector field sampled with spacing ``delta``."""
    h, w, _ = field_grid.shape

    dx = field_grid[:, 1:, :] - field_grid[:, :-1, :]
    dy = field_grid[1:, :, :] - field_grid[:-1, :, :]
    grad_mag = np.concatenate([np.linalg.norm(dx, axis=-1).ravel(), np.linalg.norm(dy, axis=-1).ravel()]) / delta
    finite_diff_roughness = float(grad_mag.mean())

    if h >= 3 and w >= 3:
        lap = (
            field_grid[2:, 1:-1, :]
            + field_grid[:-2, 1:-1, :]
            + field_grid[1:-1, 2:, :]
            + field_grid[1:-1, :-2, :]
            - 4.0 * field_grid[1:-1, 1:-1, :]
        ) / (delta**2)
        second_order_roughness = float(np.linalg.norm(lap, axis=-1).mean())
    else:
        second_order_roughness = float("nan")

    def _angles_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
        denom = np.maximum(nu * nv, 1e-12)
        cos = np.clip((u * v).sum(-1) / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    ang = np.concatenate(
        [
            _angles_deg(field_grid[:, 1:, :], field_grid[:, :-1, :]).ravel(),
            _angles_deg(field_grid[1:, :, :], field_grid[:-1, :, :]).ravel(),
        ]
    )
    angular_variation_degrees = float(ang.mean())

    mag = np.linalg.norm(field_grid, axis=-1)
    mag_diff = np.concatenate([np.abs(mag[:, 1:] - mag[:, :-1]).ravel(), np.abs(mag[1:, :] - mag[:-1, :]).ravel()])
    magnitude_variation = float(mag_diff.mean())

    return FieldRoughnessMetrics(
        finite_diff_roughness=finite_diff_roughness,
        second_order_roughness=second_order_roughness,
        angular_variation_degrees=angular_variation_degrees,
        magnitude_variation=magnitude_variation,
    )


def evaluate_fields_on_grid(
    repulsion_model: RepulsionField,
    trajectory: ReferenceTrajectory,
    a: float,
    b: float,
    t: float,
    grid_points: np.ndarray,
    device: torch.device,
    low_m: int,
    high_m: int,
    seed: int,
    clip: float | None = 4.0,
    exact_max_n: int = 200_000,
) -> dict[str, np.ndarray | None]:
    """The four fields to compare at one grid + one time: the learned
    ``b_phi``, an ordinary low-M Monte-Carlo teacher, a high-M Monte-Carlo
    teacher, and (when the reference set is small enough) the exact
    all-reference mean field."""
    h, w, _ = grid_points.shape
    flat = grid_points.reshape(-1, 2).astype(np.float32)
    y = torch.as_tensor(flat, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    n = trajectory.positions.shape[1]

    repulsion_model = repulsion_model.to(device).eval()
    with torch.no_grad():
        t_t = torch.full((flat.shape[0],), float(t), dtype=torch.float32, device=device)
        b_phi = repulsion_model(y, t_t).cpu().numpy().reshape(h, w, 2)

        idx_low = rng.integers(0, n, size=min(low_m, n))
        y_low = torch.as_tensor(trajectory.positions_at(t, indices=idx_low), dtype=torch.float32, device=device)
        low_m_teacher = mean_negative_field(y, y_low, a, b, clip=clip).cpu().numpy().reshape(h, w, 2)

        idx_high = rng.integers(0, n, size=min(high_m, n))
        y_high = torch.as_tensor(trajectory.positions_at(t, indices=idx_high), dtype=torch.float32, device=device)
        high_m_teacher = mean_negative_field(y, y_high, a, b, clip=clip).cpu().numpy().reshape(h, w, 2)

        exact = None
        if n <= exact_max_n:
            exact = (
                exact_all_reference_mean_field(y, trajectory, float(t), a, b, device, clip=clip)
                .cpu()
                .numpy()
                .reshape(h, w, 2)
            )

    return {"b_phi": b_phi, "low_m_teacher": low_m_teacher, "high_m_teacher": high_m_teacher, "exact": exact}


def field_smoothness_report(
    repulsion_model: RepulsionField,
    trajectory: ReferenceTrajectory,
    a: float,
    b: float,
    reference_embedding: np.ndarray,
    device: torch.device,
    ts: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    resolution: int = 24,
    low_m: int = 32,
    high_m: int = 2048,
    seed: int = 0,
    clip: float | None = 4.0,
    exact_max_n: int = 200_000,
) -> dict:
    """Section 2.1: roughness of B_phi vs. low-M/high-M/exact teachers, at
    each ``t`` in ``ts`` and aggregated ("overall") across ``ts``."""
    grid_points, delta = make_reference_grid(reference_embedding, resolution)
    per_t: dict[str, dict[str, dict]] = {}
    for t in ts:
        fields = evaluate_fields_on_grid(
            repulsion_model, trajectory, a, b, t, grid_points, device, low_m, high_m, seed, clip, exact_max_n
        )
        per_t[f"t={t:.2f}"] = {
            name: compute_field_roughness(field, delta).to_dict() for name, field in fields.items() if field is not None
        }

    overall: dict[str, dict[str, list[float]]] = {}
    for methods in per_t.values():
        for method, metrics in methods.items():
            bucket = overall.setdefault(method, {})
            for k, v in metrics.items():
                bucket.setdefault(k, []).append(v)
    overall_summary = {
        method: {k: float(np.nanmean(v)) for k, v in metrics.items()} for method, metrics in overall.items()
    }

    return {"per_t": per_t, "overall": overall_summary, "grid_resolution": resolution, "grid_delta": delta, "ts": list(ts)}


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _compare_vector_fields(pred: np.ndarray, target: np.ndarray, seed: int = 0) -> dict:
    residual = pred - target
    mse = (residual**2).sum(axis=-1)
    pred_n = np.linalg.norm(pred, axis=-1)
    target_n = np.linalg.norm(target, axis=-1)
    denom = np.maximum(pred_n * target_n, 1e-12)
    cosine = (pred * target).sum(-1) / denom
    mag_err = np.abs(pred_n - target_n)
    ci_lo, ci_hi = _bootstrap_mean_ci(mse, seed=seed)
    return {
        "mse_mean": float(mse.mean()),
        "mse_median": float(np.median(mse)),
        "mse_ci95": [ci_lo, ci_hi],
        "cosine_mean": float(cosine.mean()),
        "magnitude_error_mean": float(mag_err.mean()),
    }


def monte_carlo_denoising_test(
    repulsion_model: RepulsionField,
    trajectory: ReferenceTrajectory,
    a: float,
    b: float,
    reference_embedding: np.ndarray,
    device: torch.device,
    n_eval_points: int = 300,
    low_m: int = 32,
    r_repeats: int = 16,
    seed: int = 0,
    clip: float | None = 4.0,
    exact_max_n: int = 200_000,
    t: float | None = None,
) -> dict:
    """Section 2.2. At a fixed set of ``(y, t)`` evaluation locations: does
    the learned ``B_phi`` approximate the population mean field better than
    one ordinary finite-sample (low-M) Monte-Carlo estimate does, relative to
    ``B_mean`` (the average of ``r_repeats`` independent low-M estimates) and,
    where feasible, the exact all-reference field? If ``t`` is given, every
    evaluation location shares that fixed time (used to report this test
    broken down "by t"); otherwise each location gets an independently
    sampled time (the mixed, "overall" version of this test)."""
    rng = np.random.default_rng(seed)
    n = trajectory.positions.shape[1]
    t_min, t_max = float(trajectory.times[0]), float(trajectory.times[-1])

    grid_points, _ = make_reference_grid(reference_embedding, resolution=int(np.ceil(np.sqrt(n_eval_points))) + 1)
    flat = grid_points.reshape(-1, 2)
    n_take = min(n_eval_points, flat.shape[0])
    sel = rng.choice(flat.shape[0], size=n_take, replace=False)
    y_eval = flat[sel].astype(np.float32)
    t_eval = (
        np.full(y_eval.shape[0], float(t), dtype=np.float32)
        if t is not None
        else rng.uniform(t_min, t_max, size=y_eval.shape[0]).astype(np.float32)
    )

    y_t = torch.as_tensor(y_eval, dtype=torch.float32, device=device)
    repulsion_model = repulsion_model.to(device).eval()

    with torch.no_grad():
        t_t = torch.as_tensor(t_eval, dtype=torch.float32, device=device)
        b_phi = repulsion_model(y_t, t_t).cpu().numpy()

        mc_estimates = np.empty((r_repeats, y_eval.shape[0], 2), dtype=np.float32)
        for r in range(r_repeats):
            idx = rng.integers(0, n, size=(y_eval.shape[0], low_m))
            times_repeated = np.repeat(t_eval, low_m)
            pos = trajectory.positions_at_many(times_repeated, idx.reshape(-1)).reshape(y_eval.shape[0], low_m, -1)
            pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device)
            mc_estimates[r] = g_minus(y_t.unsqueeze(1), pos_t, a, b, clip=clip).mean(dim=1).cpu().numpy()

        b_mean = mc_estimates.mean(axis=0)
        b_single = mc_estimates[0]

        b_exact = None
        if n <= exact_max_n:
            b_exact = exact_all_reference_mean_field(y_t, trajectory, t_eval, a, b, device, clip=clip).cpu().numpy()

    result: dict = {
        "n_eval_points": int(y_eval.shape[0]),
        "low_m": low_m,
        "r_repeats": r_repeats,
        "vs_b_mean": {
            "b_phi": _compare_vector_fields(b_phi, b_mean, seed=seed),
            "single_mc_estimate": _compare_vector_fields(b_single, b_mean, seed=seed),
        },
    }
    if b_exact is not None:
        result["vs_exact"] = {
            "b_phi": _compare_vector_fields(b_phi, b_exact, seed=seed),
            "single_mc_estimate": _compare_vector_fields(b_single, b_exact, seed=seed),
        }
    result["b_phi_better_than_single_mc_vs_mean"] = bool(
        result["vs_b_mean"]["b_phi"]["mse_mean"] < result["vs_b_mean"]["single_mc_estimate"]["mse_mean"]
    )
    return result


def denoising_report_by_t(
    repulsion_model: RepulsionField,
    trajectory: ReferenceTrajectory,
    a: float,
    b: float,
    reference_embedding: np.ndarray,
    device: torch.device,
    ts: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    **kwargs,
) -> dict:
    """`monte_carlo_denoising_test`, once per fixed `t` in `ts` plus once with
    independently sampled ("mixed") times -- feeds `figures/field_denoising_error_by_t.png`."""
    per_t = {
        f"t={t:.2f}": monte_carlo_denoising_test(repulsion_model, trajectory, a, b, reference_embedding, device, t=t, **kwargs)
        for t in ts
    }
    mixed = monte_carlo_denoising_test(repulsion_model, trajectory, a, b, reference_embedding, device, t=None, **kwargs)
    return {"per_t": per_t, "mixed": mixed, "ts": list(ts)}


# ---------------------------------------------------------------------------
# 2.3: per-query "did this query land somewhere weird" diagnostics
# ---------------------------------------------------------------------------


@dataclass
class TailSummary:
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    worst_5pct_mean: float
    worst_1pct_mean: float

    def to_dict(self) -> dict:
        return asdict(self)


def tail_summary(values: np.ndarray) -> TailSummary:
    """`values` must already be oriented so that *larger = worse* (a recall
    *deficit* ``1 - recall``, not recall itself), so every percentile here
    means the same thing regardless of which underlying metric it summarizes."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        nan = float("nan")
        return TailSummary(nan, nan, nan, nan, nan, nan, nan)
    sorted_v = np.sort(v)
    n = sorted_v.shape[0]
    n5 = max(1, int(np.ceil(0.05 * n)))
    n1 = max(1, int(np.ceil(0.01 * n)))
    return TailSummary(
        mean=float(v.mean()),
        median=float(np.median(v)),
        p90=float(np.percentile(v, 90)),
        p95=float(np.percentile(v, 95)),
        p99=float(np.percentile(v, 99)),
        worst_5pct_mean=float(sorted_v[-n5:].mean()),
        worst_1pct_mean=float(sorted_v[-n1:].mean()),
    )


def multi_k_recall_and_ndcg(
    query_high_dim: np.ndarray,
    reference_high_dim: np.ndarray,
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 15, 30),
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Query-to-reference Recall@k for every k in `ks` (prefixes of the same
    ranked lists, so recall is monotone-consistent across k), plus a single
    NDCG@max(ks) using binary relevance (true high-dim top-k membership) with
    the *embedding-space* ranking as the ranking being scored -- rank-weighted
    neighbor preservation, distinct from the retriever's own NDCG (which
    scores *candidate* ranking against fuzzy weights, not final coordinates)."""
    k_max = max(ks)
    n_ref = reference_high_dim.shape[0]
    k_max = min(k_max, n_ref)
    true_idx, _ = chunked_exact_knn(query_high_dim, reference_high_dim, k=k_max)
    embed_idx, _ = chunked_exact_knn(query_embedding, reference_embedding, k=k_max)
    n_q = query_high_dim.shape[0]

    recall_by_k = {k: np.empty(n_q) for k in ks}
    ndcg = np.empty(n_q)
    for i in range(n_q):
        true_full = true_idx[i]
        embed_full = embed_idx[i]
        for k in ks:
            k_eff = min(k, k_max)
            recall_by_k[k][i] = len(set(true_full[:k_eff].tolist()) & set(embed_full[:k_eff].tolist())) / k
        true_set = set(true_full.tolist())
        relevance = np.array([1.0 if j in true_set else 0.0 for j in embed_full])
        ranks = np.arange(1, len(relevance) + 1)
        dcg = float(np.sum(relevance / np.log2(ranks + 1)))
        ideal = np.ones(min(k_max, len(true_set)))
        idcg = float(np.sum(ideal / np.log2(np.arange(1, len(ideal) + 1) + 1))) if len(ideal) else 0.0
        ndcg[i] = dcg / idcg if idcg > 0 else 0.0

    return recall_by_k, ndcg


def fuzzy_neighborhood_discrepancy(
    query_embedding: np.ndarray,
    neighbor_ids: list[np.ndarray],
    neighbor_weights: list[np.ndarray],
    reference_embedding: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per query: weighted MSE and weighted "attractive-edge" BCE between
    Phi(y*, y_j) and 1, over its true reference neighbors j, weighted by
    mu_{*->j}. Interpretation: "the query should be close in 2D to the
    reference points UMAP says it belongs to.\""""
    n_q = query_embedding.shape[0]
    wmse = np.full(n_q, np.nan)
    wbce = np.full(n_q, np.nan)
    for i in range(n_q):
        ids = neighbor_ids[i]
        w = np.asarray(neighbor_weights[i], dtype=np.float64)
        if len(ids) == 0 or w.sum() <= 0:
            continue
        diff = query_embedding[i][None, :] - reference_embedding[ids]
        q = np.sum(diff * diff, axis=1)
        phi_vals = 1.0 / (1.0 + a * np.power(np.maximum(q, 1e-12), b))
        w_norm = w / w.sum()
        wmse[i] = float(np.sum(w_norm * (phi_vals - 1.0) ** 2))
        wbce[i] = float(np.sum(w_norm * -np.log(np.clip(phi_vals, 1e-12, 1.0))))
    return wmse, wbce


def local_displacement_metric(
    query_embedding: np.ndarray,
    neighbor_ids: list[np.ndarray],
    neighbor_weights: list[np.ndarray],
    reference_embedding: np.ndarray,
) -> np.ndarray:
    """``||y* - weighted_barycenter(true neighbors)|| / local_scale``, where
    ``local_scale`` is the mean distance of those same neighbors from their
    own barycenter. A diagnostic only -- never used as the sole judgment of
    embedding quality."""
    n_q = query_embedding.shape[0]
    out = np.full(n_q, np.nan)
    for i in range(n_q):
        ids = neighbor_ids[i]
        w = np.asarray(neighbor_weights[i], dtype=np.float64)
        if len(ids) == 0 or w.sum() <= 0:
            continue
        pts = reference_embedding[ids]
        w_norm = (w / w.sum())[:, None]
        barycenter = (w_norm * pts).sum(axis=0)
        local_scale = max(float(np.linalg.norm(pts - barycenter[None, :], axis=1).mean()), 1e-8)
        out[i] = float(np.linalg.norm(query_embedding[i] - barycenter) / local_scale)
    return out


def class_conditional_periphery_percentile(
    query_embedding: np.ndarray,
    query_labels: np.ndarray,
    reference_embedding: np.ndarray,
    reference_labels: np.ndarray,
) -> np.ndarray:
    """For each query: radial distance from its own label's reference-embedding
    centroid, expressed as a percentile relative to same-label reference
    points' own radii from that centroid. A diagnostic, not a ground-truth
    metric of correctness."""
    out = np.full(query_embedding.shape[0], np.nan)
    for label in np.unique(query_labels):
        ref_mask = reference_labels == label
        if not np.any(ref_mask):
            continue
        ref_pts = reference_embedding[ref_mask]
        centroid = ref_pts.mean(axis=0)
        ref_radii = np.linalg.norm(ref_pts - centroid[None, :], axis=1)
        q_mask = query_labels == label
        q_radii = np.linalg.norm(query_embedding[q_mask] - centroid[None, :], axis=1)
        out[q_mask] = np.array([float(np.mean(ref_radii <= r)) * 100.0 for r in q_radii])
    return out


def repulsion_accumulation_score(
    query_embedding: np.ndarray,
    neighbor_ids: list[np.ndarray],
    reference_embedding: np.ndarray,
) -> np.ndarray:
    """Optional diagnostic (Section 2.3, item 7). This document's own
    operationalization of the qualitative "repulsion effect" Islam and
    Fleischer describe in "On Out-of-sample Embedding in UMAP" (points pushed
    toward a cluster's periphery when repulsive forces align) -- NOT a
    verified reproduction of a specific numbered equation from that paper.
    For each query: the norm of the mean *unit* direction from its true
    reference neighbors toward the query's final position. Near 0 => neighbors
    surround the query from many directions; near 1 => the query sits
    consistently on one side of all of them (the qualitative symptom the
    paper describes)."""
    n_q = query_embedding.shape[0]
    out = np.full(n_q, np.nan)
    for i in range(n_q):
        ids = neighbor_ids[i]
        if len(ids) == 0:
            continue
        directions = query_embedding[i][None, :] - reference_embedding[ids]
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        unit = directions / np.maximum(norms, 1e-12)
        out[i] = float(np.linalg.norm(unit.mean(axis=0)))
    return out


def compute_per_query_diagnostics(
    method_name: str,
    query_high_dim: np.ndarray,
    reference_high_dim: np.ndarray,
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    neighbor_ids: list[np.ndarray],
    neighbor_weights: list[np.ndarray],
    a: float,
    b: float,
    ks: tuple[int, ...] = (5, 10, 15, 30),
    query_labels: np.ndarray | None = None,
    reference_labels: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Runs every Section-2.3 diagnostic for one already-embedded method and
    returns (per-query DataFrame, {metric_name: TailSummary dict})``.
    `neighbor_ids`/`neighbor_weights` are this method's own retrieved
    reference neighbors + mu weights per query (the same ones `embed_one`
    used, or the exact/oracle equivalent for baselines without a retriever)."""
    recall_by_k, ndcg = multi_k_recall_and_ndcg(
        query_high_dim, reference_high_dim, query_embedding, reference_embedding, ks=ks
    )
    wmse, wbce = fuzzy_neighborhood_discrepancy(query_embedding, neighbor_ids, neighbor_weights, reference_embedding, a, b)
    displacement = local_displacement_metric(query_embedding, neighbor_ids, neighbor_weights, reference_embedding)
    accumulation = repulsion_accumulation_score(query_embedding, neighbor_ids, reference_embedding)

    df = pd.DataFrame({"method": method_name, "query_index": np.arange(query_high_dim.shape[0])})
    for k in ks:
        df[f"recall_at_{k}"] = recall_by_k[k]
    df["ndcg"] = ndcg
    df["fuzzy_weighted_mse"] = wmse
    df["fuzzy_weighted_bce"] = wbce
    df["local_displacement"] = displacement
    df["repulsion_accumulation_score"] = accumulation

    if query_labels is not None and reference_labels is not None:
        df["periphery_percentile"] = class_conditional_periphery_percentile(
            query_embedding, query_labels, reference_embedding, reference_labels
        )
        df["label"] = query_labels

    tails: dict[str, dict] = {}
    for k in ks:
        tails[f"recall_at_{k}_deficit"] = tail_summary(1.0 - df[f"recall_at_{k}"].to_numpy()).to_dict()
    tails["ndcg_deficit"] = tail_summary(1.0 - df["ndcg"].to_numpy()).to_dict()
    tails["fuzzy_weighted_mse"] = tail_summary(df["fuzzy_weighted_mse"].to_numpy()).to_dict()
    tails["local_displacement"] = tail_summary(df["local_displacement"].to_numpy()).to_dict()
    tails["repulsion_accumulation_score"] = tail_summary(df["repulsion_accumulation_score"].to_numpy()).to_dict()

    return df, tails
