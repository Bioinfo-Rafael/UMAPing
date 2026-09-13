"""Common baseline interface for Experiments B/C/D (spec Section 4).

Every baseline runner returns either a `BaselineResult` or a
`BaselineUnavailable` (when an optional dependency isn't installed, or no
official implementation is vendored) -- the experiment runner iterates over
a fixed list without special-casing any one of them, and a missing optional
dependency never breaks the whole run or the base package install.

Baselines that already exist in this codebase (standard umap-learn, ours,
ours + oracle neighbors, no-repulsion, the exact/high-M repulsion diagnostic)
are *reused* here via `umaping.baselines` / `umaping.inference`, never
reimplemented -- see each function's docstring for exactly what it calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from umaping.baselines import embed_all_queries, run_standard_umap
from umaping.config import Config
from umaping.data.preprocessing import PreparedDataset
from umaping.graph import chunked_exact_knn, query_fuzzy_weights
from umaping.inference import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    name: str
    reference_embedding: np.ndarray
    query_embedding: np.ndarray
    neighbor_ids: list[np.ndarray] | None = None
    neighbor_weights: list[np.ndarray] | None = None
    fit_time_seconds: float | None = None
    mean_query_latency_seconds: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class BaselineUnavailable:
    name: str
    reason: str


BaselineOutcome = BaselineResult | BaselineUnavailable


# ---------------------------------------------------------------------------
# 1. Standard umap-learn (reused verbatim from umaping.baselines)
# ---------------------------------------------------------------------------


def run_standard_umap_baseline(prepared: PreparedDataset, cfg: Config) -> BaselineResult:
    start = time.perf_counter()
    ref_emb, query_emb = run_standard_umap(
        prepared.reference_features,
        prepared.query_features,
        cfg.umap.n_neighbors,
        cfg.umap.min_dist,
        cfg.umap.spread,
        cfg.umap.embedding_dim,
        cfg.seed,
    )
    elapsed = time.perf_counter() - start
    return BaselineResult(
        name="standard_umap", reference_embedding=ref_emb, query_embedding=query_emb, fit_time_seconds=elapsed
    )


# ---------------------------------------------------------------------------
# 2. Reduced-repulsion UMAP OOS variant
# ---------------------------------------------------------------------------


def run_reduced_repulsion_umap_baseline(
    prepared: PreparedDataset, cfg: Config, negative_sample_rate_scale: float = 0.2
) -> BaselineResult:
    """Fits `umap.UMAP` exactly as `run_standard_umap` does (so the reference
    embedding is comparable), then temporarily overrides the fitted reducer's
    public `negative_sample_rate` attribute -- which `UMAP.transform()` reads
    for its own transform-time optimization (verified directly against the
    installed `umap-learn` source: both call sites inside `transform()`
    reference `self.negative_sample_rate`) -- to a smaller value before
    calling `.transform()`, then restores it. This is a best-effort,
    practical reproduction of "reduce repulsion specifically at OOS-embedding
    time" (the idea motivating the recent OOS-UMAP literature), not a
    byte-for-byte reimplementation of any one paper's own code."""
    import umap

    start = time.perf_counter()
    reducer = umap.UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        min_dist=cfg.umap.min_dist,
        spread=cfg.umap.spread,
        n_components=cfg.umap.embedding_dim,
        random_state=cfg.seed,
    )
    ref_emb = np.asarray(reducer.fit_transform(prepared.reference_features), dtype=np.float32)
    fit_time = time.perf_counter() - start

    original_rate = reducer.negative_sample_rate
    reducer.negative_sample_rate = max(original_rate * negative_sample_rate_scale, 0.0)
    try:
        query_emb = np.asarray(reducer.transform(prepared.query_features), dtype=np.float32)
    finally:
        reducer.negative_sample_rate = original_rate

    return BaselineResult(
        name="reduced_repulsion_umap",
        reference_embedding=ref_emb,
        query_embedding=query_emb,
        fit_time_seconds=fit_time,
        extra={"negative_sample_rate_scale": negative_sample_rate_scale, "original_negative_sample_rate": original_rate},
    )


# ---------------------------------------------------------------------------
# 3. Weighted kNN interpolation
# ---------------------------------------------------------------------------


def run_weighted_knn_baseline(
    prepared: PreparedDataset, reference_embedding: np.ndarray, cfg: Config
) -> BaselineResult:
    """A classic out-of-sample-extension baseline: place each query at the
    UMAP-fuzzy-weighted average of its true high-dimensional k-NN reference
    points' positions in an *already-computed* 2D reference embedding
    (passed in by the caller -- typically `standard_umap`'s own reference
    layout, so this isolates "does umap-learn's own transform-time
    optimization matter, vs. simply interpolating from neighbors" as the
    comparison). No optimization of any kind happens at query time."""
    start = time.perf_counter()
    knn_idx, knn_dist = chunked_exact_knn(prepared.query_features, prepared.reference_features, k=cfg.umap.n_neighbors)
    weights, _, _ = query_fuzzy_weights(
        knn_dist,
        local_connectivity=cfg.umap.local_connectivity,
        n_iter=cfg.umap.smooth_knn_n_iter,
        bandwidth=cfg.umap.smooth_knn_bandwidth,
        min_k_dist_scale=cfg.umap.smooth_knn_min_k_dist_scale,
        eps=cfg.umap.eps,
    )
    weights_norm = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    query_emb = np.einsum("qk,qkd->qd", weights_norm, reference_embedding[knn_idx]).astype(np.float32)
    elapsed = time.perf_counter() - start

    neighbor_ids = [knn_idx[i] for i in range(knn_idx.shape[0])]
    neighbor_weights = [weights[i] for i in range(weights.shape[0])]
    return BaselineResult(
        name="weighted_knn",
        reference_embedding=reference_embedding,
        query_embedding=query_emb,
        neighbor_ids=neighbor_ids,
        neighbor_weights=neighbor_weights,
        mean_query_latency_seconds=elapsed / max(prepared.query_features.shape[0], 1),
    )


# ---------------------------------------------------------------------------
# 4. Parametric UMAP (optional: requires umap-learn's tensorflow-based extra)
# ---------------------------------------------------------------------------


def run_parametric_umap_baseline(prepared: PreparedDataset, cfg: Config) -> BaselineOutcome:
    """`umap.parametric_umap.ParametricUMAP` -- part of the `umap-learn`
    package itself, but only importable when `tensorflow` is also installed.
    Gracefully reports unavailability rather than failing the whole run (or
    the base package install) when it isn't."""
    try:
        from umap.parametric_umap import ParametricUMAP
    except ImportError as exc:
        return BaselineUnavailable(
            name="parametric_umap",
            reason=(
                "umap.parametric_umap.ParametricUMAP requires the optional 'tensorflow' dependency, "
                f"not installed in this environment ({exc}). Install with: pip install tensorflow"
            ),
        )

    start = time.perf_counter()
    reducer = ParametricUMAP(
        n_neighbors=cfg.umap.n_neighbors,
        min_dist=cfg.umap.min_dist,
        n_components=cfg.umap.embedding_dim,
        random_state=cfg.seed,
    )
    ref_emb = np.asarray(reducer.fit_transform(prepared.reference_features), dtype=np.float32)
    fit_time = time.perf_counter() - start
    query_start = time.perf_counter()
    query_emb = np.asarray(reducer.transform(prepared.query_features), dtype=np.float32)
    query_time = time.perf_counter() - query_start

    return BaselineResult(
        name="parametric_umap",
        reference_embedding=ref_emb,
        query_embedding=query_emb,
        fit_time_seconds=fit_time,
        mean_query_latency_seconds=query_time / max(prepared.query_features.shape[0], 1),
    )


# ---------------------------------------------------------------------------
# 5-6. NUMAP/Sep-SpectralNet and ParamRepulsor -- no vendored/pip-installable
# official implementation located during this implementation. Documented as
# unavailable (per spec: "gracefully mark a baseline as unavailable ... do
# not copy large third-party repos into this repository ... create
# adapters/wrappers"), with instructions for running the official code
# out-of-repo rather than a guessed/fabricated reimplementation.
# ---------------------------------------------------------------------------


def run_numap_baseline(prepared: PreparedDataset, cfg: Config) -> BaselineOutcome:
    return BaselineUnavailable(
        name="numap_sep_spectralnet",
        reason=(
            "No official pip package located. NUMAP/Sep-SpectralNet is the method introduced in "
            "Ben-Ari, Yacobi, and Shaham (2025), 'Generalizable Spectral Embedding with an Application "
            "to UMAP' (TMLR; arXiv:2501.11305) -- a specific repository URL for their code was not "
            "located/verified during this implementation. To integrate it: clone the authors' official "
            "release once you have located it, install it in an isolated environment, and add a thin "
            "subprocess-based adapter here following the pattern described in experiments/README.md "
            "(never vendor a large third-party repo into this one)."
        ),
    )


def run_param_repulsor_baseline(prepared: PreparedDataset, cfg: Config) -> BaselineOutcome:
    return BaselineUnavailable(
        name="param_repulsor",
        reason=(
            "No official pip package located, and this implementation could not verify a specific "
            "source repository for 'ParamRepulsor' during development (external network access to "
            "confirm one was not exercised, per this task's own no-download constraint). Before "
            "enabling this baseline, locate and cite the official paper/repository, then add a thin "
            "subprocess-based adapter here following the pattern described in experiments/README.md."
        ),
    )


# ---------------------------------------------------------------------------
# 7-10. Ours / ours + oracle neighbors / no-repulsion / exact repulsion
# diagnostic -- all reuse InferenceEngine + embed_all_queries directly.
# ---------------------------------------------------------------------------


def run_ours_baselines(
    run_dir: str | Path, device: torch.device, cfg: Config, reference_embedding_ours: np.ndarray, seed: int
) -> list[BaselineResult]:
    """Baselines 7 ("ours"), 8 ("ours + oracle neighbors"), and 9
    ("no-repulsion ablation"): each is exactly an `InferenceEngine` mode
    already implemented in `inference.py`, reused as-is."""
    from umaping.data.preprocessing import load_prepared_dataset
    from umaping.pipeline import run_layout

    layout = run_layout(run_dir)
    prepared = load_prepared_dataset(layout["cache"] / "prepared_dataset.npz")

    results = []
    engine_full = InferenceEngine.load(run_dir, device, seed=seed)
    emb, ids, w, latencies, _ = embed_all_queries(engine_full, prepared.query_features)
    results.append(
        BaselineResult(
            "ours", reference_embedding_ours, emb, neighbor_ids=ids, neighbor_weights=w,
            mean_query_latency_seconds=float(np.mean(latencies)),
        )
    )

    engine_oracle = InferenceEngine.load(run_dir, device, neighbor_source="oracle", seed=seed)
    emb, ids, w, latencies, _ = embed_all_queries(engine_oracle, prepared.query_features)
    results.append(
        BaselineResult(
            "ours_oracle_neighbors", reference_embedding_ours, emb, neighbor_ids=ids, neighbor_weights=w,
            mean_query_latency_seconds=float(np.mean(latencies)),
        )
    )

    engine_no_rep = InferenceEngine.load(run_dir, device, use_repulsion=False, seed=seed)
    emb, ids, w, latencies, _ = embed_all_queries(engine_no_rep, prepared.query_features)
    results.append(
        BaselineResult(
            "no_repulsion", reference_embedding_ours, emb, neighbor_ids=ids, neighbor_weights=w,
            mean_query_latency_seconds=float(np.mean(latencies)),
        )
    )
    return results


def run_exact_repulsion_diagnostic_baseline(
    run_dir: str | Path, device: torch.device, cfg: Config, reference_embedding_ours: np.ndarray, seed: int, query_subset_n: int = 50
) -> BaselineResult:
    """Baseline 10: `InferenceEngine(repulsion_mode="exact")` (every
    reference point contributes exactly once, no Monte-Carlo sampling at
    all -- see `dynamics.exact_all_reference_mean_field`), restricted to a
    query subset since it is O(N) per query. Isolates how much of any
    end-to-end error is attributable to the *learned* repulsion
    approximation, as opposed to the mean-field approximation itself."""
    from umaping.data.preprocessing import load_prepared_dataset
    from umaping.pipeline import run_layout

    layout = run_layout(run_dir)
    prepared = load_prepared_dataset(layout["cache"] / "prepared_dataset.npz")
    n_query = prepared.query_features.shape[0]
    subset_n = min(query_subset_n, n_query)
    subset_idx = np.random.default_rng(seed).choice(n_query, size=subset_n, replace=False)

    engine = InferenceEngine.load(run_dir, device, repulsion_mode="exact", seed=seed)
    emb, ids, w, latencies, _ = embed_all_queries(engine, prepared.query_features[subset_idx])
    return BaselineResult(
        name="exact_repulsion_diagnostic",
        reference_embedding=reference_embedding_ours,
        query_embedding=emb,
        neighbor_ids=ids,
        neighbor_weights=w,
        mean_query_latency_seconds=float(np.mean(latencies)),
        extra={"query_subset_indices": subset_idx.tolist()},
    )
