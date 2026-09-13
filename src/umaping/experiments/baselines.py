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
# 5. NUMAP / Sep-SpectralNet (Ben-Ari, Yacobi, and Shaham, 2025) --
# official repository https://github.com/shaham-lab/NUMAP, official PyPI
# package `numap` (pinned to 0.2.3; see pyproject.toml's `external-baselines`
# extra). Not vendored -- this is a thin adapter around the pip-installed
# package's own public API, imported lazily so the base install never
# requires it. The exact constructor/fit/transform contract below was
# verified directly against the official source at commit
# afec4b9277d0bac3a09e89c202a319ab895af237
# (github.com/shaham-lab/NUMAP/blob/afec4b9.../src/numap/numap.py), not
# guessed from the README alone:
#
#   NUMAP.fit(X)                     -- X: torch.Tensor, reference features only, no labels.
#   NUMAP.transform(X, is_train=...) -- is_train=True reuses the exact spectral
#                                        embedding computed for X during fit
#                                        (the correct call for the *reference*
#                                        set itself); is_train=False (the
#                                        default) routes new points through a
#                                        kNN-regressor (or GrEASE, if
#                                        use_grease=True) out-of-sample
#                                        extension -- the correct call for
#                                        *query* points. Passing the wrong
#                                        value for either would silently
#                                        change which extension mechanism is
#                                        exercised, so both calls are made
#                                        explicitly rather than relying on
#                                        the default.
#
# `use_grease=True` and `use_residual_connections=True` are both optional
# constructor flags (default False in the official package) enabling,
# respectively, the paper's own generalizable spectral-embedding mechanism
# (GrEASE) for the out-of-sample extension and a residual connection from
# the spectral initialization through the trained encoder -- the
# "generalizable NUMAP configuration" this baseline is meant to represent,
# per this task's own instructions.
# ---------------------------------------------------------------------------


def run_numap_baseline(prepared: PreparedDataset, cfg: Config, device: torch.device | None = None) -> BaselineOutcome:
    try:
        from numap import NUMAP
    except ImportError as exc:
        return BaselineUnavailable(
            name="numap_sep_spectralnet",
            reason=(
                "The official NUMAP/Sep-SpectralNet package is not installed in this environment "
                f"({exc}). Repository: https://github.com/shaham-lab/NUMAP . "
                "Install with: pip install numap==0.2.3 (see pyproject.toml's 'external-baselines' extra)."
            ),
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_components = cfg.umap.embedding_dim
    constructor_kwargs = {
        "n_neighbors": cfg.umap.n_neighbors,
        "min_dist": cfg.umap.min_dist,
        "n_components": n_components,
        "se_dim": n_components,  # matched to n_components so the residual connection adds the actual spectral coordinates
        "se_neighbors": cfg.umap.n_neighbors,
        "random_state": cfg.seed,
        "use_se": True,
        "use_grease": True,
        "use_residual_connections": True,
        "num_gpus": 1 if device.type == "cuda" else 0,
    }
    model = NUMAP(**constructor_kwargs)

    x_ref = torch.as_tensor(np.ascontiguousarray(prepared.reference_features, dtype=np.float32))
    x_query = torch.as_tensor(np.ascontiguousarray(prepared.query_features, dtype=np.float32))

    start = time.perf_counter()
    model.fit(x_ref)  # reference features only -- query is never passed here
    fit_time = time.perf_counter() - start

    reference_embedding = np.asarray(model.transform(x_ref, is_train=True), dtype=np.float32)
    query_start = time.perf_counter()
    query_embedding = np.asarray(model.transform(x_query, is_train=False), dtype=np.float32)
    query_time = time.perf_counter() - query_start

    return BaselineResult(
        name="numap_sep_spectralnet",
        reference_embedding=reference_embedding,
        query_embedding=query_embedding,
        fit_time_seconds=fit_time,
        mean_query_latency_seconds=query_time / max(prepared.query_features.shape[0], 1),
        extra={
            "package": "numap",
            "version_pin": "0.2.3",
            "source_repository": "https://github.com/shaham-lab/NUMAP",
            "source_commit": "afec4b9277d0bac3a09e89c202a319ab895af237",
            "constructor_kwargs": dict(constructor_kwargs),
        },
    )


# ---------------------------------------------------------------------------
# 6. ParamRepulsor (Huang et al.) -- official repository
# https://github.com/hyhuang00/ParamRepulsor, official PyPI package
# `parampacmap` (pinned to 0.1.0; requires Python <3.12 -- see
# pyproject.toml's `external-baselines` extra). Not vendored. Verified
# directly against the official source at commit
# be8df72b1ac9041be3aae3d99f16f0d392b492dc
# (github.com/hyhuang00/ParamRepulsor/blob/be8df72.../src/parampacmap/parampacmap.py):
#
#   ParamPaCMAP.fit(X)       -- X: plain numpy array, reference features only.
#   ParamPaCMAP.transform(X) -- called separately for reference and query.
#
# `apply_pca=True` is the official default, but only actually triggers an
# *internal* PCA when the input has more than 100 dimensions
# (`_scale_input`); since this project's `PreparedDataset` features are
# already a fixed, reference-only-fitted representation (e.g. 256-D for
# COIL), leaving the official default on would silently re-derive a
# *different* representation for this one baseline. `apply_pca=False,
# apply_scale=None` are passed explicitly, per this task's instructions, so
# ParamRepulsor is compared on exactly the same input representation as
# every other baseline. Everything else -- `n_FP`, `n_MN`, `loss_weight`,
# and critically `weight_schedule`/`const_schedule` (the ParamRepulsor-
# specific repulsion weighting that distinguishes it from plain PaCMAP,
# defaulting to `paramrep_weight_schedule`/`paramrep_const_schedule`),
# `num_epochs`, `embedding_init`, and the default "ANN" backbone -- is left
# at the official default.
# ---------------------------------------------------------------------------


def run_param_repulsor_baseline(prepared: PreparedDataset, cfg: Config) -> BaselineOutcome:
    try:
        import parampacmap
    except ImportError as exc:
        return BaselineUnavailable(
            name="param_repulsor",
            reason=(
                "The official ParamRepulsor package is not installed in this environment "
                f"({exc}). Repository: https://github.com/hyhuang00/ParamRepulsor . "
                "Install with: pip install parampacmap==0.1.0 (requires Python <3.12; "
                "see pyproject.toml's 'external-baselines' extra)."
            ),
        )

    constructor_kwargs = {
        "n_components": cfg.umap.embedding_dim,
        "n_neighbors": cfg.umap.n_neighbors,
        "apply_pca": False,
        "apply_scale": None,
        "seed": cfg.seed,
    }
    model = parampacmap.ParamPaCMAP(**constructor_kwargs)

    x_ref = np.ascontiguousarray(prepared.reference_features, dtype=np.float32)
    x_query = np.ascontiguousarray(prepared.query_features, dtype=np.float32)

    start = time.perf_counter()
    model.fit(x_ref)  # reference features only -- query is never passed here
    fit_time = time.perf_counter() - start

    reference_embedding = np.asarray(model.transform(x_ref), dtype=np.float32)
    query_start = time.perf_counter()
    query_embedding = np.asarray(model.transform(x_query), dtype=np.float32)
    query_time = time.perf_counter() - query_start

    return BaselineResult(
        name="param_repulsor",
        reference_embedding=reference_embedding,
        query_embedding=query_embedding,
        fit_time_seconds=fit_time,
        mean_query_latency_seconds=query_time / max(prepared.query_features.shape[0], 1),
        extra={
            "package": "parampacmap",
            "version_pin": "0.1.0",
            "source_repository": "https://github.com/hyhuang00/ParamRepulsor",
            "source_commit": "be8df72b1ac9041be3aae3d99f16f0d392b492dc",
            "constructor_kwargs": dict(constructor_kwargs),
        },
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
