"""Baselines and ablations (README "Baselines" section).

This module is a deliberate, documented addition beyond the literal file
tree in the spec: it keeps `cli.py`/`pipeline.py` thin by giving the
baseline/ablation *runners* (as opposed to the shared metric code in
evaluation/*.py) one home.

    1. Standard umap-learn      -- run_standard_umap
    2. Spectral-only            -- embed_all_queries_spectral_only (+ engine.spectral_embedder.embed for reference)
    3. Ours + ORACLE neighbors  -- InferenceEngine(neighbor_source="oracle")   + embed_all_queries
    4. Ours (full method)       -- InferenceEngine(neighbor_source="learned")  + embed_all_queries
    5. No-repulsion ablation    -- InferenceEngine(use_repulsion=False)        + embed_all_queries
    6. Repulsion-oracle diag.   -- InferenceEngine(repulsion_mode="oracle_mc") + embed_all_queries (subset)
    (optional) Transductive UMAP on reference+query, visualization only -- run_transductive_umap
"""

from __future__ import annotations

import time

import numpy as np
import umap

from umaping.inference import InferenceEngine


def embed_all_queries(
    engine: InferenceEngine,
    query_features: np.ndarray,
    return_trajectories: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray] | None]:
    """Runs `embed_one` independently over every query -- batching here is
    just a Python loop timing each call separately, since "batch" must never
    become a query-query interaction. Returns (embeddings, neighbor_ids per
    query, per-query latency seconds, optional per-query trajectories)."""
    n_queries = query_features.shape[0]
    embedding_dim = engine.trajectory.positions.shape[2]
    embeddings = np.empty((n_queries, embedding_dim), dtype=np.float32)
    neighbor_ids: list[np.ndarray] = []
    latencies = np.empty(n_queries, dtype=np.float64)
    trajectories: list[np.ndarray] | None = [] if return_trajectories else None

    for i in range(n_queries):
        start = time.perf_counter()
        result = engine.embed_one(query_features[i], return_trajectory=return_trajectories)
        latencies[i] = time.perf_counter() - start
        embeddings[i] = result.embedding
        neighbor_ids.append(result.neighbor_ids)
        if return_trajectories:
            trajectories.append(result.trajectory)

    return embeddings, neighbor_ids, latencies, trajectories


def embed_all_queries_spectral_only(engine: InferenceEngine, query_features: np.ndarray) -> np.ndarray:
    """Baseline #2. A single batched forward pass is fine here (no flow
    integration happens at all, so there is no cross-query state to keep
    independent -- each row's output depends only on that row)."""
    return engine.spectral_embedder.embed(query_features)


def run_standard_umap(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline #1: fit umap-learn on the reference set, transform held-out
    queries via its own (inductive) `.transform()`."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=embedding_dim,
        random_state=seed,
    )
    reference_embedding = reducer.fit_transform(reference_features)
    query_embedding = reducer.transform(query_features)
    return np.asarray(reference_embedding, dtype=np.float32), np.asarray(query_embedding, dtype=np.float32)


def run_transductive_umap(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Optional, visualization-only: fits UMAP jointly on reference + query.
    NOT the inductive target -- this uses query data during graph
    construction, which the inductive method never does. Only meaningful as
    a qualitative visual reference point."""
    combined = np.concatenate([reference_features, query_features], axis=0)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=embedding_dim,
        random_state=seed,
    )
    combined_embedding = np.asarray(reducer.fit_transform(combined), dtype=np.float32)
    n_ref = reference_features.shape[0]
    return combined_embedding[:n_ref], combined_embedding[n_ref:]
