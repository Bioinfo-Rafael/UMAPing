"""Final embedding-quality metrics (docs/method.md, Evaluation section).

The primary metric respects the fixed-reference, pointwise setting: for each
query, does its neighborhood in the final 2D embedding (searched only against
the reference set) recover its true high-dimensional reference neighborhood?
Everything else here is secondary context (reference-only trustworthiness,
label kNN accuracy, latency, and Procrustes-aligned coordinate RMSE for
comparing two independently-fit coordinate systems)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import trustworthiness as sk_trustworthiness
from sklearn.neighbors import KNeighborsClassifier

from umaping.graph import chunked_exact_knn


@dataclass
class EmbeddingMetrics:
    neighborhood_recall_at_k: float
    reference_trustworthiness: float | None
    label_knn_accuracy: float | None
    mean_query_latency_seconds: float | None
    n_queries: int
    k: int

    def to_dict(self) -> dict:
        return asdict(self)


def query_to_reference_neighborhood_recall(
    query_high_dim: np.ndarray,
    reference_high_dim: np.ndarray,
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    k: int,
) -> tuple[float, np.ndarray]:
    """For each query: does its k-NN in the 2D embedding (searched only
    against the reference embedding) overlap with its true high-dimensional
    k-NN (searched only against the reference features)?"""
    true_idx, _ = chunked_exact_knn(query_high_dim, reference_high_dim, k=k)
    embed_idx, _ = chunked_exact_knn(query_embedding, reference_embedding, k=k)
    per_query = np.array(
        [len(set(true_idx[i].tolist()) & set(embed_idx[i].tolist())) / k for i in range(query_high_dim.shape[0])]
    )
    return float(per_query.mean()), per_query


def label_knn_accuracy(
    reference_embedding: np.ndarray,
    reference_labels: np.ndarray,
    query_embedding: np.ndarray,
    query_labels: np.ndarray,
    n_neighbors: int = 15,
) -> float:
    """kNN classifier fit on the reference embedding/labels, evaluated on
    the query embedding -- object ID for COIL, celltype for pancreas."""
    clf = KNeighborsClassifier(n_neighbors=min(n_neighbors, reference_embedding.shape[0]))
    clf.fit(reference_embedding, reference_labels)
    preds = clf.predict(query_embedding)
    return float(np.mean(preds == query_labels))


def procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Align `source` onto `target` via orthogonal Procrustes (rotation +
    reflection only; both mean-centered first, no scaling). Returns the
    aligned source and the resulting coordinate RMSE -- use this before ever
    comparing raw coordinates across two independently-fit 2D embeddings."""
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)
    r, _ = orthogonal_procrustes(source_c, target_c)
    aligned = source_c @ r
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target_c) ** 2, axis=1))))
    return aligned, rmse


def evaluate_embedding(
    query_high_dim: np.ndarray,
    reference_high_dim: np.ndarray,
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    k: int,
    reference_labels: np.ndarray | None = None,
    query_labels: np.ndarray | None = None,
    trustworthiness_n_neighbors: int = 15,
    mean_query_latency_seconds: float | None = None,
) -> tuple[EmbeddingMetrics, np.ndarray]:
    recall, per_query_recall = query_to_reference_neighborhood_recall(
        query_high_dim, reference_high_dim, query_embedding, reference_embedding, k=k
    )

    trust = None
    n_neighbors = min(trustworthiness_n_neighbors, reference_high_dim.shape[0] - 2)
    if n_neighbors >= 1:
        # Trustworthiness on the reference embedding only -- a standard,
        # single-homogeneous-set computation. We deliberately do not extend
        # this to query points: doing so would require ranking query-query
        # distances, which the inductive method itself never computes.
        trust = float(sk_trustworthiness(reference_high_dim, reference_embedding, n_neighbors=n_neighbors))

    acc = None
    if reference_labels is not None and query_labels is not None:
        acc = label_knn_accuracy(reference_embedding, reference_labels, query_embedding, query_labels)

    metrics = EmbeddingMetrics(
        neighborhood_recall_at_k=recall,
        reference_trustworthiness=trust,
        label_knn_accuracy=acc,
        mean_query_latency_seconds=mean_query_latency_seconds,
        n_queries=query_high_dim.shape[0],
        k=k,
    )
    return metrics, per_query_recall
