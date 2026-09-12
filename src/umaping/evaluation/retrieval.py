"""Retriever metrics (docs/method.md, Evaluation section).

Ground truth for every held-out query is always its exact Euclidean kNN
against the fixed reference set X -- never against other query points."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from umaping.graph import chunked_exact_knn, query_fuzzy_weights
from umaping.inference import InferenceEngine


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    candidate_recall_at_m: float
    fuzzy_weighted_recall: float
    ndcg_at_k: float
    mean_query_latency_seconds: float
    n_queries: int
    k: int
    m: int

    def to_dict(self) -> dict:
        return asdict(self)


def _ndcg(relevance_in_rank_order: np.ndarray, ideal_relevance_sorted_desc: np.ndarray) -> float:
    ranks = np.arange(1, len(relevance_in_rank_order) + 1)
    discounts = np.log2(ranks + 1)
    dcg = float(np.sum(relevance_in_rank_order / discounts))
    ideal_ranks = np.arange(1, len(ideal_relevance_sorted_desc) + 1)
    idcg = float(np.sum(ideal_relevance_sorted_desc / np.log2(ideal_ranks + 1)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    engine: InferenceEngine,
    query_features: np.ndarray,
    reference_features: np.ndarray,
    k: int,
) -> tuple[RetrievalMetrics, pd.DataFrame]:
    """For each query: ground-truth exact top-k in the reference set, vs. the
    engine's learned-retriever candidate pool + exact rerank. `engine`'s own
    `retrieve_neighbors` is reused verbatim (same candidate pool size, same
    backend) so this measures exactly what `embed_one` would do."""
    n_queries = query_features.shape[0]
    true_idx, true_dist = chunked_exact_knn(query_features, reference_features, k=k)
    true_weights, _, _ = query_fuzzy_weights(
        true_dist,
        local_connectivity=engine.cfg.local_connectivity,
        n_iter=engine.cfg.smooth_knn_n_iter,
        bandwidth=engine.cfg.smooth_knn_bandwidth,
        min_k_dist_scale=engine.cfg.smooth_knn_min_k_dist_scale,
        eps=engine.cfg.eps,
    )

    rows = []
    for i in range(n_queries):
        x = query_features[i]
        true_set = set(true_idx[i].tolist())
        true_mu = dict(zip(true_idx[i].tolist(), true_weights[i].tolist()))

        start = time.perf_counter()
        retrieved_ids, _ = engine.retrieve_neighbors(x)
        elapsed = time.perf_counter() - start

        # Candidate-pool recall needs the pre-rerank candidate set too.
        with torch.no_grad():
            x_t = torch.as_tensor(x[None, :], dtype=torch.float32, device=engine.device)
            q = engine.retriever.encode_query(x_t).cpu().numpy()
        candidate_ids, _ = engine.neighbor_index.search(q, engine.cfg.candidate_pool_size)
        candidate_set = set(candidate_ids[0][candidate_ids[0] >= 0].tolist())

        retrieved_set = set(retrieved_ids.tolist())
        recovered = true_set & retrieved_set

        recall_k = len(recovered) / k
        candidate_recall_m = len(true_set & candidate_set) / k
        true_total_mass = sum(true_mu.values())
        recovered_mass = sum(true_mu[j] for j in recovered)
        weighted_recall = recovered_mass / true_total_mass if true_total_mass > 0 else 0.0

        relevance_in_rank_order = np.array([true_mu.get(j, 0.0) for j in retrieved_ids])
        ideal = np.array(sorted(true_mu.values(), reverse=True))
        ndcg = _ndcg(relevance_in_rank_order, ideal)

        rows.append(
            {
                "query_index": i,
                "recall_at_k": recall_k,
                "candidate_recall_at_m": candidate_recall_m,
                "fuzzy_weighted_recall": weighted_recall,
                "ndcg_at_k": ndcg,
                "latency_seconds": elapsed,
            }
        )

    df = pd.DataFrame(rows)
    metrics = RetrievalMetrics(
        recall_at_k=float(df["recall_at_k"].mean()),
        candidate_recall_at_m=float(df["candidate_recall_at_m"].mean()),
        fuzzy_weighted_recall=float(df["fuzzy_weighted_recall"].mean()),
        ndcg_at_k=float(df["ndcg_at_k"].mean()),
        mean_query_latency_seconds=float(df["latency_seconds"].mean()),
        n_queries=n_queries,
        k=k,
        m=engine.cfg.candidate_pool_size,
    )
    return metrics, df
