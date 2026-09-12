"""Synthetic 'mock' dataset -- not one of the three real research datasets.

Exists purely so the *entire* pipeline (preprocessing -> graph -> retriever
-> spectral -> flow -> repulsion -> evaluate -> analyze) can be smoke-tested
fully locally in seconds: no download, no real training, no GPU needed. See
configs/mock.yaml. Generates a small Gaussian-blob mixture split into an
independent reference set and query set drawn from the same clusters, so
retrieval/recall/label-accuracy metrics are meaningful rather than
degenerate.
"""

from __future__ import annotations

import numpy as np

from umaping.data.preprocessing import PreparedDataset


def prepare_mock_dataset(
    n_reference: int = 200,
    n_query: int = 50,
    input_dim: int = 16,
    n_clusters: int = 5,
    seed: int = 0,
) -> PreparedDataset:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(n_clusters, input_dim)).astype(np.float32)

    def _sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, n_clusters, size=n)
        noise = rng.normal(scale=1.0, size=(n, input_dim)).astype(np.float32)
        return (centers[labels] + noise).astype(np.float32), labels

    reference_features, reference_cluster = _sample(n_reference)
    query_features, query_cluster = _sample(n_query)

    return PreparedDataset(
        reference_features=reference_features,
        query_features=query_features,
        reference_labels={"cluster": reference_cluster},
        query_labels={"cluster": query_cluster},
        input_dim=input_dim,
    )
