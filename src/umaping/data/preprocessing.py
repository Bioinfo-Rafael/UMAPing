"""Shared reference-only preprocessing utilities.

Every fitted transform in this module (:class:`FittedPCA`) is fit on the
reference split only, then applied -- never refit -- to the query split.
This is the one invariant that must never be violated anywhere in the
codebase: no query point may influence HVG selection, PCA, the UMAP graph,
retriever/spectral/flow/repulsion training, or reference trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FittedPCA:
    mean: np.ndarray  # (D,)
    components: np.ndarray  # (n_components, D)
    n_components: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) @ self.components.T).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            n_components=np.asarray(self.n_components),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FittedPCA":
        data = np.load(path)
        return cls(mean=data["mean"], components=data["components"], n_components=int(data["n_components"]))


def fit_pca(x_reference: np.ndarray, n_components: int, seed: int = 0) -> FittedPCA:
    """Fit PCA on the reference set only; returns a small, dependency-light
    (numpy-only after fitting) transform to apply to reference and query."""
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(np.asarray(x_reference, dtype=np.float64))
    return FittedPCA(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        n_components=n_components,
    )


@dataclass
class PreparedDataset:
    """Generic container populated by each dataset loader (coil.py,
    pancreas.py): fixed-dimensional reference/query features plus whatever
    per-point labels that dataset carries for evaluation/plots."""

    reference_features: np.ndarray  # (N_ref, D)
    query_features: np.ndarray  # (N_query, D)
    reference_labels: dict[str, np.ndarray]
    query_labels: dict[str, np.ndarray]
    input_dim: int


def assert_no_overlap(reference_ids: np.ndarray, query_ids: np.ndarray) -> None:
    overlap = np.intersect1d(reference_ids, query_ids)
    if overlap.size > 0:
        raise ValueError(f"Reference/query split overlaps at {overlap.size} indices, e.g. {overlap[:10]}")


def check_finite(x: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(x)):
        raise FloatingPointError(f"Non-finite values encountered in '{name}'")


def save_prepared_dataset(path: str | Path, dataset: PreparedDataset) -> None:
    payload: dict[str, Any] = {
        "reference_features": dataset.reference_features,
        "query_features": dataset.query_features,
        "input_dim": np.asarray(dataset.input_dim),
    }
    for key, value in dataset.reference_labels.items():
        payload[f"reference_label__{key}"] = value
    for key, value in dataset.query_labels.items():
        payload[f"query_label__{key}"] = value
    np.savez(path, **payload)


def load_prepared_dataset(path: str | Path) -> PreparedDataset:
    data = np.load(path, allow_pickle=False)
    reference_labels = {
        k[len("reference_label__") :]: data[k] for k in data.files if k.startswith("reference_label__")
    }
    query_labels = {k[len("query_label__") :]: data[k] for k in data.files if k.startswith("query_label__")}
    return PreparedDataset(
        reference_features=data["reference_features"],
        query_features=data["query_features"],
        reference_labels=reference_labels,
        query_labels=query_labels,
        input_dim=int(data["input_dim"]),
    )
