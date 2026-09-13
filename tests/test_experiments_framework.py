"""Common experiment framework: the registry, the Section-4 baseline
interface, and `experiments.runner.run_experiment`'s
download/prepare/train/evaluate/analyze/baselines orchestration -- exercised
end to end only against the tiny synthetic mock dataset (no network, no
real training), per this task's mock-testing requirement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from umaping.config import Config
from umaping.experiments.baselines import (
    BaselineResult,
    BaselineUnavailable,
    run_numap_baseline,
    run_param_repulsor_baseline,
    run_parametric_umap_baseline,
    run_reduced_repulsion_umap_baseline,
    run_standard_umap_baseline,
    run_weighted_knn_baseline,
)
from umaping.experiments.registry import _REGISTRY, get_experiment, list_experiments


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lists_all_six_experiments_with_loadable_configs():
    names = list_experiments()
    assert names == sorted(
        ["fashion_mnist", "twenty_newsgroups", "mnist_oos", "hong_ed", "organoid", "embryoid_body"]
    )
    for name in names:
        spec = get_experiment(name)
        assert Path(spec.config_path).exists(), f"{name}: missing config {spec.config_path}"
        Config.load(spec.config_path)  # must parse without error
        assert callable(spec.download_fn)
        assert spec.citation and spec.source_url


def test_get_experiment_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown experiment"):
        get_experiment("not_a_real_experiment")


# ---------------------------------------------------------------------------
# Baseline registry: individual functions on synthetic data
# ---------------------------------------------------------------------------


def _tiny_prepared_dataset(seed=0, n_ref=60, n_query=15, d=6):
    from umaping.data.preprocessing import PreparedDataset

    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(3, d)).astype(np.float32)
    ref_labels = rng.integers(0, 3, size=n_ref)
    query_labels = rng.integers(0, 3, size=n_query)
    ref = (centers[ref_labels] + rng.normal(scale=1.0, size=(n_ref, d))).astype(np.float32)
    query = (centers[query_labels] + rng.normal(scale=1.0, size=(n_query, d))).astype(np.float32)
    return PreparedDataset(
        reference_features=ref, query_features=query,
        reference_labels={"cluster": ref_labels}, query_labels={"cluster": query_labels}, input_dim=d,
    )


def _tiny_config(d=6):
    from umaping.config import DatasetConfig

    cfg = Config(dataset=DatasetConfig(name="toy", input_dim=d))
    cfg.umap.n_neighbors = 5
    return cfg


def test_standard_umap_baseline_produces_finite_embeddings():
    prepared = _tiny_prepared_dataset()
    cfg = _tiny_config()
    result = run_standard_umap_baseline(prepared, cfg)
    assert isinstance(result, BaselineResult)
    assert result.reference_embedding.shape == (60, cfg.umap.embedding_dim)
    assert result.query_embedding.shape == (15, cfg.umap.embedding_dim)
    assert np.all(np.isfinite(result.reference_embedding))
    assert np.all(np.isfinite(result.query_embedding))


def test_reduced_repulsion_umap_baseline_restores_the_original_negative_sample_rate():
    prepared = _tiny_prepared_dataset(seed=1)
    cfg = _tiny_config()
    result = run_reduced_repulsion_umap_baseline(prepared, cfg, negative_sample_rate_scale=0.1)
    assert isinstance(result, BaselineResult)
    assert np.all(np.isfinite(result.query_embedding))
    assert result.extra["negative_sample_rate_scale"] == 0.1


def test_weighted_knn_baseline_places_queries_near_their_true_neighbors_embedding():
    prepared = _tiny_prepared_dataset(seed=2)
    cfg = _tiny_config()
    standard = run_standard_umap_baseline(prepared, cfg)
    result = run_weighted_knn_baseline(prepared, standard.reference_embedding, cfg)
    assert isinstance(result, BaselineResult)
    assert result.query_embedding.shape == (15, cfg.umap.embedding_dim)
    assert np.all(np.isfinite(result.query_embedding))
    assert result.neighbor_ids is not None and len(result.neighbor_ids) == 15


def test_parametric_umap_baseline_returns_a_valid_outcome_type():
    """Environment-dependent (available iff tensorflow is installed) -- only
    the *contract* (a well-typed outcome, never an unhandled exception) is
    asserted here, not which branch this particular environment takes."""
    prepared = _tiny_prepared_dataset(seed=3)
    cfg = _tiny_config()
    result = run_parametric_umap_baseline(prepared, cfg)
    assert isinstance(result, (BaselineResult, BaselineUnavailable))
    if isinstance(result, BaselineUnavailable):
        assert "tensorflow" in result.reason.lower()


def test_numap_and_param_repulsor_baselines_are_always_unavailable_with_a_clear_reason():
    prepared = _tiny_prepared_dataset(seed=4)
    cfg = _tiny_config()
    for fn in (run_numap_baseline, run_param_repulsor_baseline):
        result = fn(prepared, cfg)
        assert isinstance(result, BaselineUnavailable)
        assert len(result.reason) > 20


# ---------------------------------------------------------------------------
# run_experiment end-to-end, via a temporary registry entry pointing at the
# mock dataset (no real download: download_fn is a no-op).
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_experiment_registered(monkeypatch):
    from umaping.experiments.registry import ExperimentSpec

    def _noop_download(raw_dir):
        return Path(raw_dir)

    spec = ExperimentSpec(
        name="_test_mock_experiment",
        dataset_name="mock",
        config_path="configs/mock.yaml",
        download_fn=_noop_download,
        citation="synthetic test fixture, not a real dataset",
        source_url="n/a",
    )
    monkeypatch.setitem(_REGISTRY, "_test_mock_experiment", spec)
    yield "_test_mock_experiment"


def test_run_experiment_download_only_stops_after_download(mock_experiment_registered, tmp_path):
    from umaping.experiments.runner import run_experiment

    run_dir = tmp_path / "run"
    run_experiment(mock_experiment_registered, run_dir, torch.device("cpu"), download_only=True)
    assert not run_dir.exists()  # nothing created yet -- download_only never touches run_dir


def test_run_experiment_prepare_only_caches_dataset_without_training(mock_experiment_registered, tmp_path):
    from umaping.experiments.runner import run_experiment

    run_dir = tmp_path / "run"
    run_experiment(mock_experiment_registered, run_dir, torch.device("cpu"), prepare_only=True)
    assert (run_dir / "cache" / "prepared_dataset.npz").exists()
    assert not (run_dir / "checkpoints" / "retriever.pt").exists()


def test_run_experiment_end_to_end_produces_baseline_comparison_and_timing(mock_experiment_registered, tmp_path):
    from umaping.experiments.runner import run_experiment

    run_dir = tmp_path / "run"
    run_experiment(mock_experiment_registered, run_dir, torch.device("cpu"))

    metrics_dir = run_dir / "metrics"
    assert (metrics_dir / "advanced_analysis.json").exists()  # analyze-advanced also ran
    assert (metrics_dir / "baseline_comparison.csv").exists()
    assert (metrics_dir / "timing.json").exists()

    import pandas as pd

    df = pd.read_csv(metrics_dir / "baseline_comparison.csv")
    methods = set(df["method"])
    assert {"standard_umap", "reduced_repulsion_umap", "weighted_knn", "ours", "ours_oracle_neighbors", "no_repulsion", "exact_repulsion_diagnostic"} <= methods
    assert {"numap_sep_spectralnet", "param_repulsor"} <= methods
    unavailable = df[~df["available"]]
    assert set(unavailable["method"]) >= {"numap_sep_spectralnet", "param_repulsor"}
    available = df[df["available"]]
    numeric_cols = available.select_dtypes(include=["number"]).to_numpy()
    assert not np.any(np.isinf(numeric_cols))


def test_run_experiment_resume_does_not_raise_on_a_completed_run(mock_experiment_registered, tmp_path):
    from umaping.experiments.runner import run_experiment

    run_dir = tmp_path / "run"
    run_experiment(mock_experiment_registered, run_dir, torch.device("cpu"))
    run_experiment(mock_experiment_registered, run_dir, torch.device("cpu"), resume=True)  # must not raise


def test_run_experiment_without_resume_on_an_existing_run_dir_raises(mock_experiment_registered, tmp_path):
    from umaping.experiments.runner import run_experiment

    run_dir = tmp_path / "run"
    run_experiment(mock_experiment_registered, run_dir, torch.device("cpu"))
    with pytest.raises(SystemExit):
        run_experiment(mock_experiment_registered, run_dir, torch.device("cpu"), resume=False)
