"""NUMAP/Sep-SpectralNet and ParamRepulsor baseline adapters
(experiments/baselines.py::run_numap_baseline, run_param_repulsor_baseline).

Neither official package (`numap`, `parampacmap`) is installed in this
environment, and this task explicitly forbids installing them while
implementing -- so every test here either (a) exercises the genuine
"dependency not installed" path (both packages really are absent here), or
(b) injects a fake module via `sys.modules`/monkeypatch that mimics just
enough of the real, source-verified API (see baselines.py's docstrings for
exactly which commit each contract was verified against) to check the
adapter's own logic: reference-only fitting, separate reference/query
transform calls, constructor-argument plumbing, and output shape/finiteness.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from umaping.config import Config, DatasetConfig
from umaping.data.preprocessing import PreparedDataset
from umaping.experiments.baselines import (
    BaselineResult,
    BaselineUnavailable,
    run_numap_baseline,
    run_param_repulsor_baseline,
)


def _tiny_prepared_dataset(seed=0, n_ref=40, n_query=12, d=5) -> PreparedDataset:
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


def _tiny_config(d=5, seed=7) -> Config:
    cfg = Config(dataset=DatasetConfig(name="toy", input_dim=d))
    cfg.umap.n_neighbors = 6
    cfg.umap.min_dist = 0.25
    cfg.umap.embedding_dim = 2
    cfg.seed = seed
    return cfg


# ---------------------------------------------------------------------------
# Fake `numap` module -- mimics the source-verified contract:
#   NUMAP(**kwargs).fit(X: torch.Tensor)              -- reference only
#   NUMAP(...).transform(X: torch.Tensor, is_train=..) -- called separately
# ---------------------------------------------------------------------------


class _FakeNUMAP:
    instances: list["_FakeNUMAP"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls: list[torch.Tensor] = []
        self.transform_calls: list[tuple[torch.Tensor, bool]] = []
        self._n_components = kwargs.get("n_components", 2)
        _FakeNUMAP.instances.append(self)

    def fit(self, X):
        assert isinstance(X, torch.Tensor), "NUMAP.fit must receive a torch.Tensor, per the official API"
        self.fit_calls.append(X)

    def transform(self, X, is_train=False):
        assert isinstance(X, torch.Tensor)
        self.transform_calls.append((X, is_train))
        value = 1.0 if is_train else 2.0
        return torch.full((X.shape[0], self._n_components), value, dtype=torch.float32)


@pytest.fixture
def fake_numap_module(monkeypatch):
    _FakeNUMAP.instances.clear()
    module = types.ModuleType("numap")
    module.NUMAP = _FakeNUMAP
    monkeypatch.setitem(sys.modules, "numap", module)
    yield _FakeNUMAP
    _FakeNUMAP.instances.clear()


# ---------------------------------------------------------------------------
# Fake `parampacmap` module -- mimics the source-verified contract:
#   ParamPaCMAP(**kwargs).fit(X: np.ndarray)      -- reference only
#   ParamPaCMAP(...).transform(X: np.ndarray)     -- called separately
# ---------------------------------------------------------------------------


class _FakeParamPaCMAP:
    instances: list["_FakeParamPaCMAP"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls: list[np.ndarray] = []
        self.transform_calls: list[np.ndarray] = []
        self._n_components = kwargs.get("n_components", 2)
        _FakeParamPaCMAP.instances.append(self)

    def fit(self, X):
        assert isinstance(X, np.ndarray), "ParamPaCMAP.fit must receive a numpy array, per the official API"
        self.fit_calls.append(X.copy())

    def transform(self, X):
        assert isinstance(X, np.ndarray)
        self.transform_calls.append(X.copy())
        return np.zeros((X.shape[0], self._n_components), dtype=np.float32)


@pytest.fixture
def fake_parampacmap_module(monkeypatch):
    _FakeParamPaCMAP.instances.clear()
    module = types.ModuleType("parampacmap")
    module.ParamPaCMAP = _FakeParamPaCMAP
    monkeypatch.setitem(sys.modules, "parampacmap", module)
    yield _FakeParamPaCMAP
    _FakeParamPaCMAP.instances.clear()


# ---------------------------------------------------------------------------
# Missing-dependency behavior (genuinely not installed in this environment)
# ---------------------------------------------------------------------------


def test_numap_baseline_is_unavailable_with_exact_install_command_when_not_installed():
    prepared = _tiny_prepared_dataset()
    cfg = _tiny_config()
    result = run_numap_baseline(prepared, cfg)
    assert isinstance(result, BaselineUnavailable)
    assert result.name == "numap_sep_spectralnet"
    assert "pip install numap==0.2.3" in result.reason
    assert "github.com/shaham-lab/NUMAP" in result.reason


def test_param_repulsor_baseline_is_unavailable_with_exact_install_command_when_not_installed():
    prepared = _tiny_prepared_dataset()
    cfg = _tiny_config()
    result = run_param_repulsor_baseline(prepared, cfg)
    assert isinstance(result, BaselineUnavailable)
    assert result.name == "param_repulsor"
    assert "pip install parampacmap==0.1.0" in result.reason
    assert "github.com/hyhuang00/ParamRepulsor" in result.reason


# ---------------------------------------------------------------------------
# NUMAP adapter, with the fake module installed
# ---------------------------------------------------------------------------


def test_numap_baseline_fits_reference_only_and_transforms_separately(fake_numap_module):
    prepared = _tiny_prepared_dataset(seed=1, n_ref=30, n_query=9, d=4)
    cfg = _tiny_config(d=4)

    result = run_numap_baseline(prepared, cfg, device=torch.device("cpu"))
    assert isinstance(result, BaselineResult)

    instance = fake_numap_module.instances[-1]
    assert len(instance.fit_calls) == 1
    np.testing.assert_allclose(instance.fit_calls[0].numpy(), prepared.reference_features)

    assert len(instance.transform_calls) == 2
    (call_ref, is_train_ref), (call_query, is_train_query) = instance.transform_calls
    assert is_train_ref is True
    assert is_train_query is False
    np.testing.assert_allclose(call_ref.numpy(), prepared.reference_features)
    np.testing.assert_allclose(call_query.numpy(), prepared.query_features)


def test_numap_baseline_output_shapes_and_finiteness(fake_numap_module):
    prepared = _tiny_prepared_dataset(seed=2, n_ref=25, n_query=8, d=6)
    cfg = _tiny_config(d=6)
    result = run_numap_baseline(prepared, cfg, device=torch.device("cpu"))

    assert result.reference_embedding.shape == (25, cfg.umap.embedding_dim)
    assert result.query_embedding.shape == (8, cfg.umap.embedding_dim)
    assert np.all(np.isfinite(result.reference_embedding))
    assert np.all(np.isfinite(result.query_embedding))
    # The fake distinguishes is_train via value -- confirms embed_one call routing indirectly.
    assert np.allclose(result.reference_embedding, 1.0)
    assert np.allclose(result.query_embedding, 2.0)


def test_numap_baseline_passes_expected_constructor_kwargs(fake_numap_module):
    prepared = _tiny_prepared_dataset(seed=3, n_ref=20, n_query=5, d=4)
    cfg = _tiny_config(d=4, seed=42)
    result = run_numap_baseline(prepared, cfg, device=torch.device("cpu"))

    instance = fake_numap_module.instances[-1]
    assert instance.kwargs["n_neighbors"] == cfg.umap.n_neighbors
    assert instance.kwargs["min_dist"] == cfg.umap.min_dist
    assert instance.kwargs["n_components"] == cfg.umap.embedding_dim
    assert instance.kwargs["se_dim"] == cfg.umap.embedding_dim
    assert instance.kwargs["random_state"] == 42
    assert instance.kwargs["use_se"] is True
    assert instance.kwargs["use_grease"] is True
    assert instance.kwargs["use_residual_connections"] is True
    assert instance.kwargs["num_gpus"] == 0  # cpu device in this test

    assert result.extra["constructor_kwargs"] == instance.kwargs
    assert result.extra["package"] == "numap"
    assert result.extra["version_pin"] == "0.2.3"


def test_numap_baseline_uses_cuda_num_gpus_when_device_is_cuda(fake_numap_module):
    prepared = _tiny_prepared_dataset(seed=4, n_ref=15, n_query=4, d=3)
    cfg = _tiny_config(d=3)
    run_numap_baseline(prepared, cfg, device=torch.device("cuda"))
    instance = fake_numap_module.instances[-1]
    assert instance.kwargs["num_gpus"] == 1


# ---------------------------------------------------------------------------
# ParamRepulsor adapter, with the fake module installed
# ---------------------------------------------------------------------------


def test_param_repulsor_baseline_fits_reference_only_and_transforms_separately(fake_parampacmap_module):
    prepared = _tiny_prepared_dataset(seed=5, n_ref=35, n_query=11, d=4)
    cfg = _tiny_config(d=4)

    result = run_param_repulsor_baseline(prepared, cfg)
    assert isinstance(result, BaselineResult)

    instance = fake_parampacmap_module.instances[-1]
    assert len(instance.fit_calls) == 1
    np.testing.assert_allclose(instance.fit_calls[0], prepared.reference_features)

    assert len(instance.transform_calls) == 2
    np.testing.assert_allclose(instance.transform_calls[0], prepared.reference_features)
    np.testing.assert_allclose(instance.transform_calls[1], prepared.query_features)


def test_param_repulsor_baseline_disables_official_pca_and_scaling(fake_parampacmap_module):
    prepared = _tiny_prepared_dataset(seed=6, n_ref=20, n_query=6, d=4)
    cfg = _tiny_config(d=4)
    run_param_repulsor_baseline(prepared, cfg)

    instance = fake_parampacmap_module.instances[-1]
    assert instance.kwargs["apply_pca"] is False
    assert instance.kwargs["apply_scale"] is None


def test_param_repulsor_baseline_passes_expected_constructor_kwargs(fake_parampacmap_module):
    prepared = _tiny_prepared_dataset(seed=7, n_ref=18, n_query=5, d=3)
    cfg = _tiny_config(d=3, seed=99)
    result = run_param_repulsor_baseline(prepared, cfg)

    instance = fake_parampacmap_module.instances[-1]
    assert instance.kwargs["n_components"] == cfg.umap.embedding_dim
    assert instance.kwargs["n_neighbors"] == cfg.umap.n_neighbors
    assert instance.kwargs["seed"] == 99

    assert result.extra["constructor_kwargs"] == instance.kwargs
    assert result.extra["package"] == "parampacmap"
    assert result.extra["version_pin"] == "0.1.0"


def test_param_repulsor_baseline_output_shapes_and_finiteness(fake_parampacmap_module):
    prepared = _tiny_prepared_dataset(seed=8, n_ref=22, n_query=7, d=5)
    cfg = _tiny_config(d=5)
    result = run_param_repulsor_baseline(prepared, cfg)

    assert result.reference_embedding.shape == (22, cfg.umap.embedding_dim)
    assert result.query_embedding.shape == (7, cfg.umap.embedding_dim)
    assert np.all(np.isfinite(result.reference_embedding))
    assert np.all(np.isfinite(result.query_embedding))


# ---------------------------------------------------------------------------
# Baseline registry integration: run_experiment_baselines actually invokes
# these adapters (not just the two functions in isolation).
# ---------------------------------------------------------------------------


def test_experiment_baseline_registry_invokes_numap_and_param_repulsor_when_available(
    fake_numap_module, fake_parampacmap_module, tmp_path, monkeypatch
):
    import pandas as pd

    from umaping.experiments.registry import _REGISTRY, ExperimentSpec
    from umaping.experiments.runner import run_experiment

    def _noop_download(raw_dir):
        from pathlib import Path

        return Path(raw_dir)

    spec = ExperimentSpec(
        name="_test_mock_with_external_baselines",
        dataset_name="mock",
        config_path="configs/mock.yaml",
        download_fn=_noop_download,
        citation="synthetic test fixture, not a real dataset",
        source_url="n/a",
    )
    monkeypatch.setitem(_REGISTRY, "_test_mock_with_external_baselines", spec)

    run_dir = tmp_path / "run"
    run_experiment("_test_mock_with_external_baselines", run_dir, torch.device("cpu"))

    df = pd.read_csv(run_dir / "metrics" / "baseline_comparison.csv")
    row_numap = df[df["method"] == "numap_sep_spectralnet"].iloc[0]
    row_param_repulsor = df[df["method"] == "param_repulsor"].iloc[0]
    assert bool(row_numap["available"]) is True
    assert bool(row_param_repulsor["available"]) is True

    # The fakes must actually have been invoked (not skipped/bypassed).
    assert len(_FakeNUMAP.instances) == 1
    assert len(_FakeParamPaCMAP.instances) == 1
    assert _FakeParamPaCMAP.instances[-1].kwargs["apply_pca"] is False
