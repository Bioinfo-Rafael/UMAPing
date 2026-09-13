"""Pancreas batch correction (`data/pancreas.py`, `pipeline.py::stage_preprocess`).

Written under an explicit "do not run Python/pytest locally" constraint --
authored and statically reviewed, not executed. scvi-tools is mocked via a
fake `scvi` module injected into `sys.modules` (per that task's instruction
to mock scVI so unit tests do not require expensive training); COIL/mock
preprocessing is untouched by this feature and is not re-tested here (see
the existing `tests/test_data_splits.py` for those datasets' own tests,
which remain valid unmodified).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from umaping.config import Config, DatasetConfig


def _make_synthetic_pancreas_adata(rng, n_ref, n_query, n_genes, query_lam=5.0):
    import anndata as ad

    tech = np.array(
        ["batchA"] * (n_ref // 2)
        + ["batchB"] * (n_ref - n_ref // 2)
        + ["smartseq2"] * (n_query // 2)
        + ["celseq2"] * (n_query - n_query // 2)
    )
    celltype = rng.choice(["alpha", "beta", "gamma"], size=n_ref + n_query)
    counts_ref = rng.poisson(lam=5.0, size=(n_ref, n_genes)).astype(np.float32)
    counts_query = rng.poisson(lam=query_lam, size=(n_query, n_genes)).astype(np.float32)
    counts = np.concatenate([counts_ref, counts_query], axis=0)

    adata = ad.AnnData(X=counts.copy())
    adata.layers["counts"] = counts.copy()
    adata.obs["tech"] = tech
    adata.obs["celltype"] = celltype
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    return adata


# ---------------------------------------------------------------------------
# Fake scvi-tools module: records every call so tests can assert on the
# exact sequence/arguments this implementation is documented to use, without
# requiring scvi-tools (or any real training) to be installed/run.
# ---------------------------------------------------------------------------


class _FakeSCVIModel:
    calls: list[tuple[str, dict]] = []

    def __init__(self, adata, n_latent=50, **kwargs):
        self.adata = adata
        self.n_latent = n_latent
        type(self).calls.append(("__init__", {"n_obs": adata.n_obs, "n_latent": n_latent}))

    def train(self, max_epochs=None, **kwargs):
        type(self).calls.append(("train", {"max_epochs": max_epochs, "n_obs": self.adata.n_obs}))

    def get_latent_representation(self):
        rng = np.random.default_rng(0)
        return rng.normal(size=(self.adata.n_obs, self.n_latent)).astype(np.float32)

    @staticmethod
    def setup_anndata(adata, layer=None, batch_key=None):
        _FakeSCVIModel.calls.append(
            (
                "setup_anndata",
                {
                    "layer": layer,
                    "batch_key": batch_key,
                    "n_obs": adata.n_obs,
                    "tech_values": sorted(set(adata.obs[batch_key].astype(str).tolist())) if batch_key else None,
                },
            )
        )

    @staticmethod
    def prepare_query_anndata(adata, reference_model, inplace: bool = True):
        # Matches the real scvi-tools API exactly (verified against the
        # installed source): with the default inplace=True, this mutates
        # `adata` in place and returns None -- it does NOT hand back a new
        # AnnData in that mode. A caller that (incorrectly) used the return
        # value here would get None, exactly reproducing the real bug this
        # fake is designed to catch.
        _FakeSCVIModel.calls.append(("prepare_query_anndata", {"n_obs": adata.n_obs, "inplace": inplace}))
        adata.uns["_prepare_query_anndata_called"] = True
        if not inplace:
            return adata
        return None

    @classmethod
    def load_query_data(cls, adata, reference_model):
        _FakeSCVIModel.calls.append(("load_query_data", {"n_obs": adata.n_obs}))
        return cls(adata, n_latent=reference_model.n_latent)


@pytest.fixture
def fake_scvi_module(monkeypatch):
    _FakeSCVIModel.calls.clear()
    module = types.ModuleType("scvi")
    module.settings = types.SimpleNamespace(seed=None)
    module.model = types.SimpleNamespace(SCVI=_FakeSCVIModel)
    monkeypatch.setitem(sys.modules, "scvi", module)
    yield module
    _FakeSCVIModel.calls.clear()


# ---------------------------------------------------------------------------
# _reference_only_hvg_selection: shared by both preprocessing paths
# ---------------------------------------------------------------------------


def test_reference_only_hvg_selection_is_invariant_to_query_expression():
    from umaping.data.pancreas import _reference_only_hvg_selection
    import scanpy as sc

    rng = np.random.default_rng(1)
    n_ref, n_query, n_genes = 60, 40, 100
    adata_a = _make_synthetic_pancreas_adata(rng, n_ref, n_query, n_genes, query_lam=5.0)
    adata_a.X = adata_a.layers["counts"].copy()
    sc.pp.normalize_total(adata_a, target_sum=1e4)
    sc.pp.log1p(adata_a)
    reference_mask = np.zeros(n_ref + n_query, dtype=bool)
    reference_mask[:n_ref] = True

    genes_a = _reference_only_hvg_selection(adata_a, reference_mask, n_hvg=20)

    adata_b = _make_synthetic_pancreas_adata(rng, n_ref, n_query, n_genes, query_lam=500.0)  # wildly different query
    adata_b.layers["counts"][:n_ref] = adata_a.layers["counts"][:n_ref]  # keep reference identical
    adata_b.X = adata_b.layers["counts"].copy()
    sc.pp.normalize_total(adata_b, target_sum=1e4)
    sc.pp.log1p(adata_b)
    genes_b = _reference_only_hvg_selection(adata_b, reference_mask, n_hvg=20)

    np.testing.assert_array_equal(sorted(genes_a.tolist()), sorted(genes_b.tolist()))


# ---------------------------------------------------------------------------
# prepare_pancreas_dataset: batch_correction dispatch
# ---------------------------------------------------------------------------


def test_batch_correction_false_uses_legacy_pca_path():
    from umaping.data.pancreas import prepare_pancreas_dataset

    rng = np.random.default_rng(3)
    adata = _make_synthetic_pancreas_adata(rng, n_ref=60, n_query=40, n_genes=80)

    prepared, pca = prepare_pancreas_dataset(
        raw_dir="unused", n_hvg=15, n_pcs=5, seed=0, adata=adata, batch_correction=False
    )

    assert pca is not None
    assert prepared.input_dim == 5
    assert prepared.reference_features.shape == (60, 5)
    assert prepared.query_features.shape == (40, 5)


def test_batch_correction_default_is_false():
    from umaping.data.pancreas import prepare_pancreas_dataset

    rng = np.random.default_rng(4)
    adata = _make_synthetic_pancreas_adata(rng, n_ref=30, n_query=20, n_genes=50)
    prepared, pca = prepare_pancreas_dataset(raw_dir="unused", n_hvg=10, n_pcs=4, seed=0, adata=adata)
    assert pca is not None  # legacy path, not the scVI path


def test_batch_correction_true_routes_to_scvi_scarches_path(fake_scvi_module):
    from umaping.data.pancreas import prepare_pancreas_dataset

    rng = np.random.default_rng(5)
    n_ref, n_query = 60, 40
    adata = _make_synthetic_pancreas_adata(rng, n_ref=n_ref, n_query=n_query, n_genes=80)

    prepared, pca = prepare_pancreas_dataset(
        raw_dir="unused",
        n_hvg=15,
        n_pcs=5,
        seed=0,
        adata=adata,
        batch_correction=True,
        scvi_n_latent=7,
        scvi_max_epochs=11,
        scarches_max_epochs=13,
    )

    assert pca is None
    assert prepared.input_dim == 7
    assert prepared.reference_features.shape == (n_ref, 7)
    assert prepared.query_features.shape == (n_query, 7)
    assert np.all(np.isfinite(prepared.reference_features))
    assert np.all(np.isfinite(prepared.query_features))

    call_names = [name for name, _ in _FakeSCVIModel.calls]
    assert call_names == [
        "setup_anndata",
        "__init__",
        "train",
        "prepare_query_anndata",
        "load_query_data",
        "__init__",
        "train",
    ]

    setup_call = dict(_FakeSCVIModel.calls[0][1])
    assert setup_call["layer"] == "counts"
    assert setup_call["batch_key"] == "tech"
    assert setup_call["n_obs"] == n_ref
    assert "smartseq2" not in setup_call["tech_values"]
    assert "celseq2" not in setup_call["tech_values"]

    reference_init_call = dict(_FakeSCVIModel.calls[1][1])
    assert reference_init_call["n_obs"] == n_ref
    assert reference_init_call["n_latent"] == 7

    reference_train_call = dict(_FakeSCVIModel.calls[2][1])
    assert reference_train_call["max_epochs"] == 11

    prepare_query_call = dict(_FakeSCVIModel.calls[3][1])
    assert prepare_query_call["n_obs"] == n_query

    load_query_call = dict(_FakeSCVIModel.calls[4][1])
    assert load_query_call["n_obs"] == n_query

    query_init_call = dict(_FakeSCVIModel.calls[5][1])
    assert query_init_call["n_obs"] == n_query
    assert query_init_call["n_latent"] == 7  # inherited from the reference model, not re-specified

    query_train_call = dict(_FakeSCVIModel.calls[6][1])
    assert query_train_call["max_epochs"] == 13


def test_batch_correction_true_never_exposes_query_cells_to_reference_setup(fake_scvi_module):
    """The reference scVI model must be set up/trained on reference cells
    only -- the AnnData passed to setup_anndata/__init__/train must never
    contain a query-technology cell."""
    from umaping.data.pancreas import prepare_pancreas_dataset

    rng = np.random.default_rng(6)
    adata = _make_synthetic_pancreas_adata(rng, n_ref=50, n_query=30, n_genes=60)
    prepare_pancreas_dataset(raw_dir="unused", n_hvg=10, n_pcs=4, seed=0, adata=adata, batch_correction=True)

    setup_call = dict(_FakeSCVIModel.calls[0][1])
    assert setup_call["n_obs"] == 50
    assert set(setup_call["tech_values"]) <= {"batchA", "batchB"}


def test_batch_correction_true_uses_configured_seed(fake_scvi_module):
    from umaping.data.pancreas import prepare_pancreas_dataset
    import scvi  # the fake module injected by fake_scvi_module

    rng = np.random.default_rng(7)
    adata = _make_synthetic_pancreas_adata(rng, n_ref=40, n_query=20, n_genes=50)
    prepare_pancreas_dataset(raw_dir="unused", n_hvg=10, n_pcs=4, seed=1234, adata=adata, batch_correction=True)
    assert scvi.settings.seed == 1234


# ---------------------------------------------------------------------------
# pipeline.py::stage_preprocess -- config plumbing, including the critical
# "missing key means legacy behavior" backward-compatibility contract.
# ---------------------------------------------------------------------------


def _pancreas_config(params: dict) -> Config:
    return Config(dataset=DatasetConfig(name="pancreas", raw_dir="unused", input_dim=50, params=params))


def _patch_prepare_pancreas_dataset(monkeypatch, captured: dict):
    import umaping.pipeline as pipeline_module
    from umaping.data.preprocessing import PreparedDataset

    def fake_prepare_pancreas_dataset(**kwargs):
        captured.update(kwargs)
        cfg_input_dim = kwargs.get("scvi_n_latent") if kwargs.get("batch_correction") else kwargs.get("n_pcs")
        prepared = PreparedDataset(
            reference_features=np.zeros((3, cfg_input_dim), dtype=np.float32),
            query_features=np.zeros((2, cfg_input_dim), dtype=np.float32),
            reference_labels={"celltype": np.array(["a", "b", "a"]), "tech": np.array(["x", "x", "y"])},
            query_labels={"celltype": np.array(["a", "b"]), "tech": np.array(["q", "q"])},
            input_dim=cfg_input_dim,
        )
        return prepared, None

    monkeypatch.setattr(pipeline_module, "prepare_pancreas_dataset", fake_prepare_pancreas_dataset)


def _run_stage_preprocess(cfg, tmp_path, monkeypatch, captured):
    import umaping.pipeline as pipeline_module

    _patch_prepare_pancreas_dataset(monkeypatch, captured)
    layout = {"cache": tmp_path / "cache", "memory": tmp_path / "memory"}
    layout["cache"].mkdir()
    layout["memory"].mkdir()
    pipeline_module.stage_preprocess(cfg, layout)


def test_stage_preprocess_missing_batch_correction_key_defaults_to_false(tmp_path, monkeypatch):
    cfg = _pancreas_config(params={"n_hvg": 10, "n_pcs": 50})  # no "batch_correction" key at all
    captured: dict = {}
    _run_stage_preprocess(cfg, tmp_path, monkeypatch, captured)
    assert captured["batch_correction"] is False


def test_stage_preprocess_explicit_batch_correction_false(tmp_path, monkeypatch):
    cfg = _pancreas_config(params={"n_hvg": 10, "n_pcs": 50, "batch_correction": False})
    captured: dict = {}
    _run_stage_preprocess(cfg, tmp_path, monkeypatch, captured)
    assert captured["batch_correction"] is False


def test_stage_preprocess_explicit_batch_correction_true(tmp_path, monkeypatch):
    cfg = _pancreas_config(
        params={
            "n_hvg": 10,
            "n_pcs": 50,
            "batch_correction": True,
            "scvi_n_latent": 50,
            "scvi_max_epochs": 400,
            "scarches_max_epochs": 200,
        }
    )
    captured: dict = {}
    _run_stage_preprocess(cfg, tmp_path, monkeypatch, captured)
    assert captured["batch_correction"] is True
    assert captured["scvi_n_latent"] == 50
    assert captured["scvi_max_epochs"] == 400
    assert captured["scarches_max_epochs"] == 200


def test_stage_preprocess_defaults_scvi_params_when_batch_correction_true_without_them(tmp_path, monkeypatch):
    cfg = _pancreas_config(params={"batch_correction": True})
    captured: dict = {}
    _run_stage_preprocess(cfg, tmp_path, monkeypatch, captured)
    assert captured["batch_correction"] is True
    assert captured["scvi_n_latent"] == 50
    assert captured["scvi_max_epochs"] == 400
    assert captured["scarches_max_epochs"] == 200


def test_stage_preprocess_rejects_unsupported_batch_correction_method(tmp_path, monkeypatch):
    import umaping.pipeline as pipeline_module

    cfg = _pancreas_config(params={"batch_correction": True, "batch_correction_method": "bogus_method"})
    captured: dict = {}
    _patch_prepare_pancreas_dataset(monkeypatch, captured)
    layout = {"cache": tmp_path / "cache", "memory": tmp_path / "memory"}
    layout["cache"].mkdir()
    layout["memory"].mkdir()
    with pytest.raises(ValueError, match="Unsupported"):
        pipeline_module.stage_preprocess(cfg, layout)


def test_stage_preprocess_batch_correction_false_ignores_bogus_method_key(tmp_path, monkeypatch):
    """batch_correction_method is only validated when batch_correction is
    actually True; an irrelevant/bogus value must not block the legacy
    path (a behavioral check: this call must simply succeed)."""
    cfg = _pancreas_config(params={"batch_correction": False, "batch_correction_method": "bogus_method"})
    captured: dict = {}
    _run_stage_preprocess(cfg, tmp_path, monkeypatch, captured)  # must not raise
    assert captured["batch_correction"] is False
