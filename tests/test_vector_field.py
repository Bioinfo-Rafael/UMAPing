"""Total learned UMAP dynamics field (evaluation/vector_field.py) and its
pancreas-only integration into `pipeline.py::run_analysis`.

Written under an explicit "do not run Python/pytest locally" constraint
(see RUNTIME_CHECKS.md's follow-up note for this branch) -- these tests were
authored, not executed, and reviewed only by static reading. scVelo/scanpy
are mocked via fake modules injected into `sys.modules` (per that same
task's instruction to mock scVI/scVelo so unit tests do not require
expensive training/computation); `anndata` itself is used directly (already
a real dependency exercised by `tests/test_data_splits.py`'s pancreas
tests), and `umap_forces.g_plus` is used directly as an independent
cross-check of `compute_total_field`'s attraction term.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from umaping.evaluation.vector_field import (
    build_vector_field_anndata,
    compute_total_field,
    compute_velocity_embedding_on_umap,
    generate_umap_vector_field_figures,
)
from umaping.graph import full_edges, row_degree
from umaping.models.repulsion import RepulsionField
from umaping.umap_forces import g_plus


# ---------------------------------------------------------------------------
# 1. compute_total_field: the reused force equation
# ---------------------------------------------------------------------------


def _toy_symmetric_graph(n=6, seed=0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.uniform(0.1, 1.0, size=(n, n)).astype(np.float32)
    dense = (dense + dense.T) / 2.0
    np.fill_diagonal(dense, 0.0)
    dense[dense < 0.6] = 0.0  # sparsify
    return sp.csr_matrix(dense)


def test_compute_total_field_matches_independent_manual_recomputation():
    """Independently recomputes A_i via a plain Python loop over `g_plus`
    (not `scatter_attraction`, so this is a genuine cross-check, not a
    tautology) and B_phi via the same trained RepulsionField, then compares
    against `compute_total_field`'s output under the identical clip
    convention."""
    torch.manual_seed(0)
    n, d = 6, 2
    w = _toy_symmetric_graph(n=n, seed=1)
    row_np, col_np, weight_np = full_edges(w)
    degree_np = row_degree(w)

    y = torch.randn(n, d)
    a, b = 1.2, 0.9
    t = 0.5
    negative_sample_rate = 5.0
    grad_clip = 4.0

    model = RepulsionField(embedding_dim=d, hidden_dim=8, n_residual_blocks=1, time_embed_dim=4)
    model.eval()

    row_t = torch.as_tensor(row_np, dtype=torch.long)
    col_t = torch.as_tensor(col_np, dtype=torch.long)
    weight_t = torch.as_tensor(weight_np, dtype=torch.float32)
    degree_t = torch.as_tensor(degree_np, dtype=torch.float32)

    result = compute_total_field(
        y, row_t, col_t, weight_t, degree_t, model, a, b, t,
        negative_sample_rate=negative_sample_rate, grad_clip=grad_clip,
    )

    # Independent recomputation of A_i via a plain accumulation loop.
    manual_a = torch.zeros(n, d)
    for r, c, wgt in zip(row_np.tolist(), col_np.tolist(), weight_np.tolist()):
        contrib = wgt * g_plus(y[r].unsqueeze(0), y[c].unsqueeze(0), a, b, clip=grad_clip)[0]
        manual_a[r] += contrib

    with torch.no_grad():
        t_tensor = torch.full((n,), t, dtype=torch.float32)
        manual_b = model(y, t_tensor)

    manual_force = manual_a + negative_sample_rate * degree_t.unsqueeze(-1) * manual_b
    manual_force = manual_force.clamp(min=-grad_clip, max=grad_clip)

    torch.testing.assert_close(result, manual_force, atol=1e-5, rtol=1e-4)


def test_compute_total_field_output_shape_and_finiteness():
    n, d = 10, 2
    w = _toy_symmetric_graph(n=n, seed=2)
    row_t = torch.as_tensor(full_edges(w)[0], dtype=torch.long)
    col_t = torch.as_tensor(full_edges(w)[1], dtype=torch.long)
    weight_t = torch.as_tensor(full_edges(w)[2], dtype=torch.float32)
    degree_t = torch.as_tensor(row_degree(w), dtype=torch.float32)
    y = torch.randn(n, d)
    model = RepulsionField(embedding_dim=d, hidden_dim=8, n_residual_blocks=1, time_embed_dim=4)

    for t in (0.0, 0.5, 1.0):
        result = compute_total_field(y, row_t, col_t, weight_t, degree_t, model, 1.0, 1.0, t, negative_sample_rate=5.0)
        assert result.shape == (n, d)
        assert torch.all(torch.isfinite(result))


def test_compute_total_field_respects_aggregate_clip():
    """A pathologically large repulsion output must still be clipped to
    [-grad_clip, grad_clip] in the final aggregate, per the documented
    (`dynamics.py`/`inference.py`-matching) convention."""

    class _HugeRepulsion(torch.nn.Module):
        def forward(self, y, t):
            return torch.full_like(y, 1000.0)

    n, d = 4, 2
    w = _toy_symmetric_graph(n=n, seed=3)
    row_t = torch.as_tensor(full_edges(w)[0], dtype=torch.long)
    col_t = torch.as_tensor(full_edges(w)[1], dtype=torch.long)
    weight_t = torch.as_tensor(full_edges(w)[2], dtype=torch.float32)
    degree_t = torch.as_tensor(row_degree(w), dtype=torch.float32)
    y = torch.randn(n, d)

    result = compute_total_field(
        y, row_t, col_t, weight_t, degree_t, _HugeRepulsion(), 1.0, 1.0, 0.5,
        negative_sample_rate=5.0, grad_clip=4.0,
    )
    assert torch.all(result <= 4.0 + 1e-6)
    assert torch.all(result >= -4.0 - 1e-6)


# ---------------------------------------------------------------------------
# 2. build_vector_field_anndata: observation order + shape preservation
# ---------------------------------------------------------------------------


def test_build_vector_field_anndata_preserves_order_and_content():
    n = 7
    rng = np.random.default_rng(4)
    coordinates = rng.normal(size=(n, 2)).astype(np.float32)
    velocity = rng.normal(size=(n, 2)).astype(np.float32)
    celltype = np.array([f"type_{i % 3}" for i in range(n)])
    tech = np.array([f"tech_{i % 2}" for i in range(n)])

    adata = build_vector_field_anndata(coordinates, velocity, celltype, tech, vkey="velocity")

    np.testing.assert_array_equal(np.asarray(adata.X), coordinates)
    np.testing.assert_array_equal(np.asarray(adata.layers["X"]), coordinates)
    np.testing.assert_array_equal(np.asarray(adata.obsm["X_umap"]), coordinates)
    np.testing.assert_array_equal(np.asarray(adata.layers["velocity"]), velocity)
    np.testing.assert_array_equal(adata.obs["celltype"].astype(str).to_numpy(), celltype)
    np.testing.assert_array_equal(adata.obs["tech"].astype(str).to_numpy(), tech)
    assert isinstance(adata.obs["celltype"].dtype, pd.CategoricalDtype)
    assert isinstance(adata.obs["tech"].dtype, pd.CategoricalDtype)
    assert adata.layers["velocity"].shape == adata.X.shape


def test_build_vector_field_anndata_rejects_velocity_shape_mismatch():
    coordinates = np.zeros((5, 2), dtype=np.float32)
    bad_velocity = np.zeros((5, 3), dtype=np.float32)
    celltype = np.array(["a"] * 5)
    tech = np.array(["t"] * 5)
    with pytest.raises(ValueError, match="does not match"):
        build_vector_field_anndata(coordinates, bad_velocity, celltype, tech)


def test_build_vector_field_anndata_rejects_wrong_label_length():
    coordinates = np.zeros((5, 2), dtype=np.float32)
    velocity = np.zeros((5, 2), dtype=np.float32)
    celltype = np.array(["a"] * 4)  # wrong length
    tech = np.array(["t"] * 5)
    with pytest.raises(ValueError, match="one entry per reference cell"):
        build_vector_field_anndata(coordinates, velocity, celltype, tech)


def test_build_vector_field_anndata_rejects_non_2d_coordinates():
    coordinates = np.zeros((5, 3), dtype=np.float32)  # not 2-D
    velocity = np.zeros((5, 3), dtype=np.float32)
    celltype = np.array(["a"] * 5)
    tech = np.array(["t"] * 5)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        build_vector_field_anndata(coordinates, velocity, celltype, tech)


# ---------------------------------------------------------------------------
# 3. compute_velocity_embedding_on_umap: X_umap invariance + shape checks,
# against fake scanpy/scvelo modules (never the real, potentially expensive,
# packages).
# ---------------------------------------------------------------------------


def _fake_scanpy_module(monkeypatch):
    module = types.ModuleType("scanpy")
    pp = types.SimpleNamespace(neighbors=lambda adata, n_neighbors=15, use_rep="X": adata.uns.__setitem__("neighbors", {}))
    module.pp = pp
    monkeypatch.setitem(sys.modules, "scanpy", module)
    return module


def _fake_scvelo_module(monkeypatch, *, mutate_umap: bool = False, version: str = "0.3.0"):
    module = types.ModuleType("scvelo")
    module.__version__ = version

    def velocity_graph(adata, vkey="velocity", xkey="X", backend="loky", n_jobs=1):
        adata.uns["_velocity_graph_called_with"] = dict(vkey=vkey, xkey=xkey, backend=backend, n_jobs=n_jobs)
        if mutate_umap:
            adata.obsm["X_umap"] = np.asarray(adata.obsm["X_umap"]) + 1.0

    def velocity_embedding(adata, basis="umap", vkey="velocity"):
        adata.obsm[f"{vkey}_{basis}"] = np.zeros((adata.n_obs, 2), dtype=np.float32)

    module.tl = types.SimpleNamespace(velocity_graph=velocity_graph, velocity_embedding=velocity_embedding)
    module.pl = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "scvelo", module)
    return module


def _toy_field_anndata(n=8, vkey="velocity"):
    rng = np.random.default_rng(5)
    coordinates = rng.normal(size=(n, 2)).astype(np.float32)
    velocity = rng.normal(size=(n, 2)).astype(np.float32)
    celltype = np.array([f"type_{i % 3}" for i in range(n)])
    tech = np.array(["techA"] * n)
    return build_vector_field_anndata(coordinates, velocity, celltype, tech, vkey=vkey)


def test_compute_velocity_embedding_on_umap_succeeds_and_returns_unchanged_umap(monkeypatch):
    _fake_scanpy_module(monkeypatch)
    _fake_scvelo_module(monkeypatch, mutate_umap=False)
    adata = _toy_field_anndata()
    original_umap = np.asarray(adata.obsm["X_umap"]).copy()

    returned = compute_velocity_embedding_on_umap(adata, vkey="velocity", n_neighbors=3)

    np.testing.assert_array_equal(returned, original_umap)
    np.testing.assert_array_equal(np.asarray(adata.obsm["X_umap"]), original_umap)
    assert "velocity_umap" in adata.obsm
    assert np.asarray(adata.obsm["velocity_umap"]).shape == (adata.n_obs, 2)


def test_compute_velocity_embedding_on_umap_raises_if_x_umap_is_mutated(monkeypatch):
    _fake_scanpy_module(monkeypatch)
    _fake_scvelo_module(monkeypatch, mutate_umap=True)
    adata = _toy_field_anndata()

    with pytest.raises(RuntimeError, match="X_umap"):
        compute_velocity_embedding_on_umap(adata, vkey="velocity", n_neighbors=3)


def test_compute_velocity_embedding_on_umap_raises_without_x_umap(monkeypatch):
    _fake_scanpy_module(monkeypatch)
    _fake_scvelo_module(monkeypatch)
    adata = _toy_field_anndata()
    del adata.obsm["X_umap"]

    with pytest.raises(KeyError):
        compute_velocity_embedding_on_umap(adata, vkey="velocity")


def test_compute_velocity_embedding_on_umap_raises_on_velocity_shape_mismatch(monkeypatch):
    _fake_scanpy_module(monkeypatch)
    _fake_scvelo_module(monkeypatch)
    adata = _toy_field_anndata()
    adata.layers["velocity"] = np.zeros((adata.n_obs, 5), dtype=np.float32)  # wrong width

    with pytest.raises(ValueError, match="velocity layer shape mismatch"):
        compute_velocity_embedding_on_umap(adata, vkey="velocity")


def test_compute_velocity_embedding_on_umap_applies_0_2_5_ragged_array_patch(monkeypatch):
    """When scvelo reports version 0.2.5, the ragged-array compatibility
    patch must be applied around velocity_graph and restored afterward."""
    _fake_scanpy_module(monkeypatch)
    scv_module = _fake_scvelo_module(monkeypatch, version="0.2.5")

    graph_module = types.ModuleType("scvelo.tools.velocity_graph")
    calls = []

    def fake_parallelize(*args, **kwargs):
        calls.append(kwargs.get("as_array"))
        return []

    graph_module.parallelize = fake_parallelize
    # `importlib.import_module` consults `sys.modules` for the *exact* dotted
    # name first (see importlib._bootstrap._find_and_load) and returns it
    # immediately if present, without needing to also resolve/import the
    # "scvelo.tools" parent package -- so only this one entry is needed.
    monkeypatch.setitem(sys.modules, "scvelo.tools.velocity_graph", graph_module)

    adata = _toy_field_anndata()
    compute_velocity_embedding_on_umap(adata, vkey="velocity")

    # The patch must be restored (not left dangling) after the call.
    assert graph_module.parallelize is fake_parallelize
    assert scv_module.__version__ == "0.2.5"


# ---------------------------------------------------------------------------
# 4. generate_umap_vector_field_figures: shared palette + expected filenames,
# with the scVelo-dependent steps stubbed out.
# ---------------------------------------------------------------------------


def test_generate_umap_vector_field_figures_uses_shared_palette_and_expected_filenames(monkeypatch):
    import umaping.evaluation.vector_field as vf

    computed_calls = []
    plot_calls = []

    def fake_compute(adata, vkey="velocity", n_neighbors=15, n_jobs=None):
        computed_calls.append(adata)
        return np.asarray(adata.obsm["X_umap"]).copy()

    def fake_plot_triplet(adata, output_dir, *, vkey, title, filenames):
        plot_calls.append(
            {
                "celltype_colors": list(adata.uns.get("celltype_colors", [])),
                "categories": [str(c) for c in adata.obs["celltype"].cat.categories],
                "vkey": vkey,
                "title": title,
                "filenames": tuple(filenames),
            }
        )
        return [vf.Path(output_dir) / name for name in filenames]

    monkeypatch.setattr(vf, "compute_velocity_embedding_on_umap", fake_compute)
    monkeypatch.setattr(vf, "plot_velocity_triplet", fake_plot_triplet)

    n = 9
    rng = np.random.default_rng(6)
    celltype = np.array([f"type_{i % 3}" for i in range(n)])
    tech = np.array(["techA"] * n)
    coordinates_by_t = {
        "t000": rng.normal(size=(n, 2)).astype(np.float32),
        "t050": rng.normal(size=(n, 2)).astype(np.float32),
        "t100": rng.normal(size=(n, 2)).astype(np.float32),
    }
    velocity_by_t = {k: rng.normal(size=(n, 2)).astype(np.float32) for k in coordinates_by_t}

    created = generate_umap_vector_field_figures(coordinates_by_t, velocity_by_t, celltype, tech, "/tmp/figures")

    assert len(plot_calls) == 3
    assert len(created) == 9  # 3 files per t x 3 t's

    # Every call must share the exact same palette/category assignment.
    first = plot_calls[0]
    for call in plot_calls[1:]:
        assert call["celltype_colors"] == first["celltype_colors"]
        assert call["categories"] == first["categories"]

    filenames_by_suffix = {suffix: call["filenames"] for suffix, call in zip(("t000", "t050", "t100"), plot_calls)}
    assert filenames_by_suffix["t000"] == ("vector_field_t000_stream.png", "vector_field_t000_arrow.png", "vector_field_t000_grid.png")
    assert filenames_by_suffix["t050"] == ("vector_field_t050_stream.png", "vector_field_t050_arrow.png", "vector_field_t050_grid.png")
    assert filenames_by_suffix["t100"] == ("vector_field_t100_stream.png", "vector_field_t100_arrow.png", "vector_field_t100_grid.png")


def test_generate_umap_vector_field_figures_rejects_mismatched_t_keys():
    n = 4
    celltype = np.array(["a"] * n)
    tech = np.array(["t"] * n)
    coordinates_by_t = {"t000": np.zeros((n, 2), dtype=np.float32)}  # missing t050/t100
    velocity_by_t = {"t000": np.zeros((n, 2), dtype=np.float32)}
    with pytest.raises(ValueError, match="t000"):
        generate_umap_vector_field_figures(coordinates_by_t, velocity_by_t, celltype, tech, "/tmp/figures")


# ---------------------------------------------------------------------------
# 5. pipeline.py::generate_pancreas_vector_field_figures_if_applicable --
# pancreas-only gating, isolated from the rest of run_analysis.
# ---------------------------------------------------------------------------


def test_non_pancreas_dataset_skips_vector_field_generation_entirely():
    """For any non-pancreas dataset, the function must return immediately,
    before touching any of its other (here: deliberately invalid/unusable)
    arguments -- confirms no pancreas/scVelo-specific assumption leaks into
    other datasets' analysis."""
    from umaping.config import Config, DatasetConfig
    from umaping.pipeline import generate_pancreas_vector_field_figures_if_applicable

    cfg = Config(dataset=DatasetConfig(name="coil20", input_dim=8))
    result = generate_pancreas_vector_field_figures_if_applicable(
        cfg,
        layout="not-a-real-layout",  # would break immediately if ever touched
        device="not-a-real-device",
        prepared="not-a-real-prepared-dataset",
        trajectory="not-a-real-trajectory",
        repulsion_model="not-a-real-model",
        a=1.0,
        b=1.0,
    )
    assert result is None


def test_pancreas_missing_celltype_or_tech_labels_skips_with_a_warning(caplog):
    from umaping.config import Config, DatasetConfig
    from umaping.data.preprocessing import PreparedDataset
    from umaping.pipeline import generate_pancreas_vector_field_figures_if_applicable

    cfg = Config(dataset=DatasetConfig(name="pancreas", input_dim=8))
    prepared = PreparedDataset(
        reference_features=np.zeros((3, 8), dtype=np.float32),
        query_features=np.zeros((1, 8), dtype=np.float32),
        reference_labels={},  # no "celltype"/"tech"
        query_labels={},
        input_dim=8,
    )
    result = generate_pancreas_vector_field_figures_if_applicable(
        cfg, layout={}, device="cpu", prepared=prepared, trajectory=None, repulsion_model=None, a=1.0, b=1.0
    )
    assert result is None
    assert any("celltype/tech" in message for message in caplog.messages)


def test_pancreas_dataset_computes_expected_field_dicts_for_all_three_t(monkeypatch, tmp_path):
    """Confirms the pancreas branch loads graph_symmetric.npz from run
    memory, computes the total field at t=0/0.5/1 via `compute_total_field`,
    and forwards exactly those three keyed dicts (in the reference-cell
    order given) to `generate_umap_vector_field_figures` -- without needing
    real scVelo (that call is stubbed and inspected here)."""
    import umaping.pipeline as pipeline_module
    from umaping.config import Config, DatasetConfig
    from umaping.data.preprocessing import PreparedDataset
    from umaping.dynamics import ReferenceTrajectory

    n = 5
    rng = np.random.default_rng(7)
    w = _toy_symmetric_graph(n=n, seed=7)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    sp.save_npz(str(memory_dir / "graph_symmetric.npz"), w)

    captured = {}

    def fake_generate_umap_vector_field_figures(coordinates_by_t, velocity_by_t, celltype, tech, figures_dir):
        captured["coordinates_by_t"] = coordinates_by_t
        captured["velocity_by_t"] = velocity_by_t
        captured["celltype"] = celltype
        captured["tech"] = tech
        return []

    monkeypatch.setattr(
        pipeline_module, "generate_umap_vector_field_figures", fake_generate_umap_vector_field_figures
    )

    cfg = Config(dataset=DatasetConfig(name="pancreas", input_dim=8))
    celltype = np.array([f"type_{i % 2}" for i in range(n)])
    tech = np.array(["inDrop1"] * n)
    prepared = PreparedDataset(
        reference_features=np.zeros((n, 8), dtype=np.float32),
        query_features=np.zeros((2, 8), dtype=np.float32),
        reference_labels={"celltype": celltype, "tech": tech},
        query_labels={"celltype": celltype[:2], "tech": tech[:2]},
        input_dim=8,
    )
    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    positions = rng.normal(size=(3, n, 2)).astype(np.float32)
    trajectory = ReferenceTrajectory(times=times, positions=positions)
    repulsion_model = RepulsionField(embedding_dim=2, hidden_dim=8, n_residual_blocks=1, time_embed_dim=4)

    result = pipeline_module.generate_pancreas_vector_field_figures_if_applicable(
        cfg,
        layout={"memory": memory_dir, "figures": tmp_path / "figures"},
        device=torch.device("cpu"),
        prepared=prepared,
        trajectory=trajectory,
        repulsion_model=repulsion_model,
        a=1.0,
        b=1.0,
    )

    assert result == []
    assert set(captured["coordinates_by_t"]) == {"t000", "t050", "t100"}
    assert set(captured["velocity_by_t"]) == {"t000", "t050", "t100"}
    for suffix in ("t000", "t050", "t100"):
        assert captured["coordinates_by_t"][suffix].shape == (n, 2)
        assert captured["velocity_by_t"][suffix].shape == (n, 2)
        assert np.all(np.isfinite(captured["coordinates_by_t"][suffix]))
        assert np.all(np.isfinite(captured["velocity_by_t"][suffix]))
    np.testing.assert_array_equal(captured["celltype"], celltype)
    np.testing.assert_array_equal(captured["tech"], tech)
    # t000 must equal the exact stored checkpoint at t=0 (an exact lookup,
    # per ReferenceTrajectory.positions_at's own documented behavior).
    np.testing.assert_allclose(captured["coordinates_by_t"]["t000"], positions[0])
