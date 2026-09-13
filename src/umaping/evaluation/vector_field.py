"""scVelo-style visualization of the total learned UMAP dynamics field
(analytic attraction + learned repulsion), pancreas-only for now.

Two independent halves:

1. `compute_total_field` -- the force equation itself. This is *exactly* the
   field `dynamics.py::simulate_reference_dynamics` integrates and
   `inference.py::InferenceEngine.embed_one` evaluates in its "learned"
   repulsion mode, reused verbatim (`dynamics.scatter_attraction`, the
   trained `RepulsionField`, the same aggregate-clip convention) -- never a
   new force equation invented for visualization:

       F_i(t) = A_i(t) + negative_sample_rate * degree_i * B_phi(y_i(t), t)
       A_i(t) = sum_j W_ij g_plus(y_i(t), y_j(t))            (scatter_attraction)
       B_phi_i(t) = RepulsionField(y_i(t), t)

2. The scVelo plumbing (`build_vector_field_anndata`,
   `compute_velocity_embedding_on_umap`, `_velocity_plot`,
   `plot_velocity_triplet`) -- ported from the tested pattern in
   https://github.com/Bioinfo-Rafael/scDiffusionODE
   (work/20260830/hematopoietic_viz/{core,plotting}.py), adapted for a field
   that already lives in the final 2-D UMAP coordinate system: that
   reference implementation computes its own PCA + neighbors + UMAP for
   high-dimensional gene expression before calling scVelo; here, the
   neighbor graph scVelo needs before `velocity_graph` is instead built
   directly on the existing `Y_ref(t)` representation itself
   (`sc.pp.neighbors(..., use_rep="X")`), and no new PCA/UMAP embedding is
   introduced -- the displayed coordinates stay exactly `Y_ref(t)`, and an
   assertion (mirroring the reference implementation's own) checks this
   directly after every scVelo call.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from umaping.dynamics import scatter_attraction  # noqa: E402
from umaping.models.repulsion import RepulsionField  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_VKEY = "velocity"


# ---------------------------------------------------------------------------
# 1. The force equation (reused, not reinvented)
# ---------------------------------------------------------------------------


def compute_total_field(
    y: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    degree: torch.Tensor,
    repulsion_model: RepulsionField,
    a: float,
    b: float,
    t: float,
    negative_sample_rate: float,
    grad_clip: float | None = 4.0,
    eps_attr: float = 1e-12,
) -> torch.Tensor:
    """F_i(t) = A_i(t) + negative_sample_rate * degree_i * B_phi(y_i(t), t).

    `row, col, weight` are the *full* (both-direction) sparse edges of the
    symmetric graph W (`graph.py::full_edges`, as loaded from
    `<run_dir>/memory/graph_symmetric.npz` -- never rebuilt from scratch);
    `degree` is `graph.py::row_degree(W)`. `y`: `(N, d)` positions at time
    `t` (typically `ReferenceTrajectory.positions_at(t)`). Clipping mirrors
    `dynamics.py::simulate_reference_dynamics` and
    `inference.py::InferenceEngine.embed_one`'s "learned" repulsion mode
    exactly: `scatter_attraction` already clips each edge's attractive
    contribution before summing; the raw `B_phi` output is left unclipped
    (as in `embed_one`'s learned path); only the final aggregate `F_i` is
    clipped once, component-wise, to `[-grad_clip, grad_clip]`."""
    a_term = scatter_attraction(y, row, col, weight, a, b, eps=eps_attr, clip=grad_clip)
    t_tensor = torch.full((y.shape[0],), float(t), dtype=torch.float32, device=y.device)
    with torch.no_grad():
        b_term = repulsion_model(y, t_tensor)
    force = a_term + negative_sample_rate * degree.unsqueeze(-1) * b_term
    if grad_clip is not None:
        force = force.clamp(min=-grad_clip, max=grad_clip)
    return force


# ---------------------------------------------------------------------------
# 2. AnnData construction (scDiffusionODE's build_sampling_anndata / core.py
# convention, simplified: our field is already the final 2-D representation)
# ---------------------------------------------------------------------------


def build_vector_field_anndata(
    coordinates: np.ndarray,
    velocity: np.ndarray,
    celltype: np.ndarray,
    tech: np.ndarray,
    vkey: str = _DEFAULT_VKEY,
):
    """`coordinates`, `velocity`: `(N, 2)`, `Y_ref(t)` and `F(t)` at a single
    shared time `t`, in the same reference-cell order as `celltype`/`tech`
    (both taken directly from `PreparedDataset.reference_labels`, never
    relabeled). `adata.X` holds `coordinates` directly (this field already
    lives in the 2-D UMAP coordinate system -- no separate high-dimensional
    representation exists to store instead), `adata.layers["X"]` is the
    scDiffusionODE-convention copy of it (scVelo's `xkey="X"`), and
    `adata.obsm["X_umap"]` is set to the same coordinates so scVelo's
    plotting functions display exactly `Y_ref(t)`."""
    import anndata as ad

    coordinates = np.ascontiguousarray(coordinates, dtype=np.float32)
    velocity = np.ascontiguousarray(velocity, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"coordinates must be (N, 2); got {coordinates.shape}")
    if velocity.shape != coordinates.shape:
        raise ValueError(f"velocity shape {velocity.shape} does not match coordinates shape {coordinates.shape}")

    celltype = np.asarray(celltype).astype(str)
    tech = np.asarray(tech).astype(str)
    if len(celltype) != coordinates.shape[0] or len(tech) != coordinates.shape[0]:
        raise ValueError("celltype/tech must have exactly one entry per reference cell, in the same order")

    adata = ad.AnnData(X=coordinates.copy())
    adata.layers["X"] = adata.X.copy()
    adata.layers[vkey] = velocity
    adata.obsm["X_umap"] = coordinates.copy()
    adata.obs["celltype"] = pd.Categorical(celltype)
    adata.obs["tech"] = pd.Categorical(tech)
    return adata


def _default_categorical_palette(categories: Sequence[str]) -> list[str]:
    """A fixed, deterministic categorical palette (matplotlib's own "tab20"),
    computed once from the reference celltype labels and applied identically
    to every t's AnnData -- guarantees the same category -> color mapping
    across the t=0/0.5/1 figures, rather than relying on each plot call's
    own (also deterministic, but separately re-derived) default."""
    cmap = plt.get_cmap("tab20")
    return [to_hex(cmap(i % 20)) for i in range(len(categories))]


def assign_shared_celltype_palette(adata, categories: Sequence[str], colors: Sequence[str]) -> None:
    adata.obs["celltype"] = pd.Categorical(adata.obs["celltype"].astype(str), categories=list(categories))
    adata.uns["celltype_colors"] = list(colors)


# ---------------------------------------------------------------------------
# 3. scVelo velocity graph/embedding -- ported from scDiffusionODE's
# hematopoietic_viz/plotting.py::compute_velocity_embeddings, adapted so the
# neighbor graph is built on the existing 2-D representation instead of a
# fresh PCA + UMAP.
# ---------------------------------------------------------------------------


def compute_velocity_embedding_on_umap(
    adata,
    vkey: str = _DEFAULT_VKEY,
    n_neighbors: int = 15,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Builds the neighbor graph scVelo's `velocity_graph` needs directly on
    the shared `Y_ref(t)` representation (`use_rep="X"`, never a fresh PCA
    or UMAP fit -- the displayed coordinates must stay exactly `Y_ref(t)`),
    then calls `scv.tl.velocity_graph` / `scv.tl.velocity_embedding` exactly
    as `compute_velocity_embeddings` does in the scDiffusionODE reference
    implementation, including its scVelo==0.2.5 ragged-array compatibility
    patch. Returns the (unchanged) `X_umap` coordinates, after asserting
    they are in fact unchanged."""
    import os

    import scanpy as sc
    import scvelo as scv

    if "X_umap" not in adata.obsm:
        raise KeyError("adata.obsm['X_umap'] must be set before computing the velocity embedding")
    original_umap = np.asarray(adata.obsm["X_umap"]).copy()
    if "X" not in adata.layers:
        adata.layers["X"] = adata.X.copy()
    if vkey not in adata.layers or adata.layers[vkey].shape != adata.shape:
        raise ValueError(f"velocity layer shape mismatch: {vkey}")

    if "neighbors" not in adata.uns:
        used_neighbors = max(1, min(int(n_neighbors), adata.n_obs - 1))
        sc.pp.neighbors(adata, n_neighbors=used_neighbors, use_rep="X")

    n_jobs = int(n_jobs) if n_jobs is not None else (os.cpu_count() or 1)
    graph_kwargs = dict(vkey=vkey, xkey="X", backend="loky", n_jobs=n_jobs)
    if str(getattr(scv, "__version__", "")) == "0.2.5":
        # scVelo 0.2.5 asks NumPy to make a homogeneous array from its
        # variable-length parallel graph chunks. NumPy >=1.24 correctly
        # rejects that ragged conversion. Preserve the list expected by
        # scVelo's following zip(*res), scoped to this one call only --
        # ported verbatim from the scDiffusionODE reference implementation.
        graph_module = importlib.import_module("scvelo.tools.velocity_graph")
        original_parallelize = graph_module.parallelize

        def list_parallelize(*args, **kwargs):
            kwargs["as_array"] = False
            return original_parallelize(*args, **kwargs)

        graph_module.parallelize = list_parallelize
        try:
            scv.tl.velocity_graph(adata, **graph_kwargs)
        finally:
            graph_module.parallelize = original_parallelize
    else:
        scv.tl.velocity_graph(adata, **graph_kwargs)

    scv.tl.velocity_embedding(adata, basis="umap", vkey=vkey)

    if not np.array_equal(original_umap, np.asarray(adata.obsm["X_umap"])):
        raise RuntimeError(
            "scVelo velocity computation changed adata.obsm['X_umap'] -- the displayed "
            "coordinates must stay exactly Y_ref(t)."
        )
    embedded_key = f"{vkey}_umap"
    if embedded_key not in adata.obsm:
        raise RuntimeError(f"scVelo did not create {embedded_key}")
    if np.asarray(adata.obsm[embedded_key]).shape != (adata.n_obs, 2):
        raise RuntimeError(f"invalid velocity embedding shape for {vkey}")
    return original_umap


# ---------------------------------------------------------------------------
# 4. Plotting -- ported from scDiffusionODE's hematopoietic_viz/plotting.py::
# _velocity_plot / plot_velocity_triplet, including its scVelo==0.2.5
# categorical-legend compatibility patch.
# ---------------------------------------------------------------------------


def _velocity_plot(adata, *, vkey: str, kind: str, title: str, palette, ax) -> None:
    import scvelo as scv

    common = dict(
        basis="umap", vkey=vkey, color="celltype", palette=palette,
        show=False, legend_loc="right margin", size=5, alpha=1, ax=ax, title=title,
    )
    grid_neighbors = max(1, int(adata.n_obs / 50))

    def draw():
        if kind == "stream":
            scv.pl.velocity_embedding_stream(adata, n_neighbors=grid_neighbors, **common)
        elif kind == "arrow":
            scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, **common)
        elif kind == "grid":
            scv.pl.velocity_embedding_grid(adata, n_neighbors=grid_neighbors, **common)
        else:
            raise ValueError(f"unknown velocity plot kind: {kind}")

    if str(getattr(scv, "__version__", "")) == "0.2.5":
        # pandas removed assignment to Categorical.categories, which scVelo
        # 0.2.5 performs only while constructing its legend. Keep the colored
        # scatter, suppress that legacy helper locally, then draw the same
        # category legend with supported Matplotlib primitives -- ported
        # verbatim from the scDiffusionODE reference implementation.
        scatter_module = importlib.import_module("scvelo.plotting.scatter")
        original_set_legend = scatter_module.set_legend
        scatter_module.set_legend = lambda *args, **kwargs: None
        try:
            draw()
        finally:
            scatter_module.set_legend = original_set_legend
        categories = [str(value) for value in adata.obs["celltype"].cat.categories]
        colors = list(adata.uns.get("celltype_colors", []))
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=5, color=color, label=name)
            for name, color in zip(categories, colors)
        ]
        if handles:
            ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        draw()


def plot_velocity_triplet(adata, output_dir, *, vkey: str, title: str, filenames: Sequence[str]) -> list[Path]:
    if len(filenames) != 3:
        raise ValueError("stream/arrow/grid require exactly three filenames")
    palette = list(adata.uns.get("celltype_colors", [])) or None
    created: list[Path] = []
    for kind, filename in zip(("stream", "arrow", "grid"), filenames):
        fig, ax = plt.subplots(figsize=(10, 8))
        _velocity_plot(adata, vkey=vkey, kind=kind, title=f"{title} ({kind})", palette=palette, ax=ax)
        path = Path(output_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        created.append(path)
    return created


# ---------------------------------------------------------------------------
# 5. Top-level orchestration, called from pipeline.py::run_analysis
# ---------------------------------------------------------------------------


def generate_umap_vector_field_figures(
    coordinates_by_t: dict[str, np.ndarray],
    velocity_by_t: dict[str, np.ndarray],
    celltype: np.ndarray,
    tech: np.ndarray,
    figures_dir: str | Path,
    vkey: str = _DEFAULT_VKEY,
    n_neighbors: int = 15,
    n_jobs: int | None = None,
) -> list[Path]:
    """One AnnData per time point (`t000`, `t050`, `t100` -- the same keys
    both dicts must share), each producing a stream/arrow/grid figure
    triplet under `figures_dir`, colored by `celltype` with one shared
    palette across all three time points. Never colors by `tech`."""
    expected_keys = {"t000", "t050", "t100"}
    if set(coordinates_by_t) != expected_keys or set(velocity_by_t) != expected_keys:
        raise ValueError(f"coordinates_by_t/velocity_by_t must both have exactly the keys {expected_keys}")

    categories = sorted(set(np.asarray(celltype).astype(str).tolist()))
    colors = _default_categorical_palette(categories)

    created: list[Path] = []
    for suffix, t_label in [("t000", "t=0.00"), ("t050", "t=0.50"), ("t100", "t=1.00")]:
        adata = build_vector_field_anndata(coordinates_by_t[suffix], velocity_by_t[suffix], celltype, tech, vkey=vkey)
        assign_shared_celltype_palette(adata, categories, colors)
        compute_velocity_embedding_on_umap(adata, vkey=vkey, n_neighbors=n_neighbors, n_jobs=n_jobs)
        created.extend(
            plot_velocity_triplet(
                adata,
                figures_dir,
                vkey=vkey,
                title=f"Total learned UMAP dynamics field ({t_label})",
                filenames=(
                    f"vector_field_{suffix}_stream.png",
                    f"vector_field_{suffix}_arrow.png",
                    f"vector_field_{suffix}_grid.png",
                ),
            )
        )
    return created
