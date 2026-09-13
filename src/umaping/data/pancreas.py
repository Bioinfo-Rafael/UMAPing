"""Pancreas scRNA-seq reference-mapping dataset (scvi-tools / scIB tutorial
convention).

Canonical file, verified by fetching the live scvi-tools scArches
reference-mapping tutorial notebook and the scIB figshare deposit (see
RUNTIME_CHECKS.md): ~16382 cells x ~19093 genes, `obs['tech']` in
`{inDrop1..4, smartseq2, smarter, celseq, celseq2, fluidigmc1}`,
`obs['celltype']`, raw counts in `layers['counts']`.

Primary source is the current scvi-tools tutorial's own re-host
(`exampledata.scverse.org`); falls back to the original scIB figshare
deposit (`human_pancreas_norm_complexBatch.h5ad`, the same underlying data)
if the primary is unreachable. NOTE (verified): `exampledata.scverse.org`
returns 403 to a bare `Python-urllib` User-Agent -- `requests`'s default UA
(used throughout, via utils/io.download_file) works fine.

All HVG selection and PCA fitting happens on the reference split only (see
`prepare_pancreas_dataset`); the query split is only ever *transformed*
through those already-fitted objects.

Two preprocessing paths, selected by `batch_correction` (default False here;
the checked-in `configs/pancreas.yaml` sets it to True -- see
`pipeline.py::stage_preprocess` for why a *missing* key in an old saved
run's config.yaml must still resolve to False):

- `batch_correction=False` (legacy path, `_prepare_pancreas_legacy_pca`):
  the original reference-only PCA preprocessing, numerically unchanged.
- `batch_correction=True` (`_prepare_pancreas_scvi_scarches`): a scVI
  reference model (batch_key="tech") trained on reference cells only, then
  scArches-style query mapping (the current scvi-tools `ArchesMixin`
  implementation: `SCVI.prepare_query_anndata` + `SCVI.load_query_data`) for
  the held-out query technologies. **Important, stated explicitly rather
  than silently implied**: scArches query adaptation trains the query model
  against the query batch itself (unsupervised domain adaptation of a
  frozen reference architecture) -- this is a materially different, and
  weaker, leakage guarantee than the legacy path's strict fit-on-reference/
  transform-only-on-query PCA. See README.md's "Pancreas batch correction"
  section for the same note in user-facing form.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from umaping.data.preprocessing import (
    FittedPCA,
    PreparedDataset,
    assert_no_overlap,
    check_finite,
    fit_pca,
)
from umaping.utils.io import download_file, ensure_dir

logger = logging.getLogger(__name__)

_PRIMARY_URL = "https://exampledata.scverse.org/scvi-tools/pancreas.h5ad"
_FALLBACK_URL = "https://ndownloader.figshare.com/files/24539828"
_EXPECTED_N_OBS = 16382
_EXPECTED_N_VARS = 19093
_MIN_EXPECTED_BYTES = 50_000_000
DEFAULT_QUERY_TECH = ("smartseq2", "celseq2")


def download_pancreas(raw_dir: str | Path) -> Path:
    """Idempotent download with a figshare fallback if the primary
    scvi-tools-hosted URL is unreachable."""
    raw_dir = ensure_dir(raw_dir)
    dest = raw_dir / "pancreas.h5ad"
    if dest.exists() and dest.stat().st_size >= _MIN_EXPECTED_BYTES:
        logger.info("%s already present; skipping download.", dest)
        return dest
    try:
        return download_file(_PRIMARY_URL, dest, min_expected_bytes=_MIN_EXPECTED_BYTES)
    except Exception:
        logger.warning(
            "Primary pancreas URL (%s) failed; falling back to the scIB figshare deposit.",
            _PRIMARY_URL,
            exc_info=True,
        )
        return download_file(_FALLBACK_URL, dest, min_expected_bytes=_MIN_EXPECTED_BYTES)


def load_pancreas_anndata(raw_dir: str | Path) -> ad.AnnData:
    path = download_pancreas(raw_dir)
    adata = ad.read_h5ad(path)

    missing_obs = {"tech", "celltype"} - set(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"pancreas.h5ad is missing expected obs columns: {missing_obs}")
    if "counts" not in adata.layers:
        raise ValueError("pancreas.h5ad is missing the expected 'counts' layer.")
    if adata.n_obs != _EXPECTED_N_OBS or adata.n_vars != _EXPECTED_N_VARS:
        logger.warning(
            "pancreas.h5ad shape (%d, %d) does not match the expected (%d, %d); proceeding anyway.",
            adata.n_obs,
            adata.n_vars,
            _EXPECTED_N_OBS,
            _EXPECTED_N_VARS,
        )
    return adata


def split_pancreas(
    tech: np.ndarray, query_tech: tuple[str, ...] = DEFAULT_QUERY_TECH
) -> tuple[np.ndarray, np.ndarray]:
    """Reference-mapping-style split: `query_tech` technologies are the
    held-out query set, every other technology is the fixed reference X."""
    query_mask = np.isin(tech, list(query_tech))
    reference_mask = ~query_mask
    if not np.any(reference_mask) or not np.any(query_mask):
        raise ValueError(
            f"Pancreas reference/query split is degenerate for query_tech={query_tech} "
            f"(observed tech values: {sorted(set(tech.tolist()))})."
        )
    return reference_mask, query_mask


def _densify(x) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def _reference_only_hvg_selection(adata: ad.AnnData, reference_mask: np.ndarray, n_hvg: int) -> np.ndarray:
    """Highly-variable-gene selection fit on the reference split only,
    shared verbatim by both preprocessing paths below (the legacy PCA path
    and the scVI/scArches path both need exactly this reference-only gene
    set, applied identically to reference and query -- this helper exists
    so that invariant is expressed once, not duplicated). `adata.X` must
    already hold the normalize_total+log1p representation `sc.pp.
    highly_variable_genes(..., flavor="seurat")` expects."""
    adata_ref = adata[reference_mask].copy()
    sc.pp.highly_variable_genes(adata_ref, n_top_genes=n_hvg, batch_key="tech", flavor="seurat")
    hvg_genes = adata_ref.var_names[adata_ref.var["highly_variable"]].to_numpy()
    if len(hvg_genes) == 0:
        raise ValueError("Highly-variable-gene selection on the reference set produced zero genes.")
    return hvg_genes


def _prepare_pancreas_legacy_pca(
    adata: ad.AnnData,
    reference_mask: np.ndarray,
    query_mask: np.ndarray,
    n_hvg: int,
    n_pcs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, FittedPCA]:
    """`batch_correction=False`: the original preprocessing path, numerically
    unchanged from this module's pre-batch-correction implementation."""
    # Steps 1-3: raw counts -> normalize_total(1e4) -> log1p. Both are purely
    # per-cell (row-wise) transforms, so applying them before the split (once,
    # to every cell independently) carries no query-into-reference leakage.
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Step 4: HVG selection fit on the reference split only.
    hvg_genes = _reference_only_hvg_selection(adata, reference_mask, n_hvg)

    # Step 5: apply the same reference-selected gene set to the query split.
    x_ref_hvg = _densify(adata[reference_mask][:, hvg_genes].X)
    x_query_hvg = _densify(adata[query_mask][:, hvg_genes].X)
    check_finite(x_ref_hvg, "pancreas reference HVG matrix")
    check_finite(x_query_hvg, "pancreas query HVG matrix")

    # Steps 6-7: PCA fit on the reference split only, query merely transformed.
    pca = fit_pca(x_ref_hvg, n_components=n_pcs, seed=seed)
    x_ref = pca.transform(x_ref_hvg)
    x_query = pca.transform(x_query_hvg)
    return x_ref, x_query, pca


def _prepare_pancreas_scvi_scarches(
    adata: ad.AnnData,
    reference_mask: np.ndarray,
    query_mask: np.ndarray,
    n_hvg: int,
    scvi_n_latent: int,
    scvi_max_epochs: int,
    scarches_max_epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`batch_correction=True`: a scVI reference model (`batch_key="tech"`)
    trained on reference cells only, then the current scvi-tools scArches
    implementation (`scvi.model.SCVI.prepare_query_anndata` +
    `scvi.model.SCVI.load_query_data`, `ArchesMixin` -- verified directly
    against the installed scvi-tools source, not guessed) adapts the frozen
    reference model to the held-out query technologies. The query latent
    space comes from that adapted (query) model, so it stays aligned with
    the reference latent space by construction -- no manual registry/
    category surgery, and no independent second model trained on the query
    cells alone.

    HVG selection reuses `_reference_only_hvg_selection` on a *separate*,
    normalize_total+log1p'd copy of `adata` (mirroring the legacy path's own
    steps 1-3 for that purpose only); the actual scVI models are trained on
    **raw counts** (`layer="counts"`), scVI's own documented convention,
    restricted to that same reference-selected gene set for both splits.
    """
    import scvi

    scvi.settings.seed = seed

    adata_for_hvg = adata.copy()
    adata_for_hvg.X = adata_for_hvg.layers["counts"].copy()
    sc.pp.normalize_total(adata_for_hvg, target_sum=1e4)
    sc.pp.log1p(adata_for_hvg)
    hvg_genes = _reference_only_hvg_selection(adata_for_hvg, reference_mask, n_hvg)

    # Raw-count AnnData, restricted to the reference-selected gene set, used
    # for both the reference and query scVI models.
    adata_hvg = adata[:, hvg_genes].copy()
    adata_hvg.X = adata_hvg.layers["counts"].copy()

    adata_ref = adata_hvg[reference_mask].copy()
    adata_query = adata_hvg[query_mask].copy()

    logger.info(
        "Training scVI reference model (batch_key='tech', n_latent=%d, max_epochs=%d) on "
        "%d reference cells...",
        scvi_n_latent,
        scvi_max_epochs,
        adata_ref.n_obs,
    )
    scvi.model.SCVI.setup_anndata(adata_ref, layer="counts", batch_key="tech")
    reference_model = scvi.model.SCVI(adata_ref, n_latent=scvi_n_latent)
    reference_model.train(max_epochs=scvi_max_epochs)

    logger.info(
        "Adapting the frozen reference model to %d query cells (scArches, max_epochs=%d)...",
        adata_query.n_obs,
        scarches_max_epochs,
    )
    query_adata_prepared = scvi.model.SCVI.prepare_query_anndata(adata_query, reference_model)
    query_model = scvi.model.SCVI.load_query_data(query_adata_prepared, reference_model)
    query_model.train(max_epochs=scarches_max_epochs)

    x_ref = np.asarray(reference_model.get_latent_representation(), dtype=np.float32)
    x_query = np.asarray(query_model.get_latent_representation(), dtype=np.float32)
    return x_ref, x_query


def prepare_pancreas_dataset(
    raw_dir: str | Path,
    n_hvg: int,
    n_pcs: int,
    seed: int = 0,
    query_tech: tuple[str, ...] = DEFAULT_QUERY_TECH,
    adata: ad.AnnData | None = None,
    batch_correction: bool = False,
    scvi_n_latent: int = 50,
    scvi_max_epochs: int = 400,
    scarches_max_epochs: int = 200,
) -> tuple[PreparedDataset, FittedPCA | None]:
    """`adata`: inject an already-loaded AnnData (used by tests to check the
    reference-only-fitting invariant without downloading real data); when
    omitted, the real dataset is downloaded/loaded from `raw_dir` as usual.

    `batch_correction`: selects the preprocessing path (see this module's
    docstring and `_prepare_pancreas_legacy_pca`/`_prepare_pancreas_scvi_scarches`).
    Defaults to False *here* -- the checked-in `configs/pancreas.yaml` sets
    it to True, but this function's own default, and `pipeline.py::
    stage_preprocess`'s handling of a config that omits the key entirely,
    must both stay False so old saved runs' `config.yaml` (predating this
    parameter) keep reproducing their original behavior.

    Returns `(prepared, pca)`, with `pca` `None` under `batch_correction=True`
    (there is no PCA object in that path -- do not treat this as an error).
    """
    if adata is None:
        adata = load_pancreas_anndata(raw_dir)
    else:
        adata = adata.copy()
    tech = adata.obs["tech"].astype(str).to_numpy()
    celltype = adata.obs["celltype"].astype(str).to_numpy()

    reference_mask, query_mask = split_pancreas(tech, query_tech=query_tech)
    assert_no_overlap(np.where(reference_mask)[0], np.where(query_mask)[0])

    if batch_correction:
        logger.info(
            "Pancreas preprocessing mode: batch_correction=True (scVI/scArches). This is the "
            "batch-corrected analysis mode; unlike batch_correction=False (the original strict "
            "reference-fit/query-transform PCA preprocessing), scArches query mapping performs "
            "unsupervised query-domain adaptation of the frozen reference model -- see README.md."
        )
        x_ref, x_query = _prepare_pancreas_scvi_scarches(
            adata,
            reference_mask,
            query_mask,
            n_hvg=n_hvg,
            scvi_n_latent=scvi_n_latent,
            scvi_max_epochs=scvi_max_epochs,
            scarches_max_epochs=scarches_max_epochs,
            seed=seed,
        )
        check_finite(x_ref, "pancreas reference scVI latent representation")
        check_finite(x_query, "pancreas query scVI latent representation")
        pca = None
        input_dim = x_ref.shape[1]
    else:
        logger.info("Pancreas preprocessing mode: batch_correction=False (legacy reference-only PCA).")
        x_ref, x_query, pca = _prepare_pancreas_legacy_pca(
            adata, reference_mask, query_mask, n_hvg=n_hvg, n_pcs=n_pcs, seed=seed
        )
        input_dim = n_pcs

    prepared = PreparedDataset(
        reference_features=x_ref,
        query_features=x_query,
        reference_labels={"celltype": celltype[reference_mask], "tech": tech[reference_mask]},
        query_labels={"celltype": celltype[query_mask], "tech": tech[query_mask]},
        input_dim=input_dim,
    )
    return prepared, pca
