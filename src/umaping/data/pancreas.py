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


def prepare_pancreas_dataset(
    raw_dir: str | Path,
    n_hvg: int,
    n_pcs: int,
    seed: int = 0,
    query_tech: tuple[str, ...] = DEFAULT_QUERY_TECH,
    adata: ad.AnnData | None = None,
) -> tuple[PreparedDataset, FittedPCA]:
    """`adata`: inject an already-loaded AnnData (used by tests to check the
    reference-only-fitting invariant without downloading real data); when
    omitted, the real dataset is downloaded/loaded from `raw_dir` as usual."""
    if adata is None:
        adata = load_pancreas_anndata(raw_dir)
    else:
        adata = adata.copy()
    tech = adata.obs["tech"].astype(str).to_numpy()
    celltype = adata.obs["celltype"].astype(str).to_numpy()

    reference_mask, query_mask = split_pancreas(tech, query_tech=query_tech)
    assert_no_overlap(np.where(reference_mask)[0], np.where(query_mask)[0])

    # Steps 1-3: raw counts -> normalize_total(1e4) -> log1p. Both are purely
    # per-cell (row-wise) transforms, so applying them before the split (once,
    # to every cell independently) carries no query-into-reference leakage.
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Step 4: HVG selection fit on the reference split only.
    adata_ref = adata[reference_mask].copy()
    sc.pp.highly_variable_genes(adata_ref, n_top_genes=n_hvg, batch_key="tech", flavor="seurat")
    hvg_genes = adata_ref.var_names[adata_ref.var["highly_variable"]].to_numpy()
    if len(hvg_genes) == 0:
        raise ValueError("Highly-variable-gene selection on the reference set produced zero genes.")

    # Step 5: apply the same reference-selected gene set to the query split.
    x_ref_hvg = _densify(adata_ref[:, hvg_genes].X)
    x_query_hvg = _densify(adata[query_mask][:, hvg_genes].X)
    check_finite(x_ref_hvg, "pancreas reference HVG matrix")
    check_finite(x_query_hvg, "pancreas query HVG matrix")

    # Steps 6-7: PCA fit on the reference split only, query merely transformed.
    pca = fit_pca(x_ref_hvg, n_components=n_pcs, seed=seed)
    x_ref = pca.transform(x_ref_hvg)
    x_query = pca.transform(x_query_hvg)

    prepared = PreparedDataset(
        reference_features=x_ref,
        query_features=x_query,
        reference_labels={"celltype": celltype[reference_mask], "tech": tech[reference_mask]},
        query_labels={"celltype": celltype[query_mask], "tech": tech[query_mask]},
        input_dim=n_pcs,
    )
    return prepared, pca
