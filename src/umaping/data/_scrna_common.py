"""Shared scRNA-seq loading/preprocessing helpers for Experiment D
(`organoid.py`, `embryoid_body.py`). Both reuse the exact reference-only
HVG + PCA pipeline `data/pancreas.py` already implements (see that module's
docstring for the full reasoning), generalized to an arbitrary caller-given
"state"/"time" grouping column instead of pancreas's fixed `tech`/`celltype`
columns -- kept here, not duplicated in each dataset module, per "do not
duplicate code that already exists."
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from umaping.data.preprocessing import PreparedDataset, assert_no_overlap, check_finite, fit_pca

logger = logging.getLogger(__name__)


def load_anndata_any_format(path: Path) -> ad.AnnData:
    """Loads a single-cell file whose exact format was not confirmed by an
    actual download during implementation (see `organoid.py`/
    `embryoid_body.py`'s module docstrings for why) -- dispatches on file
    extension across the formats scanpy/anndata read natively. Raises with a
    clear message (rather than guessing) for anything else, so a real run
    that hits an unrecognized format fails loudly instead of silently
    misreading the file."""
    suffix = "".join(path.suffixes[-2:]) if path.suffixes[-2:][:1] == [".csv"] or path.suffixes[-2:][:1] == [".mtx"] else path.suffix
    suffix = suffix.lower()
    if suffix == ".h5ad":
        return ad.read_h5ad(path)
    if suffix == ".loom":
        return sc.read_loom(path)
    if suffix in (".csv", ".csv.gz"):
        return sc.read_csv(str(path)).T
    if suffix in (".mtx", ".mtx.gz"):
        return sc.read_mtx(str(path))
    raise ValueError(
        f"Don't know how to load '{path.name}' (unrecognized extension '{suffix}'). Inspect the actual "
        "downloaded file's format and extend `data/_scrna_common.py::load_anndata_any_format` for it."
    )


_MIN_SUBSTRING_CANDIDATE_LENGTH = 3  # see data/hong_ed.py::_detect_column's identical guard:
# a candidate shorter than this is only ever matched exactly, never as a
# substring (a short candidate as a substring risks spurious matches).


def detect_grouping_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        if len(cand) < _MIN_SUBSTRING_CANDIDATE_LENGTH:
            continue
        for col_lower, col in lower.items():
            if cand in col_lower:
                return col
    return None


def prepare_continuous_scrna_dataset(
    adata: ad.AnnData,
    grouping_column: str,
    query_groups: tuple[str, ...] | None,
    n_hvg: int,
    n_pcs: int,
    seed: int,
    label_columns: tuple[str, ...] = (),
    artifacts: dict | None = None,
) -> PreparedDataset:
    """Reference-only HVG + PCA, splitting on `grouping_column` (a state/time
    metadata column): if `query_groups` is given, those groups are held out
    entirely (the "D-hard" regime -- a genuinely unseen state/time region);
    if `query_groups` is None, an approximately-stratified random ~20% split
    is used instead (the "D-easy" regime, where every state remains
    represented in the reference)."""
    groups = adata.obs[grouping_column].astype(str).to_numpy()

    if query_groups is not None:
        query_mask = np.isin(groups, list(query_groups))
    else:
        rng = np.random.default_rng(seed)
        query_mask = np.zeros(adata.n_obs, dtype=bool)
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            n_query_g = max(1, int(round(0.2 * len(idx)))) if len(idx) > 1 else 0
            if n_query_g:
                query_mask[rng.choice(idx, size=n_query_g, replace=False)] = True
    reference_mask = ~query_mask
    if not np.any(reference_mask) or not np.any(query_mask):
        raise ValueError(
            f"Reference/query split is degenerate for grouping_column='{grouping_column}', "
            f"query_groups={query_groups} (observed groups: {sorted(set(groups.tolist()))})."
        )
    assert_no_overlap(np.where(reference_mask)[0], np.where(query_mask)[0])

    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    # Per-cell transforms (data/pancreas.py's own reasoning): safe to apply
    # before the split since they use no cross-cell statistics.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata_ref = adata[reference_mask].copy()
    n_hvg_eff = min(n_hvg, adata_ref.n_vars - 1)
    sc.pp.highly_variable_genes(adata_ref, n_top_genes=n_hvg_eff, flavor="seurat")
    hvg_genes = adata_ref.var_names[adata_ref.var["highly_variable"]].to_numpy()
    if len(hvg_genes) == 0:
        raise ValueError("Highly-variable-gene selection on the reference set produced zero genes.")

    def _densify(x) -> np.ndarray:
        return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)

    x_ref_hvg = _densify(adata_ref[:, hvg_genes].X)
    x_query_hvg = _densify(adata[query_mask][:, hvg_genes].X)
    check_finite(x_ref_hvg, "reference HVG matrix")
    check_finite(x_query_hvg, "query HVG matrix")

    n_pcs_eff = max(1, min(n_pcs, x_ref_hvg.shape[0] - 1, x_ref_hvg.shape[1] - 1))
    pca = fit_pca(x_ref_hvg, n_components=n_pcs_eff, seed=seed)
    x_ref = pca.transform(x_ref_hvg)
    x_query = pca.transform(x_query_hvg)
    if artifacts is not None:
        artifacts.update(hvg_genes=np.asarray(hvg_genes, dtype=str), pca_mean=pca.mean,
                         pca_components=pca.components, reference_indices=np.flatnonzero(reference_mask),
                         query_indices=np.flatnonzero(query_mask), normalization_target=1e4)

    reference_labels = {grouping_column: groups[reference_mask]}
    query_labels = {grouping_column: groups[query_mask]}
    for col in label_columns:
        if col in adata.obs.columns:
            vals = adata.obs[col].astype(str).to_numpy()
            reference_labels[col] = vals[reference_mask]
            query_labels[col] = vals[query_mask]
        else:
            logger.warning("Requested label column '%s' not found in obs; skipping.", col)

    return PreparedDataset(
        reference_features=x_ref,
        query_features=x_query,
        reference_labels=reference_labels,
        query_labels=query_labels,
        input_dim=n_pcs_eff,
    )
