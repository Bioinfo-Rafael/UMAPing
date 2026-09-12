"""Tests #8 and #9: dataset reference/query splits must never leak.

* COIL: the every-4th-pose holdout must not overlap reference/query, and
  every object class must appear on both sides.
* Pancreas: HVG selection and PCA fitting must depend only on the reference
  split -- changing the query cells' data must not change them at all.

Written to be run remotely (see RUNTIME_CHECKS.md); not executed here.
"""

from __future__ import annotations

import numpy as np

from umaping.data.coil import compute_pose_ranks, split_coil
from umaping.data.pancreas import prepare_pancreas_dataset


def test_coil_split_has_no_overlap_and_every_object_on_both_sides():
    entries = [(f"dummy_{obj}_{pose}", obj, pose) for obj in range(1, 4) for pose in range(72)]
    pose_rank = compute_pose_ranks(entries)
    object_id = np.array([e[1] for e in entries])

    reference_mask, query_mask = split_coil(pose_rank, holdout_period=4)

    assert not np.any(reference_mask & query_mask)
    assert np.all(reference_mask | query_mask)
    assert abs(reference_mask.mean() - 0.75) < 1e-9
    assert abs(query_mask.mean() - 0.25) < 1e-9

    for obj in np.unique(object_id):
        obj_mask = object_id == obj
        assert np.any(reference_mask & obj_mask), f"object {obj} missing from reference"
        assert np.any(query_mask & obj_mask), f"object {obj} missing from query"


def test_coil_pose_rank_is_a_permutation_of_0_to_71_per_object():
    """Sanity check on the rank helper itself: COIL-20's suffix is a raw
    sequence index while COIL-100's is literal degrees, so the rank -- not
    the raw parsed value -- is what the split keys off."""
    entries = [(f"dummy_{pose}", 1, pose * 5) for pose in range(72)]  # COIL-100-style degree labels
    ranks = compute_pose_ranks(entries)
    assert sorted(ranks.tolist()) == list(range(72))


def _make_synthetic_pancreas_adata(rng: np.random.Generator, n_ref: int, n_query: int, n_genes: int, query_lam: float):
    import anndata as ad

    tech = np.array(["batchA"] * (n_ref // 2) + ["batchB"] * (n_ref - n_ref // 2) + ["smartseq2"] * (n_query // 2) + ["celseq2"] * (n_query - n_query // 2))
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


def test_pancreas_hvg_and_pca_are_invariant_to_query_data():
    """Changing the query cells' expression data must not change the
    reference-fitted HVG selection or PCA transform at all."""
    rng = np.random.default_rng(0)
    n_ref, n_query, n_genes = 60, 40, 100

    adata_a = _make_synthetic_pancreas_adata(rng, n_ref, n_query, n_genes, query_lam=5.0)
    prepared_a, pca_a = prepare_pancreas_dataset(raw_dir="unused", n_hvg=20, n_pcs=5, seed=0, adata=adata_a)

    # Wildly different query-side data (10x the count rate); reference cells
    # (first n_ref rows) are untouched.
    adata_b = adata_a.copy()
    rng2 = np.random.default_rng(123)
    new_query_counts = rng2.poisson(lam=50.0, size=(n_query, n_genes)).astype(np.float32)
    adata_b.X[n_ref:, :] = new_query_counts
    adata_b.layers["counts"][n_ref:, :] = new_query_counts

    prepared_b, pca_b = prepare_pancreas_dataset(raw_dir="unused", n_hvg=20, n_pcs=5, seed=0, adata=adata_b)

    np.testing.assert_allclose(pca_a.mean, pca_b.mean)
    np.testing.assert_allclose(pca_a.components, pca_b.components)
    np.testing.assert_allclose(prepared_a.reference_features, prepared_b.reference_features)
    # The reference-side labels shouldn't have moved either.
    np.testing.assert_array_equal(prepared_a.reference_labels["celltype"], prepared_b.reference_labels["celltype"])


def test_pancreas_split_has_no_overlap():
    rng = np.random.default_rng(1)
    adata = _make_synthetic_pancreas_adata(rng, n_ref=60, n_query=40, n_genes=50, query_lam=5.0)
    prepared, _pca = prepare_pancreas_dataset(raw_dir="unused", n_hvg=15, n_pcs=5, seed=0, adata=adata)

    assert prepared.reference_features.shape[0] == 60
    assert prepared.query_features.shape[0] == 40
    assert set(prepared.reference_labels["tech"].tolist()).isdisjoint({"smartseq2", "celseq2"})
    assert set(prepared.query_labels["tech"].tolist()) <= {"smartseq2", "celseq2"}
