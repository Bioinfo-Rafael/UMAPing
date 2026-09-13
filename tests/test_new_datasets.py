"""Experiment B/C/D dataset loaders: split integrity + reference-only-fitting
invariants, exercised entirely with injected synthetic data (no network, no
torchvision/pyreadr/Figshare) via each loader's `train_data`/`test_data`/`df`/
`adata` injection parameter -- mirroring `test_data_splits.py`'s existing
pattern for `data/pancreas.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from umaping.data.fashion_mnist import prepare_fashion_mnist_dataset
from umaping.data.mnist_oos import prepare_mnist_dataset
from umaping.data.twenty_newsgroups import prepare_twenty_newsgroups_dataset


# ---------------------------------------------------------------------------
# Fashion-MNIST / MNIST (structurally identical loaders)
# ---------------------------------------------------------------------------


def _synthetic_image_split(rng, n, size=28):
    images = rng.uniform(0, 1, size=(n, size, size)).astype(np.float32)
    labels = rng.integers(0, 10, size=n)
    return images, labels


@pytest.mark.parametrize("prepare_fn", [prepare_fashion_mnist_dataset, prepare_mnist_dataset])
def test_image_dataset_reference_only_pca_is_invariant_to_query_data(prepare_fn):
    rng = np.random.default_rng(0)
    train = _synthetic_image_split(rng, 80)
    test_a = _synthetic_image_split(rng, 20)

    prepared_a, pca_a = prepare_fn(raw_dir="unused", pca_dim=5, seed=0, train_data=train, test_data=test_a)

    rng2 = np.random.default_rng(999)
    test_b = _synthetic_image_split(rng2, 20)
    prepared_b, pca_b = prepare_fn(raw_dir="unused", pca_dim=5, seed=0, train_data=train, test_data=test_b)

    np.testing.assert_allclose(pca_a.mean, pca_b.mean)
    np.testing.assert_allclose(pca_a.components, pca_b.components)
    np.testing.assert_allclose(prepared_a.reference_features, prepared_b.reference_features)
    assert prepared_a.reference_features.shape == (80, 5)
    assert prepared_a.query_features.shape == (20, 5)


@pytest.mark.parametrize("prepare_fn", [prepare_fashion_mnist_dataset, prepare_mnist_dataset])
def test_image_dataset_subsampling_is_reproducible_and_shrinks_the_split(prepare_fn):
    rng = np.random.default_rng(1)
    train = _synthetic_image_split(rng, 100)
    test = _synthetic_image_split(rng, 40)

    prepared_1, _ = prepare_fn(
        raw_dir="unused", pca_dim=4, seed=3, train_data=train, test_data=test,
        n_reference_subsample=30, n_query_subsample=10,
    )
    prepared_2, _ = prepare_fn(
        raw_dir="unused", pca_dim=4, seed=3, train_data=train, test_data=test,
        n_reference_subsample=30, n_query_subsample=10,
    )
    assert prepared_1.reference_features.shape[0] == 30
    assert prepared_1.query_features.shape[0] == 10
    np.testing.assert_array_equal(prepared_1.reference_features, prepared_2.reference_features)


def test_fashion_mnist_labels_are_class_key():
    rng = np.random.default_rng(2)
    train = _synthetic_image_split(rng, 20)
    test = _synthetic_image_split(rng, 10)
    prepared, _ = prepare_fashion_mnist_dataset(raw_dir="unused", pca_dim=3, seed=0, train_data=train, test_data=test)
    assert "class" in prepared.reference_labels and "class" in prepared.query_labels


def test_mnist_labels_are_digit_key():
    rng = np.random.default_rng(2)
    train = _synthetic_image_split(rng, 20)
    test = _synthetic_image_split(rng, 10)
    prepared, _ = prepare_mnist_dataset(raw_dir="unused", pca_dim=3, seed=0, train_data=train, test_data=test)
    assert "digit" in prepared.reference_labels and "digit" in prepared.query_labels


# ---------------------------------------------------------------------------
# 20 Newsgroups
# ---------------------------------------------------------------------------

_VOCAB = ["space", "car", "engine", "graphics", "faith", "hockey", "computer", "orbit", "religion", "puck"]


def _synthetic_docs(rng, n):
    docs = [" ".join(rng.choice(_VOCAB, size=15)) for _ in range(n)]
    labels = rng.integers(0, 4, size=n)
    return docs, labels


def test_twenty_newsgroups_reference_only_fitting_is_invariant_to_query_text():
    rng = np.random.default_rng(4)
    train = _synthetic_docs(rng, 60)
    test_a = _synthetic_docs(rng, 20)

    prepared_a, pipeline_a = prepare_twenty_newsgroups_dataset(
        raw_dir="unused", n_components=5, seed=0, train_data=train, test_data=test_a
    )
    rng2 = np.random.default_rng(777)
    test_b = _synthetic_docs(rng2, 20)
    prepared_b, pipeline_b = prepare_twenty_newsgroups_dataset(
        raw_dir="unused", n_components=5, seed=0, train_data=train, test_data=test_b
    )

    assert pipeline_a.vectorizer.vocabulary_ == pipeline_b.vectorizer.vocabulary_
    np.testing.assert_allclose(pipeline_a.svd.components_, pipeline_b.svd.components_)
    np.testing.assert_allclose(prepared_a.reference_features, prepared_b.reference_features)
    assert prepared_a.reference_features.shape[0] == 60
    assert prepared_a.query_features.shape[0] == 20


def test_twenty_newsgroups_labels_are_newsgroup_key():
    rng = np.random.default_rng(5)
    train = _synthetic_docs(rng, 30)
    test = _synthetic_docs(rng, 10)
    prepared, _ = prepare_twenty_newsgroups_dataset(raw_dir="unused", n_components=4, seed=0, train_data=train, test_data=test)
    assert "newsgroup" in prepared.reference_labels and "newsgroup" in prepared.query_labels


def test_twenty_newsgroups_raises_on_empty_split_after_subsampling():
    rng = np.random.default_rng(6)
    train = _synthetic_docs(rng, 10)
    test = _synthetic_docs(rng, 10)
    with pytest.raises(ValueError):
        prepare_twenty_newsgroups_dataset(
            raw_dir="unused", n_components=2, seed=0, train_data=train, test_data=test, n_reference_subsample=0
        )


# ---------------------------------------------------------------------------
# Hong ED
# ---------------------------------------------------------------------------


def _synthetic_hong_ed_df(n=200, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_patients = n // 2
    patient_id = rng.integers(0, n_patients, size=n)  # some patients contribute >1 visit
    return pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": rng.normal(50, 15, size=n),
            "heart_rate": rng.normal(80, 10, size=n),
            "triage_category": rng.choice(["A", "B", "C"], size=n),
            "disposition": rng.integers(0, 2, size=n),
        }
    )


def test_hong_ed_never_uses_outcome_or_id_as_a_feature():
    from umaping.data.hong_ed import prepare_hong_ed_dataset

    df = _synthetic_hong_ed_df()
    prepared, meta = prepare_hong_ed_dataset(raw_dir="unused", seed=0, df=df)
    assert meta.outcome_column == "disposition"
    assert meta.id_column == "patient_id"
    # 1 numeric (age) + 1 numeric (heart_rate) + 3 one-hot (triage_category) = 5.
    assert prepared.reference_features.shape[1] == 5
    assert prepared.input_dim == 5


def test_hong_ed_group_split_keeps_every_patient_on_one_side_only():
    from umaping.data.hong_ed import prepare_hong_ed_dataset

    df = _synthetic_hong_ed_df(n=300, seed=1)
    prepared, meta = prepare_hong_ed_dataset(raw_dir="unused", seed=0, df=df)
    assert meta.split_mode == "group"

    # Reconstruct which patient each row belonged to isn't directly exposed on
    # PreparedDataset, so re-derive the split the same way and check directly.
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    ref_idx, query_idx = next(splitter.split(df, groups=df["patient_id"]))
    ref_patients = set(df.iloc[ref_idx]["patient_id"].tolist())
    query_patients = set(df.iloc[query_idx]["patient_id"].tolist())
    assert ref_patients.isdisjoint(query_patients)


def test_hong_ed_falls_back_to_random_split_when_no_id_column_present():
    from umaping.data.hong_ed import prepare_hong_ed_dataset

    df = _synthetic_hong_ed_df().drop(columns=["patient_id"])
    prepared, meta = prepare_hong_ed_dataset(raw_dir="unused", seed=0, df=df)
    assert meta.id_column is None
    assert meta.split_mode == "stratified_random_fallback"


def test_hong_ed_query_only_category_does_not_expand_reference_feature_space():
    from umaping.data.hong_ed import prepare_hong_ed_dataset

    df = _synthetic_hong_ed_df(n=200, seed=2)
    # Force a category ("D") that appears only among a few rows likely to land in query.
    df.loc[df.index[:5], "triage_category"] = "D"
    prepared, _ = prepare_hong_ed_dataset(raw_dir="unused", seed=0, df=df)
    # One-hot width is fixed by the *reference* split's observed categories.
    assert prepared.reference_features.shape[1] == prepared.query_features.shape[1]


def test_hong_ed_subset_mode_reduces_pool_size_reproducibly():
    from umaping.data.hong_ed import prepare_hong_ed_dataset

    df = _synthetic_hong_ed_df(n=500, seed=3)
    prepared_1, _ = prepare_hong_ed_dataset(raw_dir="unused", seed=0, df=df, subset_mode="small", small_n=50)
    prepared_2, _ = prepare_hong_ed_dataset(raw_dir="unused", seed=0, df=df, subset_mode="small", small_n=50)
    total_1 = prepared_1.reference_features.shape[0] + prepared_1.query_features.shape[0]
    assert total_1 == 50
    np.testing.assert_array_equal(prepared_1.reference_features, prepared_2.reference_features)


def test_hong_ed_raises_when_no_outcome_column_detected():
    from umaping.data.hong_ed import prepare_hong_ed_dataset

    df = _synthetic_hong_ed_df().drop(columns=["disposition"])
    with pytest.raises(ValueError, match="outcome"):
        prepare_hong_ed_dataset(raw_dir="unused", seed=0, df=df)


# ---------------------------------------------------------------------------
# Organoid / Embryoid body (share data/_scrna_common.py)
# ---------------------------------------------------------------------------


def _synthetic_continuous_scrna_adata(n=120, n_genes=60, n_groups=4, seed=0):
    import anndata as ad

    rng = np.random.default_rng(seed)
    groups = rng.choice([f"state_{i}" for i in range(n_groups)], size=n)
    counts = rng.poisson(lam=4.0, size=(n, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=counts.copy())
    adata.layers["counts"] = counts.copy()
    adata.obs["time"] = groups
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    return adata


def test_prepare_continuous_scrna_dataset_d_easy_keeps_every_group_on_both_sides():
    from umaping.data._scrna_common import prepare_continuous_scrna_dataset

    adata = _synthetic_continuous_scrna_adata(seed=0)
    prepared = prepare_continuous_scrna_dataset(
        adata, grouping_column="time", query_groups=None, n_hvg=20, n_pcs=5, seed=0
    )
    ref_groups = set(prepared.reference_labels["time"].tolist())
    query_groups = set(prepared.query_labels["time"].tolist())
    all_groups = set(adata.obs["time"].tolist())
    assert ref_groups == all_groups
    assert query_groups == all_groups


def test_prepare_continuous_scrna_dataset_d_hard_holds_out_the_requested_group_entirely():
    from umaping.data._scrna_common import prepare_continuous_scrna_dataset

    adata = _synthetic_continuous_scrna_adata(seed=1)
    held_out = "state_0"
    prepared = prepare_continuous_scrna_dataset(
        adata, grouping_column="time", query_groups=(held_out,), n_hvg=20, n_pcs=5, seed=0
    )
    assert held_out not in set(prepared.reference_labels["time"].tolist())
    assert set(prepared.query_labels["time"].tolist()) == {held_out}


def test_prepare_continuous_scrna_dataset_hvg_pca_invariant_to_query_perturbation():
    from umaping.data._scrna_common import prepare_continuous_scrna_dataset

    adata_a = _synthetic_continuous_scrna_adata(seed=2)
    prepared_a = prepare_continuous_scrna_dataset(
        adata_a.copy(), grouping_column="time", query_groups=("state_0",), n_hvg=15, n_pcs=4, seed=0
    )

    adata_b = adata_a.copy()
    query_mask = (adata_b.obs["time"] == "state_0").to_numpy()
    rng = np.random.default_rng(555)
    new_counts = rng.poisson(lam=40.0, size=(query_mask.sum(), adata_b.n_vars)).astype(np.float32)
    adata_b.layers["counts"][query_mask] = new_counts
    prepared_b = prepare_continuous_scrna_dataset(
        adata_b, grouping_column="time", query_groups=("state_0",), n_hvg=15, n_pcs=4, seed=0
    )

    np.testing.assert_allclose(prepared_a.reference_features, prepared_b.reference_features, atol=1e-4)


def test_detect_grouping_column_matches_by_substring_and_returns_none_when_absent():
    from umaping.data._scrna_common import detect_grouping_column

    assert detect_grouping_column(["cell_id", "sample_labels", "gene_count"], ("time", "sample_labels")) == "sample_labels"
    assert detect_grouping_column(["a", "b"], ("time", "day")) is None


def test_organoid_and_embryoid_body_prepare_functions_delegate_to_shared_helper_with_injected_adata():
    from umaping.data.embryoid_body import prepare_embryoid_body_dataset
    from umaping.data.organoid import prepare_organoid_dataset

    adata = _synthetic_continuous_scrna_adata(seed=3)
    prepared_o, col_o = prepare_organoid_dataset(raw_dir="unused", n_hvg=15, n_pcs=4, seed=0, adata=adata.copy())
    prepared_e, col_e = prepare_embryoid_body_dataset(raw_dir="unused", n_hvg=15, n_pcs=4, seed=0, adata=adata.copy())
    assert col_o == "time" and col_e == "time"
    assert prepared_o.reference_features.shape[1] == 4
    assert prepared_e.reference_features.shape[1] == 4


def test_organoid_raises_when_no_grouping_column_is_detectable():
    import anndata as ad

    from umaping.data.organoid import prepare_organoid_dataset

    rng = np.random.default_rng(9)
    adata = ad.AnnData(X=rng.poisson(lam=3.0, size=(30, 10)).astype(np.float32))
    adata.layers["counts"] = adata.X.copy()
    adata.var_names = [f"g{i}" for i in range(10)]
    # No column named anything like time/state/well/etc.
    adata.obs["unrelated_metadata"] = rng.integers(0, 3, size=30)

    with pytest.raises(ValueError, match="grouping column"):
        prepare_organoid_dataset(raw_dir="unused", adata=adata)
