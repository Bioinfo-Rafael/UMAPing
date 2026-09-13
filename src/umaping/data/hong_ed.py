"""Hong et al. Emergency Department admission-prediction dataset (Experiment C).

Public repository: https://github.com/yaleemmlc/admissionprediction
File: ``Results/5v_cleandf.RData``

**Caveat, stated explicitly because this could not be verified by an actual
download during implementation (see the "do not download real data locally"
constraint this module was written under):** the exact branch name
(``main`` vs. ``master``) and the RData file's precise column schema were not
confirmed live. `download_hong_ed` tries both common branch names via
`raw.githubusercontent.com`; `prepare_hong_ed_dataset` auto-detects a likely
outcome column and a likely patient-identifier column by name-matching
common candidates (rather than hardcoding column names this implementation
cannot verify), and *fails loudly* if it cannot find a usable outcome
column, rather than silently guessing wrong. Confirm both on the first real
run (see experiments/README.md's reproducibility section) and correct the
candidate-name lists below if needed -- this is analogous to
`RUNTIME_CHECKS.md`'s "verified sources, not yet exercised" entries for the
three original datasets.

Reading ``.RData`` requires the optional `pyreadr` dependency (see
pyproject.toml's `experiments` extra), imported lazily.

Never uses the outcome column or any detected patient-identifier column as
an input feature. Categorical columns are one-hot encoded and numeric
columns z-scored, both fit on the reference split only; patient-level
grouping (if an identifier column is found) is used for the reference/query
split via `GroupShuffleSplit` so no patient appears on both sides -- if no
identifier column is found, this falls back to a stratified-by-outcome
random split and says so explicitly in the returned metadata, rather than
silently claiming patient-level generalization it cannot support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from umaping.data.preprocessing import PreparedDataset, check_finite
from umaping.utils.io import download_file, ensure_dir

logger = logging.getLogger(__name__)

CITATION = (
    "Hong, W. S., Haimovich, A. D., and Taylor, R. A. (2018). Predicting hospital admission at "
    "emergency department triage using machine learning. PLOS ONE 13(7): e0201016. "
    "https://doi.org/10.1371/journal.pone.0201016 . Data: https://github.com/yaleemmlc/admissionprediction"
)
SOURCE_URL = "https://github.com/yaleemmlc/admissionprediction (Results/5v_cleandf.RData)"
_BRANCH_CANDIDATES = ("master", "main")
_RAW_PATH = "Results/5v_cleandf.RData"

_OUTCOME_CANDIDATES = ("disposition", "admit", "admission", "outcome", "y", "label")
_ID_CANDIDATES = ("patient_id", "patientid", "mrn", "encounter_id", "encounterid", "visit_id", "id")


def _require_pyreadr():
    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover - exercised only when pyreadr is absent
        raise ImportError(
            "Reading the Hong ED dataset's .RData file requires the optional 'pyreadr' dependency, "
            "not installed in this environment. Install it with: pip install -e '.[experiments]' "
            "(or `pip install pyreadr`)."
        ) from exc
    return pyreadr


def download_hong_ed(raw_dir: str | Path) -> Path:
    """Idempotent; tries `master` then `main` (see module docstring -- the
    real default branch was not confirmed live)."""
    raw_dir = ensure_dir(raw_dir)
    dest = raw_dir / "5v_cleandf.RData"
    if dest.exists():
        logger.info("%s already present; skipping download.", dest)
        return dest
    last_exc: Exception | None = None
    for branch in _BRANCH_CANDIDATES:
        url = f"https://raw.githubusercontent.com/yaleemmlc/admissionprediction/{branch}/{_RAW_PATH}"
        try:
            return download_file(url, dest, min_expected_bytes=1000)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: try the next branch name
            logger.warning("Download from branch '%s' failed (%s); trying the next candidate.", branch, exc)
            last_exc = exc
    raise RuntimeError(
        f"Could not download the Hong ED dataset from any of {_BRANCH_CANDIDATES}. "
        "The repository's default branch name or file path may have changed -- verify at "
        f"https://github.com/yaleemmlc/admissionprediction/tree/main/Results"
    ) from last_exc


def load_hong_ed_dataframe(raw_dir: str | Path) -> pd.DataFrame:
    pyreadr = _require_pyreadr()
    path = download_hong_ed(raw_dir)
    result = pyreadr.read_r(str(path))
    if len(result) == 0:
        raise ValueError(f"{path} contained no R objects.")
    # An RData file can hold several objects; take the first (the expected
    # case for a single-dataframe .RData export like this one) and log the
    # rest so a schema surprise is visible rather than silently discarded.
    keys = list(result.keys())
    if len(keys) > 1:
        logger.warning("RData file contained multiple objects %s; using the first ('%s').", keys, keys[0])
    return result[keys[0]]


_MIN_SUBSTRING_CANDIDATE_LENGTH = 3  # a candidate shorter than this (e.g. "y", "id") is only
# ever matched exactly (below), never as a substring -- "y" as a substring
# would spuriously match almost any column ("category", "day", ...).


def _detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
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


@dataclass
class HongEDSplitMetadata:
    outcome_column: str
    id_column: str | None
    split_mode: str  # "group" or "stratified_random_fallback"
    n_reference: int
    n_query: int
    dropped_columns: list[str] = field(default_factory=list)


def prepare_hong_ed_dataset(
    raw_dir: str | Path,
    query_fraction: float = 0.2,
    subset_mode: str = "full",  # "small" (fast dev), "medium", or "full"
    small_n: int = 500,
    medium_n: int = 5000,
    seed: int = 0,
    df: pd.DataFrame | None = None,
) -> tuple[PreparedDataset, HongEDSplitMetadata]:
    """`df`: optionally inject an already-loaded DataFrame -- used by tests
    to check the reference-only-fitting/no-leakage invariants without
    downloading or requiring `pyreadr`; omit to load the real file from
    `raw_dir` (downloading first if needed)."""
    if df is None:
        df = load_hong_ed_dataframe(raw_dir)
    df = df.reset_index(drop=True).copy()

    outcome_col = _detect_column(list(df.columns), _OUTCOME_CANDIDATES)
    if outcome_col is None:
        raise ValueError(
            f"Could not auto-detect an outcome/admission column among {list(df.columns)}. "
            f"Tried candidates {_OUTCOME_CANDIDATES}; update this list once the real schema is confirmed "
            "(see this module's docstring)."
        )
    id_col = _detect_column(list(df.columns), _ID_CANDIDATES)

    y = df[outcome_col]
    if y.dtype == object or y.dtype.name == "category":
        y = y.astype("category").cat.codes.to_numpy()
    else:
        y = y.to_numpy()

    # Reproducible subsetting, from the reference side (per spec: "All
    # subsets must sample from the reference side reproducibly") -- applied
    # to the whole pool *before* the split so query size stays proportional.
    if subset_mode in ("small", "medium"):
        target_n = small_n if subset_mode == "small" else medium_n
        if target_n < len(df):
            rng = np.random.default_rng(seed)
            keep_idx = rng.choice(len(df), size=target_n, replace=False)
            keep_idx.sort()
            df = df.iloc[keep_idx].reset_index(drop=True)
            y = y[keep_idx]
    elif subset_mode != "full":
        raise ValueError(f"Unknown subset_mode '{subset_mode}', expected 'small', 'medium', or 'full'.")

    if id_col is not None:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=query_fraction, random_state=seed)
        ref_idx, query_idx = next(splitter.split(df, groups=df[id_col]))
        split_mode = "group"
    else:
        from sklearn.model_selection import train_test_split

        logger.warning(
            "No patient-identifier column detected among %s; falling back to a random "
            "(non-patient-grouped) split. Patient-level generalization is NOT claimed for this split.",
            list(df.columns),
        )
        all_idx = np.arange(len(df))
        stratify = y if len(np.unique(y)) > 1 and len(np.unique(y)) < len(y) else None
        ref_idx, query_idx = train_test_split(
            all_idx, test_size=query_fraction, random_state=seed, stratify=stratify
        )
        split_mode = "stratified_random_fallback"

    feature_cols = [c for c in df.columns if c != outcome_col and c != id_col]
    df_ref, df_query = df.iloc[ref_idx], df.iloc[query_idx]
    y_ref, y_query = y[ref_idx], y[query_idx]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Numeric: z-score, fit (mean/std) on the reference split only.
    ref_numeric = df_ref[numeric_cols].astype(np.float64).fillna(0.0)
    query_numeric = df_query[numeric_cols].astype(np.float64).fillna(0.0)
    mean = ref_numeric.mean(axis=0)
    std = ref_numeric.std(axis=0).replace(0.0, 1.0)
    ref_numeric_z = ((ref_numeric - mean) / std).to_numpy(dtype=np.float32)
    query_numeric_z = ((query_numeric - mean) / std).to_numpy(dtype=np.float32)

    # Categorical: one-hot, with the category set fixed by the reference
    # split only -- a query-only category maps to all-zeros, never expands
    # the feature space to accommodate it.
    ref_dummy_blocks, query_dummy_blocks = [], []
    for col in categorical_cols:
        categories = pd.Categorical(df_ref[col].astype(str)).categories
        ref_cat = pd.Categorical(df_ref[col].astype(str), categories=categories)
        query_cat = pd.Categorical(df_query[col].astype(str), categories=categories)
        ref_dummy_blocks.append(pd.get_dummies(ref_cat).to_numpy(dtype=np.float32))
        query_dummy_blocks.append(pd.get_dummies(query_cat).to_numpy(dtype=np.float32))

    ref_parts = [ref_numeric_z] + ref_dummy_blocks
    query_parts = [query_numeric_z] + query_dummy_blocks
    x_ref = np.concatenate(ref_parts, axis=1) if ref_parts else np.zeros((len(df_ref), 0), dtype=np.float32)
    x_query = np.concatenate(query_parts, axis=1) if query_parts else np.zeros((len(df_query), 0), dtype=np.float32)

    check_finite(x_ref, "hong_ed reference features")
    check_finite(x_query, "hong_ed query features")

    prepared = PreparedDataset(
        reference_features=x_ref,
        query_features=x_query,
        reference_labels={"admitted": y_ref},
        query_labels={"admitted": y_query},
        input_dim=x_ref.shape[1],
    )
    metadata = HongEDSplitMetadata(
        outcome_column=outcome_col,
        id_column=id_col,
        split_mode=split_mode,
        n_reference=x_ref.shape[0],
        n_query=x_query.shape[0],
        dropped_columns=[c for c in [outcome_col, id_col] if c is not None],
    )
    return prepared, metadata
