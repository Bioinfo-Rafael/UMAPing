"""20 Newsgroups reference/query experiment.

Official source: `sklearn.datasets.fetch_20newsgroups` (scikit-learn's own
official loader; `scikit-learn` is already a core dependency of this
project, so no extra optional install is needed for this dataset).
Reference = official `subset="train"` (11,314 documents); query = official
`subset="test"` (7,532 documents) -- the canonical Lang (1995) train/test
split scikit-learn ships, never resampled into a single pool.

Reference-fitted TF-IDF -> reference-fitted TruncatedSVD -> query transform
(the "controlled feature pipeline" the spec calls for). Headers/footers/
quotes are stripped by default (`remove=("headers","footers","quotes")`),
matching scikit-learn's own documented recommendation for this dataset (its
un-stripped metadata otherwise makes classification/neighbor tasks close to
trivial and not representative of the text content itself).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from umaping.data.preprocessing import PreparedDataset, check_finite

logger = logging.getLogger(__name__)

CITATION = (
    "Lang, K. (1995). NewsWeeder: Learning to Filter Netnews. Proceedings of the 12th "
    "International Conference on Machine Learning (ICML 1995), pp. 331-339. Accessed here "
    "via scikit-learn's official `sklearn.datasets.fetch_20newsgroups` loader."
)
SOURCE_URL = "http://qwone.com/~jason/20Newsgroups/ (fetched via sklearn.datasets.fetch_20newsgroups)"


@dataclass
class FittedTextPipeline:
    """Reference-only-fitted TF-IDF + TruncatedSVD, bundled for reuse/inspection."""

    vectorizer: TfidfVectorizer
    svd: TruncatedSVD

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.svd.transform(self.vectorizer.transform(texts)).astype(np.float32)


def download_twenty_newsgroups(raw_dir: str | Path) -> Path:
    """Idempotent: `fetch_20newsgroups(..., data_home=raw_dir)` caches to
    disk and skips re-downloading on subsequent calls."""
    from sklearn.datasets import fetch_20newsgroups

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetch_20newsgroups(data_home=str(raw_dir), subset="train", remove=("headers", "footers", "quotes"))
    fetch_20newsgroups(data_home=str(raw_dir), subset="test", remove=("headers", "footers", "quotes"))
    return raw_dir


def _load_official_split(raw_dir: Path, subset: str, remove: tuple[str, ...]) -> tuple[list[str], np.ndarray]:
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(data_home=str(raw_dir), subset=subset, remove=remove)
    return list(bunch.data), np.asarray(bunch.target)


def prepare_twenty_newsgroups_dataset(
    raw_dir: str | Path,
    n_components: int = 100,
    max_features: int = 20000,
    remove: tuple[str, ...] = ("headers", "footers", "quotes"),
    seed: int = 0,
    n_reference_subsample: int | None = None,
    n_query_subsample: int | None = None,
    train_data: tuple[list[str], np.ndarray] | None = None,
    test_data: tuple[list[str], np.ndarray] | None = None,
) -> tuple[PreparedDataset, FittedTextPipeline]:
    """`train_data`/`test_data`: optionally inject already-loaded
    ``(texts, labels)`` pairs -- used by tests without a real network fetch;
    omit both to load the real, official split from `raw_dir`."""
    texts_ref, y_ref = train_data if train_data is not None else _load_official_split(Path(raw_dir), "train", remove)
    texts_query, y_query = test_data if test_data is not None else _load_official_split(Path(raw_dir), "test", remove)
    texts_ref, y_ref = list(texts_ref), np.asarray(y_ref)
    texts_query, y_query = list(texts_query), np.asarray(y_query)

    rng = np.random.default_rng(seed)
    if n_reference_subsample is not None and n_reference_subsample < len(texts_ref):
        idx = rng.choice(len(texts_ref), size=n_reference_subsample, replace=False)
        texts_ref = [texts_ref[i] for i in idx]
        y_ref = y_ref[idx]
    if n_query_subsample is not None and n_query_subsample < len(texts_query):
        idx = rng.choice(len(texts_query), size=n_query_subsample, replace=False)
        texts_query = [texts_query[i] for i in idx]
        y_query = y_query[idx]

    if len(texts_ref) == 0 or len(texts_query) == 0:
        raise ValueError("20 Newsgroups reference or query split is empty after subsampling.")

    # Reference-only fitting: TF-IDF vocabulary + IDF weights, then the SVD
    # basis, both fit on the reference split only; the query split is only
    # ever transformed through them.
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    x_ref_tfidf = vectorizer.fit_transform(texts_ref)
    n_components_eff = max(1, min(n_components, x_ref_tfidf.shape[1] - 1, x_ref_tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components_eff, random_state=seed)
    x_ref = svd.fit_transform(x_ref_tfidf).astype(np.float32)
    x_query = svd.transform(vectorizer.transform(texts_query)).astype(np.float32)

    check_finite(x_ref, "20newsgroups reference features")
    check_finite(x_query, "20newsgroups query features")

    prepared = PreparedDataset(
        reference_features=x_ref,
        query_features=x_query,
        reference_labels={"newsgroup": y_ref},
        query_labels={"newsgroup": y_query},
        input_dim=n_components_eff,
    )
    return prepared, FittedTextPipeline(vectorizer=vectorizer, svd=svd)
