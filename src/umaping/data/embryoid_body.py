"""Embryoid body development dataset (Experiment D: continuous single-cell
structure).

Public source (Figshare article, resolved via the Figshare API -- see
`data/_figshare.py` -- rather than a guessed direct file URL):
https://figshare.com/articles/dataset/Embryoid_body_development/23737416
(article id 23737416). This is the human embryonic-stem-cell embryoid-body
differentiation time-course dataset (Moon et al. 2019, used to introduce
PHATE), commonly cited across five sampled time windows spanning ~27 days.

**Caveat, stated explicitly because this could not be verified by an actual
download during implementation:** the exact file format and precise `obs`
metadata column name for the time-window label were not confirmed live. The
grouping column is auto-detected from `_GROUPING_CANDIDATES` below by
substring match, and this function fails loudly (rather than guessing) if
none of them match any observed column. Confirm on the first real run and
extend the candidate list if needed.

Never claims the resulting inductive embedding, or UMAPing's own
optimization-time vector field, corresponds to true developmental
trajectories or RNA velocity -- this dataset tests only whether a
fixed-reference inductive embedding preserves a *known* continuous
structure (successive sampling time windows) without artificially
fragmenting it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from umaping.data._figshare import download_figshare_article
from umaping.data._scrna_common import detect_grouping_column, load_anndata_any_format, prepare_continuous_scrna_dataset
from umaping.data.preprocessing import PreparedDataset

logger = logging.getLogger(__name__)

CITATION = (
    "Moon, K. R., van Dijk, D., Wang, Z., Gigante, S., Burkhardt, D. B., Chen, W. S., Yim, K., "
    "Elzen, A. van den, Hirn, M. J., Coifman, R. R., Ivanova, N. B., Wolf, G., and Krishnaswamy, S. "
    "(2019). Visualizing structure and transitions in high-dimensional biological data. Nature "
    "Biotechnology, 37, 1482-1492. Data (this deposit): "
    "https://figshare.com/articles/dataset/Embryoid_body_development/23737416"
)
SOURCE_URL = "https://figshare.com/articles/dataset/Embryoid_body_development/23737416"
ARTICLE_ID = 23737416
_GROUPING_CANDIDATES = ("time", "day", "sample_labels", "timepoint", "time_point", "stage", "state", "cell_type", "celltype")


def download_embryoid_body(raw_dir: str | Path) -> Path:
    """Idempotent: resolves the article's actual files via the Figshare API,
    then downloads each (skipping any already present)."""
    paths = download_figshare_article(ARTICLE_ID, raw_dir)
    return paths[0]


def prepare_embryoid_body_dataset(
    raw_dir: str | Path,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    seed: int = 0,
    query_groups: tuple[str, ...] | None = None,
    adata=None,
) -> tuple[PreparedDataset, str]:
    """`adata`: optionally inject an already-loaded AnnData -- used by tests
    without a Figshare download; omit to load the real file from `raw_dir`.
    `query_groups=None` uses the stratified "D-easy" split; pass explicit
    time-window values (e.g. one held-out intermediate window) for the
    "D-hard" split. Returns ``(prepared, grouping_column)``."""
    if adata is None:
        path = download_embryoid_body(raw_dir)
        adata = load_anndata_any_format(path)

    grouping_column = detect_grouping_column(list(adata.obs.columns), _GROUPING_CANDIDATES)
    if grouping_column is None:
        raise ValueError(
            f"Could not auto-detect a state/time grouping column among {list(adata.obs.columns)}. "
            f"Tried candidates {_GROUPING_CANDIDATES}; update this list once the real schema is confirmed "
            "(see this module's docstring)."
        )

    prepared = prepare_continuous_scrna_dataset(
        adata, grouping_column=grouping_column, query_groups=query_groups, n_hvg=n_hvg, n_pcs=n_pcs, seed=seed
    )
    return prepared, grouping_column
