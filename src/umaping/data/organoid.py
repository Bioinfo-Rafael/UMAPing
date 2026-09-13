"""Murine intestinal organoid scEU-seq dataset (Experiment D: continuous
single-cell structure).

Public source (Figshare article, resolved via the Figshare API -- see
`data/_figshare.py` -- rather than a guessed direct file URL):
https://figshare.com/articles/dataset/Murine_intestinal_organoid_scEU-seq_data_-_Raw/23737170
(article id 23737170).

**Caveat, stated explicitly because this could not be verified by an actual
download during implementation:** the exact file format inside this article
and its precise `obs` metadata column names were not confirmed live.
`load_anndata_any_format` (data/_scrna_common.py) dispatches on whatever
file extension is actually downloaded; the grouping column used for the
reference/query split is auto-detected from `_GROUPING_CANDIDATES` below by
substring match, and this function fails loudly (rather than guessing) if
none of them match any observed column. Confirm both on the first real run
and extend the candidate list if needed.

Never claims the resulting inductive embedding, or UMAPing's own
optimization-time vector field, corresponds to RNA velocity or true
developmental/biological time -- this dataset is used only to test whether
a fixed-reference inductive embedding preserves a *known* continuous
structure (successive scEU-seq labeling time points) without artificially
fragmenting it, per the experiment's own stated purpose.
"""

from __future__ import annotations

import logging
from pathlib import Path

from umaping.data._figshare import download_figshare_article
from umaping.data._scrna_common import detect_grouping_column, load_anndata_any_format, prepare_continuous_scrna_dataset
from umaping.data.preprocessing import PreparedDataset

logger = logging.getLogger(__name__)

CITATION = (
    "Battich, N., Beumer, J., de Barbanson, B., Krenning, L., Baron, C. S., Tanenbaum, M. E., "
    "Clevers, H., and van Oudenaarden, A. (2020). Sequencing metabolically labeled transcripts "
    "identifies a longevity control pathway in mammalian intestinal stem cells. Science, "
    "367(6483), 1151-1156. Data (this deposit): "
    "https://figshare.com/articles/dataset/Murine_intestinal_organoid_scEU-seq_data_-_Raw/23737170"
)
SOURCE_URL = "https://figshare.com/articles/dataset/Murine_intestinal_organoid_scEU-seq_data_-_Raw/23737170"
ARTICLE_ID = 23737170
_GROUPING_CANDIDATES = ("well", "time", "labeling_time", "pulse", "chase", "state", "cell_type", "celltype")


def download_organoid(raw_dir: str | Path) -> Path:
    """Idempotent: resolves the article's actual files via the Figshare API,
    then downloads each (skipping any already present)."""
    paths = download_figshare_article(ARTICLE_ID, raw_dir)
    return paths[0]


def prepare_organoid_dataset(
    raw_dir: str | Path,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    seed: int = 0,
    query_groups: tuple[str, ...] | None = None,
    adata=None,
) -> tuple[PreparedDataset, str]:
    """`adata`: optionally inject an already-loaded AnnData -- used by tests
    to check the reference-only-fitting/no-leakage invariants without a
    Figshare download; omit to load the real file from `raw_dir` (downloading
    first if needed). `query_groups=None` uses the stratified "D-easy" split;
    pass explicit group values for the "D-hard" held-out-region split.
    Returns ``(prepared, grouping_column)``, the latter recorded so callers
    know which auto-detected column the split/labels are keyed on."""
    if adata is None:
        path = download_organoid(raw_dir)
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
