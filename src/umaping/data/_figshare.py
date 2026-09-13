"""Shared Figshare-API download helper for the two Experiment-D scRNA-seq
datasets (`organoid.py`, `embryoid_body.py`).

A Figshare *article* page (e.g. the URLs given for these two datasets) is
not itself a direct file download link -- the actual files, and their
`ndownloader.figshare.com` download URLs, are only available through
Figshare's public API (`https://api.figshare.com/v2/articles/{id}`). This
module queries that API and downloads the resulting file list, rather than
guessing a direct `ndownloader` URL the way `data/pancreas.py`'s fallback
does for a URL a human already resolved by hand.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from umaping.utils.io import download_file, ensure_dir

logger = logging.getLogger(__name__)

_API_URL = "https://api.figshare.com/v2/articles/{article_id}"


def list_figshare_files(article_id: int, timeout: int = 60) -> list[dict[str, Any]]:
    """The article's file metadata (name, size, `download_url`) via Figshare's
    public API -- never guessed from the article's human-facing page URL."""
    response = requests.get(_API_URL.format(article_id=article_id), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    files = payload.get("files", [])
    if not files:
        raise ValueError(f"Figshare article {article_id} returned no files via the API.")
    return files


def download_figshare_article(article_id: int, raw_dir: str | Path, name_contains: str | None = None) -> list[Path]:
    """Downloads every file attached to a Figshare article (or, if
    `name_contains` is given, only files whose name contains that substring,
    case-insensitively) into `raw_dir`. Idempotent via `utils.io.download_file`."""
    raw_dir = ensure_dir(raw_dir)
    files = list_figshare_files(article_id)
    if name_contains is not None:
        files = [f for f in files if name_contains.lower() in f["name"].lower()]
        if not files:
            raise ValueError(f"No Figshare file names contained '{name_contains}' for article {article_id}.")

    paths = []
    for f in files:
        dest = raw_dir / f["name"]
        min_bytes = max(1, int(f.get("size", 1)) // 2)  # loose lower bound; a true checksum isn't provided cheaply here
        paths.append(download_file(f["download_url"], dest, min_expected_bytes=min_bytes))
    return paths
