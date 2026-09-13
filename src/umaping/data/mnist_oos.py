"""MNIST reference/query experiment (Experiment C: out-of-sample repulsion /
cluster-periphery effect).

Official source: `torchvision.datasets.MNIST` (torchvision's own download +
parsing of the canonical LeCun/Cortes/Burges release). `torchvision` is an
optional dependency (see pyproject.toml's `experiments` extra) -- imported
lazily so importing this module never requires it unless this dataset is
actually used.

Reference = the official 60,000-image train split; query = the official
10,000-image test split. Reference-only PCA (default 50 components) is fit
on the (optionally subsampled) reference split's flattened, [0,1]-scaled
pixels; the query split is only ever *transformed* through it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from umaping.data.preprocessing import FittedPCA, PreparedDataset, check_finite, fit_pca

logger = logging.getLogger(__name__)

CITATION = (
    "LeCun, Y., Cortes, C., and Burges, C. J. C. The MNIST Database of Handwritten Digits. "
    "http://yann.lecun.com/exdb/mnist/ . Accessed here via torchvision.datasets.MNIST."
)
SOURCE_URL = "http://yann.lecun.com/exdb/mnist/ (fetched via torchvision.datasets.MNIST)"


def _require_torchvision():
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:  # pragma: no cover - exercised only when torchvision is absent
        raise ImportError(
            "MNIST requires the optional 'torchvision' dependency, not installed in this "
            "environment. Install it with: pip install -e '.[experiments]' (or `pip install torchvision`)."
        ) from exc
    return MNIST


def download_mnist(raw_dir: str | Path) -> Path:
    """Idempotent: `torchvision.datasets.MNIST(..., download=True)` already
    skips re-downloading files already present under `raw_dir`."""
    MNIST = _require_torchvision()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    MNIST(root=str(raw_dir), train=True, download=True)
    MNIST(root=str(raw_dir), train=False, download=True)
    return raw_dir


def _load_official_split(raw_dir: Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    MNIST = _require_torchvision()
    ds = MNIST(root=str(raw_dir), train=train, download=False)
    images = ds.data.numpy().astype(np.float32) / 255.0
    labels = ds.targets.numpy().astype(np.int64)
    return images, labels


def prepare_mnist_dataset(
    raw_dir: str | Path,
    use_pca: bool = True,
    pca_dim: int = 50,
    seed: int = 0,
    n_reference_subsample: int | None = None,
    n_query_subsample: int | None = None,
    train_data: tuple[np.ndarray, np.ndarray] | None = None,
    test_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[PreparedDataset, FittedPCA | None]:
    """`train_data`/`test_data`: optionally inject already-loaded
    ``(images, labels)`` pairs -- used by tests without torchvision or a real
    download; omit both to load the real, official split from `raw_dir`."""
    x_ref_raw, y_ref = train_data if train_data is not None else _load_official_split(Path(raw_dir), train=True)
    x_query_raw, y_query = test_data if test_data is not None else _load_official_split(Path(raw_dir), train=False)

    x_ref_raw = np.asarray(x_ref_raw, dtype=np.float32).reshape(x_ref_raw.shape[0], -1)
    x_query_raw = np.asarray(x_query_raw, dtype=np.float32).reshape(x_query_raw.shape[0], -1)
    y_ref = np.asarray(y_ref)
    y_query = np.asarray(y_query)

    rng = np.random.default_rng(seed)
    if n_reference_subsample is not None and n_reference_subsample < x_ref_raw.shape[0]:
        idx = rng.choice(x_ref_raw.shape[0], size=n_reference_subsample, replace=False)
        x_ref_raw, y_ref = x_ref_raw[idx], y_ref[idx]
    if n_query_subsample is not None and n_query_subsample < x_query_raw.shape[0]:
        idx = rng.choice(x_query_raw.shape[0], size=n_query_subsample, replace=False)
        x_query_raw, y_query = x_query_raw[idx], y_query[idx]

    if x_ref_raw.shape[0] == 0 or x_query_raw.shape[0] == 0:
        raise ValueError("MNIST reference or query split is empty after subsampling.")
    check_finite(x_ref_raw, "mnist reference features")
    check_finite(x_query_raw, "mnist query features")

    if use_pca:
        pca = fit_pca(x_ref_raw, n_components=pca_dim, seed=seed)
        x_ref, x_query = pca.transform(x_ref_raw), pca.transform(x_query_raw)
        input_dim = pca_dim
    else:
        pca = None
        x_ref, x_query = x_ref_raw.astype(np.float32), x_query_raw.astype(np.float32)
        input_dim = x_ref_raw.shape[1]

    prepared = PreparedDataset(
        reference_features=x_ref,
        query_features=x_query,
        reference_labels={"digit": y_ref},
        query_labels={"digit": y_query},
        input_dim=input_dim,
    )
    return prepared, pca
