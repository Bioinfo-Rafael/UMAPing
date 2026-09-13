"""Filesystem helpers: atomic writes, JSON/YAML I/O, and checkpoint markers.

Every "expensive stage" in the pipeline (graph construction, retriever
training, spectral training, reference-flow + repulsion training) writes its
output through :func:`atomic_write_bytes` / :func:`atomic_torch_save` and then
drops a ``<name>.done`` marker via :func:`mark_done`. Resuming a run is just
checking :func:`is_done` before redoing the work.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
import scipy.sparse as sp
import torch
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(path: str | Path, write_fn) -> None:
    """Write via a same-directory temp file + ``os.replace`` so a crash mid-write
    never leaves a corrupt file at ``path``."""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    _atomic_write(path, lambda f: f.write(data))


def save_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    payload = json.dumps(obj, indent=indent, default=str).encode("utf-8")
    atomic_write_bytes(path, payload)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(path: str | Path, obj: Any) -> None:
    payload = yaml.safe_dump(obj, sort_keys=False).encode("utf-8")
    atomic_write_bytes(path, payload)


def load_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    def _write(f):
        torch.save(obj, f)

    _atomic_write(path, _write)


def atomic_pickle_save(obj: Any, path: str | Path) -> None:
    _atomic_write(path, lambda f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL))


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    def _write(f):
        np.savez(f, **arrays)

    _atomic_write(path, _write)


def load_npz(path: str | Path) -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=False)


def atomic_save_sparse(path: str | Path, matrix: sp.spmatrix) -> None:
    def _write(f):
        sp.save_npz(f, matrix.tocsr())

    _atomic_write(path, _write)


def load_sparse(path: str | Path) -> sp.csr_matrix:
    return sp.load_npz(path)


def guard_run_dir(run_dir: str | Path, resume: bool) -> None:
    """Refuse to silently reuse a run directory that already has completed
    stages unless `resume` is explicitly set. Shared by `cli.py` (the
    `train`/`pipeline` commands) and `experiments/runner.py`."""
    metadata_path = Path(run_dir) / "metadata.json"
    if metadata_path.exists() and not resume:
        raise SystemExit(
            f"Run directory '{run_dir}' already contains a previous run (metadata.json exists). "
            "Pass --resume to continue it, or point --run-dir at an empty directory to start fresh."
        )


def mark_done(stage_dir: str | Path, stage_name: str) -> None:
    marker = Path(stage_dir) / f"{stage_name}.done"
    atomic_write_bytes(marker, b"ok")


def is_done(stage_dir: str | Path, stage_name: str) -> bool:
    return (Path(stage_dir) / f"{stage_name}.done").exists()


def download_file(url: str, dest: str | Path, min_expected_bytes: int = 1_000_000, timeout: int = 120) -> Path:
    """Streamed download to `dest` via a same-directory `.part` file, renamed
    atomically on success. Idempotent: skips re-downloading if `dest`
    already exists and looks complete (this is deliberately a size check,
    not a checksum, so it stays cheap on every resumed run)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size >= min_expected_bytes:
        logger.info("%s already present (%d bytes); skipping download.", dest, dest.stat().st_size)
        return dest
    ensure_dir(dest.parent)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def collect_package_versions() -> dict[str, str]:
    """Best-effort snapshot of installed package versions, recorded into
    ``metadata.json`` whenever a run actually executes (never fabricated
    ahead of time)."""
    import importlib.metadata as importlib_metadata

    packages = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "torch",
        "umap-learn",
        "pynndescent",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "pillow",
        "requests",
        "anndata",
        "scanpy",
        "umaping",
    ]
    versions: dict[str, str] = {"python": sys.version}
    for name in packages:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions
