"""COIL-20 / COIL-100 (Columbia CAVE) download, parsing, and reference/query
split.

Official archives (verified against the live Columbia CAVE pages and by
downloading + inspecting both zips; see RUNTIME_CHECKS.md):

    COIL-20:  https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip
              1440 grayscale 128x128 PNGs, "obj{N}__{idx}.png", N=1..20,
              idx = 0..71 a *sequential pose index* (NOT degrees).
    COIL-100: https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip
              7200 RGB 128x128 PNGs, "obj{N}__{angle}.png", N=1..100,
              angle = 0, 5, ..., 355 -- literal degrees. The archive also
              ships two leftover non-image files (a ppm->png conversion
              script and its backup) that the "*.png" glob below already
              excludes.

COIL-20's suffix is a raw sequence index while COIL-100's is literal
degrees -- both encode the same 72-step rotational sequence, so sorting by
the parsed integer ascending and taking its *rank* within each object
recovers the same "pose position in the rotation" for both datasets, and
that rank is what the every-4th-pose holdout below keys off.
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from umaping.data.preprocessing import (
    FittedPCA,
    PreparedDataset,
    assert_no_overlap,
    check_finite,
    fit_pca,
)
from umaping.utils.io import download_file, ensure_dir

logger = logging.getLogger(__name__)

_URLS = {
    "coil20": "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
    "coil100": "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip",
}
_EXPECTED_COUNT = {"coil20": 1440, "coil100": 7200}
_EXPECTED_SUBDIR = {"coil20": "coil-20-proc", "coil100": "coil-100"}
_IS_GRAYSCALE = {"coil20": True, "coil100": False}
_FILENAME_RE = re.compile(r"^obj(\d+)__(\d+)\.png$", re.IGNORECASE)
_IMAGE_SIZE = 128


def _extract_zip(zip_path: Path, extract_dir: Path, expected_count: int) -> Path:
    ensure_dir(extract_dir)
    if len(list(extract_dir.rglob("*.png"))) < expected_count:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return extract_dir


def download_coil(dataset: str, raw_dir: str | Path) -> Path:
    """Idempotent: skips the download/extraction if already present. Returns
    the directory directly containing the `obj*__*.png` files."""
    if dataset not in _URLS:
        raise ValueError(f"Unknown COIL dataset '{dataset}', expected one of {sorted(_URLS)}")
    raw_dir = ensure_dir(raw_dir)
    zip_path = raw_dir / f"{dataset}.zip"
    download_file(_URLS[dataset], zip_path)
    image_dir = _extract_zip(zip_path, raw_dir / "extracted", _EXPECTED_COUNT[dataset])
    return image_dir


def parse_coil_directory(image_dir: Path, expected_count: int) -> list[tuple[Path, int, int]]:
    entries = []
    for path in sorted(image_dir.glob("*.png")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            logger.warning("Skipping unrecognized file in COIL directory: %s", path.name)
            continue
        entries.append((path, int(match.group(1)), int(match.group(2))))
    if len(entries) != expected_count:
        raise ValueError(f"Expected {expected_count} COIL images under {image_dir}, found {len(entries)}.")
    return entries


def compute_pose_ranks(entries: list[tuple[Path, int, int]]) -> np.ndarray:
    object_ids = np.array([e[1] for e in entries])
    pose_values = np.array([e[2] for e in entries])
    ranks = np.zeros(len(entries), dtype=np.int64)
    for obj in np.unique(object_ids):
        idx = np.where(object_ids == obj)[0]
        order = np.argsort(pose_values[idx], kind="stable")
        ranks[idx[order]] = np.arange(len(idx))
    return ranks


def load_images(entries: list[tuple[Path, int, int]], grayscale: bool) -> np.ndarray:
    shape = (len(entries), _IMAGE_SIZE, _IMAGE_SIZE) if grayscale else (len(entries), _IMAGE_SIZE, _IMAGE_SIZE, 3)
    images = np.empty(shape, dtype=np.float32)
    for i, (path, _, _) in enumerate(entries):
        img = Image.open(path)
        img = img.convert("L") if grayscale else img.convert("RGB")
        if img.size != (_IMAGE_SIZE, _IMAGE_SIZE):
            raise ValueError(f"{path} has unexpected size {img.size}, expected {_IMAGE_SIZE}x{_IMAGE_SIZE}")
        images[i] = np.asarray(img, dtype=np.float32) / 255.0
    return images


def split_coil(pose_rank: np.ndarray, holdout_period: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Every `holdout_period`-th pose (by rotational rank) is held out as the
    query set; everything else is the fixed reference set X. With the
    default period 4 over 72 poses this gives 18 query / 54 reference poses
    per object (25% / 75%). Ground-truth query neighbors are always found by
    searching the reference set only."""
    query_mask = (pose_rank % holdout_period) == 0
    reference_mask = ~query_mask
    return reference_mask, query_mask


def prepare_coil_dataset(
    dataset: str,
    raw_dir: str | Path,
    use_pca: bool,
    pca_dim: int,
    seed: int = 0,
    holdout_period: int = 4,
) -> tuple[PreparedDataset, FittedPCA | None]:
    image_dir = download_coil(dataset, raw_dir)
    entries = parse_coil_directory(image_dir, _EXPECTED_COUNT[dataset])
    pose_rank = compute_pose_ranks(entries)
    object_id = np.array([e[1] for e in entries])
    images = load_images(entries, grayscale=_IS_GRAYSCALE[dataset])
    features_flat = images.reshape(images.shape[0], -1)

    reference_mask, query_mask = split_coil(pose_rank, holdout_period=holdout_period)
    assert_no_overlap(np.where(reference_mask)[0], np.where(query_mask)[0])

    ref_objects = set(np.unique(object_id[reference_mask]).tolist())
    query_objects = set(np.unique(object_id[query_mask]).tolist())
    missing = query_objects - ref_objects
    if missing:
        raise ValueError(f"Object classes present in query but absent from reference: {sorted(missing)}")

    x_ref_raw = features_flat[reference_mask]
    x_query_raw = features_flat[query_mask]
    check_finite(x_ref_raw, "coil reference features")
    check_finite(x_query_raw, "coil query features")

    if use_pca:
        pca = fit_pca(x_ref_raw, n_components=pca_dim, seed=seed)
        x_ref = pca.transform(x_ref_raw)
        x_query = pca.transform(x_query_raw)
        input_dim = pca_dim
    else:
        pca = None
        x_ref, x_query = x_ref_raw.astype(np.float32), x_query_raw.astype(np.float32)
        input_dim = x_ref_raw.shape[1]

    prepared = PreparedDataset(
        reference_features=x_ref,
        query_features=x_query,
        reference_labels={"object_id": object_id[reference_mask], "pose_rank": pose_rank[reference_mask]},
        query_labels={"object_id": object_id[query_mask], "pose_rank": pose_rank[query_mask]},
        input_dim=input_dim,
    )
    return prepared, pca
