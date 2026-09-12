"""Tests for models/spectral.py + training/spectral.py: output dimensions
and the frozen, reference-only calibration contract (no live graph access
needed to embed a point once trained).

Written to be run remotely (see RUNTIME_CHECKS.md); not executed here.
"""

from __future__ import annotations

import numpy as np
import torch

from umaping.config import SpectralConfig
from umaping.graph import build_reference_graph
from umaping.models.spectral import SpectralCalibration, SpectralEmbedder, SpectralEncoderNet
from umaping.training.spectral import compute_calibration, train_spectral

_EMBEDDING_DIM = 2
_RAW_OUTPUT_DIM = _EMBEDDING_DIM + 1


def _tiny_setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, d = 40, 5
    features = rng.normal(size=(n, d)).astype(np.float32)
    graph = build_reference_graph(features, n_neighbors=5)
    return features, graph.w


def _train_tiny_spectral(features, w, seed: int = 0):
    model = SpectralEncoderNet(input_dim=features.shape[1], hidden_dims=[16], raw_output_dim=_RAW_OUTPUT_DIM)
    cfg = SpectralConfig(hidden_dims=[16], raw_output_dim=_RAW_OUTPUT_DIM, steps=10, edge_batch_size=64, log_every=5)
    train_spectral(model, features, w, cfg, device=torch.device("cpu"), seed=seed, progress=False)
    calibration, reference_embedding = compute_calibration(
        model, features, w, embedding_dim=_EMBEDDING_DIM, calibration_target_scale=10.0, device=torch.device("cpu")
    )
    return model, calibration, reference_embedding


def test_spectral_embed_one_has_expected_dimensions():
    features, w = _tiny_setup()
    model, calibration, reference_embedding = _train_tiny_spectral(features, w)

    assert reference_embedding.shape == (features.shape[0], _EMBEDDING_DIM)
    assert calibration.whitening.shape == (_RAW_OUTPUT_DIM, _RAW_OUTPUT_DIM)
    assert calibration.projection.shape == (_RAW_OUTPUT_DIM, _EMBEDDING_DIM)
    assert calibration.mean.shape == (_EMBEDDING_DIM,)

    embedder = SpectralEmbedder(model, calibration, device="cpu")
    y_one = embedder.embed_one(features[0])
    assert y_one.shape == (_EMBEDDING_DIM,)
    assert np.all(np.isfinite(y_one))

    y_batch = embedder.embed(features[:3])
    assert y_batch.shape == (3, _EMBEDDING_DIM)
    np.testing.assert_allclose(y_batch[0], y_one, rtol=1e-5, atol=1e-6)


def test_spectral_calibration_save_load_round_trip(tmp_path):
    features, w = _tiny_setup(seed=1)
    _model, calibration, _ = _train_tiny_spectral(features, w, seed=1)

    path = tmp_path / "calibration.npz"
    calibration.save(path)
    reloaded = SpectralCalibration.load(path)

    np.testing.assert_allclose(reloaded.whitening, calibration.whitening)
    np.testing.assert_allclose(reloaded.projection, calibration.projection)
    np.testing.assert_allclose(reloaded.mean, calibration.mean)
    assert reloaded.scale == calibration.scale


def test_spectral_embed_one_uses_only_frozen_state_no_live_graph_needed():
    """Once trained + calibrated, a fresh SpectralEmbedder built only from
    the trained model + saved calibration must reproduce embed_one's output
    with no access whatsoever to the training graph or reference features."""
    features, w = _tiny_setup(seed=2)
    model, calibration, _ = _train_tiny_spectral(features, w, seed=2)

    query_point = features[7].copy()
    embedder = SpectralEmbedder(model, calibration, device="cpu")
    y_before = embedder.embed_one(query_point)

    # Simulate the graph/reference features being unavailable at inference
    # time: build a *new* embedder from nothing but the frozen model +
    # calibration, never touching `w` or `features` again.
    del w, features
    fresh_embedder = SpectralEmbedder(model, calibration, device="cpu")
    y_after = fresh_embedder.embed_one(query_point)

    np.testing.assert_allclose(y_after, y_before, rtol=1e-6, atol=1e-7)


def test_whitened_reference_outputs_are_orthonormal():
    """Z^T Z / N should be (near-)exactly I after whitening, by construction
    -- this is what makes the projected operator H's eigendecomposition
    meaningful (Rayleigh-Ritz over an orthonormal basis)."""
    features, w = _tiny_setup(seed=3)
    model, calibration, _ = _train_tiny_spectral(features, w, seed=3)

    from umaping.models.mlp import batched_forward

    raw = batched_forward(model, features, torch.device("cpu"))
    z = raw @ calibration.whitening
    gram = (z.T @ z) / z.shape[0]
    np.testing.assert_allclose(gram, np.eye(_RAW_OUTPUT_DIM), atol=1e-4)
