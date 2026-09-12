"""Tests for inference.py:

* embed_one(x*) must never depend on any other query point, and must never
  mutate the frozen reference trajectory (tests #7);
* a saved model can be reloaded from disk and used for single-point
  inference (test #10).

Written to be run remotely (see RUNTIME_CHECKS.md); not executed here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from umaping.config import Config, DatasetConfig, FlowConfig, RepulsionConfig, RetrieverConfig, SpectralConfig
from umaping.graph import build_reference_graph, row_degree
from umaping.inference import InferenceConfig, InferenceEngine
from umaping.models.repulsion import RepulsionField
from umaping.models.retriever import DualEncoder, ExactChunkedIndex
from umaping.models.spectral import SpectralEmbedder, SpectralEncoderNet
from umaping.training.flow import build_reference_trajectory, train_repulsion_field
from umaping.training.retriever import train_retriever
from umaping.training.spectral import compute_calibration, train_spectral
from umaping.umap_forces import find_ab_params
from umaping.utils.io import atomic_torch_save

_EMBEDDING_DIM = 2
_N_STEPS = 8
_HIDDEN = [16]
_RETRIEVAL_DIM = 8
_REPULSION_HIDDEN = 16
_REPULSION_BLOCKS = 2
_TIME_EMBED_DIM = 8


def _build_toy_engine(seed: int = 0) -> tuple[InferenceEngine, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = 30, 6
    features = rng.normal(size=(n, d)).astype(np.float32)
    device = torch.device("cpu")

    graph = build_reference_graph(features, n_neighbors=5)

    retriever = DualEncoder(input_dim=d, hidden_dims=_HIDDEN, retrieval_dim=_RETRIEVAL_DIM, temperature=0.1)
    retriever_cfg = RetrieverConfig(
        hidden_dims=_HIDDEN, retrieval_dim=_RETRIEVAL_DIM, batch_size=8, n_random_negatives=4, steps=5, log_every=5
    )
    train_retriever(retriever, features, graph.mu, retriever_cfg, device, seed=seed, progress=False)
    raw_keys = retriever.encode_key(torch.as_tensor(features)).detach().numpy()
    neighbor_index = ExactChunkedIndex(raw_keys, temperature=retriever_cfg.temperature, device="cpu")

    spectral_model = SpectralEncoderNet(input_dim=d, hidden_dims=_HIDDEN, raw_output_dim=_EMBEDDING_DIM + 1)
    spectral_cfg = SpectralConfig(
        hidden_dims=_HIDDEN, raw_output_dim=_EMBEDDING_DIM + 1, steps=10, edge_batch_size=64, log_every=5
    )
    train_spectral(spectral_model, features, graph.w, spectral_cfg, device, seed=seed, progress=False)
    calibration, reference_embedding = compute_calibration(
        spectral_model, features, graph.w, embedding_dim=_EMBEDDING_DIM, calibration_target_scale=10.0, device=device
    )
    spectral_embedder = SpectralEmbedder(spectral_model, calibration, device="cpu")

    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    flow_cfg = FlowConfig(n_steps=_N_STEPS, initial_alpha=1.0, dynamics_negative_samples=8, checkpoint_stride=1)
    trajectory = build_reference_trajectory(
        reference_embedding, graph.w, a, b, negative_sample_rate=5.0, cfg=flow_cfg, device=device, seed=seed
    )

    repulsion_model = RepulsionField(
        embedding_dim=_EMBEDDING_DIM,
        hidden_dim=_REPULSION_HIDDEN,
        n_residual_blocks=_REPULSION_BLOCKS,
        time_embed_dim=_TIME_EMBED_DIM,
    )
    repulsion_cfg = RepulsionConfig(
        hidden_dim=_REPULSION_HIDDEN,
        n_residual_blocks=_REPULSION_BLOCKS,
        time_embed_dim=_TIME_EMBED_DIM,
        steps=10,
        batch_size=16,
        teacher_negative_samples=8,
        log_every=5,
    )
    row_mass = row_degree(graph.w)
    train_repulsion_field(repulsion_model, trajectory, a, b, repulsion_cfg, row_mass, device, seed=seed, progress=False)

    inf_cfg = InferenceConfig(
        n_neighbors=5, candidate_pool_size=10, negative_sample_rate=5.0, n_steps=_N_STEPS, initial_alpha=1.0, a=a, b=b
    )
    engine = InferenceEngine(
        retriever=retriever,
        neighbor_index=neighbor_index,
        reference_features=features,
        spectral_embedder=spectral_embedder,
        reference_trajectory=trajectory,
        repulsion_field=repulsion_model,
        cfg=inf_cfg,
        device=device,
        seed=seed,
    )
    return engine, features


def test_embed_one_is_independent_of_other_queries():
    engine, features = _build_toy_engine(seed=0)
    rng = np.random.default_rng(99)
    query_a = rng.normal(size=features.shape[1]).astype(np.float32)
    query_b = rng.normal(size=features.shape[1]).astype(np.float32)

    result_a_first = engine.embed_one(query_a)
    engine.embed_one(query_b)  # a different query, processed in between
    result_a_second = engine.embed_one(query_a)

    np.testing.assert_allclose(result_a_first.embedding, result_a_second.embedding, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result_a_first.neighbor_ids, result_a_second.neighbor_ids)


def test_embed_one_oracle_mc_repulsion_is_reproducible_and_query_independent():
    """Regression test: the oracle_mc repulsion diagnostic's Monte-Carlo
    sampling must be a fresh, locally-seeded draw per embed_one call, never a
    generator whose state persists (and is thus perturbed by) other queries
    processed earlier on the same engine."""
    engine, features = _build_toy_engine(seed=8)
    engine.repulsion_mode = "oracle_mc"
    engine.oracle_mc_samples = 10

    rng = np.random.default_rng(50)
    query_a = rng.normal(size=features.shape[1]).astype(np.float32)
    query_b = rng.normal(size=features.shape[1]).astype(np.float32)

    result_alone = engine.embed_one(query_a)
    result_alone_again = engine.embed_one(query_a)
    np.testing.assert_allclose(result_alone.embedding, result_alone_again.embedding, rtol=1e-6, atol=1e-7)

    engine.embed_one(query_b)  # a different query processed in between
    result_after_other_query = engine.embed_one(query_a)
    np.testing.assert_allclose(result_alone.embedding, result_after_other_query.embedding, rtol=1e-6, atol=1e-7)


def test_embed_one_does_not_mutate_reference_trajectory():
    engine, features = _build_toy_engine(seed=1)
    positions_before = engine.trajectory.positions.copy()

    rng = np.random.default_rng(5)
    for _ in range(5):
        query = rng.normal(size=features.shape[1]).astype(np.float32)
        engine.embed_one(query)

    np.testing.assert_array_equal(engine.trajectory.positions, positions_before)


def test_embed_one_output_shape_and_finiteness():
    engine, features = _build_toy_engine(seed=2)
    query = np.random.default_rng(3).normal(size=features.shape[1]).astype(np.float32)
    result = engine.embed_one(query)

    assert result.embedding.shape == (_EMBEDDING_DIM,)
    assert np.all(np.isfinite(result.embedding))
    assert len(result.neighbor_ids) == engine.cfg.n_neighbors
    assert len(result.neighbor_weights) == engine.cfg.n_neighbors


def test_embed_one_rejects_wrong_input_dimension():
    engine, features = _build_toy_engine(seed=4)
    wrong_dim_query = np.zeros(features.shape[1] + 1, dtype=np.float32)
    with pytest.raises(ValueError):
        engine.embed_one(wrong_dim_query)


def test_saved_model_can_be_reloaded_and_used_for_single_point_inference(tmp_path):
    """Test #10: persist every artifact InferenceEngine.load expects, reload
    a brand-new engine from disk only, and confirm embed_one works."""
    engine, features = _build_toy_engine(seed=6)
    d = features.shape[1]

    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "memory").mkdir(parents=True)

    np.save(run_dir / "memory" / "reference_features.npy", engine.reference_features)

    atomic_torch_save(
        {
            "state_dict": engine.retriever.state_dict(),
            "hparams": {
                "input_dim": d,
                "hidden_dims": _HIDDEN,
                "retrieval_dim": _RETRIEVAL_DIM,
                "activation": "gelu",
                "temperature": 0.1,
            },
        },
        run_dir / "checkpoints" / "retriever.pt",
    )
    raw_keys = engine.retriever.encode_key(torch.as_tensor(features)).detach().numpy()
    np.save(run_dir / "memory" / "retriever_keys.npy", raw_keys.astype(np.float32))

    atomic_torch_save(
        {
            "state_dict": engine.spectral_embedder.encoder.state_dict(),
            "hparams": {
                "input_dim": d,
                "hidden_dims": _HIDDEN,
                "raw_output_dim": _EMBEDDING_DIM + 1,
                "activation": "gelu",
            },
        },
        run_dir / "checkpoints" / "spectral_encoder.pt",
    )
    engine.spectral_embedder.calibration.save(run_dir / "memory" / "spectral_calibration.npz")
    engine.trajectory.save(run_dir / "memory" / "reference_trajectory.npz")

    atomic_torch_save(
        {
            "state_dict": engine.repulsion_field.state_dict(),
            "hparams": {
                "embedding_dim": _EMBEDDING_DIM,
                "hidden_dim": _REPULSION_HIDDEN,
                "n_residual_blocks": _REPULSION_BLOCKS,
                "time_embed_dim": _TIME_EMBED_DIM,
                "activation": "gelu",
            },
        },
        run_dir / "checkpoints" / "repulsion_field.pt",
    )

    cfg = Config(dataset=DatasetConfig(name="toy", input_dim=d))
    cfg.umap.n_neighbors = engine.cfg.n_neighbors
    cfg.umap.embedding_dim = _EMBEDDING_DIM
    cfg.umap.negative_sample_rate = engine.cfg.negative_sample_rate
    cfg.flow.n_steps = engine.cfg.n_steps
    cfg.flow.initial_alpha = engine.cfg.initial_alpha
    cfg.retriever.temperature = 0.1
    cfg.retriever.candidate_multiplier = 1
    cfg.retriever.min_candidates = engine.cfg.candidate_pool_size
    cfg.spectral.raw_output_dim = _EMBEDDING_DIM + 1
    cfg.eval.k = engine.cfg.n_neighbors  # keep Config.validate() happy (min_candidates >= eval.k)
    cfg.save(run_dir / "config.yaml")

    reloaded = InferenceEngine.load(run_dir, device=torch.device("cpu"))
    query = np.random.default_rng(7).normal(size=d).astype(np.float32)
    result = reloaded.embed_one(query)

    assert result.embedding.shape == (_EMBEDDING_DIM,)
    assert np.all(np.isfinite(result.embedding))
