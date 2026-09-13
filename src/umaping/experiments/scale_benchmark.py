"""Scale benchmark (spec Section 8): offline vs. online cost as the
reference set size N grows.

Uses the synthetic mock dataset (`data/mock.py`) at increasing N -- no real
dataset is downloaded or trained at scale locally, and this is explicitly
*not* meant to claim O(1) inference: reference-neighbor retrieval is still
O(N) per query in the current default exact backend (`models/retriever.py::
ExactChunkedIndex`). The point is to measure actual wall-clock scaling, not
assert a complexity class ahead of time.

Offline cost is broken down per training stage (mirroring
`pipeline.py::run_training`'s own stages); online cost is broken down across
the sub-steps `InferenceEngine`'s public API actually exposes as separable
calls (query encoding, candidate search, the full retrieve-and-rerank call,
spectral encoding, and the full `embed_one` call, which is the only number
that also includes the Euler flow-integration loop -- that loop is not
separately instrumented, since isolating it would mean adding benchmarking
-specific instrumentation to the stable `embed_one` inference path itself).
"""

from __future__ import annotations

import logging
import platform
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from umaping.config import FlowConfig, RepulsionConfig, RetrieverConfig, SpectralConfig
from umaping.data.mock import prepare_mock_dataset
from umaping.graph import build_reference_graph, row_degree
from umaping.inference import InferenceConfig, InferenceEngine
from umaping.models.mlp import batched_forward
from umaping.models.repulsion import RepulsionField
from umaping.models.retriever import DualEncoder, build_neighbor_index
from umaping.models.spectral import SpectralEmbedder, SpectralEncoderNet
from umaping.training.flow import build_reference_trajectory, train_repulsion_field
from umaping.training.retriever import train_retriever
from umaping.training.spectral import compute_calibration, train_spectral
from umaping.umap_forces import find_ab_params
from umaping.utils.io import save_json
from umaping.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB. `ru_maxrss` is
    KB on Linux but bytes on macOS (both documented `getrusage` platform
    quirks) -- converted here so the reported unit is consistent."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024.0 * 1024.0) if platform.system() == "Darwin" else raw / 1024.0


@dataclass
class ScaleBenchmarkPoint:
    n_reference: int
    n_query: int
    preprocessing_seconds: float
    graph_construction_seconds: float
    retriever_training_seconds: float
    spectral_training_seconds: float
    reference_dynamics_seconds: float
    repulsion_training_seconds: float
    offline_total_seconds: float
    query_encode_seconds_per_query: float
    candidate_search_seconds_per_query: float
    retrieve_and_rerank_seconds_per_query: float
    spectral_encode_seconds_per_query: float
    full_embed_one_seconds_per_query: float
    peak_rss_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_scale_benchmark(
    n_reference_values: tuple[int, ...] = (200, 500, 1000, 2000),
    n_query: int = 50,
    input_dim: int = 16,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    ann_backend: str = "exact",
    steps: int = 30,
) -> list[ScaleBenchmarkPoint]:
    points: list[ScaleBenchmarkPoint] = []

    for n_ref in n_reference_values:
        set_seed(seed)
        n_neighbors = min(15, max(2, n_ref - 1))

        t0 = time.perf_counter()
        prepared = prepare_mock_dataset(n_reference=n_ref, n_query=n_query, input_dim=input_dim, n_clusters=5, seed=seed)
        t_preprocess = time.perf_counter() - t0

        t0 = time.perf_counter()
        graph = build_reference_graph(prepared.reference_features, n_neighbors=n_neighbors)
        t_graph = time.perf_counter() - t0

        retriever_cfg = RetrieverConfig(
            hidden_dims=[32], retrieval_dim=16, batch_size=min(32, n_ref), n_random_negatives=min(16, n_ref),
            steps=steps, log_every=10_000, ann_backend=ann_backend, candidate_multiplier=4, min_candidates=min(64, n_ref),
        )
        retriever = DualEncoder(
            input_dim=input_dim, hidden_dims=retriever_cfg.hidden_dims, retrieval_dim=retriever_cfg.retrieval_dim,
            temperature=retriever_cfg.temperature,
        )
        t0 = time.perf_counter()
        train_retriever(retriever, prepared.reference_features, graph.mu, retriever_cfg, device, seed=seed, progress=False)
        t_retriever = time.perf_counter() - t0

        raw_keys = batched_forward(retriever.key_encoder, prepared.reference_features, device)
        keys = (raw_keys / np.maximum(np.linalg.norm(raw_keys, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        neighbor_index = build_neighbor_index(
            ann_backend, keys, temperature=retriever_cfg.temperature, device=str(device), seed=seed
        )

        spectral_cfg = SpectralConfig(
            hidden_dims=[32], raw_output_dim=3, steps=steps, edge_batch_size=min(256, max(graph.w.nnz, 1)), log_every=10_000
        )
        spectral_model = SpectralEncoderNet(
            input_dim=input_dim, hidden_dims=spectral_cfg.hidden_dims, raw_output_dim=spectral_cfg.raw_output_dim
        )
        t0 = time.perf_counter()
        train_spectral(spectral_model, prepared.reference_features, graph.w, spectral_cfg, device, seed=seed, progress=False)
        t_spectral = time.perf_counter() - t0

        calibration, reference_embedding = compute_calibration(
            spectral_model, prepared.reference_features, graph.w, embedding_dim=2, calibration_target_scale=10.0, device=device
        )
        spectral_embedder = SpectralEmbedder(spectral_model, calibration, device=str(device))

        a, b = find_ab_params(spread=1.0, min_dist=0.1)
        flow_cfg = FlowConfig(n_steps=max(10, steps), dynamics_negative_samples=min(16, n_ref))
        t0 = time.perf_counter()
        trajectory = build_reference_trajectory(
            reference_embedding, graph.w, a, b, negative_sample_rate=5.0, cfg=flow_cfg, device=device, seed=seed
        )
        t_dynamics = time.perf_counter() - t0

        repulsion_cfg = RepulsionConfig(
            hidden_dim=32, n_residual_blocks=2, time_embed_dim=8, steps=steps, batch_size=min(32, n_ref),
            teacher_negative_samples=min(16, n_ref), log_every=10_000,
        )
        repulsion_model = RepulsionField(
            embedding_dim=2, hidden_dim=repulsion_cfg.hidden_dim, n_residual_blocks=repulsion_cfg.n_residual_blocks,
            time_embed_dim=repulsion_cfg.time_embed_dim,
        )
        row_mass = row_degree(graph.w)
        t0 = time.perf_counter()
        train_repulsion_field(repulsion_model, trajectory, a, b, repulsion_cfg, row_mass, device, seed=seed, progress=False)
        t_repulsion = time.perf_counter() - t0

        offline_total = t_preprocess + t_graph + t_retriever + t_spectral + t_dynamics + t_repulsion

        inf_cfg = InferenceConfig(
            n_neighbors=n_neighbors, candidate_pool_size=retriever_cfg.min_candidates, negative_sample_rate=5.0,
            n_steps=flow_cfg.n_steps, initial_alpha=1.0, a=a, b=b,
        )
        engine = InferenceEngine(
            retriever=retriever, neighbor_index=neighbor_index, reference_features=prepared.reference_features,
            spectral_embedder=spectral_embedder, reference_trajectory=trajectory, repulsion_field=repulsion_model,
            cfg=inf_cfg, device=device, seed=seed,
        )

        query_encode_t, candidate_search_t, retrieve_rerank_t, spectral_t, full_t = [], [], [], [], []
        for q in prepared.query_features:
            t0 = time.perf_counter()
            with torch.no_grad():
                x_t = torch.as_tensor(q[None, :], dtype=torch.float32, device=device)
                qv = engine.retriever.encode_query(x_t).cpu().numpy()
            query_encode_t.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            engine.neighbor_index.search(qv, engine.cfg.candidate_pool_size)
            candidate_search_t.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            engine.retrieve_neighbors(q)
            retrieve_rerank_t.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            engine.spectral_embedder.embed_one(q)
            spectral_t.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            engine.embed_one(q)
            full_t.append(time.perf_counter() - t0)

        point = ScaleBenchmarkPoint(
            n_reference=n_ref,
            n_query=n_query,
            preprocessing_seconds=t_preprocess,
            graph_construction_seconds=t_graph,
            retriever_training_seconds=t_retriever,
            spectral_training_seconds=t_spectral,
            reference_dynamics_seconds=t_dynamics,
            repulsion_training_seconds=t_repulsion,
            offline_total_seconds=offline_total,
            query_encode_seconds_per_query=float(np.mean(query_encode_t)),
            candidate_search_seconds_per_query=float(np.mean(candidate_search_t)),
            retrieve_and_rerank_seconds_per_query=float(np.mean(retrieve_rerank_t)),
            spectral_encode_seconds_per_query=float(np.mean(spectral_t)),
            full_embed_one_seconds_per_query=float(np.mean(full_t)),
            peak_rss_mb=_peak_rss_mb(),
        )
        points.append(point)
        logger.info(
            "n_reference=%d: offline_total=%.3fs, full_embed_one=%.5fs/query, peak_rss=%.1fMB",
            n_ref, offline_total, point.full_embed_one_seconds_per_query, point.peak_rss_mb,
        )

    return points


def run_and_save_scale_benchmark(output_dir: str | Path, **kwargs) -> list[ScaleBenchmarkPoint]:
    from umaping.evaluation.plotting import plot_memory_vs_n, plot_offline_cost_vs_n, plot_query_latency_vs_n

    output_dir = Path(output_dir)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    points = run_scale_benchmark(**kwargs)
    payload = [p.to_dict() for p in points]
    save_json(output_dir / "metrics" / "scale_benchmark.json", payload)

    plot_query_latency_vs_n(payload, output_dir / "figures" / "scale_query_latency_vs_n.png")
    plot_memory_vs_n(payload, output_dir / "figures" / "scale_memory_vs_n.png")
    plot_offline_cost_vs_n(payload, output_dir / "figures" / "scale_offline_cost_vs_n.png")
    return points
