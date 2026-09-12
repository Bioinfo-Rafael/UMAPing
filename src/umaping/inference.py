"""Single-point inference (Algorithm 1, Section 7; docs/method.md).

`InferenceEngine.embed_one(x_star)` is the only contract that matters here:
its result must never depend on any other query point, and the fixed
reference trajectory is read-only. There is no query-query graph, attention,
or message passing anywhere in this module -- batch convenience methods are
pure vectorizations of independent per-point work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from umaping.config import Config
from umaping.dynamics import ReferenceTrajectory, alpha_schedule, mean_negative_field, weighted_attraction
from umaping.graph import chunked_exact_knn, query_fuzzy_weights
from umaping.models.repulsion import RepulsionField
from umaping.models.retriever import DualEncoder, NeighborIndex, build_neighbor_index
from umaping.models.spectral import SpectralCalibration, SpectralEmbedder, SpectralEncoderNet
from umaping.umap_forces import find_ab_params


@dataclass
class InferenceConfig:
    n_neighbors: int
    candidate_pool_size: int
    negative_sample_rate: float
    n_steps: int
    initial_alpha: float
    a: float
    b: float
    local_connectivity: float = 1.0
    smooth_knn_n_iter: int = 64
    smooth_knn_bandwidth: float = 1.0
    smooth_knn_min_k_dist_scale: float = 1e-3
    eps: float = 1e-8
    grad_clip: float | None = 4.0


@dataclass
class QueryResult:
    embedding: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_weights: np.ndarray
    trajectory: np.ndarray | None = None


class InferenceEngine:
    """Bundles the frozen reference memory + trained models needed for
    single-point embedding. Everything is loaded once at construction;
    `embed_one` afterwards touches only `x_star` plus this frozen state --
    never the training-time graph, never another query."""

    def __init__(
        self,
        retriever: DualEncoder,
        neighbor_index: NeighborIndex,
        reference_features: np.ndarray,
        spectral_embedder: SpectralEmbedder,
        reference_trajectory: ReferenceTrajectory,
        repulsion_field: RepulsionField,
        cfg: InferenceConfig,
        device: torch.device,
        neighbor_source: Literal["learned", "oracle"] = "learned",
        repulsion_mode: Literal["learned", "oracle_mc"] = "learned",
        use_repulsion: bool = True,
        oracle_mc_samples: int = 2048,
        seed: int = 0,
    ) -> None:
        self.retriever = retriever.to(device).eval()
        self.neighbor_index = neighbor_index
        self.reference_features = np.ascontiguousarray(reference_features, dtype=np.float32)
        self.spectral_embedder = spectral_embedder
        self.trajectory = reference_trajectory
        self.repulsion_field = repulsion_field.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.neighbor_source = neighbor_source
        self.repulsion_mode = repulsion_mode
        self.use_repulsion = use_repulsion
        self.oracle_mc_samples = oracle_mc_samples
        # NOT a stateful `self._rng`: a generator stored on the engine and
        # advanced across calls would make embed_one's oracle_mc sampling
        # depend on how many other embed_one calls (for other queries) ran
        # earlier on this same engine -- a real query-query dependency.
        # embed_one instead builds a fresh generator from this seed at the
        # start of every call (see below).
        self._seed = seed

    # -- candidate retrieval -------------------------------------------------

    def retrieve_neighbors(self, x_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exactly k neighbor (ids, Euclidean distances), already reranked.
        Public (not just an embed_one implementation detail) because
        evaluation/retrieval.py needs this stage in isolation."""
        if self.neighbor_source == "oracle":
            idx, dist = chunked_exact_knn(x_star[None, :], self.reference_features, k=self.cfg.n_neighbors)
            return idx[0], dist[0]

        with torch.no_grad():
            x_t = torch.as_tensor(x_star[None, :], dtype=torch.float32, device=self.device)
            q = self.retriever.encode_query(x_t).cpu().numpy()
        candidate_ids, _ = self.neighbor_index.search(q, self.cfg.candidate_pool_size)
        candidate_ids = candidate_ids[0]
        candidate_ids = candidate_ids[candidate_ids >= 0]

        diff = self.reference_features[candidate_ids] - x_star[None, :]
        dists = np.sqrt(np.maximum((diff * diff).sum(axis=1), 0.0))
        order = np.argsort(dists, kind="stable")[: self.cfg.n_neighbors]
        return candidate_ids[order], dists[order]

    # -- repulsion ------------------------------------------------------------

    def _repulsion(self, y_star: torch.Tensor, t: float, s_star: float, rng: np.random.Generator) -> torch.Tensor:
        if not self.use_repulsion:
            return torch.zeros_like(y_star)

        if self.repulsion_mode == "learned":
            t_t = torch.as_tensor([t], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                b_phi = self.repulsion_field(y_star.unsqueeze(0), t_t)[0]
        elif self.repulsion_mode == "oracle_mc":
            n = self.trajectory.positions.shape[1]
            m = min(self.oracle_mc_samples, n)
            idx = rng.integers(0, n, size=m)
            y_neg = torch.as_tensor(
                self.trajectory.positions_at(t, indices=idx), dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                b_phi = mean_negative_field(y_star, y_neg, self.cfg.a, self.cfg.b, clip=self.cfg.grad_clip)
        else:
            raise ValueError(f"Unknown repulsion_mode '{self.repulsion_mode}'")

        return self.cfg.negative_sample_rate * s_star * b_phi

    # -- public contract --------------------------------------------------

    def embed_one(self, x_star: np.ndarray, return_trajectory: bool = False) -> QueryResult:
        """`x_star` must already be in the fixed preprocessed feature space
        (the same PCA/HVG transform fit on the reference set). Never reads
        any other query point; the reference trajectory is read-only."""
        x_star = np.asarray(x_star, dtype=np.float32).reshape(-1)
        if x_star.shape[0] != self.reference_features.shape[1]:
            raise ValueError(
                f"x_star has dim {x_star.shape[0]}, but this model instance expects "
                f"{self.reference_features.shape[1]} (fixed per-dataset input dimensionality)."
            )

        neighbor_ids, neighbor_dists = self.retrieve_neighbors(x_star)
        weights, _, _ = query_fuzzy_weights(
            neighbor_dists[None, :],
            local_connectivity=self.cfg.local_connectivity,
            n_iter=self.cfg.smooth_knn_n_iter,
            bandwidth=self.cfg.smooth_knn_bandwidth,
            min_k_dist_scale=self.cfg.smooth_knn_min_k_dist_scale,
            eps=self.cfg.eps,
        )
        weights = weights[0]
        s_star = float(weights.sum())

        y_star = torch.as_tensor(self.spectral_embedder.embed_one(x_star), dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        # Fresh, local to this call: reseeded from the engine's fixed base
        # seed every time, so oracle_mc sampling is reproducible per query
        # and never depends on what other queries were embedded earlier.
        rng = np.random.default_rng(self._seed)

        traj_record = [y_star.detach().cpu().numpy().copy()] if return_trajectory else None

        for e in range(self.cfg.n_steps):
            t_e = e / self.cfg.n_steps
            alpha_e = alpha_schedule(e, self.cfg.n_steps, self.cfg.initial_alpha)

            y_neighbors = torch.as_tensor(
                self.trajectory.positions_at(t_e, indices=neighbor_ids), dtype=torch.float32, device=self.device
            )
            v_attr = weighted_attraction(y_star, y_neighbors, weights_t, self.cfg.a, self.cfg.b, clip=self.cfg.grad_clip)
            v_rep = self._repulsion(y_star, t_e, s_star, rng)
            v = v_attr + v_rep
            # Clipped twice, deliberately -- see the matching comment in
            # dynamics.py::simulate_reference_dynamics (same reasoning,
            # applied identically here for the single query point).
            if self.cfg.grad_clip is not None:
                v = v.clamp(min=-self.cfg.grad_clip, max=self.cfg.grad_clip)

            y_star = y_star + alpha_e * v
            if return_trajectory:
                traj_record.append(y_star.detach().cpu().numpy().copy())

        embedding = y_star.detach().cpu().numpy()
        if not np.all(np.isfinite(embedding)):
            raise FloatingPointError(f"Non-finite embedding produced for a query point: {embedding}")

        trajectory_arr = np.stack(traj_record, axis=0) if return_trajectory else None
        return QueryResult(embedding=embedding, neighbor_ids=neighbor_ids, neighbor_weights=weights, trajectory=trajectory_arr)

    def embed_spectral_only(self, x_star: np.ndarray) -> np.ndarray:
        """Baseline #2: y* = SpectralEncoder(x*), no flow integration."""
        x_star = np.asarray(x_star, dtype=np.float32).reshape(-1)
        return self.spectral_embedder.embed_one(x_star)

    # -- construction from a run directory ---------------------------------

    @classmethod
    def load(cls, run_dir: str | Path, device: torch.device, **mode_kwargs) -> "InferenceEngine":
        run_dir = Path(run_dir)
        cfg = Config.load(run_dir / "config.yaml")

        reference_features = np.load(run_dir / "memory" / "reference_features.npy")

        retriever_ckpt = torch.load(run_dir / "checkpoints" / "retriever.pt", map_location="cpu", weights_only=False)
        retriever = DualEncoder(**retriever_ckpt["hparams"])
        retriever.load_state_dict(retriever_ckpt["state_dict"])

        retriever_keys = np.load(run_dir / "memory" / "retriever_keys.npy")
        neighbor_index = build_neighbor_index(
            cfg.retriever.ann_backend, retriever_keys, temperature=cfg.retriever.temperature, device=str(device)
        )

        spectral_ckpt = torch.load(run_dir / "checkpoints" / "spectral_encoder.pt", map_location="cpu", weights_only=False)
        spectral_net = SpectralEncoderNet(**spectral_ckpt["hparams"])
        spectral_net.load_state_dict(spectral_ckpt["state_dict"])
        calibration = SpectralCalibration.load(run_dir / "memory" / "spectral_calibration.npz")
        spectral_embedder = SpectralEmbedder(spectral_net, calibration, device=str(device))

        trajectory = ReferenceTrajectory.load(run_dir / "memory" / "reference_trajectory.npz")

        repulsion_ckpt = torch.load(run_dir / "checkpoints" / "repulsion_field.pt", map_location="cpu", weights_only=False)
        repulsion_field = RepulsionField(**repulsion_ckpt["hparams"])
        repulsion_field.load_state_dict(repulsion_ckpt["state_dict"])

        a, b = find_ab_params(cfg.umap.spread, cfg.umap.min_dist)
        inf_cfg = InferenceConfig(
            n_neighbors=cfg.umap.n_neighbors,
            candidate_pool_size=cfg.candidate_pool_size(),
            negative_sample_rate=cfg.umap.negative_sample_rate,
            n_steps=cfg.flow.n_steps,
            initial_alpha=cfg.flow.initial_alpha,
            a=a,
            b=b,
            local_connectivity=cfg.umap.local_connectivity,
            smooth_knn_n_iter=cfg.umap.smooth_knn_n_iter,
            smooth_knn_bandwidth=cfg.umap.smooth_knn_bandwidth,
            smooth_knn_min_k_dist_scale=cfg.umap.smooth_knn_min_k_dist_scale,
            eps=cfg.umap.eps,
            grad_clip=cfg.flow.grad_clip,
        )
        return cls(
            retriever=retriever,
            neighbor_index=neighbor_index,
            reference_features=reference_features,
            spectral_embedder=spectral_embedder,
            reference_trajectory=trajectory,
            repulsion_field=repulsion_field,
            cfg=inf_cfg,
            device=device,
            **mode_kwargs,
        )
