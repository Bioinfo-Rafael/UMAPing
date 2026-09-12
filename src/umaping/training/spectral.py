"""Algorithm 3: Spectral Encoder training (docs/method.md).

Trains a SpectralNet-style pointwise encoder on the *symmetric* graph W by
minimizing the graph Dirichlet energy over sampled edge minibatches, with a
soft orthogonality penalty encouraging ``Z^T Z / N ~= I``. After training,
`compute_calibration` performs the reference-only post-hoc steps (exact
whitening, projecting onto the small operator H's non-trivial eigenvectors,
scale calibration) that turn the raw network into a deployable
`x -> y0` map with no query-time graph access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import trange

from umaping.config import SpectralConfig
from umaping.graph import normalized_laplacian, row_degree, upper_triangular_edges
from umaping.models.mlp import batched_forward
from umaping.models.spectral import SpectralCalibration, SpectralEncoderNet


@dataclass
class SpectralTrainState:
    step: int = 0
    dirichlet_losses: list[float] = field(default_factory=list)
    ortho_losses: list[float] = field(default_factory=list)
    total_losses: list[float] = field(default_factory=list)


def train_spectral(
    model: SpectralEncoderNet,
    features: np.ndarray,
    w: sp.csr_matrix,
    cfg: SpectralConfig,
    device: torch.device,
    seed: int = 0,
    resume_state: SpectralTrainState | None = None,
    progress: bool = True,
) -> SpectralTrainState:
    features_t = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
    degree = torch.as_tensor(row_degree(w), dtype=torch.float32, device=device)
    inv_sqrt_degree = 1.0 / torch.sqrt(degree.clamp(min=1e-12))

    row, col, weight = upper_triangular_edges(w)
    row_t = torch.as_tensor(row, dtype=torch.long, device=device)
    col_t = torch.as_tensor(col, dtype=torch.long, device=device)
    weight_t = torch.as_tensor(weight, dtype=torch.float32, device=device)
    n_edges = row_t.shape[0]
    if n_edges == 0:
        raise ValueError("Symmetric graph W has no edges; cannot train the spectral encoder.")

    rng = np.random.default_rng(seed + (resume_state.step if resume_state else 0))
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    state = resume_state or SpectralTrainState()
    r = cfg.raw_output_dim
    eye_r = torch.eye(r, device=device)
    batch_size = min(cfg.edge_batch_size, n_edges)

    iterator = trange(
        state.step, cfg.steps, disable=not progress, desc="spectral", initial=state.step, total=cfg.steps
    )
    for step in iterator:
        batch_edges = rng.integers(0, n_edges, size=batch_size)
        b_row = row_t[batch_edges]
        b_col = col_t[batch_edges]
        b_w = weight_t[batch_edges]

        unique_nodes, inverse = torch.unique(torch.cat([b_row, b_col]), return_inverse=True)
        z_unique = model(features_t[unique_nodes])  # (U, r), raw network output
        inv_row, inv_col = inverse[: b_row.shape[0]], inverse[b_row.shape[0] :]

        z_i = z_unique[inv_row] * inv_sqrt_degree[b_row].unsqueeze(-1)
        z_j = z_unique[inv_col] * inv_sqrt_degree[b_col].unsqueeze(-1)
        diff = z_i - z_j
        dirichlet = (b_w * (diff * diff).sum(-1)).mean()

        # Soft penalty only: normalized by this batch's unique (degree-biased,
        # since nodes come from sampled edges) node count, not the true N --
        # a cheap stochastic proxy for the N-normalized target. It only needs
        # to keep training from collapsing; compute_calibration() below
        # re-whitens exactly against the true N afterwards regardless.
        u = unique_nodes.shape[0]
        gram = (z_unique.T @ z_unique) / u
        ortho = ((gram - eye_r) ** 2).sum()

        loss = dirichlet + cfg.orthogonality_weight * ortho

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        state.step = step + 1
        state.dirichlet_losses.append(float(dirichlet.item()))
        state.ortho_losses.append(float(ortho.item()))
        state.total_losses.append(float(loss.item()))
        if progress and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            iterator.set_postfix(dirichlet=float(dirichlet.item()), ortho=float(ortho.item()))

    return state


def compute_calibration(
    model: SpectralEncoderNet,
    features: np.ndarray,
    w: sp.csr_matrix,
    embedding_dim: int,
    calibration_target_scale: float,
    device: torch.device,
) -> tuple[SpectralCalibration, np.ndarray]:
    """Reference-only post-hoc steps: whiten raw outputs so Z^TZ/N is exactly
    I, form H = Z^T L_sym Z, discard the trivial mode, retain the next
    `embedding_dim` modes, and calibrate scale. Returns the calibration
    object plus the calibrated reference embedding it implies."""
    raw = batched_forward(model, features, device)  # (N, r)
    n, r = raw.shape

    cov = (raw.T @ raw) / n
    eigval_c, eigvec_c = np.linalg.eigh(cov)
    eigval_c = np.clip(eigval_c, 1e-10, None)
    whitening = eigvec_c @ np.diag(1.0 / np.sqrt(eigval_c))  # (r, r): Z=raw@whitening has Z^TZ/N = I
    z = raw @ whitening

    l_sym = normalized_laplacian(w, row_degree(w))
    h = z.T @ (l_sym @ z)
    h = (h + h.T) / 2.0  # enforce exact numerical symmetry before eigh

    eigval_h, eigvec_h = np.linalg.eigh(h)  # ascending
    if r < embedding_dim + 1:
        raise ValueError(f"raw_output_dim={r} must be >= embedding_dim + 1 ({embedding_dim + 1})")
    discarded_eigenvalue = float(eigval_h[0])
    retained_eigenvalues = eigval_h[1 : 1 + embedding_dim].astype(np.float32)
    projection = eigvec_h[:, 1 : 1 + embedding_dim].astype(np.float32)

    y0_raw = z @ projection
    mean = y0_raw.mean(axis=0).astype(np.float32)
    centered = y0_raw - mean
    max_abs = float(np.abs(centered).max())
    scale = calibration_target_scale / max(max_abs, 1e-12)

    calibration = SpectralCalibration(
        whitening=whitening.astype(np.float32),
        projection=projection,
        mean=mean,
        scale=float(scale),
        retained_eigenvalues=retained_eigenvalues,
        discarded_eigenvalue=discarded_eigenvalue,
    )
    calibrated_reference_embedding = (centered * scale).astype(np.float32)
    return calibration, calibrated_reference_embedding
