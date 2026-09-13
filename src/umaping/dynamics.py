"""Reference mean UMAP dynamics (Algorithm 4, part 1; docs/method.md).

Builds the fixed reference trajectory ``Y_ref(t)`` once, offline, from the
symmetric graph W and a spectral initialization, by integrating the
*expected* UMAP field (analytic attraction over all graph edges + a
Monte-Carlo mean negative field) instead of replaying stochastic per-edge
SGD. Also hosts the force primitives shared verbatim with single-query
inference (``weighted_attraction``, ``mean_negative_field``, the alpha
schedule): the only things that differ at inference are the source (one
unseen point vs. all N reference points) and that the reference trajectory
itself is read-only there.

Time convention (used everywhere, training and inference alike): step
``e in {0, ..., n_steps}`` maps to normalized time ``t_e = e / n_steps in
[0, 1]``, and ``alpha_e = initial_alpha * (1 - t_e)`` is the learning rate
used for the Euler update from state ``e`` to state ``e + 1``:

    y_i^{e+1} = y_i^e + alpha_e * F_i^e
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from umaping.umap_forces import g_plus, g_minus


def alpha_schedule(step: int, n_steps: int, initial_alpha: float) -> float:
    return initial_alpha * (1.0 - step / n_steps)


def scatter_attraction(
    y: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    a: float,
    b: float,
    eps: float = 1e-12,
    clip: float | None = 4.0,
) -> torch.Tensor:
    """A_i = sum_j W_ij g_plus(y_i, y_j), via scatter-add over sparse graph
    edges (row, col, weight); no N x N matrix is ever formed."""
    contrib = weight.unsqueeze(-1) * g_plus(y[row], y[col], a, b, eps=eps, clip=clip)
    out = torch.zeros_like(y)
    out.index_add_(0, row, contrib)
    return out


def weighted_attraction(
    y_star: torch.Tensor,
    y_neighbors: torch.Tensor,
    weights: torch.Tensor,
    a: float,
    b: float,
    eps: float = 1e-12,
    clip: float | None = 4.0,
) -> torch.Tensor:
    """v_attr = sum_j mu_{*->j} g_plus(y*, y_j) for a single query point
    against its k retrieved neighbors -- the single-query analogue of
    scatter_attraction."""
    contrib = weights.unsqueeze(-1) * g_plus(y_star.unsqueeze(0), y_neighbors, a, b, eps=eps, clip=clip)
    return contrib.sum(0)


def mean_negative_field(
    y: torch.Tensor,
    y_candidates: torch.Tensor,
    a: float,
    b: float,
    eps: float = 1e-3,
    clip: float | None = 4.0,
) -> torch.Tensor:
    """B(y) = mean_c g_minus(y, y_c) over a shared candidate set, for every
    row of y at once (y: (d,) or (N, d); y_candidates: (M, d)). This is the
    O(N*M) mean-field approximation used in place of an O(N^2) pairwise
    repulsion matrix."""
    single = y.ndim == 1
    if single:
        y = y.unsqueeze(0)
    field = g_minus(y.unsqueeze(1), y_candidates.unsqueeze(0), a, b, eps=eps, clip=clip)  # (N, M, d)
    out = field.mean(dim=1)
    return out.squeeze(0) if single else out


def exact_all_reference_mean_field(
    y: torch.Tensor,
    trajectory: "ReferenceTrajectory",
    t: float | np.ndarray,
    a: float,
    b: float,
    device: torch.device,
    eps: float = 1e-3,
    clip: float | None = 4.0,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """The *exact* all-reference mean field, ``B_X(y, t) = mean_c g_minus(y, y_c(t))``
    over every reference point exactly once -- unlike `mean_negative_field`
    called on a randomly-drawn subset (as `InferenceEngine`'s ``oracle_mc``
    repulsion mode and the reference-dynamics/repulsion-training loops both
    do, which sample *with replacement* and therefore only approximate this
    quantity), this never samples: it is a deterministic function of the
    trajectory and (y, t) alone. Chunked over the reference axis so an
    ``(n_eval, N, d)`` tensor is never fully materialized for large ``N``.
    Used by `InferenceEngine`'s ``"exact"`` repulsion mode (an O(N)-per-query
    diagnostic baseline, not the default learned inference path) and by
    `evaluation/advanced.py`'s field-smoothness/denoising diagnostics.

    ``y``: ``(n_eval, d)``. ``t``: a single shared time, or a ``(n_eval,)``
    array of per-point times."""
    n = trajectory.positions.shape[1]
    n_eval = y.shape[0]
    t_arr = np.full(n_eval, float(t), dtype=np.float64) if np.isscalar(t) else np.asarray(t, dtype=np.float64)
    total = torch.zeros((n_eval, y.shape[-1]), dtype=torch.float32, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_n = end - start
        times_rep = np.repeat(t_arr, chunk_n)
        idx_rep = np.tile(np.arange(start, end), n_eval)
        chunk_positions = trajectory.positions_at_many(times_rep, idx_rep).reshape(n_eval, chunk_n, -1)
        chunk_positions_t = torch.as_tensor(chunk_positions, dtype=torch.float32, device=device)
        field = g_minus(y.unsqueeze(1), chunk_positions_t, a, b, eps=eps, clip=clip)  # (n_eval, chunk_n, d)
        total = total + field.sum(dim=1)

    return total / n


def simulate_reference_dynamics(
    y0: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    degree: torch.Tensor,
    a: float,
    b: float,
    negative_sample_rate: float,
    n_steps: int,
    initial_alpha: float,
    m_neg: int,
    generator: torch.Generator,
    eps_attr: float = 1e-12,
    eps_rep: float = 1e-3,
    grad_clip: float | None = 4.0,
    checkpoint_stride: int = 1,
) -> "ReferenceTrajectory":
    """Vectorized Euler integration of the expected UMAP field over the whole
    fixed reference set. Frozen once built: query inference only ever reads
    from the returned trajectory, never recomputes or mutates it."""
    n = y0.shape[0]
    y = y0.clone()
    checkpoint_steps = set(range(0, n_steps + 1, max(checkpoint_stride, 1))) | {0, n_steps}

    times: list[float] = []
    checkpoints: list[torch.Tensor] = []
    if 0 in checkpoint_steps:
        times.append(0.0)
        checkpoints.append(y.clone())

    for e in range(n_steps):
        alpha_e = alpha_schedule(e, n_steps, initial_alpha)
        neg_idx = torch.randint(0, n, (m_neg,), generator=generator).to(y.device)
        y_neg = y[neg_idx]

        # Clipped twice, deliberately: g_plus/g_minus already clip each
        # individual pairwise contribution (mirroring umap-learn's own
        # per-edge SGD clip), but a high-degree point's *summed* A_i can
        # still exceed the clip bound after summing many bounded terms --
        # this second clamp bounds the aggregate step size itself, and is
        # applied identically here and in inference.py's embed_one loop.
        a_term = scatter_attraction(y, row, col, weight, a, b, eps=eps_attr, clip=grad_clip)
        b_term = mean_negative_field(y, y_neg, a, b, eps=eps_rep, clip=grad_clip)
        force = a_term + negative_sample_rate * degree.unsqueeze(-1) * b_term
        if grad_clip is not None:
            force = force.clamp(min=-grad_clip, max=grad_clip)

        y = y + alpha_e * force

        if (e + 1) in checkpoint_steps:
            times.append((e + 1) / n_steps)
            checkpoints.append(y.clone())

    positions = torch.stack(checkpoints, dim=0).cpu().numpy().astype(np.float32)
    # float64, not float32: query-side code computes t_e = e / n_steps in plain
    # Python (float64) arithmetic, and exact-checkpoint lookups in `_bracket`
    # rely on that matching this array bit-for-bit rather than up to float32
    # rounding error.
    times_arr = np.asarray(times, dtype=np.float64)
    return ReferenceTrajectory(times=times_arr, positions=positions)


@dataclass
class ReferenceTrajectory:
    """Fixed reference memory: ``positions[e]`` is Y_ref at normalized time
    ``times[e]``. Frozen after `training/flow.py` builds it; inference only
    reads from it (never rebuilt or mutated per-query)."""

    times: np.ndarray  # (T,)
    positions: np.ndarray  # (T, N, embedding_dim)

    def _bracket(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.clip(t, self.times[0], self.times[-1])
        idx_hi = np.searchsorted(self.times, t, side="right")
        idx_hi = np.clip(idx_hi, 1, len(self.times) - 1)
        idx_lo = idx_hi - 1
        t_lo, t_hi = self.times[idx_lo], self.times[idx_hi]
        denom = np.where(t_hi == t_lo, 1.0, t_hi - t_lo)
        frac = np.where(t_hi == t_lo, 0.0, (t - t_lo) / denom)
        return idx_lo, idx_hi, frac

    def positions_at(self, t: float, indices: np.ndarray | None = None) -> np.ndarray:
        """Linear interpolation between stored checkpoints at a single shared
        time t in [0, 1] (exact lookup when every step was checkpointed, as
        is the default), for `indices` (or all N reference points if None).
        Used at inference: one query time step, many retrieved neighbors."""
        idx_lo, idx_hi, frac = self._bracket(np.asarray([t], dtype=np.float64))
        lo = self.positions[idx_lo[0]] if indices is None else self.positions[idx_lo[0], indices]
        hi = self.positions[idx_hi[0]] if indices is None else self.positions[idx_hi[0], indices]
        return lo + frac[0] * (hi - lo)

    def positions_at_many(self, times: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Vectorized, *paired* lookup: position of reference point
        `indices[i]` at time `times[i]`, independently per i. Used by the
        repulsion-field teacher generator, where every sampled example has
        its own random time."""
        times = np.asarray(times, dtype=np.float64)
        indices = np.asarray(indices)
        idx_lo, idx_hi, frac = self._bracket(times)
        lo = self.positions[idx_lo, indices]
        hi = self.positions[idx_hi, indices]
        return lo + frac[..., None] * (hi - lo)

    def save(self, path: str | Path) -> None:
        np.savez(path, times=self.times, positions=self.positions)

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceTrajectory":
        data = np.load(path)
        return cls(times=data["times"], positions=data["positions"])


class TorchTrajectoryView:
    """Device-resident view of a `ReferenceTrajectory`'s checkpoints, for
    training loops that need many per-step interpolated lookups against
    potentially large position arrays (currently: `train_repulsion_field`'s
    teacher generation, which looks up positions for a batch plus
    batch*teacher_negative_samples reference points every step). Built once;
    every subsequent lookup stays entirely on `device` -- only small index
    and time tensors ever need to already be there, never the interpolated
    position data itself round-tripping through host memory."""

    def __init__(self, trajectory: ReferenceTrajectory, device: torch.device):
        self.times = torch.as_tensor(trajectory.times, dtype=torch.float64, device=device)
        self.positions = torch.as_tensor(trajectory.positions, dtype=torch.float32, device=device)

    def positions_at_many(self, times: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Torch analogue of `ReferenceTrajectory.positions_at_many`: paired
        lookup, position of reference point `indices[i]` at time
        `times[i]`, independently per i. `times`/`indices` must already be
        on the same device as this view."""
        t = times.to(torch.float64).clamp(min=self.times[0], max=self.times[-1])
        idx_hi = torch.searchsorted(self.times, t, right=True)
        idx_hi = idx_hi.clamp(min=1, max=self.times.shape[0] - 1)
        idx_lo = idx_hi - 1
        t_lo = self.times[idx_lo]
        t_hi = self.times[idx_hi]
        denom = torch.where(t_hi == t_lo, torch.ones_like(t_hi), t_hi - t_lo)
        frac = torch.where(t_hi == t_lo, torch.zeros_like(t_hi), (t - t_lo) / denom).to(torch.float32)
        lo = self.positions[idx_lo, indices]
        hi = self.positions[idx_hi, indices]
        return lo + frac.unsqueeze(-1) * (hi - lo)
