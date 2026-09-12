"""Algorithm 4: reference mean UMAP dynamics + repulsion-field distillation
(docs/method.md).

Two stages, both pure functions of their inputs (no filesystem access; see
pipeline.py for checkpointing):

1. `build_reference_trajectory` integrates the deterministic expected-SGD
   field over the fixed reference set once, producing the frozen
   `ReferenceTrajectory` that both this module's teacher generator and
   inference.py's analytic attraction step read from.
2. `train_repulsion_field` distills the Monte-Carlo mean negative field
   B_X(y, t) into `RepulsionField`, sampling fresh (position, time, negative
   set) teacher examples directly from that trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import trange

from umaping.config import FlowConfig, RepulsionConfig
from umaping.dynamics import ReferenceTrajectory, simulate_reference_dynamics
from umaping.graph import full_edges, row_degree
from umaping.models.repulsion import RepulsionField
from umaping.umap_forces import g_minus
from umaping.utils.seed import make_generator


def build_reference_trajectory(
    y0: np.ndarray,
    w: sp.csr_matrix,
    a: float,
    b: float,
    negative_sample_rate: float,
    cfg: FlowConfig,
    device: torch.device,
    seed: int = 0,
) -> ReferenceTrajectory:
    row, col, weight = full_edges(w)
    degree = row_degree(w)

    y0_t = torch.as_tensor(np.asarray(y0, dtype=np.float32), device=device)
    row_t = torch.as_tensor(row, dtype=torch.long, device=device)
    col_t = torch.as_tensor(col, dtype=torch.long, device=device)
    weight_t = torch.as_tensor(weight, dtype=torch.float32, device=device)
    degree_t = torch.as_tensor(degree, dtype=torch.float32, device=device)
    generator = make_generator(seed)

    return simulate_reference_dynamics(
        y0=y0_t,
        row=row_t,
        col=col_t,
        weight=weight_t,
        degree=degree_t,
        a=a,
        b=b,
        negative_sample_rate=negative_sample_rate,
        n_steps=cfg.n_steps,
        initial_alpha=cfg.initial_alpha,
        m_neg=cfg.dynamics_negative_samples,
        generator=generator,
        grad_clip=cfg.grad_clip,
        checkpoint_stride=cfg.checkpoint_stride,
    )


@dataclass
class RepulsionTrainState:
    step: int = 0
    losses: list[float] = field(default_factory=list)


def train_repulsion_field(
    model: RepulsionField,
    trajectory: ReferenceTrajectory,
    a: float,
    b: float,
    cfg: RepulsionConfig,
    row_mass: np.ndarray,
    device: torch.device,
    seed: int = 0,
    resume_state: RepulsionTrainState | None = None,
    progress: bool = True,
    grad_clip: float | None = 4.0,
) -> RepulsionTrainState:
    """`grad_clip` should match the `flow.grad_clip` used to build
    `trajectory` (see `build_reference_trajectory`): the teacher target below
    is the same `g_minus` mean field the reference dynamics were actually
    integrated with, so a mismatched clip would train B_phi to imitate a
    field the trajectory never experienced."""
    n = trajectory.positions.shape[1]
    t_min, t_max = float(trajectory.times[0]), float(trajectory.times[-1])

    rng = np.random.default_rng(seed + (resume_state.step if resume_state else 0))
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    row_mass_t = torch.as_tensor(np.asarray(row_mass, dtype=np.float32), device=device)

    state = resume_state or RepulsionTrainState()
    iterator = trange(
        state.step, cfg.steps, disable=not progress, desc="repulsion", initial=state.step, total=cfg.steps
    )

    for step in iterator:
        b_idx = rng.integers(0, n, size=cfg.batch_size)
        t_vals = rng.uniform(t_min, t_max, size=cfg.batch_size).astype(np.float32)

        base_positions = trajectory.positions_at_many(t_vals, b_idx)
        jitter = rng.normal(scale=cfg.jitter_sigma, size=base_positions.shape).astype(np.float32)
        y_query = (base_positions + jitter).astype(np.float32)

        neg_idx = rng.integers(0, n, size=(cfg.batch_size, cfg.teacher_negative_samples))
        times_repeated = np.repeat(t_vals, cfg.teacher_negative_samples)
        neg_positions = trajectory.positions_at_many(times_repeated, neg_idx.reshape(-1))
        neg_positions = neg_positions.reshape(cfg.batch_size, cfg.teacher_negative_samples, -1)

        y_query_t = torch.as_tensor(y_query, device=device)
        t_t = torch.as_tensor(t_vals, device=device)
        neg_positions_t = torch.as_tensor(neg_positions, dtype=torch.float32, device=device)

        with torch.no_grad():
            # Teacher target: fresh, independent negative sample per example,
            # never backpropagated through (stop-gradient by construction).
            target = g_minus(y_query_t.unsqueeze(1), neg_positions_t, a, b, clip=grad_clip).mean(dim=1)

        pred = model(y_query_t, t_t)
        residual_sq = (pred - target) ** 2
        if cfg.weight_by_row_mass:
            w_row = row_mass_t[torch.as_tensor(b_idx, device=device)]
            loss = (residual_sq.sum(-1) * w_row).mean() / w_row.mean().clamp(min=1e-12)
        else:
            loss = residual_sq.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        state.step = step + 1
        state.losses.append(float(loss.item()))
        if progress and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            iterator.set_postfix(loss=float(loss.item()))

    return state
