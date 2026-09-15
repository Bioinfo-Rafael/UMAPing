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
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import trange

from umaping.config import FlowConfig, RepulsionConfig
from umaping.dynamics import ReferenceTrajectory, TorchTrajectoryView, simulate_reference_dynamics
from umaping.graph import full_edges, row_degree
from umaping.models.repulsion import RepulsionField
from umaping.umap_forces import g_minus
from umaping.utils.seed import make_generator

if TYPE_CHECKING:
    from umaping.repulsion_estimators.base import Teacher


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
    teacher: Teacher | None = None,
    query_sampler: Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    callback: Callable[[int, float, RepulsionField], None] | None = None,
    optimizer_state: dict | None = None,
    checkpoint_callback: Callable | None = None,
) -> RepulsionTrainState:
    """`grad_clip` should match the `flow.grad_clip` used to build
    `trajectory` (see `build_reference_trajectory`): the teacher target below
    is the same `g_minus` mean field the reference dynamics were actually
    integrated with, so a mismatched clip would train B_phi to imitate a
    field the trajectory never experienced. Optional ``teacher`` replaces
    only the target estimator; ``query_sampler(step)`` supplies a shared
    (anchor, time, jittered position) sequence and ``callback`` observes
    updates. With all three omitted, the original uniform RNG/order is kept.

    Position lookups happen through a device-resident `TorchTrajectoryView`
    (dynamics.py) rather than `ReferenceTrajectory`'s own numpy
    `positions_at_many`: this loop needs two such lookups (batch_size, and
    batch_size * teacher_negative_samples reference points) *every* step, so
    keeping the interpolation itself on-device avoids round-tripping large
    position arrays through host memory on every step -- only small index
    and time arrays (numpy, for exact reproducibility of which
    points/times are sampled) ever cross that boundary."""
    n = trajectory.positions.shape[1]
    t_min, t_max = float(trajectory.times[0]), float(trajectory.times[-1])
    traj_view = TorchTrajectoryView(trajectory, device)

    rng = np.random.default_rng(seed + (resume_state.step if resume_state else 0))
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    row_mass_t = torch.as_tensor(np.asarray(row_mass, dtype=np.float32), device=device)

    state = resume_state or RepulsionTrainState()
    iterator = trange(
        state.step, cfg.steps, disable=not progress, desc="repulsion", initial=state.step, total=cfg.steps
    )

    for step in iterator:
        if query_sampler is None:
            b_idx = torch.as_tensor(rng.integers(0, n, size=cfg.batch_size), dtype=torch.long, device=device)
            t_vals = torch.as_tensor(rng.uniform(t_min, t_max, size=cfg.batch_size), dtype=torch.float32, device=device)
            base_positions = traj_view.positions_at_many(t_vals, b_idx)
            y_query_t = base_positions + torch.randn_like(base_positions) * cfg.jitter_sigma
        else:
            b_idx, t_vals, y_query_t = query_sampler(step)

        with torch.no_grad():
            # Teacher target: fresh, independent negative sample per example,
            # never backpropagated through (stop-gradient by construction).
            if teacher is None:
                neg_idx = torch.as_tensor(
                    rng.integers(0, n, size=cfg.batch_size * cfg.teacher_negative_samples), dtype=torch.long, device=device
                )
                times_repeated = t_vals.repeat_interleave(cfg.teacher_negative_samples)
                neg_positions_t = traj_view.positions_at_many(times_repeated, neg_idx).reshape(
                    cfg.batch_size, cfg.teacher_negative_samples, -1
                )
                target = g_minus(y_query_t.unsqueeze(1), neg_positions_t, a, b, clip=grad_clip).mean(dim=1)
            else:
                target = teacher.estimate(y_query_t, t_vals, b_idx).detach()
                if target.shape != y_query_t.shape or not torch.isfinite(target).all():
                    raise FloatingPointError('Invalid repulsion teacher target')

        pred = model(y_query_t, t_vals)
        residual_sq = (pred - target) ** 2
        if cfg.weight_by_row_mass:
            w_row = row_mass_t[b_idx]
            loss = (residual_sq.sum(-1) * w_row).mean() / w_row.mean().clamp(min=1e-12)
        else:
            loss = residual_sq.mean()
        if teacher is not None and not torch.isfinite(loss):
            raise FloatingPointError('Non-finite repulsion training loss')

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        state.step = step + 1
        state.losses.append(float(loss.item()))
        if callback is not None:
            callback(state.step, float(loss.item()), model)
        if checkpoint_callback is not None:
            checkpoint_callback(state, model, optimizer)
        if progress and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            iterator.set_postfix(loss=float(loss.item()))

    return state
