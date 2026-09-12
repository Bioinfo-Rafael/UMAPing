"""Repulsive-field (B_phi) evaluation (docs/method.md, Evaluation section):
held-out (y, t) locations with freshly, independently sampled Monte-Carlo
teacher fields -- vector MSE/RMSE, cosine similarity, magnitude error."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from umaping.dynamics import ReferenceTrajectory
from umaping.models.repulsion import RepulsionField
from umaping.umap_forces import g_minus


@dataclass
class FieldMetrics:
    vector_mse: float
    vector_rmse: float
    mean_cosine_similarity: float
    mean_magnitude_error: float
    n_eval_points: int
    teacher_negative_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_repulsion_field(
    model: RepulsionField,
    trajectory: ReferenceTrajectory,
    a: float,
    b: float,
    n_eval_points: int,
    teacher_negative_samples: int,
    device: torch.device,
    seed: int = 0,
    jitter_sigma: float = 0.1,
    grad_clip: float | None = 4.0,
) -> tuple[FieldMetrics, pd.DataFrame]:
    """`grad_clip` should match the `flow.grad_clip` the reference trajectory
    (and B_phi's own training) used, so this evaluates against the same
    teacher field B_phi was actually distilled from."""
    rng = np.random.default_rng(seed)
    n = trajectory.positions.shape[1]
    t_min, t_max = float(trajectory.times[0]), float(trajectory.times[-1])

    b_idx = rng.integers(0, n, size=n_eval_points)
    t_vals = rng.uniform(t_min, t_max, size=n_eval_points).astype(np.float32)
    base = trajectory.positions_at_many(t_vals, b_idx)
    jitter = rng.normal(scale=jitter_sigma, size=base.shape).astype(np.float32)
    y_eval = (base + jitter).astype(np.float32)

    neg_idx = rng.integers(0, n, size=(n_eval_points, teacher_negative_samples))
    times_repeated = np.repeat(t_vals, teacher_negative_samples)
    neg_positions = trajectory.positions_at_many(times_repeated, neg_idx.reshape(-1))
    neg_positions = neg_positions.reshape(n_eval_points, teacher_negative_samples, -1)

    y_eval_t = torch.as_tensor(y_eval, dtype=torch.float32, device=device)
    t_t = torch.as_tensor(t_vals, dtype=torch.float32, device=device)
    neg_t = torch.as_tensor(neg_positions, dtype=torch.float32, device=device)

    model = model.to(device).eval()
    with torch.no_grad():
        target = g_minus(y_eval_t.unsqueeze(1), neg_t, a, b, clip=grad_clip).mean(dim=1)
        pred = model(y_eval_t, t_t)

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    residual = pred_np - target_np
    per_point_mse = (residual**2).sum(axis=1)
    pred_norm = np.linalg.norm(pred_np, axis=1)
    target_norm = np.linalg.norm(target_np, axis=1)
    denom = np.maximum(pred_norm * target_norm, 1e-12)
    cosine = (pred_np * target_np).sum(axis=1) / denom
    magnitude_error = np.abs(pred_norm - target_norm)

    df = pd.DataFrame(
        {
            "t": t_vals,
            "mse": per_point_mse,
            "cosine_similarity": cosine,
            "magnitude_error": magnitude_error,
            "pred_norm": pred_norm,
            "target_norm": target_norm,
        }
    )
    metrics = FieldMetrics(
        vector_mse=float(per_point_mse.mean()),
        vector_rmse=float(np.sqrt(per_point_mse.mean())),
        mean_cosine_similarity=float(np.mean(cosine)),
        mean_magnitude_error=float(np.mean(magnitude_error)),
        n_eval_points=n_eval_points,
        teacher_negative_samples=teacher_negative_samples,
    )
    return metrics, df
