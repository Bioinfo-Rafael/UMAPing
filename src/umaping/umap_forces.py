"""The low-dimensional UMAP kernel and its analytic attractive/repulsive
gradients (Algorithm 1 / Section 4 of docs/method.md).

    Phi(y, z) = 1 / (1 + a * ||y - z||^(2b)) = 1 / (1 + a * q^b),  q = ||y-z||^2

``find_ab_params`` and the gradient-coefficient formulas below were verified
against the current umap-learn source (v0.5.12, ``umap/umap_.py`` and
``umap/layouts.py``; see RUNTIME_CHECKS.md): the attractive coefficient
``-2ab*q^(b-1)/(1+a*q^b)``, the repulsive coefficient
``2b/((q+eps)*(1+a*q^b))`` with ``eps=0.001``, and elementwise clipping of the
final per-dimension force to ``[-4, 4]`` all match upstream exactly. Only the
*aggregation* differs from upstream: umap-learn applies this per sampled edge
inside asynchronous SGD, whereas here it is summed/averaged over sparse graph
edges or Monte-Carlo negative samples to build a deterministic mean field
(dynamics.py), so clipping is applied per-pair here too, before aggregation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import curve_fit


def find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    """Fit (a, b) so Phi matches the umap-learn target curve: 1.0 below
    min_dist, exp(-(x-min_dist)/spread) beyond it. Reimplemented (rather than
    imported from umap.umap_) to avoid depending on a private API; verified
    line-for-line identical to the current upstream implementation."""

    def curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros_like(xv)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


def squared_distance(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    diff = y - z
    return (diff * diff).sum(-1)


# --- Autograd-friendly forms, used only by tests / analytic sanity checks to
# verify the closed-form g_plus/g_minus below against numerical autograd
# derivatives. Not on any training/inference hot path. ---------------------


def phi(y: torch.Tensor, z: torch.Tensor, a: float, b: float, eps: float = 1e-12) -> torch.Tensor:
    q = squared_distance(y, z).clamp(min=eps)
    return 1.0 / (1.0 + a * q**b)


def log_phi(y: torch.Tensor, z: torch.Tensor, a: float, b: float, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(phi(y, z, a, b, eps=eps) + eps)


def log_one_minus_phi(y: torch.Tensor, z: torch.Tensor, a: float, b: float, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(1.0 - phi(y, z, a, b, eps=eps) + eps)


# --- Closed-form directions used everywhere else (dynamics.py, inference.py) ---


def g_plus(
    y: torch.Tensor,
    z: torch.Tensor,
    a: float,
    b: float,
    eps: float = 1e-12,
    clip: float | None = 4.0,
) -> torch.Tensor:
    """grad_y log Phi(y,z) = -2ab q^(b-1) / (1 + a q^b) * (y - z)."""
    diff = y - z
    q = squared_distance(y, z).clamp(min=0.0).unsqueeze(-1)
    q_safe = q.clamp(min=eps)
    coeff = -2.0 * a * b * q_safe.pow(b - 1.0) / (1.0 + a * q_safe.pow(b))
    coeff = torch.where(q > 0.0, coeff, torch.zeros_like(coeff))
    out = coeff * diff
    if clip is not None:
        out = out.clamp(min=-clip, max=clip)
    return out


def g_minus(
    y: torch.Tensor,
    z: torch.Tensor,
    a: float,
    b: float,
    eps: float = 1e-3,
    clip: float | None = 4.0,
) -> torch.Tensor:
    """grad_y log(1 - Phi(y,z)) ~= 2b / ((q+eps) * (1 + a q^b)) * (y - z)."""
    diff = y - z
    q = squared_distance(y, z).clamp(min=0.0).unsqueeze(-1)
    coeff = 2.0 * b / ((q + eps) * (1.0 + a * q.pow(b)))
    out = coeff * diff
    if clip is not None:
        out = out.clamp(min=-clip, max=clip)
    return out
