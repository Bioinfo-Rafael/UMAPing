"""Tests for umap_forces.py and the Monte-Carlo mean-field helper in
dynamics.py.

Written to be run remotely (see RUNTIME_CHECKS.md); not executed here.
"""

from __future__ import annotations

import pytest
import torch

from umaping.dynamics import mean_negative_field
from umaping.umap_forces import find_ab_params, g_minus, g_plus, log_one_minus_phi, log_phi


def test_find_ab_params_matches_umap_learn():
    umap_module = pytest.importorskip("umap.umap_")
    a_ours, b_ours = find_ab_params(spread=1.0, min_dist=0.1)
    a_umap, b_umap = umap_module.find_ab_params(1.0, 0.1)
    assert abs(a_ours - a_umap) < 1e-6
    assert abs(b_ours - b_umap) < 1e-6


def test_g_plus_matches_autograd_of_log_phi():
    torch.manual_seed(0)
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    y = torch.randn(16, 2, requires_grad=True)
    z = torch.randn(16, 2)

    loss = log_phi(y, z, a, b).sum()
    (grad_autograd,) = torch.autograd.grad(loss, y)
    grad_closed_form = g_plus(y.detach(), z, a, b, clip=None)

    torch.testing.assert_close(grad_closed_form, grad_autograd, rtol=1e-3, atol=1e-4)


def test_g_minus_matches_autograd_of_log_one_minus_phi():
    torch.manual_seed(1)
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    # Scaled up so points are well-separated (q large enough to avoid the
    # q -> 0 boundary case, which g_minus/log_one_minus_phi handle via
    # separate epsilon conventions and are not expected to match exactly at).
    y = torch.randn(16, 2, requires_grad=True) * 3.0
    z = torch.randn(16, 2) * 3.0

    loss = log_one_minus_phi(y, z, a, b).sum()
    (grad_autograd,) = torch.autograd.grad(loss, y)
    grad_closed_form = g_minus(y.detach(), z, a, b, clip=None)

    torch.testing.assert_close(grad_closed_form, grad_autograd, rtol=1e-3, atol=1e-4)


def test_mean_negative_field_matches_brute_force_average_on_tiny_dataset():
    torch.manual_seed(2)
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    y = torch.randn(5, 2)
    candidates = torch.randn(4, 2)

    brute_force = torch.stack([torch.stack([g_minus(yi, cj, a, b) for cj in candidates]).mean(dim=0) for yi in y])
    via_helper = mean_negative_field(y, candidates, a, b)

    torch.testing.assert_close(via_helper, brute_force, rtol=1e-5, atol=1e-6)


def test_mean_negative_field_monte_carlo_converges_with_more_samples():
    """A larger random sample of a fixed reference population should give a
    Monte-Carlo mean-field estimate closer to the exhaustive population
    average than a much smaller sample does. Averaged over several trials
    per sample size so the comparison is not a coin flip on RNG luck."""
    torch.manual_seed(3)
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    y = torch.zeros(1, 2)
    population = torch.randn(3000, 2) * 1.5
    true_mean = mean_negative_field(y, population, a, b)

    def average_error(n_samples: int, n_trials: int = 8) -> float:
        errors = []
        for _ in range(n_trials):
            idx = torch.randperm(population.shape[0])[:n_samples]
            sample = population[idx]
            errors.append((mean_negative_field(y, sample, a, b) - true_mean).norm().item())
        return sum(errors) / len(errors)

    err_small = average_error(n_samples=8)
    err_large = average_error(n_samples=1500)
    assert err_large < err_small


def test_g_plus_and_g_minus_clip_bounds_output():
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    y = torch.tensor([[1000.0, 0.0]])
    z = torch.tensor([[0.0, 0.0]])
    clip = 4.0

    out_plus = g_plus(y, z, a, b, clip=clip)
    out_minus = g_minus(y, z, a, b, clip=clip)

    assert torch.all(out_plus.abs() <= clip + 1e-6)
    assert torch.all(out_minus.abs() <= clip + 1e-6)


def test_g_plus_is_zero_at_zero_distance():
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    y = torch.tensor([[1.0, 2.0]])
    z = torch.tensor([[1.0, 2.0]])
    out = g_plus(y, z, a, b, clip=None)
    torch.testing.assert_close(out, torch.zeros_like(out))
