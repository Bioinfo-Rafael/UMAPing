"""Experiment A: analysis-only diagnostics.

Written to be run against the mock dataset (no download, no real training,
seconds not minutes) both as a direct unit test of the pure functions in
`evaluation/advanced.py` and as a smoke test of the full
`analyze-advanced` path against a just-trained mock run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from umaping.config import Config
from umaping.dynamics import ReferenceTrajectory
from umaping.evaluation.advanced import (
    compute_field_roughness,
    compute_per_query_diagnostics,
    denoising_report_by_t,
    exact_all_reference_mean_field,
    field_smoothness_report,
    make_reference_grid,
    monte_carlo_denoising_test,
    tail_summary,
)
from umaping.models.repulsion import RepulsionField


# ---------------------------------------------------------------------------
# Pure-function unit tests (no filesystem, no training)
# ---------------------------------------------------------------------------


def test_compute_field_roughness_is_near_zero_for_a_constant_field():
    grid = np.ones((10, 10, 2), dtype=np.float32) * 3.0
    metrics = compute_field_roughness(grid, delta=0.5)
    assert metrics.finite_diff_roughness == pytest.approx(0.0, abs=1e-6)
    assert metrics.second_order_roughness == pytest.approx(0.0, abs=1e-6)
    assert metrics.magnitude_variation == pytest.approx(0.0, abs=1e-6)


def test_compute_field_roughness_is_large_for_a_checkerboard_field():
    h, w = 12, 12
    checker = np.indices((h, w)).sum(axis=0) % 2
    grid = np.stack([checker, checker], axis=-1).astype(np.float32) * 5.0
    smooth_grid = np.ones((h, w, 2), dtype=np.float32) * 2.5
    rough = compute_field_roughness(grid, delta=1.0)
    smooth = compute_field_roughness(smooth_grid, delta=1.0)
    assert rough.finite_diff_roughness > smooth.finite_diff_roughness


def test_tail_summary_matches_direct_percentile_computation():
    rng = np.random.default_rng(0)
    values = rng.exponential(scale=1.0, size=5000)
    summary = tail_summary(values)
    assert summary.mean == pytest.approx(float(values.mean()), rel=1e-9)
    assert summary.median == pytest.approx(float(np.median(values)), rel=1e-9)
    assert summary.p99 == pytest.approx(float(np.percentile(values, 99)), rel=1e-9)
    # Worst-1% mean must be >= p99 (it's the mean of everything at/above that tail).
    assert summary.worst_1pct_mean >= summary.p99


def test_tail_summary_handles_empty_and_all_nan_input():
    empty = tail_summary(np.array([]))
    assert np.isnan(empty.mean)
    all_nan = tail_summary(np.array([np.nan, np.nan]))
    assert np.isnan(all_nan.mean)


def _toy_trajectory(n_ref: int = 40, n_steps: int = 3, seed: int = 0) -> ReferenceTrajectory:
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 1.0, n_steps + 1)
    positions = np.stack(
        [rng.normal(scale=3.0, size=(n_ref, 2)).astype(np.float32) + 0.1 * t for t in range(n_steps + 1)], axis=0
    ).astype(np.float32)
    return ReferenceTrajectory(times=times, positions=positions)


def test_exact_all_reference_mean_field_matches_brute_force_mean():
    """The exact diagnostic must reproduce a direct numpy mean over every
    reference point exactly once -- the whole point of it vs. `oracle_mc`,
    which samples with replacement and need not equal this."""
    from umaping.umap_forces import g_minus

    traj = _toy_trajectory(n_ref=25, n_steps=2, seed=1)
    device = torch.device("cpu")
    a, b = 1.0, 1.0
    y = torch.as_tensor(np.array([[0.5, -0.3], [1.2, 2.0]], dtype=np.float32))

    result = exact_all_reference_mean_field(y, traj, t=0.5, a=a, b=b, device=device, clip=4.0, chunk_size=7)

    all_positions = torch.as_tensor(traj.positions_at(0.5), dtype=torch.float32)
    brute_force = g_minus(y.unsqueeze(1), all_positions.unsqueeze(0), a, b, clip=4.0).mean(dim=1)

    assert torch.allclose(result, brute_force, atol=1e-5)


def test_exact_all_reference_mean_field_supports_per_point_times():
    traj = _toy_trajectory(n_ref=20, n_steps=4, seed=2)
    device = torch.device("cpu")
    y = torch.as_tensor(np.random.default_rng(3).normal(size=(6, 2)).astype(np.float32))
    t_per_point = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.6], dtype=np.float64)

    result = exact_all_reference_mean_field(y, traj, t=t_per_point, a=1.0, b=1.0, device=device)
    assert result.shape == (6, 2)
    assert torch.all(torch.isfinite(result))


def test_make_reference_grid_covers_the_occupied_region_with_padding():
    ref = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 3.0]], dtype=np.float32)
    grid, delta = make_reference_grid(ref, resolution=8, pad_fraction=0.2)
    assert grid.shape == (8, 8, 2)
    assert delta > 0
    assert grid[..., 0].min() < ref[:, 0].min()
    assert grid[..., 0].max() > ref[:, 0].max()


def test_field_smoothness_report_runs_on_a_toy_repulsion_model():
    traj = _toy_trajectory(n_ref=30, n_steps=5, seed=4)
    model = RepulsionField(embedding_dim=2, hidden_dim=8, n_residual_blocks=1, time_embed_dim=4)
    report = field_smoothness_report(
        model, traj, a=1.0, b=1.0, reference_embedding=traj.positions[-1], device=torch.device("cpu"),
        ts=(0.0, 0.5, 1.0), resolution=6, low_m=5, high_m=20, seed=0,
    )
    assert set(report["overall"].keys()) == {"b_phi", "low_m_teacher", "high_m_teacher", "exact"}
    for metrics in report["overall"].values():
        for v in metrics.values():
            assert np.isfinite(v)


def test_monte_carlo_denoising_test_reports_b_phi_vs_single_estimate():
    traj = _toy_trajectory(n_ref=40, n_steps=5, seed=5)
    model = RepulsionField(embedding_dim=2, hidden_dim=8, n_residual_blocks=1, time_embed_dim=4)
    result = monte_carlo_denoising_test(
        model, traj, a=1.0, b=1.0, reference_embedding=traj.positions[-1], device=torch.device("cpu"),
        n_eval_points=20, low_m=6, r_repeats=8, seed=0,
    )
    assert "vs_b_mean" in result and "vs_exact" in result
    assert isinstance(result["b_phi_better_than_single_mc_vs_mean"], bool)
    for comparison in (result["vs_b_mean"], result["vs_exact"]):
        for field_result in comparison.values():
            assert np.isfinite(field_result["mse_mean"])
            lo, hi = field_result["mse_ci95"]
            assert lo <= field_result["mse_mean"] <= hi or np.isnan(lo)


def test_denoising_report_by_t_produces_one_entry_per_t_plus_mixed():
    traj = _toy_trajectory(n_ref=30, n_steps=4, seed=6)
    model = RepulsionField(embedding_dim=2, hidden_dim=8, n_residual_blocks=1, time_embed_dim=4)
    report = denoising_report_by_t(
        model, traj, a=1.0, b=1.0, reference_embedding=traj.positions[-1], device=torch.device("cpu"),
        ts=(0.0, 0.5, 1.0), n_eval_points=15, low_m=5, r_repeats=6, seed=0,
    )
    assert set(report["per_t"].keys()) == {"t=0.00", "t=0.50", "t=1.00"}
    assert "mixed" in report


def test_compute_per_query_diagnostics_has_no_infinities_and_sane_ranges():
    rng = np.random.default_rng(7)
    n_query, n_ref, d_high, d_low = 12, 60, 5, 2
    query_high = rng.normal(size=(n_query, d_high)).astype(np.float32)
    ref_high = rng.normal(size=(n_ref, d_high)).astype(np.float32)
    query_emb = rng.normal(size=(n_query, d_low)).astype(np.float32)
    ref_emb = rng.normal(size=(n_ref, d_low)).astype(np.float32)
    neighbor_ids = [rng.choice(n_ref, size=6, replace=False) for _ in range(n_query)]
    neighbor_weights = [rng.uniform(0.1, 1.0, size=6) for _ in range(n_query)]
    labels_q = rng.integers(0, 3, size=n_query)
    labels_ref = rng.integers(0, 3, size=n_ref)

    df, tails = compute_per_query_diagnostics(
        "toy_method", query_high, ref_high, query_emb, ref_emb, neighbor_ids, neighbor_weights,
        a=1.0, b=1.0, ks=(2, 4), query_labels=labels_q, reference_labels=labels_ref,
    )
    assert len(df) == n_query
    assert not np.any(np.isinf(df.select_dtypes(include=[np.number]).to_numpy()))
    assert (df["recall_at_2"] >= 0).all() and (df["recall_at_2"] <= 1).all()
    assert set(tails.keys()) >= {"recall_at_2_deficit", "recall_at_4_deficit", "ndcg_deficit"}


# ---------------------------------------------------------------------------
# End-to-end: analyze-advanced against a just-trained mock run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_mock_run(tmp_path_factory):
    """Trains the tiny synthetic mock config end to end (seconds, no
    download, no real data) exactly once, shared across the tests below."""
    from umaping.pipeline import run_training
    from umaping.utils.device import resolve_device
    from umaping.utils.seed import set_seed

    run_dir = tmp_path_factory.mktemp("advanced_mock_run")
    cfg = Config.load("configs/mock.yaml")
    device = resolve_device("cpu")
    set_seed(cfg.seed)
    run_training(cfg, run_dir, device)
    return run_dir


def test_analyze_advanced_fails_loudly_on_an_untrained_run_dir(tmp_path):
    from umaping.pipeline import run_advanced_analysis

    empty_dir = tmp_path / "untrained"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="missing required artifacts"):
        run_advanced_analysis(empty_dir, torch.device("cpu"))


def test_analyze_advanced_end_to_end_on_a_trained_mock_run(trained_mock_run):
    from umaping.pipeline import run_advanced_analysis
    from umaping.utils.device import resolve_device

    run_advanced_analysis(trained_mock_run, resolve_device("cpu"))

    metrics_dir = trained_mock_run / "metrics"
    figures_dir = trained_mock_run / "figures"
    assert (metrics_dir / "advanced_analysis.json").exists()
    assert (metrics_dir / "advanced_per_query.csv").exists()
    for name in [
        "field_smoothness_by_t.png",
        "field_denoising_error_by_t.png",
        "query_neighbor_recall_distribution.png",
        "query_fuzzy_error_distribution.png",
        "query_tail_failure_comparison.png",
        "query_periphery_score.png",
    ]:
        assert (figures_dir / name).exists(), f"missing figure {name}"

    import json

    import pandas as pd

    payload = json.loads((metrics_dir / "advanced_analysis.json").read_text())
    assert "field_smoothness" in payload and "field_denoising" in payload and "per_query_tail_summaries" in payload

    df = pd.read_csv(metrics_dir / "advanced_per_query.csv")
    assert not df.empty
    numeric = df.select_dtypes(include=["number"]).to_numpy()
    assert not np.any(np.isinf(numeric))


def test_analyze_advanced_does_not_modify_existing_checkpoints(trained_mock_run):
    """Analysis-only: running it must never touch the trained checkpoints."""
    from umaping.pipeline import run_advanced_analysis
    from umaping.utils.device import resolve_device

    ckpt_path = trained_mock_run / "checkpoints" / "repulsion_field.pt"
    before = ckpt_path.stat().st_mtime_ns
    run_advanced_analysis(trained_mock_run, resolve_device("cpu"))
    after = ckpt_path.stat().st_mtime_ns
    assert before == after
