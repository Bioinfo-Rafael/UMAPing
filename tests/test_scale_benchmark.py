"""Section 8 scale benchmark: tiny synthetic (mock-data) smoke test only --
never a real-data scaling run."""

from __future__ import annotations

import numpy as np
import torch

from umaping.experiments.scale_benchmark import run_and_save_scale_benchmark, run_scale_benchmark


def test_run_scale_benchmark_produces_finite_monotonic_offline_cost_fields():
    points = run_scale_benchmark(n_reference_values=(40, 80), n_query=5, input_dim=6, device=torch.device("cpu"), steps=5)
    assert len(points) == 2
    for p in points:
        d = p.to_dict()
        numeric = np.array([v for v in d.values() if isinstance(v, (int, float))])
        assert np.all(np.isfinite(numeric))
        assert p.offline_total_seconds > 0
        assert p.full_embed_one_seconds_per_query > 0
        assert p.peak_rss_mb > 0


def test_run_and_save_scale_benchmark_writes_metrics_and_figures(tmp_path):
    run_and_save_scale_benchmark(
        tmp_path, n_reference_values=(30, 60), n_query=4, input_dim=5, device=torch.device("cpu"), steps=5
    )
    assert (tmp_path / "metrics" / "scale_benchmark.json").exists()
    for name in ["scale_query_latency_vs_n.png", "scale_memory_vs_n.png", "scale_offline_cost_vs_n.png"]:
        assert (tmp_path / "figures" / name).exists()

    import json

    payload = json.loads((tmp_path / "metrics" / "scale_benchmark.json").read_text())
    assert len(payload) == 2
    assert payload[0]["n_reference"] == 30
