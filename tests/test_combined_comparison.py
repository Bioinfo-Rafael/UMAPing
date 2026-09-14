"""小さい合成保存ファイルだけで統合解析を検証する。"""
from pathlib import Path
import json
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from umaping.comparison.loading import load_results, DATASETS, digest
from umaping.comparison.statistics import paired_statistics, tail_summary, gains_vs_standard, gain_summary, holm, ranked_table
from umaping.comparison.analysis import paired_comparisons, ablations, continuous_structure
from umaping.comparison.__main__ import run_analysis, safe_output


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def saved(tmp_path):
    internal, external = tmp_path / "runs", tmp_path / "external"
    external.mkdir()
    summary = []
    for dataset in ("coil20", "coil100"):
        run = internal / dataset / "main"
        (run / "metrics").mkdir(parents=True)
        (run / "cache").mkdir()
        cfg = {"seed": 0, "dataset": {"name": dataset}, "eval": {"k": 15}}
        (run / "config.yaml").write_text(yaml.safe_dump(cfg))
        rng = np.random.default_rng(5)
        np.savez(run / "cache/prepared_dataset.npz", reference_features=rng.normal(size=(40, 3)),
                 query_features=rng.normal(size=(8, 3)), input_dim=3)
        entries, query_frames = {}, []
        for method, offset in (("standard_umap", 0), ("ours_full", .1), ("oracle_neighbors", .1), ("no_repulsion", .12)):
            recall = np.arange(8) / 30 + .2 + offset
            entries[method] = {"neighborhood_recall_at_k": float(recall.mean()), "n_queries": 8, "k": 15,
                               "mean_query_latency_seconds": .01 if method != "standard_umap" else None}
            query_frames.append(pd.DataFrame({"method": method, "query_index": np.arange(8), "recall_at_15": recall,
                                             "recall_at_5": recall / 2, "recall_at_10": recall * .8, "recall_at_30": recall * 1.1,
                                             "ndcg": recall, "local_displacement": np.ones(8) * (2 if method == "standard_umap" else 1),
                                             "periphery_percentile": np.arange(8) * 13,
                                             "repulsion_accumulation_score": np.arange(8) / 8}))
        write_json(run / "metrics/embedding.json", entries)
        pd.concat(query_frames).to_csv(run / "metrics/advanced_per_query.csv", index=False)
        pd.DataFrame([{"method": "numap_sep_spectralnet", "available": False, "reason": "old environment absent"}]).to_csv(run / "metrics/baseline_comparison.csv", index=False)
        folder = external / dataset / "numap"
        folder.mkdir(parents=True)
        recall = np.arange(8) / 30 + .35
        metrics = {"neighborhood_recall_at_k": float(recall.mean()), "recall_at_15": float(recall.mean()),
                   "n_queries": 8, "k": 15, "ndcg": float(recall.mean()), "mean_query_latency_seconds": .001}
        detail = dict(dataset=dataset, method="numap_sep_spectralnet", baseline="numap", success=True,
                      available=True, status="success", n_query=8, k=15, metrics=metrics,
                      prepared_dataset_sha256=digest(run / "cache/prepared_dataset.npz"),
                      config_sha256=digest(run / "config.yaml"))
        write_json(folder / "result.json", detail)
        pd.DataFrame({"query_index": np.arange(8)[::-1], "recall_at_15": recall[::-1], "ndcg": recall[::-1],
                      "recall_at_5": recall[::-1] / 2, "recall_at_10": recall[::-1] * .8, "recall_at_30": recall[::-1] * 1.1}).to_csv(folder / "advanced_per_query.csv", index=False)
        summary.append({**{k:v for k,v in detail.items() if k != "metrics"}, **metrics})
        summary.append(dict(dataset=dataset, method="oos_umap", available=False, success=False, status="unavailable", reason="official executable missing", recall_at_15=999))
    summary.append(dict(dataset="hong_ed", method="ours", available=True, success=True, status="success", recall_at_15=999))
    pd.DataFrame(summary).to_csv(external / "summary.csv", index=False)
    return internal, external


def test_real_schemas_alias_merge_and_availability(saved):
    internal, external = saved
    data = load_results(internal, external)
    frame = data.frame()
    assert set(frame.dataset) == {"coil20", "coil100"}
    assert "numap_sep_spectralnet" not in set(frame.method)
    assert data.rows["coil20", "numap"]["success"]
    assert data.rows["coil20", "numap"]["recall_at_15"] > 0
    assert not data.rows["coil20", "oos_umap"]["success"]
    assert np.isnan(data.rows["coil20", "oos_umap"]["recall_at_15"])
    assert np.isnan(data.rows["coil20", "ours"]["reference_trustworthiness"])
    assert data.queries["coil20", "numap"]["identity_verified"]
    assert data.queries["coil20", "numap"]["frame"].index.equals(pd.Index(range(8), name="query_index"))


def test_gain_and_zero_baseline_and_sign_test(saved):
    data = load_results(*saved)
    gains = gains_vs_standard(data.frame())
    r = gains[(gains.method == "ours") & (gains.dataset == "coil20")].iloc[0]
    assert r.absolute_gain == pytest.approx(.1)
    assert r.relative_gain == pytest.approx(.1 / (np.arange(8) / 30 + .2).mean())
    summary = gain_summary(gains)
    assert summary["wins"] == 2 and summary["sign_test_p"] == .25
    frame = data.frame()
    frame.loc[frame.method == "standard_umap", "recall_at_15"] = 0
    assert gains_vs_standard(frame).relative_gain.isna().all()


def test_paired_bootstrap_and_exact_permutation_direction():
    r = paired_statistics(np.ones(5), np.zeros(5), "recall_at_15", seed=1, resamples=10000)
    assert r["ci_low"] == r["ci_high"] == 1
    assert r["p_value"] == 2 / 32
    assert r["test"] == "exact_sign_flip"
    low = paired_statistics(np.zeros(5), np.ones(5), "local_displacement", seed=1, resamples=10000)
    assert low["mean_improvement"] == 1 and low["fraction_ours_better"] == 1
    zero = paired_statistics(np.ones(5), np.ones(5), "ndcg", resamples=100)
    assert zero["p_value"] == 1 and zero["fraction_tied"] == 1


def test_paired_monte_carlo_deterministic_nan_and_holm():
    x = np.r_[np.linspace(.1, 1, 30), np.nan]
    y = np.zeros_like(x)
    a = paired_statistics(x, y, "ndcg", seed=123, resamples=10000)
    assert a == paired_statistics(x, y, "ndcg", seed=123, resamples=10000)
    assert a["n_pairs"] == 30 and a["n_dropped"] == 1
    assert a["p_value"] >= 1 / 10001 and a["test"].startswith("monte_carlo")
    np.testing.assert_allclose(holm([.01, .03, .04, np.nan]), [.03, .06, .06, np.nan])
    with pytest.raises(ValueError):
        paired_statistics([1], [1, 2], "ndcg")


def test_tail_uses_low_recall_high_displacement_and_ceil():
    recall = tail_summary(np.arange(100) / 100)
    assert recall["worst_5pct_mean"] == .02
    assert recall["worst_1pct_mean"] == 0
    displacement = tail_summary(np.arange(100), higher=False)
    assert displacement["worst_5pct_mean"] == 97
    assert tail_summary([.7, np.nan])["worst_5pct_mean"] == .7


def test_duplicate_summary_and_conflicting_aggregate_rejected(saved):
    internal, external = saved
    path = external / "summary.csv"
    original = pd.read_csv(path)
    pd.concat([original, original.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="重複"):
        load_results(internal, external)
    original.to_csv(path, index=False)
    path = internal / "coil20/main/metrics/baseline_comparison.csv"
    pd.DataFrame([dict(method="ours", available=True, neighborhood_recall_at_k=.999, k=15, n_queries=8)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="矛盾"):
        load_results(internal, external)


def test_external_cache_mismatch_rejected(saved):
    internal, external = saved
    path = internal / "coil20/main/cache/prepared_dataset.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="cache hash"):
        load_results(internal, external)


def test_query_duplicate_and_mismatched_ids_not_paired(saved):
    internal, external = saved
    path = external / "coil20/numap/advanced_per_query.csv"
    q = pd.read_csv(path)
    q.loc[0, "query_index"] = 100
    q.to_csv(path, index=False)
    data = load_results(internal, external)
    pairs, skipped = paired_comparisons(data, seed=1, resamples=100)
    assert not ((pairs.dataset == "coil20") & (pairs.baseline == "numap")).any()
    assert any("query ID" in x for x in skipped)
    q.loc[0, "query_index"] = q.loc[1, "query_index"]
    q.to_csv(path, index=False)
    with pytest.raises(ValueError, match="重複"):
        load_results(internal, external)


def test_query_count_mismatch_is_error(saved):
    internal, external = saved
    path = external / "coil20/numap/advanced_per_query.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="件数"):
        load_results(internal, external)


def test_internal_evaluate_and_advanced_phases_preserved(saved):
    internal, external = saved
    path = internal / "coil20/main/metrics/advanced_per_query.csv"
    q = pd.read_csv(path)
    q.loc[q.method == "standard_umap", "recall_at_15"] += .001
    q.to_csv(path, index=False)
    data = load_results(internal, external)
    r = data.rows["coil20", "standard_umap"]
    assert r["per_query_phase_recall15_mean"] - r["recall_at_15"] == pytest.approx(.001)
    assert any("追加解析段階" in n for n in data.notes)


def test_sparse_rank_panel_exposes_counts_and_no_missing_zeros(saved):
    frame = load_results(*saved).frame()
    table, ranks, info = ranked_table(frame, "recall_at_15", DATASETS)
    assert "oos_umap" not in set(table.method)
    assert table.n_datasets.eq(2).all()
    assert table.n_rank_datasets.eq(2).all()
    assert table.pancreas.isna().all()
    assert info["datasets"] == ["coil20", "coil100"]


def test_subset_indices_remapped_and_not_ranked(saved):
    internal, external = saved
    path = internal / "coil20/main/metrics/embedding.json"
    e = json.loads(path.read_text())
    e["repulsion_oracle_diagnostic"] = dict(neighborhood_recall_at_k=.5, k=15, n_queries=2, query_subset_indices=[3, 0])
    write_json(path, e)
    path = path.parent / "advanced_per_query.csv"
    q = pd.read_csv(path)
    q = pd.concat([q, pd.DataFrame(dict(method=["repulsion_oracle_diagnostic"] * 2, query_index=[0, 1], recall_at_15=[.4, .6]))])
    q.to_csv(path, index=False)
    data = load_results(internal, external)
    sub = data.queries["coil20", "repulsion_oracle_diagnostic"]["frame"]
    assert sub.index.tolist() == [0, 3] and sub.loc[3, "recall_at_15"] == .4
    table, _, _ = ranked_table(data.frame(), "recall_at_15", DATASETS)
    assert "repulsion_oracle_diagnostic" not in set(table.method)
    abl, _, _ = ablations(data, data.frame())
    diag = abl[abl.method == "repulsion_oracle_diagnostic"].iloc[0]
    assert diag.ours_matched_recall15 == pytest.approx((.3 + .4) / 2)


def test_output_guard_and_missing_summary(saved, tmp_path):
    internal, external = saved
    with pytest.raises(ValueError):
        safe_output(internal / "new", internal, external)
    (tmp_path / "alias").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_output(tmp_path / "alias/new", internal, external)
    (tmp_path / "existing-output").mkdir()
    with pytest.raises(FileExistsError):
        safe_output(tmp_path / "existing-output", internal, external)
    (external / "summary.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_results(internal, external)


def test_no_guess_of_arbitrary_time_label(saved):
    data = load_results(*saved)
    run = data.internal_root / "organoid/main/cache"
    run.mkdir(parents=True)
    np.savez(run / "prepared_dataset.npz", reference_label__state=["early", "late"], query_label__state=["middle"])
    values, reasons = continuous_structure(data, seed=1)
    assert values.empty and any("順序" in r for r in reasons)


def test_numeric_time_analysis_uses_saved_coordinates_only(saved):
    data = load_results(*saved)
    run = data.internal_root / "organoid/main/cache"
    run.mkdir(parents=True)
    tr, tq = np.arange(40), np.array([.5, 10.5, 20.5])
    np.savez(run / "prepared_dataset.npz", reference_label__time=tr, query_label__time=tq)
    folder = data.external_root / "organoid/numap"
    folder.mkdir(parents=True)
    np.savez(folder / "embeddings.npz", reference=np.column_stack([tr, np.zeros_like(tr)]),
             query=np.column_stack([tq, np.zeros_like(tq)]))
    data.rows["organoid", "numap"] = {"success": True}
    data.external_details["organoid", "numap"] = {"folder": folder, "record": {"identity_verified": True}}
    values, _ = continuous_structure(data, seed=1)
    assert len(values) == 1 and values.iloc[0].time_distance_spearman == pytest.approx(1)
    assert values.iloc[0].temporal_smoothness_mean > 0


def test_aggregate_only_external_has_no_invented_query_values(saved):
    internal, external = saved
    (external / "coil20/numap/advanced_per_query.csv").unlink()
    data = load_results(internal, external)
    assert data.rows["coil20", "numap"]["success"]
    assert ("coil20", "numap") not in data.queries
    assert any("per-query保存なし" in n for n in data.notes)


@pytest.mark.parametrize("metric", ["recall_at_15", "ndcg"])
def test_external_perquery_conflict_fails_loudly(saved, metric):
    internal, external = saved
    path = external / "coil20/numap/advanced_per_query.csv"
    frame = pd.read_csv(path)
    frame[metric] += .1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="per-query値が矛盾"):
        load_results(internal, external)


def test_external_summary_timing_conflict_fails_loudly(saved):
    internal, external = saved
    path = external / "summary.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.method == "numap_sep_spectralnet", "mean_query_latency_seconds"] = 123
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="集約結果が矛盾"):
        load_results(internal, external)


def test_full_synthetic_report_and_figures_preserve_sources(saved, tmp_path, monkeypatch):
    def snap(root):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    internal, external = saved
    before = {**snap(internal), **snap(external)}
    monkeypatch.setitem(sys.modules, "umaping.pipeline", None)
    monkeypatch.setitem(sys.modules, "umaping.inference", None)
    monkeypatch.setitem(sys.modules, "numap", None)
    output = run_analysis(internal, external, tmp_path / "analysis", resamples=100, seed=7)
    assert before == {**snap(internal), **snap(external)}
    text = (output / "report.md").read_text()
    assert "独自再構築" in text and "これは手法自体の失敗を意味しません" in text
    assert "hong_ed" not in text and "twenty_newsgroups" not in text
    assert "numap: 共通2datasetでoursのRecallが高い=0件" in text
    main = pd.read_csv(output / "tables/main_recall15.csv")
    assert "oos_umap" not in set(main.method)
    assert main.pancreas.isna().all()
    assert (output / "figures/recall15_by_dataset.png").stat().st_size > 1000
    assert (output / "figures/latency_quality_tradeoff.svg").is_file()
    assert json.loads((output / "metadata.json").read_text())["seed"] == 7
    with pytest.raises(FileExistsError):
        run_analysis(internal, external, output, resamples=100)
