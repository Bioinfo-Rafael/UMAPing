"""保存値の集計、対応検定、診断。モデルのfit/transformは呼ばない。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr

from .loading import DATASETS, METRICS, DIAGNOSTICS
from .statistics import paired_statistics, keyed_seed, holm, tail_summary, gains_vs_standard, gain_summary, ranked_table


def paired_comparisons(data, seed, resamples):
    rows, skipped = [], []
    for dataset in DATASETS:
        ours = data.queries.get((dataset, "ours"))
        if ours is None:
            skipped.append(f"{dataset}: oursのper-query値がないため対応比較不可")
            continue
        for (d, method), other in data.queries.items():
            if d != dataset or method == "ours":
                continue
            a, b = ours["frame"], other["frame"]
            scope = data.rows[d, method]["scope"]
            if not ours["identity_verified"] or not other["identity_verified"]:
                skipped.append(f"{d}/{method}: query由来を照合できず対応比較を除外")
                continue
            if scope == "subset" and b.index.isin(a.index).all():
                a = a.loc[b.index]
            if not a.index.equals(b.index):
                skipped.append(f"{d}/{method}: query ID集合または件数が違うため対応比較を除外")
                continue
            if "label" in a and "label" in b and not a.label.astype(str).equals(b.label.astype(str)):
                raise ValueError(f"同じquery IDのラベルが不一致です: {d}/{method}")
            for metric in ("recall_at_15", "ndcg", "local_displacement"):
                if metric not in a or metric not in b:
                    skipped.append(f"{d}/{method}/{metric}: 対応値なし")
                    continue
                print(f"対応検定 {d}: ours vs {method}, {metric}", flush=True)
                result = paired_statistics(a[metric], b[metric], metric,
                                           seed=keyed_seed(seed, d, method, metric), resamples=resamples)
                family = ("primary_recall15_ours_vs_standard" if method == "standard_umap" and metric == "recall_at_15"
                          else "exploratory_" + metric)
                rows.append(dict(dataset=d, baseline=method, metric=metric, scope=scope, family=family,
                                 ours_source=ours["source"], baseline_source=other["source"], **result))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["p_holm"] = np.nan
        if "p_value" not in frame:
            frame["p_value"] = np.nan
        for _, indices in frame.groupby("family").groups.items():
            frame.loc[indices, "p_holm"] = holm(frame.loc[indices, "p_value"])
    return frame, skipped


def diagnostics(data):
    tails, periphery = [], []
    for (dataset, method), entry in data.queries.items():
        frame = entry["frame"]
        scope = data.rows[dataset, method]["scope"]
        for metric in ("recall_at_15", "local_displacement"):
            if metric in frame:
                tails.append(dict(dataset=dataset, method=method, metric=metric, scope=scope,
                                  **tail_summary(frame[metric], higher=metric == "recall_at_15")))
        for metric, thresholds in (("periphery_percentile", (90, 95)), ("repulsion_accumulation_score", (.9, .95))):
            if metric not in frame:
                continue
            v = frame[metric].to_numpy(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            periphery.append(dict(dataset=dataset, method=method, metric=metric, scope=scope,
                                  n_queries=len(v), mean=float(v.mean()), median=float(np.median(v)),
                                  p90=float(np.percentile(v, 90)), p95=float(np.percentile(v, 95)),
                                  threshold_90=thresholds[0], threshold_95=thresholds[1],
                                  fraction_ge_90=float(np.mean(v >= thresholds[0])),
                                  fraction_ge_95=float(np.mean(v >= thresholds[1]))))
    return pd.DataFrame(tails), pd.DataFrame(periphery)


def ablations(data, frame):
    rows = []
    for dataset in DATASETS:
        base = data.rows.get((dataset, "ours"))
        if base is None or not base["success"]:
            continue
        for method in ("ours", "ours_oracle_neighbors", "no_repulsion", "exact_repulsion_diagnostic", "repulsion_oracle_diagnostic"):
            r = data.rows.get((dataset, method))
            if r is None or not r["success"]:
                continue
            ours_value = base["recall_at_15"]
            ndcg_base, disp_base = base["ndcg"], base["local_displacement"]
            comparison_phase = "aggregate_full"
            if r["scope"] == "subset":
                ours_value, ndcg_base, disp_base = np.nan, np.nan, np.nan
                indices = data.subsets.get((dataset, method))
                q = data.queries.get((dataset, "ours"))
                if indices is not None and q is not None and pd.Index(indices).isin(q["frame"].index).all():
                    matched = q["frame"].loc[indices]
                    ours_value = matched.recall_at_15.mean()
                    ndcg_base = matched.ndcg.mean() if "ndcg" in matched else np.nan
                    disp_base = matched.local_displacement.mean() if "local_displacement" in matched else np.nan
                    comparison_phase = "aggregate_diagnostic_vs_advanced_ours_same_subset"
                else:
                    comparison_phase = "subset_identity_unavailable"
            rows.append(dict(dataset=dataset, method=method, scope=r["scope"], n_queries=r["n_queries"],
                             recall_at_15=r["recall_at_15"], ours_matched_recall15=ours_value,
                             delta_vs_ours=r["recall_at_15"] - ours_value,
                             ndcg=r["ndcg"], ndcg_delta_vs_ours=r["ndcg"] - ndcg_base,
                             local_displacement=r["local_displacement"],
                             displacement_improvement_vs_ours=disp_base - r["local_displacement"],
                             comparison_phase=comparison_phase))
    table = pd.DataFrame(rows)
    summary = {}
    if not table.empty:
        retrieval = table[table.method.eq("ours_oracle_neighbors")].dropna(subset=["delta_vs_ours"])
        gap = retrieval.delta_vs_ours.to_numpy()
        summary["retriever"] = dict(n_datasets=len(gap), mean_absolute_gap=float(np.abs(gap).mean()) if len(gap) else None,
                                     max_absolute_gap=float(np.abs(gap).max()) if len(gap) else None)
        rep = table[table.method.eq("no_repulsion")].dropna(subset=["delta_vs_ours"])
        summary["repulsion"] = dict(ours_better=rep.loc[rep.delta_vs_ours < -1e-12, "dataset"].tolist(),
                                     no_repulsion_better=rep.loc[rep.delta_vs_ours > 1e-12, "dataset"].tolist(),
                                     tied=rep.loc[rep.delta_vs_ours.abs() <= 1e-12, "dataset"].tolist(),
                                     mean_ours_minus_no_repulsion=float(-rep.delta_vs_ours.mean()) if len(rep) else None)
    aggregate = (table.groupby(["method", "scope"])[["recall_at_15", "delta_vs_ours", "ndcg", "local_displacement"]]
                 .agg(["mean", "median", "count"]).reset_index()) if not table.empty else pd.DataFrame()
    if not aggregate.empty:
        aggregate.columns = ["_".join(filter(None, col)) for col in aggregate.columns]
    return table, aggregate, summary


def continuous_structure(data, seed):
    """保存済みの数値time/day列と外部座標だけ。任意のstate文字列は順序化しない。"""
    rows, notes = [], []
    for dataset in ("organoid", "embryoid_body"):
        cache = data.internal_root / dataset / "main/cache/prepared_dataset.npz"
        if not cache.is_file():
            notes.append(f"{dataset}: 前処理cacheなし。時間変数を確認できず連続構造解析を省略。")
            continue
        try:
            data.track(cache)
            with np.load(cache, allow_pickle=False) as stored:
                label_keys = [k.removeprefix("reference_label__") for k in stored.files if k.startswith("reference_label__")]
                # このloaderは検出したgrouping_columnを最初のラベルとして保存する。
                key = label_keys[0] if label_keys else ""
                if key.lower() not in {"time", "day", "timepoint", "time_point"}:
                    notes.append(f"{dataset}: 保存grouping列={key!r}。loaderが順序を保証しないため連続構造解析を省略。")
                    continue
                tr = np.asarray(stored["reference_label__" + key], dtype=float)
                tq = np.asarray(stored["query_label__" + key], dtype=float)
                if not np.isfinite(tr).all() or not np.isfinite(tq).all() or len(np.unique(tr)) < 2:
                    raise ValueError("時刻が非有限、またはreferenceが単一時刻")
        except (ValueError, KeyError) as exc:
            notes.append(f"{dataset}: 時刻の数値解釈不可（{exc}）。文字列から順序を推測しない。")
            continue
        notes.append(f"{dataset}: loader保存列 {key!r} の数値を時刻として使用。単位は変換せず、他datasetと平均しない。内部query座標は未保存のため再推論しない。")
        for (d, method), detail in data.external_details.items():
            if d != dataset or not data.rows[d, method]["success"]:
                continue
            path = detail["folder"] / "embeddings.npz"
            if not path.is_file() or not detail["record"].get("identity_verified"):
                notes.append(f"{d}/{method}: 保存座標またはcache照合がないため連続構造解析不可。")
                continue
            with np.load(data.track(path), allow_pickle=False) as emb:
                ref, query = emb["reference"], emb["query"]
            if len(ref) != len(tr) or len(query) != len(tq) or not np.isfinite(ref).all() or not np.isfinite(query).all():
                raise ValueError(f"保存座標と時刻が不一致です: {d}/{method}")
            from umaping.graph import chunked_exact_knn
            ids, _ = chunked_exact_knn(query, ref, k=15)
            temporal = np.abs(tq[:, None] - tr[ids]).mean(axis=1)
            rng = np.random.default_rng(keyed_seed(seed, dataset, "temporal_pairs"))
            n_pairs = min(100000, len(query) * len(ref))
            qi, ri = rng.integers(len(query), size=n_pairs), rng.integers(len(ref), size=n_pairs)
            td = np.abs(tq[qi] - tr[ri])
            distance = np.linalg.norm(query[qi] - ref[ri], axis=1)
            rho = float(spearmanr(td, distance).statistic) if np.ptp(td) and np.ptp(distance) else np.nan
            rows.append(dict(dataset=d, method=method, time_column=key, k=15,
                             temporal_smoothness_mean=float(temporal.mean()), temporal_smoothness_median=float(np.median(temporal)),
                             time_distance_spearman=rho, n_sampled_pairs=n_pairs, source=str(path)))
    return pd.DataFrame(rows), notes


def summarize(data, seed, resamples):
    frame = data.frame()
    main, rank_matrix, rank_info = ranked_table(frame, "recall_at_15", DATASETS)
    main = main.rename(columns={"mean": "mean_recall15", "median": "median_recall15"})
    secondary, secondary_ranks = [], []
    for metric in METRICS:
        table, _, info = ranked_table(frame, metric, DATASETS)
        table.insert(0, "metric", metric)
        secondary.append(table)
        secondary_ranks.append(info)
    gains = gains_vs_standard(frame)
    paired, pair_skips = paired_comparisons(data, seed, resamples)
    tails, periphery = diagnostics(data)
    ablation, ablation_aggregate, ablation_info = ablations(data, frame)
    continuous, continuous_notes = continuous_structure(data, seed)
    # 固定panelの揃った反復測定のみ。関連するpancreas二条件は独立反復でない。
    friedman = dict(status="skipped", reason="共通panelが3手法・5データセット未満、または独立性の制約。")
    independent = [d for d in rank_info["datasets"] if d != "pancreas_batch_corrected"]
    if len(rank_info["cohort"]) >= 3 and len(independent) >= 5:
        pivot = frame[frame.success & frame.scope.eq("full")].pivot(index="dataset", columns="method", values="recall_at_15")
        vals = pivot.loc[independent, rank_info["cohort"]]
        if not vals.eq(vals.iloc[:, 0], axis=0).all().all():
            test = friedmanchisquare(*(vals[m].to_numpy() for m in vals))
            friedman = dict(status="exploratory", statistic=float(test.statistic), p_value=float(test.pvalue),
                            datasets=independent, methods=list(vals), note="pancreas_batch_correctedは関連条件のため除外。少数datasetなので探索的。")
    return dict(frame=frame, main=main, secondary=pd.concat(secondary, ignore_index=True),
                ranks=rank_matrix, gains=gains, gain_summary=gain_summary(gains), paired=paired,
                tails=tails, periphery=periphery, ablations=ablation, ablation_aggregate=ablation_aggregate,
                ablation_summary=ablation_info, continuous=continuous, rank_info=rank_info,
                secondary_rank_info=secondary_ranks, friedman=friedman,
                notes=pair_skips + continuous_notes)
