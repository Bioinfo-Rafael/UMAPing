"""方向を揃えた対応比較と、データセット単位の集計。"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HIGHER = {"recall_at_5", "recall_at_10", "recall_at_15", "recall_at_30", "ndcg",
          "reference_trustworthiness", "label_knn_accuracy"}
LOWER = {"local_displacement", "fuzzy_weighted_mse", "fuzzy_weighted_bce",
         "repulsion_accumulation_score", "periphery_percentile", "fit_time_seconds",
         "mean_query_latency_seconds"}


def keyed_seed(seed: int, *keys: str) -> int:
    return int.from_bytes(hashlib.sha256((str(seed) + "/" + "/".join(keys)).encode()).digest()[:8], "little")


def paired_statistics(ours, other, metric: str, seed=0, resamples=10000) -> dict:
    x, y = np.asarray(ours, float), np.asarray(other, float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("対応比較の配列shapeが一致しません")
    if metric not in HIGHER | LOWER:
        raise ValueError(f"指標の方向が未定義です: {metric}")
    if resamples < 1:
        raise ValueError("resamplesは正数である必要があります")
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    n = len(x)
    if not n:
        return {"n_pairs": 0, "status": "no_finite_pairs"}
    d = (x - y) if metric in HIGHER else (y - x)
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    unique, counts = np.unique(d, return_counts=True)
    # Recallのように差が離散的な場合も、通常の対応bootstrapと同じ分布を使う。
    if len(unique) <= 128:
        for start in range(0, resamples, 128):
            size = min(128, resamples - start)
            boot[start:start + size] = rng.multinomial(n, counts / n, size=size) @ unique / n
    else:
        for start in range(0, resamples, 128):
            size = min(128, resamples - start)
            boot[start:start + size] = d[rng.integers(n, size=(size, n))].mean(axis=1)
    observed = abs(d.mean())
    threshold = max(0.0, observed - 1e-14)
    nonzero = d[d != 0]
    if len(nonzero) <= 16:
        ns = 2 ** len(nonzero)
        count = 0
        for start in range(0, ns, 128):
            nums = np.arange(start, min(start + 128, ns), dtype=np.uint64)
            signs = ((nums[:, None] >> np.arange(len(nonzero), dtype=np.uint64)) & 1).astype(float) * 2 - 1
            count += int(np.count_nonzero(np.abs(signs @ nonzero / n) >= threshold))
        p = count / ns
        test = "exact_sign_flip"
    else:
        ns, count = resamples, 0
        for start in range(0, ns, 128):
            size = min(128, ns - start)
            signs = rng.integers(0, 2, size=(size, len(nonzero)), dtype=np.int8) * 2 - 1
            count += int(np.count_nonzero(np.abs(signs @ nonzero / n) >= threshold))
        p = (count + 1) / (ns + 1)
        test = "monte_carlo_sign_flip_plus_one"
    return dict(status="ok", n_pairs=n, n_dropped=int((~good).sum()), ours_mean=float(x.mean()),
                baseline_mean=float(y.mean()), mean_improvement=float(d.mean()),
                median_improvement=float(np.median(d)), ci_low=float(np.quantile(boot, .025)),
                ci_high=float(np.quantile(boot, .975)), fraction_ours_better=float(np.mean(d > 0)),
                fraction_tied=float(np.mean(d == 0)), p_value=float(p), test=test,
                n_bootstrap=resamples, n_permutations=ns, seed=seed,
                direction="higher" if metric in HIGHER else "lower")


def holm(pvalues):
    p = np.asarray(pvalues, float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    if len(order):
        result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(order), 0, -1)))
    return result


def tail_summary(values, higher=True):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n_queries": 0}
    ordered = np.sort(v)
    if not higher:
        ordered = ordered[::-1]
    row = dict(n_queries=len(v), mean=float(v.mean()), median=float(np.median(v)),
               worst_5pct_mean=float(ordered[:max(1, int(np.ceil(.05 * len(v))))].mean()),
               worst_1pct_mean=float(ordered[:max(1, int(np.ceil(.01 * len(v))))].mean()))
    for q in ([5, 10] if higher else [90, 95, 99]):
        row[f"p{q}"] = float(np.percentile(v, q))
    return row


def gains_vs_standard(frame):
    rows = []
    for dataset, group in frame.groupby("dataset", sort=False):
        valid = group[group["scope"].eq("full") & group["success"]]
        base = valid[valid.method.eq("standard_umap")]
        if base.empty or not np.isfinite(base.iloc[0].recall_at_15):
            continue
        ref = float(base.iloc[0].recall_at_15)
        for _, r in valid.iterrows():
            if np.isfinite(r.recall_at_15):
                gain = float(r.recall_at_15) - ref
                rows.append(dict(dataset=dataset, method=r.method, recall_at_15=r.recall_at_15,
                                 standard_umap=ref, absolute_gain=gain,
                                 relative_gain=gain / ref if ref != 0 else np.nan))
    return pd.DataFrame(rows, columns=["dataset", "method", "recall_at_15", "standard_umap", "absolute_gain", "relative_gain"])


def gain_summary(gains):
    ours = gains[gains.method.eq("ours")]
    d = ours.absolute_gain.to_numpy(float)
    relative = ours.relative_gain.dropna().to_numpy(float)
    wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    def stat(v, fn):
        return float(fn(v)) if len(v) else None
    return dict(n_datasets=len(d), wins=wins, losses=losses, ties=len(d) - wins - losses,
                mean_absolute_gain=stat(d, np.mean), median_absolute_gain=stat(d, np.median),
                n_relative=len(relative), mean_relative_gain=stat(relative, np.mean),
                median_relative_gain=stat(relative, np.median), min_relative_gain=stat(relative, np.min),
                max_relative_gain=stat(relative, np.max), sign_test_n=wins + losses,
                sign_test_p=float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue)
                if wins + losses else None)


def ranked_table(frame, metric, datasets):
    full = frame[frame.success & frame.scope.eq("full")]
    pivot = full.pivot(index="method", columns="dataset", values=metric).reindex(columns=datasets)
    pivot = pivot.dropna(how="all")
    summary = pd.DataFrame(index=pivot.index)
    summary["mean"] = pivot.mean(axis=1)
    summary["median"] = pivot.median(axis=1)
    summary["n_datasets"] = pivot.count(axis=1)
    cohort = list(pivot.index)
    shared = list(pivot.columns[pivot.notna().all(axis=0)]) if len(cohort) else []
    rule = "全手法が共通して測定されたデータセット"
    if not shared:
        cohort = list(pivot.index[pivot.count(axis=1) >= max(1, int(np.ceil(.75 * len(datasets))))])
        shared = list(pivot.columns[pivot.loc[cohort].notna().all(axis=0)]) if cohort else []
        rule = "全手法の共通集合不足のため、75%以上のデータセットで測定された固定手法群"
    for col in ("mean_rank", "median_rank", "n_rank_datasets", "wins", "top2"):
        summary[col] = np.nan
    summary["n_rank_datasets"] = 0
    ranks = pd.DataFrame()
    if cohort and shared:
        rank_values = pivot.loc[cohort, shared].round(12)
        ranks = rank_values.rank(ascending=metric in LOWER, method="average", axis=0)
        minimum_ranks = rank_values.rank(ascending=metric in LOWER, method="min", axis=0)
        summary.loc[cohort, "mean_rank"] = ranks.mean(axis=1)
        summary.loc[cohort, "median_rank"] = ranks.median(axis=1)
        summary.loc[cohort, "n_rank_datasets"] = len(shared)
        summary.loc[cohort, "wins"] = minimum_ranks.eq(1).sum(axis=1)
        summary.loc[cohort, "top2"] = minimum_ranks.le(2).sum(axis=1)
    info = dict(metric=metric, cohort=cohort, datasets=shared, rule=rule,
                ties="順位算出時のみ小数12桁に丸めて平均順位。wins/top2は同順位を含む最小順位。原測定値は変更しない。")
    return pivot.join(summary).reset_index(), ranks, info
