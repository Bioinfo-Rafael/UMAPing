"""matplotlibだけで作る比較図。PNGと編集可能なSVGを出力する。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .loading import DATASETS

LABELS = {"ours": "UMAPing (ours)", "ours_oracle_neighbors": "Ours: oracle neighbors",
          "no_repulsion": "Ours: no repulsion", "standard_umap": "Standard UMAP",
          "reduced_repulsion_umap": "Reduced repulsion UMAP", "weighted_knn": "Weighted kNN",
          "spectral_only": "Spectral only", "numap": "NUMAP", "paramrepulsor": "ParamRepulsor",
          "parametric_umap": "Parametric UMAP"}
NAMES = {"coil20": "COIL-20", "coil100": "COIL-100", "pancreas": "Pancreas",
         "pancreas_batch_corrected": "Pancreas (corrected)", "fashion_mnist": "Fashion-MNIST",
         "mnist_oos": "MNIST OOS", "organoid": "Organoid", "embryoid_body": "Embryoid body"}


def draw_figures(result, data, output):
    paths, skips = [], []
    frame = result["frame"]
    good = frame[frame.success & frame.scope.eq("full")]
    methods = list(good.method.unique())
    colors = {m: plt.get_cmap("tab20")(i % 20) for i, m in enumerate(methods)}
    colors["ours"] = "#b83239"
    colors["standard_umap"] = "#243f65"

    def save(fig, name):
        fig.savefig(output / (name + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(output / (name + ".svg"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(name + ".png")

    def heatmap(table, value, name, title, vmin=None, vmax=None):
        if table.empty or value not in table or table[value].notna().sum() == 0:
            skips.append(name + ": 必要な測定値なし")
            return
        pivot = table.pivot(index="method", columns="dataset", values=value).reindex(columns=DATASETS).dropna(how="all")
        fig, ax = plt.subplots(figsize=(11, max(3, .47 * len(pivot) + 1.7)))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#ededed")
        vals = pivot.to_numpy(float)
        im = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(DATASETS)), [NAMES[d] for d in DATASETS], rotation=30, ha="right")
        ax.set_yticks(range(len(pivot)), [LABELS.get(m, m) for m in pivot.index])
        for i in range(len(pivot)):
            for j in range(len(DATASETS)):
                x = vals[i, j]
                text = f"{x:.3f}" if np.isfinite(x) else "—"
                color = "white" if np.isfinite(x) and im.norm(x) < .55 else "#111111"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
        ax.set_title(title, loc="left", pad=14)
        fig.colorbar(im, ax=ax, shrink=.8, pad=.02)
        save(fig, name)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.titleweight": "bold"}):
        heatmap(good, "recall_at_15", "recall15_by_dataset", "OOS neighborhood Recall@15 · higher is better", 0, 1)
        gains = result["gains"]
        ours = gains[gains.method.eq("ours")].set_index("dataset").reindex(DATASETS)
        if ours.relative_gain.notna().any():
            fig, ax = plt.subplots(figsize=(9, 4.5))
            values = ours.relative_gain * 100
            mask = values.notna()
            x = np.arange(len(DATASETS))[mask]
            ax.bar(x, values[mask], color=["#b83239" if v >= 0 else "#527f9c" for v in values[mask]])
            ax.axhline(0, color="#444444", lw=1)
            ax.set_xticks(range(len(DATASETS)), [NAMES[d] for d in DATASETS], rotation=30, ha="right")
            ax.set_ylabel("Relative gain vs standard UMAP (%)")
            ax.set_title("UMAPing: Recall@15 improvement", loc="left")
            save(fig, "relative_gain_vs_umap")
        else:
            skips.append("relative_gain_vs_umap: 有効なstandard UMAPとの比率なし")

        ranks = result["main"].dropna(subset=["mean_rank"]).sort_values("mean_rank")
        if not ranks.empty:
            fig, ax = plt.subplots(figsize=(8, max(3, len(ranks) * .45 + 1.4)))
            y = np.arange(len(ranks))
            ax.scatter(ranks.mean_rank, y, c=[colors[m] for m in ranks.method], s=60)
            ax.set_yticks(y, [f"{LABELS.get(r.method, r.method)}  (rank n={int(r.n_rank_datasets)}, measured n={int(r.n_datasets)})" for r in ranks.itertuples()])
            ax.invert_yaxis()
            ax.set_xlabel("Mean rank on the same dataset / method panel (lower is better)")
            ax.set_title("Recall@15 ranks · missing cells excluded", loc="left")
            ax.grid(axis="x", alpha=.2)
            save(fig, "mean_rank")
        else:
            skips.append("mean_rank: 固定比較panelなし")

        selected = [m for m in ("ours", "standard_umap", "numap", "paramrepulsor", "parametric_umap") if m in methods]
        cols = [f"recall_at_{k}" for k in (5, 10, 15, 30)]
        common = [d for d in DATASETS if all(
            len(good[(good.dataset == d) & (good.method == m)].dropna(subset=cols)) == 1 for m in selected)]
        if selected and common:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for method in selected:
                values = good[good.method.eq(method) & good.dataset.isin(common)][cols]
                # 全kを同じ追加解析段階で比較できる場合は、その段階のRecall15も使う。
                curves = []
                for d in common:
                    entry = data.queries.get((d, method))
                    if entry and all(c in entry["frame"] for c in cols):
                        curves.append(entry["frame"][cols].mean().to_numpy())
                    else:
                        curves.append(good[(good.method == method) & (good.dataset == d)][cols].iloc[0].to_numpy())
                ax.plot([5, 10, 15, 30], np.mean(curves, axis=0), marker="o", label=LABELS.get(method, method), color=colors[method])
            ax.set(xlabel="k", ylabel="Mean neighborhood Recall@k", xticks=[5, 10, 15, 30], ylim=(0, 1))
            ax.set_title(f"Multi-k recall · same {len(common)} datasets for all curves", loc="left")
            ax.legend(frameon=False, fontsize=9)
            ax.grid(alpha=.2)
            save(fig, "recall_multi_k")
        else:
            skips.append("recall_multi_k: 全4つのkと選択手法が揃う共通datasetなし")

        tails = result["tails"]
        if not tails.empty:
            recall = tails[tails.metric.eq("recall_at_15") & tails.scope.eq("full")]
            heatmap(recall, "worst_5pct_mean", "tail_recall15", "Lowest 5% of queries: mean Recall@15 · higher is better", 0, 1)
        else:
            skips.append("tail_recall15: per-query値なし")
        periphery = result["periphery"]
        if not periphery.empty:
            sub = periphery[periphery.metric.eq("periphery_percentile") & periphery.scope.eq("full")]
            heatmap(sub, "fraction_ge_95", "periphery_comparison", "OOS periphery diagnostic: fraction at percentile ≥95 · lower is better", 0, 1)
        else:
            skips.append("periphery_comparison: 診断値なし")

        latency = good[(good.mean_query_latency_seconds > 0) & good.recall_at_15.notna()]
        if not latency.empty:
            fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
            for ax, d in zip(axes.flat, DATASETS):
                sub = latency[latency.dataset.eq(d)]
                for r in sub.itertuples():
                    ax.scatter(r.mean_query_latency_seconds, r.recall_at_15, color=colors[r.method], s=50,
                               marker="*" if r.method == "ours" else "o")
                if len(sub):
                    ax.set_xscale("log")
                else:
                    ax.text(.5, .5, "No measured latency", transform=ax.transAxes, ha="center", fontsize=8)
                ax.set_title(NAMES[d], fontsize=10)
                ax.set_xlabel("Mean query time (s)", fontsize=9)
                ax.set_ylim(0, 1)
                ax.grid(alpha=.2)
            axes[0, 0].set_ylabel("Recall@15")
            axes[1, 0].set_ylabel("Recall@15")
            used = latency.method.unique()
            fig.legend(handles=[Line2D([], [], color=colors[m], marker="o", linestyle="", label=LABELS.get(m, m)) for m in used],
                       loc="lower center", bbox_to_anchor=(.5, -.08), ncol=4, frameon=False, fontsize=8)
            fig.suptitle("Quality / latency · timing scopes and environments differ", fontsize=13)
            fig.tight_layout()
            save(fig, "latency_quality_tradeoff")
        else:
            skips.append("latency_quality_tradeoff: 正の実測時間なし")

        for dataset in DATASETS:
            ours = data.queries.get((dataset, "ours"))
            std = data.queries.get((dataset, "standard_umap"))
            if ours and std and ours["frame"].index.equals(std["frame"].index):
                a, b = ours["frame"], std["frame"]
                if "recall_at_15" in a and "recall_at_15" in b:
                    differences = (a.recall_at_15 - b.recall_at_15).dropna()
                    if len(differences):
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(differences, bins=np.linspace(-1, 1, 32), color="#527f9c", edgecolor="white")
                        ax.axvline(0, color="#b83239", lw=1.5)
                        ax.set(xlabel="Paired Recall@15: ours − standard UMAP", ylabel="Queries")
                        ax.set_title(NAMES[dataset], loc="left")
                        save(fig, "recall15_paired_distribution_" + dataset)
            entries = [(m, data.queries.get((dataset, m))) for m in ("ours", "standard_umap", "no_repulsion")]
            valid = [(m, e["frame"].local_displacement.dropna().to_numpy()) for m, e in entries if e and "local_displacement" in e["frame"]]
            valid = [(m, v) for m, v in valid if len(v)]
            if valid:
                fig, ax = plt.subplots(figsize=(6, 4))
                for method, v in valid:
                    ax.plot(np.sort(v), np.arange(1, len(v) + 1) / len(v), label=LABELS.get(method, method), color=colors[method])
                ax.set_xscale("symlog", linthresh=.1)
                ax.set(xlabel="Local displacement (lower is better)", ylabel="Empirical CDF", ylim=(0, 1))
                ax.set_title(NAMES[dataset], loc="left")
                ax.legend(frameon=False, fontsize=8)
                save(fig, "local_displacement_" + dataset)
            for name in ("recall15_paired_distribution_" + dataset, "local_displacement_" + dataset):
                if name + ".png" not in paths:
                    skips.append(name + ": 同じqueryの値、または対象指標が不足")
    return paths, skips, common
