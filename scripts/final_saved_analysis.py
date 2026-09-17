#!/usr/bin/env python3
"""CPU-only, read-only analysis of saved Embryoid body artifacts.

No project/pipeline imports: numpy, pandas, matplotlib and the standard library
are the complete dependency set. Does not load features, models or trajectories.
Outputs are transactional per figure and the destination must not exist.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import html
import json
import platform
import re
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex

SECTIONS = ["00_overview", "01_existing_split", "02_temporal_holdout", "03_teacher_estimators",
            "04_learned_field", "05_ablations", "99_provenance"]
METHODS = ["standard_umap", "reduced_repulsion_umap", "weighted_knn", "uniform_mc", "fit_grid"]
NAMES = dict(standard_umap="Standard UMAP", reduced_repulsion_umap="Reduced repulsion UMAP",
             weighted_knn="Weighted kNN", uniform_mc="Uniform", fit_grid="FitGrid",
             no_repulsion="No repulsion", fit_grid_oracle_neighbors="Oracle neighbors",
             exact_repulsion_diagnostic="Full-sum exact (50 queries)")
COLORS = dict(zip(METHODS, ["#606a79", "#ab7bba", "#d9a441", "#238a8d", "#d36448"]))
METRICS = ["recall_at_15", "ndcg", "density_log_distortion"]
METRIC_LABEL = dict(recall_at_15="Recall@15 (%) ↑", ndcg="NDCG ↑",
                    density_log_distortion="Density log distortion ↓")
CI_NOTE = "Paired cell bootstrap; fixed models and reference; not biological replication."
FIELD_NOTE = "t = embedding optimization time, NOT biological stage."
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150, "pdf.fonttype": 42, "axes.titlepad": 10})


class BudgetExceeded(BaseException):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def named(m):
    if m.startswith("barnes_hut_"):
        return "Barnes–Hut θ="+m.rsplit("_",1)[1]
    if m.startswith("fit_grid_") and m.rsplit("_",1)[1].isdigit():
        return "FitGrid size="+m.rsplit("_",1)[1]
    if m in {"dual_raw_is","dual_hub_is","dual_topl","barnes_hut"}:
        return {"dual_raw_is":"Raw Dual IS","dual_hub_is":"Hub Dual IS","dual_topl":"Top-L","barnes_hut":"Barnes–Hut"}[m]
    return NAMES.get(m, m.replace("_", " "))


class Analysis:
    def __init__(self, args):
        self.args, self.root, self.out = args, args.root.resolve(), args.output.resolve()
        require(not self.out.exists(), f"Refusing to overwrite existing output: {self.out}")
        require(self.out != self.root, "Invalid output")
        for forbidden in (self.root / "runs", self.root / "results", self.root / "src"):
            require(not self.out.is_relative_to(forbidden), "Output must not be inside inputs or src")
        self.start = time.monotonic()
        self.inventory, self.catalog, self.skips, self.checks = {}, [], [], []
        self.contexts, self.pairs, self.region_rows = {}, [], []
        self.current_inputs, self.current_task = set(), "initialization"
        self.executed, self.completed_tasks = [], []
        self.out.mkdir(parents=True)
        for folder in SECTIONS:
            (self.out / folder).mkdir()
        self.status = "running"

    def check_time(self):
        if time.monotonic() - self.start >= max(.001, self.args.max_seconds - min(10, self.args.max_seconds * .1)):
            raise BudgetExceeded()

    def track(self, path):
        self.check_time()
        p = Path(path).resolve()
        require(p.is_file(), f"Missing saved file: {p}")
        require(p.suffix.lower() in {".csv", ".json", ".npz", ".npy", ".txt", ".png", ".pdf"}, f"Forbidden input type: {p}")
        require(not any(s in p.name for s in ("trajectory", "checkpoint", "retriever_keys", "dual_keys", "dual_queries")), f"Forbidden large/model input: {p}")
        require(p.stat().st_size <= 64 * 1024**2, f"Input exceeds 64 MiB read cap: {p}")
        key = str(p)
        self.current_inputs.add(key)
        if key not in self.inventory:
            stat = p.stat()
            self.inventory[key] = dict(path=key, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                       sha256=sha(p), tasks=set(), unchanged=None)
        self.inventory[key]["tasks"].add(self.current_task)
        return p

    def csv(self, p):
        return pd.read_csv(self.track(p))

    def js(self, p):
        return json.loads(self.track(p).read_text())

    def npz(self, p, keys):
        with np.load(self.track(p), allow_pickle=False) as z:
            return {k: z[k] for k in keys}

    def skip(self, item, reason):
        self.skips.append(dict(item=item, reason=str(reason)))
        print(f"SKIP {item}: {reason}", flush=True)

    def task(self, name, fn):
        self.current_task, self.current_inputs = name, set()
        self.executed.append(name)
        print(f"TASK {name} ({time.monotonic()-self.start:.1f}s)", flush=True)
        try:
            self.check_time()
            fn()
            self.completed_tasks.append(name)
        except BudgetExceeded:
            self.skip(name, "CLI analysis/drawing time budget reached")
            raise
        except (FileNotFoundError, KeyError, ValueError, OSError, pd.errors.EmptyDataError) as exc:
            self.skip(name, f"{type(exc).__name__}: {exc}")
        finally:
            plt.close("all")

    def save(self, section, stem, fig, table, purpose, *, ctx=None, region="final embedding",
             note="", sources=(), query_count=None, seed=None):
        self.check_time()
        folder = self.out / section
        table = pd.DataFrame(table)
        if ctx is not None:
            table = table.copy()
            for col, val in (("split", ctx["split"]), ("training_seed", ctx["seed"])):
                if col not in table:
                    table[col] = val
            seed = ctx["seed"]
            if query_count is None:
                query_count = len(ctx["query_labels"])
            sources = list(sources) + ctx["sources"]
        source_list = sorted(set(sources) | self.current_inputs)
        subtitle = (f"{ctx['split']} | training seed {seed} | queries={query_count}" if ctx else
                    f"{region} | n={query_count if query_count is not None else 'see CSV'} | seed={seed if seed is not None else 'see CSV'}")
        fig.suptitle(stem.replace("_", " "), fontsize=14, y=.99)
        fig.text(.5, .035, subtitle, ha="center", fontsize=9)
        if note:
            import textwrap
            fig.text(.5, .007, "\n".join(textwrap.wrap(note, 155)), ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .085 if len(note) < 155 else .115, 1, .865 if fig.legends else .94))
        temp = []
        committed = False
        try:
            for ext in ("png", "pdf"):
                p = folder / f".{stem}.{ext}"
                temp.append(p)
                fig.savefig(p, format=ext)
                self.check_time()
            p = folder / f".{stem}.csv"
            temp.append(p)
            table.to_csv(p, index=False)
            for p in temp:
                p.rename(folder / p.name[1:])
            committed = True
        finally:
            for p in temp:
                if p.exists():
                    p.unlink()
            if not committed:
                for ext in ("png","pdf","csv"):
                    partial=folder/f"{stem}.{ext}"
                    if partial.exists():
                        partial.unlink()
            plt.close(fig)
        self.catalog.append(dict(figure_name=stem, purpose=purpose, relative_path=f"{section}/{stem}.png",
            pdf_path=f"{section}/{stem}.pdf", csv_path=f"{section}/{stem}.csv", inputs=";".join(source_list),
            split=ctx["split"] if ctx else "see CSV", training_seed=seed if seed is not None else "see CSV",
            query_count=query_count if query_count is not None else "see CSV", evaluation_region=region,
            creation="generated", notes=note, metric_direction="Recall/NDCG/cosine ↑; RMSE/density/MAE ↓; see axes"))

    def masks(self, frame, ctx):
        masks = {"all": np.ones(len(frame), bool)}
        if ctx["temporal"]:
            labels = frame.stage.to_numpy()
            masks.update(interpolation=labels == ctx["temporal"]["interpolation_timepoint"],
                         extrapolation=np.isin(labels, ctx["temporal"]["extrapolation_timepoints"]))
        return masks

    def load_context(self, key, folder, uniform_run, fit_run, expected_seed, temporal):
        base = self.root / "runs/embryoid_body"
        comp = base / "comparisons" / folder
        manifest = self.js(comp / "manifest.json")
        require(manifest["status"] == "complete", "Comparison is incomplete")
        require(manifest.get("source_hashes_unchanged") is True, "Comparison source integrity not verified by saved manifest")
        if "source_runs" in manifest:
            require([Path(p).name for p in manifest["source_runs"]]==[uniform_run,fit_run],"Comparison source run names mismatch")
        split = self.js(base / uniform_run / "temporal_split.json") if temporal else None
        seed_evidence = []
        for run in (uniform_run, fit_run):
            meta = self.js(base / run / "metadata.json")
            if temporal:
                require(meta.get("matched_seed") == expected_seed, f"Training seed mismatch: {run}")
                other = self.js(base / run / "temporal_split.json")
                require(other == split, f"Temporal split differs: {run}")
            seed_evidence.append(dict(run=run, matched_seed=meta.get("matched_seed"), seed=meta.get("seed"),
                                      checkpoint_source=meta.get("source_checkpoint"), provenance=meta))
        if temporal:
            initial=[s["provenance"].get("initial_checkpoint_sha256") for s in seed_evidence]
            require(initial[0]==initial[1],"Matched seed metadata reports different initial checkpoints")
        labels = self.npz(base / uniform_run / "cache/prepared_dataset.npz",
                          ["reference_label__stage", "query_label__stage"])
        saved_hash=manifest.get("upstream_hashes",{}).get("cache/prepared_dataset.npz")
        if saved_hash:
            actual_hash=self.inventory[str((base/uniform_run/"cache/prepared_dataset.npz").resolve())]["sha256"]
            require(saved_hash==actual_hash,"Prepared-dataset file hash differs from saved comparison provenance")
        ref, query = labels["reference_label__stage"].astype(str), labels["query_label__stage"].astype(str)
        if split:
            order = np.asarray(split["ordered_timepoints"])
            require(np.array_equal(query, order[np.asarray(split["query_ordinals"])-1]), "Query stage/split mismatch")
            require(np.array_equal(ref, order[np.asarray(split["reference_ordinals"])-1]), "Reference stage/split mismatch")
        aggregate = self.csv(comp / "aggregate.csv")
        frames = {}
        available = ["uniform_mc", "fit_grid"] if expected_seed == 1 else METHODS + ["no_repulsion", "fit_grid_oracle_neighbors", "exact_repulsion_diagnostic"]
        for method in available:
            p = comp / f"per_query_{method}.csv"
            if not p.is_file():
                self.skip(f"{key}/{method}", "Saved per-query CSV missing; no baseline execution")
                continue
            frame = self.csv(p)
            require(not frame.query_index.duplicated().any(), f"Duplicate query ID: {p}")
            require(frame.query_index.notna().all() and np.equal(frame.query_index, frame.query_index.astype(int)).all(), "Invalid query IDs")
            frame = frame.set_index("query_index").sort_index()
            ids = frame.index.to_numpy(int)
            require(((ids >= 0) & (ids < len(query))).all(), "Query ID outside split")
            require(set(frame.method) == {method}, "Method label mismatch")
            if method != "exact_repulsion_diagnostic":
                require(np.array_equal(ids, np.arange(len(query))), f"Missing full-query IDs: {method}")
            for col in ("timepoint", "label", "stage"):
                if col in frame:
                    require(np.array_equal(frame[col].astype(str), query[ids]), f"Stage mismatch: {method}/{col}")
            frame["stage"] = query[ids]
            require(np.isfinite(frame[METRICS].to_numpy()).all(), f"Nonfinite primary metric: {method}")
            frames[method] = frame
        ctx = dict(key=key, split="temporal holdout" if temporal else "existing split", seed=expected_seed,
                   temporal=split, comp=comp, uniform_run=uniform_run, fit_run=fit_run,
                   reference_labels=ref, query_labels=query, frames=frames, aggregate=aggregate,
                   section="02_temporal_holdout" if temporal else "01_existing_split",
                   seed_evidence=seed_evidence, evaluation_seed=manifest.get("seed"))
        if not temporal:
            # A comparison evaluation seed alone does not establish a training seed.
            ctx["seed"]="not recorded in run metadata"
            seeds=[s["matched_seed"] if s["matched_seed"] is not None else s["seed"] for s in seed_evidence]
            if all(s is not None for s in seeds) and len(set(seeds))==1:
                ctx["seed"]=seeds[0]
        for method, frame in frames.items():
            for group, mask in self.masks(frame, ctx).items():
                stored = aggregate[(aggregate.method == method) & (aggregate.group == group)]
                require(len(stored) == 1, f"Missing/duplicate aggregate: {method}/{group}")
                require(int(stored.iloc[0].n_queries) == int(mask.sum()), "Aggregate query count mismatch")
                for metric in METRICS:
                    actual, saved = float(frame.loc[mask, metric].mean()), float(stored.iloc[0][metric])
                    require(np.isclose(actual, saved, rtol=1e-8, atol=1e-10), f"Aggregate mismatch: {method}/{group}/{metric}")
                    self.checks.append(dict(check="aggregate", context=key, method=method, group=group,
                                            metric=metric, n_queries=int(mask.sum()), saved=saved, recomputed=actual,
                                            absolute_error=abs(actual-saved), status="passed"))
        ctx["sources"] = sorted(self.current_inputs)
        self.contexts[key] = ctx

    def summarize_frames(self, ctx, methods, subset=None):
        rows = []
        for method in methods:
            frame = ctx["frames"][method]
            if subset is not None:
                require(pd.Index(subset).isin(frame.index).all(), "Missing matched subset ID")
                frame = frame.loc[subset]
            for group, mask in self.masks(frame, ctx).items():
                part = frame.loc[mask]
                for metric in METRICS:
                    rows.append(dict(method=method, group=group, metric=metric, n_queries=len(part),
                                     mean=part[metric].mean(), display_mean=part[metric].mean()*(100 if metric == "recall_at_15" else 1),
                                     median=part[metric].median(), stage_counts=json.dumps(part.stage.value_counts().to_dict())))
        return pd.DataFrame(rows)

    def bar_metrics(self, table, metrics=METRICS, value="display_mean"):
        groups = list(dict.fromkeys(table.group))
        fig, axes = plt.subplots(len(groups), len(metrics), figsize=(5*len(metrics), 3.1*len(groups)+.9), squeeze=False)
        for row, group in enumerate(groups):
            for col, metric in enumerate(metrics):
                sub = table[(table.group == group) & (table.metric == metric)]
                ax = axes[row, col]
                ax.barh(np.arange(len(sub)), sub[value], color=[COLORS.get(m, "#8f9caa") for m in sub.method])
                ax.set_yticks(np.arange(len(sub)), [named(m) for m in sub.method], fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel(METRIC_LABEL.get(metric, metric))
                ax.set_title(f"{group} | n={sub.n_queries.iloc[0] if len(sub) else 0}")
                ax.grid(axis="x", alpha=.2)
        return fig

    def performance(self, ctx):
        methods = [m for m in METHODS if m in ctx["frames"]]
        require(len(methods) == 5, "Five-method comparison requires all saved methods")
        table = self.summarize_frames(ctx, methods)
        self.save(ctx["section"], "performance_by_method", self.bar_metrics(table), table,
                  "同じqueryでの最終埋め込み性能", ctx=ctx,
                  note="All methods use identical query IDs. Recall is percent; groups are not independent replicates.")

    def bootstrap(self, values, labels, seed):
        # Resample complete paired vectors, independently inside each time stratum.
        rng = np.random.default_rng(seed)
        strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        n, p = values.shape
        samples = np.empty((self.args.bootstrap, p))
        batch_size = max(1, min(32, int(self.args.bootstrap_memory_mb*1024**2 / max(1, n*(8+8*p)))))
        for start in range(0, len(samples), batch_size):
            self.check_time()
            size = min(batch_size, len(samples)-start)
            total = np.zeros((size, p))
            for ids in strata:
                index = rng.integers(0, len(ids), size=(size, len(ids)))
                total += values[ids[index]].sum(axis=1)
            samples[start:start+size] = total/n
        return np.quantile(samples, [.025, .975], axis=0), batch_size

    def paired(self, ctx):
        rows = []
        comparisons = [("uniform_mc", "standard_umap"), ("fit_grid", "standard_umap"), ("fit_grid", "uniform_mc")]
        for ours, baseline in comparisons:
            if baseline not in ctx["frames"]:
                continue
            a, b = ctx["frames"][ours], ctx["frames"][baseline]
            require(a.index.equals(b.index) and a.stage.equals(b.stage), "Paired ID/stage mismatch")
            for group, mask in self.masks(a, ctx).items():
                av, bv = a.loc[mask, METRICS].to_numpy(), b.loc[mask, METRICS].to_numpy()
                signs = np.array([1, 1, -1])
                d = (av-bv)*signs
                labels = a.loc[mask, "stage"].to_numpy() if ctx["temporal"] else np.zeros(len(d))
                keyed = int.from_bytes(hashlib.sha256(f"{self.args.seed}/{ctx['key']}/{ours}/{baseline}/{group}".encode()).digest()[:4], "little")
                ci, batch = self.bootstrap(d, labels, keyed)
                for i, metric in enumerate(METRICS):
                    scale = 100 if metric == "recall_at_15" else 1
                    rows.append(dict(split=ctx["split"], training_seed=ctx["seed"], group=group, ours=ours,
                        baseline=baseline, metric=metric, n_queries=len(d), ours_mean=av[:,i].mean(), baseline_mean=bv[:,i].mean(),
                        raw_mean_difference=(av-bv)[:,i].mean(), raw_median_difference=np.median((av-bv)[:,i]),
                        mean_improvement=d[:,i].mean(), median_improvement=np.median(d[:,i]),
                        ci_low=ci[0,i], ci_high=ci[1,i], display_improvement=d[:,i].mean()*scale,
                        display_ci_low=ci[0,i]*scale, display_ci_high=ci[1,i]*scale,
                        fraction_improved=np.mean(d[:,i] > 1e-12), fraction_tied=np.mean(abs(d[:,i]) <= 1e-12),
                        fraction_worsened=np.mean(d[:,i] < -1e-12), direction="ours-baseline" if signs[i] == 1 else "baseline-ours",
                        unit="percentage points" if i == 0 else "metric units", bootstrap_iterations=self.args.bootstrap,
                        bootstrap_seed=keyed, bootstrap_batch=batch, bootstrap_method="paired percentile; fixed time-stratum counts" if ctx["temporal"] else "paired percentile; iid cells",
                        ci_scope=CI_NOTE, source="new bootstrap", stage_counts=json.dumps(a.loc[mask,"stage"].value_counts().to_dict())))
        table = pd.DataFrame(rows)
        require(not table.empty, "No saved paired metrics")
        self.pairs.extend(rows)
        groups = list(dict.fromkeys(table.group))
        fig, axes = plt.subplots(len(groups), 3, figsize=(15, len(groups)*2.8+1), squeeze=False)
        for row, group in enumerate(groups):
            for col, metric in enumerate(METRICS):
                ax = axes[row,col]
                sub = table[(table.group == group) & (table.metric == metric)]
                y = np.arange(len(sub))
                ax.errorbar(sub.display_improvement, y,
                            xerr=[sub.display_improvement-sub.display_ci_low, sub.display_ci_high-sub.display_improvement],
                            fmt="o", capsize=4, color="#356e89")
                ax.set_yticks(y, [f"{named(a)} vs {named(b)}" for a,b in zip(sub.ours,sub.baseline)], fontsize=9)
                ax.axvline(0, color="gray", lw=1)
                ax.set_xlabel(("Recall gain (pp)" if col == 0 else metric+" improvement")+" → better")
                ax.set_title(f"{group} | n={sub.n_queries.iloc[0]}")
        stem = "paired_improvement_seed_1" if ctx["key"] == "temporal1" else "paired_improvement_vs_standard_umap"
        self.save(ctx["section"], stem, fig, table, "query対応差と95% CI（FitGrid vs Uniformも掲載）", ctx=ctx,
                  note="Improvement: ours-baseline for Recall/NDCG; baseline-ours for density. "+CI_NOTE)

    def distributions(self, ctx, zero=False):
        methods = [m for m in METHODS if m in ctx["frames"]]
        groups = list(self.masks(ctx["frames"][methods[0]], ctx))
        fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 4.6), squeeze=False)
        rows = []
        for g, ax in zip(groups, axes[0]):
            for i, method in enumerate(methods):
                f = ctx["frames"][method]
                values = f.loc[self.masks(f,ctx)[g], "recall_at_15"].to_numpy()
                if zero:
                    frac = np.mean(values == 0)
                    rows.append(dict(method=method, group=g, n_queries=len(values), zero_count=int((values==0).sum()), zero_fraction=frac))
                    ax.bar(i, frac*100, color=COLORS[method])
                else:
                    x, counts = np.unique(values, return_counts=True)
                    ecdf = np.cumsum(counts)/len(values)
                    ax.step(np.r_[0,x*100,100], np.r_[0,ecdf,1], where="post", label=named(method), color=COLORS[method])
                    rows.extend(dict(method=method, group=g, n_queries=len(values), recall_percent=v*100, count=int(n), ecdf=e) for v,n,e in zip(x,counts,ecdf))
            ax.set_title(f"{g} | n={len(values)}")
            if zero:
                ax.set_xticks(range(len(methods)), [named(m) for m in methods], rotation=35, ha="right", fontsize=8)
                ax.set_ylabel("Queries with Recall@15=0 (%) ↓")
            else:
                ax.set_xlabel("Recall@15 (%) ↑")
                ax.set_ylabel("Cumulative query fraction")
                ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        self.save(ctx["section"], "zero_recall_fraction" if zero else "recall_distribution", fig, rows,
                  "Recallゼロ率" if zero else "Recallの離散ECDF", ctx=ctx)

    def recall_k(self, ctx):
        collected, sources = {}, []
        for method in ("uniform_mc", "fit_grid"):
            path = ctx["comp"] / f"{method}_evaluation/metrics/advanced_per_query.csv"
            advanced = self.csv(path)
            sources.append(str(path))
            require(not advanced.duplicated(["method", "query_index"]).any(), "Duplicate advanced query ID")
            for source_method, output_method in (("ours_full", method), ("standard_umap", "standard_umap")):
                frame = advanced[advanced.method == source_method].set_index("query_index").sort_index()
                main = ctx["frames"][output_method]
                if not frame.index.equals(main.index) or not np.allclose(frame.recall_at_15, main.recall_at_15, rtol=1e-8, atol=1e-10):
                    self.skip(f"{ctx['key']}/recall_at_k/{method}/{output_method}", "Advanced Recall@15 or query IDs differ from primary CSV; not mixed")
                    continue
                require(np.array_equal(frame.label.astype(str), main.stage), "Advanced stage mismatch")
                if output_method in collected:
                    require(np.allclose(frame[[f"recall_at_{k}" for k in (5,10,15,30)]], collected[output_method][[f"recall_at_{k}" for k in (5,10,15,30)]]), "Duplicated Standard UMAP Recall@k disagrees")
                    continue
                frame["stage"] = main.stage
                collected[output_method] = frame
        require(bool(collected), "No validated saved Recall@k values")
        rows = []
        for method, frame in collected.items():
            for group, mask in self.masks(frame,ctx).items():
                for k in (5,10,15,30):
                    rows.append(dict(method=method, group=group, k=k, n_queries=int(mask.sum()),
                                     recall=frame.loc[mask,f"recall_at_{k}"].mean(), recall_percent=100*frame.loc[mask,f"recall_at_{k}"].mean()))
        table = pd.DataFrame(rows)
        fig, axes = plt.subplots(1,table.group.nunique(), figsize=(5*table.group.nunique(),4.6), squeeze=False)
        for (group,sub), ax in zip(table.groupby("group",sort=False), axes[0]):
            for method, part in sub.groupby("method",sort=False):
                ax.plot(part.k, part.recall_percent, "o-", label=named(method), color=COLORS[method])
            ax.set(title=f"{group} | n={sub.n_queries.iloc[0]}", xlabel="k", ylabel="Recall@k (%) ↑", xticks=[5,10,15,30])
            ax.legend(fontsize=8)
        self.save(ctx["section"], "recall_at_k", fig, table, "保存済みmulti-kの照合済み比較", ctx=ctx, sources=sources,
                  note="Only stored, ID/stage/Recall@15-validated rows; repeated Standard UMAP rows deduplicated.")

    def embeddings(self, ctx, highlight=False, difference=False):
        methods = (["uniform_mc"] if difference else
                   [m for m in ("standard_umap", "uniform_mc", "fit_grid") if m in ctx["frames"]])
        arrays, paths = {}, []
        for method in methods:
            path = ctx["comp"] / "embeddings" / f"{method}.npz"
            z = self.npz(path, ["reference", "query", "query_index"])
            paths.append(str(path))
            ids = z["query_index"]
            require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer), "Invalid saved embedding IDs")
            require(len(np.unique(ids)) == len(ids), "Duplicate embedding IDs")
            require(np.array_equal(np.sort(ids), ctx["frames"][method].index), "Embedding/metric ID mismatch")
            require(z["reference"].shape == (len(ctx["reference_labels"]),2) and z["query"].shape == (len(ids),2), "Invalid coordinate dimensions")
            require(np.isfinite(z["reference"]).all() and np.isfinite(z["query"]).all(), "Nonfinite coordinates")
            z["stage"] = ctx["query_labels"][ids]
            arrays[method] = z
        if "fit_grid" in arrays:
            require(np.array_equal(arrays["uniform_mc"]["reference"], arrays["fit_grid"]["reference"]), "Uniform/FitGrid reference coordinates differ")
        stages = sorted(set(ctx["reference_labels"]) | set(ctx["query_labels"]), key=lambda s: float(s.split("-")[0]))
        if ctx["temporal"]:
            stages = ctx["temporal"]["ordered_timepoints"]
        palette = {stage:to_hex(plt.get_cmap("tab10")(i)) for i,stage in enumerate(stages)}
        nrows = 2 if highlight else 1
        fig, axes = plt.subplots(nrows,len(methods),figsize=(5*len(methods),4.6*nrows+1),squeeze=False)
        rows = []
        common = np.concatenate([v[k] for m,v in arrays.items() if m != "standard_umap" for k in ("reference","query")])
        def limits(coords):
            low, high = coords.min(0), coords.max(0)
            span = max(float(np.max(high-low)),1e-9)*1.08
            center = (low+high)/2
            return center-span/2, center+span/2
        if difference:
            a, b = ctx["frames"]["fit_grid"], ctx["frames"]["uniform_mc"]
            require(a.index.equals(b.index) and a.stage.equals(b.stage), "Recall map ID/stage mismatch")
            delta = (a.recall_at_15-b.recall_at_15)*100
            bound = max(float(delta.abs().max()), 1)
        for col, method in enumerate(methods):
            z = arrays[method]
            low, high = limits(np.concatenate([z["reference"],z["query"]]) if method == "standard_umap" else common)
            for row in range(nrows):
                ax = axes[row,col]
                group = ("interpolation" if row == 0 else "extrapolation") if highlight else "all"
                mask = np.ones(len(z["query"]),bool)
                if highlight:
                    mask = (z["stage"] == ctx["temporal"]["interpolation_timepoint"] if row == 0 else np.isin(z["stage"],ctx["temporal"]["extrapolation_timepoints"]))
                if highlight or difference:
                    ax.scatter(*z["reference"].T,s=1,c="#d7d9dc",alpha=.55,rasterized=True)
                else:
                    ax.scatter(*z["reference"].T,s=1,c=[palette[s] for s in ctx["reference_labels"]],alpha=.18,rasterized=True)
                if difference:
                    im = ax.scatter(*z["query"].T,s=3,c=delta.loc[z["query_index"]],cmap="RdBu_r",vmin=-bound,vmax=bound,rasterized=True)
                    fig.colorbar(im,ax=ax,label="FitGrid − Uniform Recall@15 (pp)",shrink=.75)
                else:
                    ax.scatter(*z["query"][mask].T,s=3,c=[palette[s] for s in z["stage"][mask]],alpha=.8,rasterized=True)
                ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel="Embedding dimension 1",ylabel="Embedding dimension 2")
                ax.set_aspect("equal",adjustable="box")
                ax.set_title(f"{named(method)} | {group}\nreference={len(z['reference']):,}, query={int(mask.sum()):,}",fontsize=10)
                for stage in stages:
                    rows.append(dict(method=method, group=group, stage=stage, color=palette[stage] if not difference else "",
                        reference_count=int(np.sum(ctx["reference_labels"] == stage)), query_count=int(np.sum(mask & (z["stage"] == stage))),
                        total_queries=len(z["query"]), sampled=False, coordinate_source=paths[col],
                        label_source=str(self.root/"runs/embryoid_body"/ctx["uniform_run"]/"cache/prepared_dataset.npz"),
                        query_alignment="query_label__stage[query_index]", coordinate_system="independent Standard UMAP" if method=="standard_umap" else "shared Uniform/FitGrid reference",
                        delta_source="matched main per_query CSVs" if difference else "", colormap="RdBu_r" if difference else "fixed stage palette",
                        color_min=-bound if difference else np.nan,color_max=bound if difference else np.nan,
                        x_min=low[0],x_max=high[0],y_min=low[1],y_max=high[1]))
        if not difference:
            fig.legend(handles=[Line2D([],[],marker="o",ls="",color=palette[s],label=s) for s in stages],
                       title="Stage (reference ∪ query)",loc="upper center",bbox_to_anchor=(.5,.94),ncol=len(stages),fontsize=9)
        section = "05_ablations" if difference else ctx["section"]
        stem = ("fit_grid_minus_uniform_recall_map_seed_0" if difference else
                f"embedding_interpolation_extrapolation_seed_{ctx['seed']}" if highlight else "embedding_by_stage")
        if difference and not ctx["temporal"]:
            stem += "_existing_split"
        note = ("All queries. Uniform query coordinates are display positions only; association is not causation." if difference else
                "All saved queries; fixed stage colors. Standard UMAP has its own coordinate system; Uniform/FitGrid share axes. Equal aspect.")
        self.save(section,stem,fig,rows,"保存2D座標とstageの対応表示",ctx=ctx,sources=paths,note=note)

    def temporal_counts(self):
        ctx = self.contexts["temporal0"]
        split = ctx["temporal"]
        saved = self.csv(self.root/"runs/embryoid_body/temporal_holdout_uniform/metrics/temporal_split.csv")
        rows = []
        for stage in split["ordered_timepoints"]:
            role = "reference" if stage in split["reference_timepoints"] else "interpolation" if stage == split["interpolation_timepoint"] else "extrapolation"
            count = int(np.sum(ctx["reference_labels"]==stage)+np.sum(ctx["query_labels"]==stage))
            require(count == split["cells_per_timepoint"][stage], "Split count mismatch")
            stored=saved[saved.timepoint==stage]
            require(len(stored)==1 and int(stored.iloc[0].n_cells)==count and stored.iloc[0].role==role,
                    "Saved temporal_split.csv counts/role differ from labels/JSON")
            rows.append(dict(stage=stage,role=role,count=count,ordered_position=split["ordered_timepoints"].index(stage),
                             saved_split_csv_columns=",".join(saved.columns)))
        table = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(9,4.7))
        colors = dict(reference="#a9aeb8",interpolation="#238a8d",extrapolation="#d36448")
        ax.bar(table.stage,table["count"],color=[colors[r] for r in table.role])
        for i,r in table.iterrows():
            ax.text(i,r["count"]+80,f"{r['count']:,}\n{r.role}",ha="center",fontsize=9)
        ax.set(ylabel="Cells",xlabel="Biological stage",ylim=(0,table["count"].max()*1.2))
        self.save("02_temporal_holdout","split_cell_counts",fig,table,"実ラベル年代順とreference/queryの役割",ctx=ctx,note="Chronology and roles read from temporal_split.json.")

    def temporal_metric(self, metric):
        ctx = self.contexts["temporal0"]
        metrics = (["temporal_neighbor_mae", "excess_temporal_neighbor_mae", "temporal_bracketing_rate"] if metric == "diagnostics" else [metric])
        rows = []
        for method in METHODS:
            frame=ctx["frames"][method]
            for stage in ctx["temporal"]["query_timepoints"]:
                sub=frame[frame.stage==stage]
                for m in metrics:
                    rows.append(dict(method=method,group=stage,metric=m,n_queries=len(sub),n_valid=int(sub[m].notna().sum()),
                                     mean=sub[m].mean(),display_mean=sub[m].mean()*(100 if m in ("recall_at_15","temporal_bracketing_rate") else 1)))
        table=pd.DataFrame(rows)
        for m in metrics:
            METRIC_LABEL.setdefault(m,{"temporal_neighbor_mae":"Temporal neighbor MAE ↓ (ordinal)","excess_temporal_neighbor_mae":"Excess temporal MAE ↓ (ordinal)","temporal_bracketing_rate":"Bracketing rate (%)"}.get(m,m))
        stem={"recall_at_15":"recall_at_15_by_timepoint_seed_0","density_log_distortion":"density_distortion_by_timepoint_seed_0","diagnostics":"temporal_neighbor_diagnostics_seed_0"}[metric]
        self.save("02_temporal_holdout",stem,self.bar_metrics(table,metrics),table,"実timepoint別の保存指標",ctx=ctx,
                  note="Bracketing is auxiliary, not proof of biological correctness. Undefined groups remain missing; one extrapolation horizon only." if metric=="diagnostics" else "Same query IDs within each timepoint. No extrapolation-horizon trend fitted.")

    def seed_comparison(self):
        summary=self.root/"runs/embryoid_body/comparisons/temporal_summary_seeds_0_1"
        meta=self.js(summary/"manifest.json")
        require(meta["selected_seeds"]==[0,1],"Unexpected selected training seeds")
        saved=self.csv(summary/"matched_seed_results.csv")
        self.csv(summary/"matched_seed_summary.csv")
        for seed,key in ((0,"temporal0"),(1,"temporal1")):
            ctx=self.contexts[key]
            for method in ("uniform_mc","fit_grid"):
                for group,mask in self.masks(ctx["frames"][method],ctx).items():
                    row=saved[(saved.seed==seed)&(saved.method==method)&(saved.group==group)]
                    require(len(row)==1 and int(row.iloc[0].n_queries)==int(mask.sum()),"Seed summary count mismatch")
                    for metric in METRICS:
                        require(np.isclose(row.iloc[0][metric],ctx["frames"][method].loc[mask,metric].mean()),"Seed summary metric mismatch")
        table=pd.DataFrame(self.pairs)
        table=table[(table.split=="temporal holdout")&(table.ours=="fit_grid")&(table.baseline=="uniform_mc")].copy()
        require(set(table.training_seed)=={0,1},"Both completed seeds required")
        fig,axes=plt.subplots(1,3,figsize=(15,4.8))
        for ax,metric in zip(axes,METRICS):
            sub=table[table.metric==metric]
            for seed,marker in ((0,"o"),(1,"s")):
                vals=sub[sub.training_seed==seed].set_index("group").loc[["all","interpolation","extrapolation"]]
                y=vals.display_improvement.to_numpy()
                ax.errorbar(np.arange(3)+(seed-.5)*.1,y,yerr=[y-vals.display_ci_low,vals.display_ci_high-y],fmt=marker,ls="",label=f"Training seed {seed}",capsize=3)
            ax.axhline(0,color="gray",lw=1)
            ax.set(xticks=np.arange(3),xticklabels=["all","interpolation","extrapolation"],ylabel="FitGrid improvement (pp)" if metric=="recall_at_15" else "FitGrid improvement",title=METRIC_LABEL[metric])
            ax.legend(fontsize=8)
        self.save("02_temporal_holdout","fit_grid_minus_uniform_by_seed",fig,table,"学習seed 0・1を別々の点として表示",region="final embedding",query_count=12543,seed="0,1",
                  note="Each seed is separate; cells are never pooled across seeds. Positive = better. "+CI_NOTE,
                  sources=self.contexts["temporal0"]["sources"]+self.contexts["temporal1"]["sources"])

    def ablation(self, ctx, exact=False):
        methods=["uniform_mc","fit_grid","exact_repulsion_diagnostic"] if exact else ["uniform_mc","fit_grid","no_repulsion","fit_grid_oracle_neighbors"]
        require(all(m in ctx["frames"] for m in methods),"Saved ablation methods missing")
        subset=ctx["frames"]["exact_repulsion_diagnostic"].index if exact else None
        if exact:
            require(len(subset)==50,"Expected saved exact diagnostic of 50 queries")
        table=self.summarize_frames(ctx,methods,subset)
        stem="exact_repulsion_matched_50_queries" if exact else "repulsion_and_neighbor_ablation"
        if not ctx["temporal"]:
            stem+="_existing_split"
        n=len(subset) if exact else len(ctx["query_labels"])
        self.save("05_ablations",stem,self.bar_metrics(table),table,"保存済みablationの同一query比較",ctx=ctx,query_count=n,
                  note="Uniform/FitGrid restricted to the SAME saved 50 IDs. Full-sum exact is distinct from legacy MC repulsion oracle." if exact else "All query IDs matched. No repulsion and oracle-neighbor ablations use the saved FitGrid-run comparisons.")

    @property
    def benchmark(self):
        return self.root/"results/embryo_repulsion_estimators/recovery_20260915T104741Z_406100"

    def teacher_metrics(self, kind):
        manifest=self.js(self.benchmark/"config/manifest.json")
        table=self.csv(self.benchmark/"metrics/estimator_metrics.csv")
        require(table.status.eq("success").all(),"Incomplete teacher benchmark")
        require(not table.method.duplicated().any(),"Duplicate teacher settings")
        table["bias_scope"]="finite-repetition sample-mean bias, not theoretical bias"
        table["evaluation"]="old teacher benchmark; different Uniform model from main"
        table["theta"]=[m.rsplit("_",1)[1] if m.startswith("barnes_hut_") else np.nan for m in table.method]
        table["grid_size"]=[m.rsplit("_",1)[1] if m.startswith("fit_grid_") else np.nan for m in table.method]
        table["build_seconds_scope"]="saved construction time; separate from query runtime"
        names=[named(m) for m in table.method]
        if kind=="rmse":
            for _,r in table.iterrows():
                decomposed=r.bias_rmse**2+r.variance
                require(np.isclose(decomposed,r.mse,rtol=1e-6,atol=1e-12),"Stored teacher MSE/bias/variance decomposition inconsistent")
                require(np.isclose(r.rmse**2,r.mse,rtol=1e-8),"Stored teacher RMSE/MSE inconsistent")
                self.checks.append(dict(check="teacher_mse_decomposition",method=r.method,saved=r.mse,recomputed=decomposed,status="passed"))
            fig,ax=plt.subplots(figsize=(10,6))
            ax.barh(names,table.rmse,color="#447e93")
            ax.set_xlabel("Teacher vector RMSE vs saved exact ↓")
            ax.invert_yaxis()
            stem="estimator_rmse_by_method"
        elif kind=="bias":
            fig,axes=plt.subplots(1,3,figsize=(17,6))
            for ax,col,title in zip(axes,["bias_rmse","variance","bias_squared_debiased"],
                ["Finite-repeat mean bias RMSE ↓","Variance (mean squared vector norm)","Debiased squared bias estimate (may be <0)"]):
                ax.barh(np.arange(len(table)),table[col],color="#447e93")
                ax.set_yticks(np.arange(len(table)),names)
                ax.set_ylim(len(table)-.5,-.5)
                for i,value in enumerate(table[col]):
                    if not np.isfinite(value):
                        ax.text(0,i,"N/A (one repetition)",va="center",fontsize=8)
                ax.set_xlabel(title,fontsize=9)
            stem="estimator_bias_and_variance"
        elif kind=="runtime":
            fig,axes=plt.subplots(1,2,figsize=(16,6))
            for i,row in table.iterrows():
                axes[0].scatter(row.runtime_seconds_per_query*1e6,row.rmse,s=40,label=names[i])
            axes[0].set(xscale="log",xlabel="Saved runtime per query (µs)",ylabel="Teacher RMSE ↓")
            axes[0].legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.01,1))
            axes[1].barh(names,table.build_seconds,color="#879aaa")
            axes[1].invert_yaxis()
            axes[1].set_xlabel("Saved construction time (s)")
            stem="estimator_runtime_vs_rmse"
        elif kind=="ess":
            # Stored ESS is a query-weighted mean across diagnostic batches/repetitions.
            part=table[table.method.isin(["dual_raw_is","dual_hub_is"])].copy()
            for _,row in part.iterrows():
                diag=self.csv(self.benchmark/"estimator_benchmark"/f"{row.method}_diagnostics.csv")
                value=np.average(diag.ess,weights=diag.n_queries)
                require(np.isclose(value,row.ess),"Stored ESS/diagnostics mismatch")
                self.checks.append(dict(check="teacher_ess",method=row.method,saved=row.ess,recomputed=value,status="passed"))
            fig,ax=plt.subplots(figsize=(8,4.5))
            ax.bar([named(m) for m in part.method],part.ess,color=["#9677aa","#447e93"])
            ax.set_ylabel("Saved mean ESS (of 64 importance samples) ↑")
            table=part
            stem="importance_sampling_ess"
        self.save("03_teacher_estimators",stem,fig,table,"旧教師推定器benchmarkの全保存設定",region="old teacher benchmark; near-reference validation",seed=manifest.get("seed"),query_count=int(table.n_queries.iloc[0]),
                  note="All saved theta/grid settings retained. Stochastic: 50 repeats × 1,000 queries; deterministic: 1 × 1,000. Recorded timing only; separate model from main.")

    def importance_weights(self):
        needed=[self.benchmark/"estimator_benchmark"/f"{m}_weights.npy" for m in ("dual_raw_is","dual_hub_is")]
        if not all(p.is_file() for p in needed):
            self.skip("teacher/importance_weight_reaggregation","Saved weight arrays unavailable locally; existing figure copied, no reconstructed histogram")
            source=self.track(self.benchmark/"figures/importance_weight_distribution.png")
            fig,ax=plt.subplots(figsize=(12,6))
            ax.imshow(plt.imread(source))
            ax.axis("off")
            table=[dict(source=str(source),status="existing figure; numerical histogram unavailable",query_count=1000,repetitions=50)]
            self.save("03_teacher_estimators","importance_weight_distribution",fig,table,"既存importance weight図の再掲",region="old teacher benchmark",seed=0,query_count=1000,
                      note="EXISTING FIGURE. Saved PNG reproduced; raw weights absent locally. Histogram values were not reconstructed.")
            self.catalog[-1]["creation"]="copy of existing figure (annotated PNG/PDF)"
            return
        rows=[]
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        bins=np.logspace(-6,5,111)
        for method,ax in zip(["dual_raw_is","dual_hub_is"],axes):
            path=self.benchmark/"estimator_benchmark"/f"{method}_weights.npy"
            weights=np.load(self.track(path),allow_pickle=False,mmap_mode="r")
            require(weights.shape==(3200000,),f"Unexpected importance weight shape {weights.shape}")
            counts=np.zeros(len(bins)-1,dtype=np.int64)
            zeros=under=over=0
            for start in range(0,len(weights),65536):
                self.check_time()
                v=np.asarray(weights[start:start+65536]).ravel()
                require(np.isfinite(v).all() and (v>=0).all(),"Invalid saved importance weights")
                zeros+=int((v==0).sum());under+=int(((v>0)&(v<bins[0])).sum());over+=int((v>bins[-1]).sum())
                counts+=np.histogram(v,bins=bins)[0]
            ax.stairs(counts/weights.size,bins,fill=True,color="#447e93")
            ax.set(xscale="log",yscale="log",xlabel="Importance weight",ylabel="Fraction of all saved weights",title=named(method))
            rows.extend(dict(method=method,bin_left=lo,bin_right=hi,count=int(n),fraction=n/weights.size,total_weights=weights.size,
                             zero_count=zeros,positive_underflow_count=under,overflow_count=over,
                             shape=str(weights.shape),layout="50 repetitions × 1000 queries, 64 samples; histogram aggregates weights") for lo,hi,n in zip(bins[:-1],bins[1:],counts))
        self.save("03_teacher_estimators","importance_weight_distribution",fig,rows,"保存importance weightの全値ヒストグラム",region="old teacher benchmark",seed=0,query_count=1000,
                  note="50 repetitions are not independent query cells. Fixed log bins; zero/underflow/overflow counts retained in CSV.")

    def benchmark_curves(self, kind):
        manifest=self.js(self.benchmark/"config/manifest.json")
        methods=["uniform_mc","dual_raw_is","dual_hub_is","dual_topl","barnes_hut","fit_grid"]
        rows=[]
        fig,ax=plt.subplots(figsize=(12,5.7))
        for method in methods:
            filename="validation.csv" if kind=="validation" else "losses.csv"
            table=self.csv(self.benchmark/"repulsion_training"/method/filename)
            require(table.step.is_unique and table.step.is_monotonic_increasing,"Invalid saved training step sequence")
            table["method"]=method
            setting=manifest["selected"][method]["name"]
            table["teacher_setting"]=setting
            col="rmse" if kind=="validation" else "loss" if kind=="loss" else "rolling_std"
            if col=="rolling_std" and col not in table:
                table[col]=table.loss.rolling(100,min_periods=2).std()
                table["rolling_definition"]="recomputed window=100 min_periods=2 ddof=1"
            else:
                table["rolling_definition"]="saved rolling_std (training implementation window=100, expanding before 100)"
            rows.extend(table.to_dict("records"))
            if kind=="loss":
                ax.plot(table.step,table.loss,alpha=.17,lw=.6)
                ax.plot(table.step,table.loss.rolling(100,min_periods=1).mean(),label=named(setting),lw=1.3)
            else:
                ax.plot(table.step,table[col],label=named(setting),lw=1.2)
        ax.set(xlabel="Training step",ylabel={"validation":"Validation RMSE against saved exact ↓","loss":"Teacher-specific training loss (100-step mean)","std":"Rolling loss standard deviation"}[kind])
        ax.legend(fontsize=9,ncol=3)
        ax.grid(alpha=.2)
        stem={"validation":"benchmark_exact_validation_rmse_by_step","loss":"benchmark_training_loss","std":"benchmark_training_loss_rolling_std"}[kind]
        self.save("04_learned_field",stem,fig,rows,"旧benchmarkの保存学習ログ",region="old benchmark training / fixed exact validation",seed=0,query_count=1000 if kind=="validation" else "training batches",
                  note="Old benchmark models are separate from main. Teacher-specific loss/std are not shared-ground-truth performance or rankings.")

    def current_training_curves(self):
        rows=[]
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        specs=[("existing","main","fit_grid"),("temporal0","temporal_holdout_uniform","temporal_holdout_fit_grid"),
               ("temporal1","temporal_seed_1_uniform","temporal_seed_1_fit_grid")]
        for ax,(key,uniform,fit) in zip(axes,specs):
            ctx=self.contexts.get(key,dict(split="existing split" if key=="existing" else "temporal holdout",seed="unverified (comparison inputs absent)"))
            for method,run in (("uniform_mc",uniform),("fit_grid",fit)):
                root=self.root/"runs/embryoid_body"/run
                if not (root/"metrics/repulsion_training.json").is_file():
                    self.skip(f"training/{run}","Saved training log unavailable")
                    continue
                if key!="existing" and key not in self.contexts:
                    metadata=self.js(root/"metadata.json")
                    expected=1 if key=="temporal1" else 0
                    require(metadata.get("matched_seed")==expected,"Training log matched_seed mismatch")
                    ctx["seed"]=expected
                training=self.js(root/"metrics/repulsion_training.json")
                path=root/"metrics/losses.csv"
                if path.is_file():
                    table=self.csv(path)
                    if "loss" not in table:
                        raise ValueError(f"No loss column in {path}")
                    origin="metrics/losses.csv"
                    require(len(table)==len(training["losses"]) and np.allclose(table.loss,training["losses"]),"Saved CSV/JSON training losses disagree")
                else:
                    require("losses" in training,"No saved training loss series")
                    data=training["losses"]
                    table=pd.DataFrame(data) if len(data) and isinstance(data[0],dict) else pd.DataFrame({"step":np.arange(1,len(data)+1),"loss":data})
                    origin="metrics/repulsion_training.json losses"
                table=table.assign(method=method,split=ctx["split"],training_seed=ctx["seed"],source=origin)
                rows.extend(table.to_dict("records"))
                ax.plot(table.step,table.loss.rolling(100,min_periods=1).mean(),label=named(method),color=COLORS[method])
            title_seed=ctx["seed"] if isinstance(ctx["seed"],int) else "unverified" if key=="existing" else "0 requested" if key=="temporal0" else "1 requested"
            ax.set(title=f"{ctx['split']}\ntraining seed: {title_seed}",xlabel="Training step",ylabel="Teacher-specific loss (100-step mean)")
            if ax.lines:
                ax.legend()
            else:
                ax.text(.5,.5,"Saved logs unavailable",transform=ax.transAxes,ha="center")
        require(bool(rows),"No saved training logs available")
        self.save("04_learned_field","current_run_training_loss",fig,rows,"現行runの保存学習ログ",region="current model training",seed="see CSV (only verified values labeled)",query_count="training logs",
                  note="Different teachers imply different training targets. These losses are not common exact-field validation performance.",
                  sources=sum([c["sources"] for c in self.contexts.values()],[]))

    def field_data(self):
        near,binned,grid,errors=[],[],[],[]
        for key,ctx in self.contexts.items():
            base=ctx["comp"]/"exact_field"
            try:
                z=self.npz(base/"validation.npz",["t","exact"])
                t,truth=z["t"].reshape(-1),z["exact"].astype(float)
                require(truth.shape==(1000,2) and t.shape==(1000,),"Unexpected near-reference evaluation shape")
                require(np.isfinite(t).all() and ((t>=0)&(t<=1)).all(),"Invalid saved optimization time")
                saved=self.csv(base/"metrics.csv")
                for method in ("uniform_mc","fit_grid"):
                    pred=np.load(self.track(base/f"{method}.npy"),allow_pickle=False).astype(float)
                    require(pred.shape==truth.shape and np.isfinite(pred).all() and np.isfinite(truth).all(),"Invalid field arrays")
                    e=np.linalg.norm(pred-truth,axis=1)
                    denom=np.linalg.norm(pred,axis=1)*np.linalg.norm(truth,axis=1)
                    valid=(np.linalg.norm(pred,axis=1)>1e-12)&(np.linalg.norm(truth,axis=1)>1e-12)
                    cosine=np.sum(pred*truth,axis=1)/np.maximum(denom,1e-24)
                    row=dict(context=key,split=ctx["split"],training_seed=ctx["seed"],method=method,region="near reference trajectory",n_points=len(t),
                             rmse=float(np.sqrt(np.mean(e**2))),cosine=float(cosine[valid].mean()),cosine_valid_count=int(valid.sum()),
                             error_median=float(np.median(e)),error_p05=float(np.quantile(e,.05)),error_p95=float(np.quantile(e,.95)),
                             rmse_definition="sqrt(mean(sum((pred-exact)^2,axis=2D)))")
                    ref=saved[saved.method==method]
                    require(len(ref)==1 and np.isclose(row["rmse"],ref.iloc[0].rmse,rtol=1e-8),"Field RMSE differs from saved metrics")
                    require(np.isclose(row["cosine"],ref.iloc[0].cosine,rtol=1e-8),"Field cosine differs from saved metrics")
                    self.checks.append(dict(check="field_metrics",context=key,method=method,saved=ref.iloc[0].rmse,recomputed=row["rmse"],status="passed"))
                    near.append(row)
                    values,counts=np.unique(e,return_counts=True)
                    errors.extend(dict(context=key,method=method,error=v,ecdf=c/len(e),n_points=len(e)) for v,c in zip(values,np.cumsum(counts)))
                    bins=np.linspace(0,1,11)
                    for i,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
                        mask=(t>=lo)&((t<hi) if i<9 else (t<=hi))
                        binned.append(dict(context=key,split=ctx["split"],training_seed=ctx["seed"],method=method,t_left=lo,t_right=hi,t_mid=(lo+hi)/2,n_points=int(mask.sum()),
                                           rmse=float(np.sqrt(np.mean(e[mask]**2))) if mask.any() else np.nan,bin_closed="left; last bin both",time_definition=FIELD_NOTE))
            except (ValueError,FileNotFoundError,KeyError) as exc:
                self.skip(f"field/{key}/near_reference",exc)
            for method in ("uniform_mc","fit_grid"):
                path=ctx["comp"]/f"{method}_evaluation/metrics/advanced_analysis.json"
                if not path.exists():
                    self.skip(f"field/{key}/{method}/spatial_grid","Saved spatial-grid exact evaluation absent; missing, not imputed")
                    continue
                data=self.js(path)["field_denoising"]
                items=list(data["per_t"].items())+[("mixed",data["mixed"])]
                for label,part in items:
                    require("vs_exact" in part,"No saved vs_exact; legacy MC teacher comparison excluded")
                    values=part["vs_exact"]["b_phi"]
                    # The legacy advanced code calls mean squared VECTOR error mse_mean.
                    rmse=float(np.sqrt(values["mse_mean"]))
                    grid.append(dict(context=key,split=ctx["split"],training_seed=ctx["seed"],method=method,region="spatial grid",time_label=label,
                                     t=float(label.split("=")[-1]) if label!="mixed" else np.nan,n_points=int(part["n_eval_points"]),rmse=rmse,
                                     cosine=values.get("cosine_mean",np.nan),rmse_definition="sqrt(saved vs_exact.b_phi.mse_mean)",source=str(path),
                                     saved_exact_metrics=json.dumps(values)))
        self.near,self.binned,self.grid,self.field_errors=map(pd.DataFrame,(near,binned,grid,errors))
        self.field_sources=sorted(self.current_inputs)

    def field_plot(self,kind):
        require(hasattr(self,"near"),"Field inputs not loaded")
        contexts=list(self.contexts)
        titles={k:f"{self.contexts[k]['split']} | seed {self.contexts[k]['seed']}" for k in contexts}
        if kind=="near":
            require(not self.near.empty,"No near-reference saved predictions")
            fig,axes=plt.subplots(1,2,figsize=(13,5))
            table=self.near.copy()
            for method in ("uniform_mc","fit_grid"):
                sub=table[table.method==method].set_index("context").reindex(contexts)
                x=np.arange(len(contexts))+(-.15 if method=="uniform_mc" else .15)
                for ax,col in zip(axes,["rmse","cosine"]):
                    ax.bar(x,sub[col],width=.28,label=named(method),color=COLORS[method])
                    ax.set_xticks(range(len(contexts)),[titles[k] for k in contexts],fontsize=8)
                    ax.set_ylabel("Vector RMSE ↓" if col=="rmse" else "Cosine ↑")
                    ax.legend()
            stem="near_reference_field_accuracy"
        elif kind=="distribution":
            table=self.field_errors
            require(not table.empty,"No field errors")
            fig,axes=plt.subplots(1,len(contexts),figsize=(5*len(contexts),4.7),squeeze=False)
            for key,ax in zip(contexts,axes[0]):
                for method in ("uniform_mc","fit_grid"):
                    sub=table[(table.context==key)&(table.method==method)]
                    ax.step(sub.error,sub.ecdf,label=named(method),color=COLORS[method])
                ax.set(title=titles[key],xlabel="Vector error norm ↓",ylabel="ECDF")
                ax.legend()
            stem="near_reference_field_error_distribution"
        elif kind in ("near_time","grid_time"):
            require(not (self.binned.empty if kind=="near_time" else self.grid.empty),"No saved evaluation by optimization time")
            table=self.binned if kind=="near_time" else self.grid[self.grid.time_label!="mixed"]
            require(not table.empty,"No saved evaluation by optimization time")
            fig,axes=plt.subplots(1,len(contexts),figsize=(5*len(contexts),4.9),squeeze=False)
            for key,ax in zip(contexts,axes[0]):
                for method in ("uniform_mc","fit_grid"):
                    sub=table[(table.context==key)&(table.method==method)]
                    x=sub.t_mid if kind=="near_time" else sub.t
                    ax.plot(x,sub.rmse,"o-",label=named(method),color=COLORS[method])
                    for tx,rmse,n in zip(x,sub.rmse,sub.n_points):
                        ax.annotate(str(n),(tx,rmse),xytext=(0,5),textcoords="offset points",fontsize=6)
                if not (table.context==key).any():
                    ax.text(.5,.5,"Not saved",ha="center",transform=ax.transAxes)
                ax.set(title=titles[key],xlabel="Embedding optimization time t",ylabel="Vector RMSE ↓")
                ax.legend(fontsize=8)
            stem="near_reference_field_error_by_optimization_time" if kind=="near_time" else "spatial_grid_field_error_by_optimization_time"
        else:
            table=pd.concat([self.near,self.grid[self.grid.time_label=="mixed"] if not self.grid.empty else self.grid],ignore_index=True)
            require(not table.empty,"No saved region metrics")
            table["fit_grid_over_uniform_rmse"]=np.nan
            for (key,region),sub in table.groupby(["context","region"]):
                indexed=sub.set_index("method")
                if {"fit_grid","uniform_mc"}.issubset(indexed.index):
                    table.loc[sub.index,"fit_grid_over_uniform_rmse"]=indexed.loc["fit_grid","rmse"]/indexed.loc["uniform_mc","rmse"]
            # Explicit missing row keeps the absent seed-1 spatial distribution visible.
            for key in contexts:
                if not ((table.context==key)&(table.region=="spatial grid")).any():
                    for m in ("uniform_mc","fit_grid"):
                        table=pd.concat([table,pd.DataFrame([dict(context=key,split=self.contexts[key]["split"],training_seed=self.contexts[key]["seed"],region="spatial grid",method=m,n_points=np.nan,rmse=np.nan,missing_reason="not saved")])],ignore_index=True)
            fig,axes=plt.subplots(2,2,figsize=(14,8.5))
            for row,region in enumerate(["near reference trajectory","spatial grid"]):
                sub=table[table.region==region]
                for method in ("uniform_mc","fit_grid"):
                    vals=sub[sub.method==method].set_index("context").reindex(contexts)
                    axes[row,0].bar(np.arange(len(contexts))+(-.15 if method=="uniform_mc" else .15),vals.rmse,width=.28,color=COLORS[method],label=named(method))
                ratio=sub[sub.method=="fit_grid"].set_index("context").reindex(contexts).fit_grid_over_uniform_rmse
                axes[row,1].bar(np.arange(len(contexts)),ratio,color=["#238a8d" if v<1 else "#d36448" for v in ratio])
                axes[row,1].axhline(1,color="gray",ls="--")
                for col in (0,1):
                    axes[row,col].set_xticks(range(len(contexts)),[titles[k] for k in contexts],fontsize=8)
                    axes[row,col].set_title(region+(" | 1,000 saved points" if row==0 else " | 300 saved points, mixed t"))
                    axes[row,col].set_ylabel("Vector RMSE ↓" if col==0 else "FitGrid / Uniform RMSE (<1 better)")
                    for i,key in enumerate(contexts):
                        if sub[(sub.context==key)].rmse.isna().all():
                            axes[row,col].text(i,.5,"not saved",ha="center",transform=axes[row,col].get_xaxis_transform(),fontsize=8)
                axes[row,0].legend()
            stem="field_accuracy_by_evaluation_region"
            self.region_rows=table.to_dict("records")
        region="near reference trajectory vs spatial grid" if kind=="region" else "spatial grid" if kind=="grid_time" else "near reference trajectory"
        self.save("04_learned_field",stem,fig,table,"保存exact値に対する学習済み場の精度",region=region,seed="0,1",query_count="1,000 near-reference / 300 grid",sources=self.field_sources,
                  note=FIELD_NOTE+" Different evaluation distributions are kept separate. Labels on time curves are point counts; seed-1 grid is missing.")

    def overview(self):
        selected=[
            ("01_existing_split/performance_by_method","existing_split_final_performance"),
            ("02_temporal_holdout/performance_by_method","temporal_seed_0_final_performance"),
            ("02_temporal_holdout/paired_improvement_vs_standard_umap","temporal_paired_improvement_and_ci"),
            ("02_temporal_holdout/embedding_interpolation_extrapolation_seed_0","temporal_interpolation_extrapolation_embeddings"),
            ("02_temporal_holdout/fit_grid_minus_uniform_by_seed","fit_grid_uniform_seed_comparison"),
            ("03_teacher_estimators/estimator_rmse_by_method","old_teacher_benchmark_rmse"),
            ("03_teacher_estimators/estimator_runtime_vs_rmse","old_teacher_benchmark_runtime_accuracy"),
            ("04_learned_field/field_accuracy_by_evaluation_region","learned_field_accuracy_by_region"),
        ]
        for source,destination in selected:
            found=next((r for r in self.catalog if r["relative_path"]==source+".png"),None)
            if found is None:
                self.skip(f"00_overview/{destination}",f"Required fixed overview source unavailable: {source}")
                continue
            for ext in ("png","pdf","csv"):
                shutil.copyfile(self.out/(source+"."+ext),self.out/"00_overview"/(destination+"."+ext))
            row=dict(found)
            row.update(figure_name=destination,relative_path=f"00_overview/{destination}.png",pdf_path=f"00_overview/{destination}.pdf",
                       csv_path=f"00_overview/{destination}.csv",creation="copy of generated figure",notes=found["notes"]+f" | Overview physical copy of {source}")
            self.catalog.append(row)

    def write_readme(self):
        lines=["# Embryoid body 保存済み成果物の追加解析", "",
               "[全図の静的ギャラリー](index.html) / [図の索引](99_provenance/figure_catalog.csv) / [スキップ理由](99_provenance/skipped_items.csv)","",
               f"実行状態：**{self.status}**。利用可能だったcomparison：{', '.join(self.contexts) or 'なし（入力未配置）'}。欠測した比較は未検証です。", "",
               "## 実行範囲", "",
               "保存済みCSV・JSON・2D座標・予測値だけをCPUで読み取りました。再学習、モデルロード、推論、UMAP fit/transform、ODE積分、exact場・高次元近傍の再計算はありません。依存パッケージは追加していません。出力は新規ディレクトリで、入力成果物とは分離しています。", "",
               "既存split、temporal seed 0・1、旧教師benchmark、参照軌道近傍の場評価、空間格子の場評価を区別します。comparison manifestのseedは評価seedです。temporalの学習seedはrun metadataのmatched_seedを照合しています。seed 2は除外しました。", "",
               "## 保存値・追加集計から直接確認できる結果", ""]
        for key,ctx in self.contexts.items():
            lines.append(f"### {ctx['split']}：学習seed {ctx['seed']}")
            lines.append(f"\nReference {len(ctx['reference_labels']):,}、query {len(ctx['query_labels']):,}。各手法のID重複・欠損とstageを検証し、主指標の平均をaggregate.csvと照合しました。\n")
            lines.append("| 方法 | 集団 | n | Recall@15 (%) | NDCG | density log distortion ↓ |\n|---|---|---:|---:|---:|---:|")
            for method in METHODS:
                if method not in ctx["frames"]:
                    continue
                f=ctx["frames"][method]
                for group,mask in self.masks(f,ctx).items():
                    p=f.loc[mask]
                    lines.append(f"| {named(method)} | {group} | {len(p):,} | {p.recall_at_15.mean()*100:.3f} | {p.ndcg.mean():.4f} | {p.density_log_distortion.mean():.4f} |")
            lines.append("")
        if self.pairs:
            lines.extend(["### 対応差", "", "Recall差はパーセントポイント（pp）です。相対改善率ではありません。", "",
                          "| split | 学習seed | 集団 | 比較 | Recall差 (pp) | 95% CI (pp) | n |", "|---|---:|---|---|---:|---|---:|"])
            for r in self.pairs:
                if r["metric"]=="recall_at_15":
                    lines.append(f"| {r['split']} | {r['training_seed']} | {r['group']} | {named(r['ours'])} − {named(r['baseline'])} | {r['display_improvement']:+.3f} | [{r['display_ci_low']:+.3f}, {r['display_ci_high']:+.3f}] | {r['n_queries']:,} |")
            lines.append("")
        if self.region_rows:
            lines.extend(["### 学習済み場：異なる評価分布", "", "| split | 学習seed | 領域 | 方法 | n | RMSE ↓ | FitGrid/Uniform RMSE |", "|---|---:|---|---|---:|---:|---:|"])
            for r in self.region_rows:
                lines.append(f"| {r['split']} | {r['training_seed']} | {r['region']} | {named(r['method'])} | {r.get('n_points',np.nan)} | {r.get('rmse',np.nan):.6g} | {r.get('fit_grid_over_uniform_rmse',np.nan):.4g} |")
            lines.extend(["", "near-referenceは参照軌道近傍1,000点、spatial gridは300点の保存された空間評価です。格子の全体値はmixed tを採用し、固定tの値は別図に載せます。RMSEは2成分の二乗誤差の和を点平均して平方根を取る定義です。tは埋め込み最適化時間であり、生物学的stageではありません。seed 1の空間評価は欠測です。", ""])
        benchmark_path=self.out/"03_teacher_estimators/estimator_rmse_by_method.csv"
        if benchmark_path.is_file():
            table=pd.read_csv(benchmark_path)
            lines.extend(["### 旧教師推定器benchmark", "", "| 設定 | 反復 | query数 | RMSE ↓ | query時間 (µs) | 構築時間 (s) |", "|---|---:|---:|---:|---:|---:|"])
            for _,r in table.iterrows():
                lines.append(f"| {r.method} | {r.repetitions} | {r.n_queries} | {r.rmse:.6g} | {r.runtime_seconds_per_query*1e6:.3f} | {r.build_seconds:.4f} |")
            lines.extend(["", "掲載するtheta・grid sizeは保存された全設定です。旧benchmarkのUniformをmainのUniformと同一視しません。stochasticは反復×query×2、deterministicは1×query×2です。biasは有限反復の標本平均の偏りであり理論的biasではありません。構築時間とquery時間はこの実験内の保存計測値のみです。", ""])
            uniform=float(table.loc[table.method=="uniform_mc","rmse"].iloc[0])
            grid=table[table.method.str.startswith("fit_grid_")]
            detail="、".join(f"{r.method}={r.rmse:.6f}" for _,r in grid.iterrows())
            lines.extend([f"保存された教師RMSEはUniform={uniform:.6f}、{detail}です。", ""])
        curve_path=self.out/"04_learned_field/benchmark_exact_validation_rmse_by_step.csv"
        if curve_path.is_file():
            last=pd.read_csv(curve_path).sort_values("step").groupby("method",sort=False).tail(1)
            lines.extend(["旧benchmarkの最終保存stepでのexact validation RMSE："+"、".join(f"{r.teacher_setting}={r.rmse:.6f}（step {r.step}）" for _,r in last.iterrows())+"。",""])
        lines.extend(["## その結果から読み取れる解釈", "",
            "通常UMAPに対する差と、Uniformに対するFitGridの差は別の比較です。正の改善量はRecall/NDCGではours−baseline、densityではbaseline−oursです。良い結果と悪い結果を同じ構成で表示し、CIが0を跨ぐ場合は固定モデルの細胞bootstrapでも差の方向は不確かです。", "",
            "場の精度が評価領域によって変わる場合、その結果は当該位置・時刻分布に条件づけられています。教師推定器の精度、学習済み場の精度、最終埋め込みのRecallは異なる評価なので、一つの優劣にまとめません。異なる教師へのtraining lossやlossの変動だけからexact精度の順位を決めません。", "",
            "既存splitとtemporalではreference数とquery集団が異なるため、数値の大小だけで難易度を断定しません。seed別の点は別の学習結果で、同じqueryをseed間で独立細胞としてプールしていません。", "",
            "## 未検証の仮説と未測定事項", "",
            "seed 2、未測定baseline、seed 1の通常UMAP・Recall@k・空間格子評価は補完していません。2つの学習seedだけでは一般的な再現性を確定できません。新しい生物学的反復、cell typeの妥当性、将来の複数外挿horizonに対する傾向、他のハードウェアでの速度優位は未検証です。bracketing rateだけで生物学的正しさを結論しません。Recall差の座標図は因果効果を示しません。", "",
            "## 統計・表示の条件", "",
            f"新規bootstrapは固定seed={self.args.seed}から比較ごとに決定的なseedを作り、{self.args.bootstrap:,}回、メモリ制限{self.args.bootstrap_memory_mb:g} MiBのバッチで実行します。対応queryのベクトルを同時再標本化し、temporalでは時間群ごとの件数を固定した層別再標本化を使います。95% CIは平均改善量のpercentile区間です。既存10,000回bootstrapのCIは再利用していません。", "",
            "CIは固定モデル・固定referenceに対する細胞bootstrapであり、生物学的反復のCIではありません。細胞間・donor間依存はモデル化していません。CSVには元の平均、符号変換前の差、中央値、改善／同点／悪化の割合、反復数・seed・batchを保存しました。", "",
            "temporalはall／interpolation／extrapolationを表示し、timepoint別名を独立集団として二重計上しません。全stageの色はreferenceとqueryの和集合で固定し、query_indexでラベルを対応付けます。全queryを描画し、PDFの点群はラスタライズしています。Uniform/FitGridのreference座標一致を確認して軸を共有し、通常UMAPは独自の座標系でequal aspect表示しています。", "",
            "exact repulsionは保存50 queryの同じID集合だけで比較します。temporal内挿／外挿件数はそのCSVのstage_countsに記録します。legacy advancedのMC repulsion oracleとは異なるfull-sum exactです。", "",
            "## 保存場所と検証", "",
            "00_overviewは固定した主要図の実体コピーです。01は既存split、02はtemporal、03は旧教師、04は学習済み場、05はablationです。各図のPNG・PDF・CSVは同じstemです。", "",
            "99_provenance/source_inventory.csvに使用入力のSHA-256と終了時照合、validation_checks.csvに数値照合、figure_catalog.csvに出所・seed・件数・領域、analysis_manifest.jsonに実行条件を保存します。入力の読取りは64 MiB以下の許可形式に制限し、大容量軌道・モデルをハッシュ目的でも読みません。", "",
            f"実行コマンド：`{' '.join(sys.argv)}`", "",
            "## スキップした項目", ""])
        lines.extend(f"- {r['item']}: {r['reason']}" for r in self.skips)
        if not self.skips:
            lines.append("なし。")
        (self.out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    def finalize(self):
        # Deadline applies to analysis/drawing; small index/provenance writes always finish.
        signal.setitimer(signal.ITIMER_REAL,0)
        analysis_seconds=time.monotonic()-self.start
        self.overview()
        records=[]
        for row in self.inventory.values():
            row=dict(row)
            p=Path(row["path"])
            row["unchanged"]=p.is_file() and p.stat().st_size==row["bytes"] and p.stat().st_mtime_ns==row["mtime_ns"] and sha(p)==row["sha256"]
            row["tasks"]=";".join(sorted(row["tasks"]))
            records.append(row)
        if any(not r["unchanged"] for r in records):
            self.status="input_changed"
            self.skip("input_integrity","At least one used source changed during analysis; see inventory")
        prov=self.out/"99_provenance"
        pd.DataFrame(records,columns=["path","bytes","mtime_ns","sha256","tasks","unchanged"]).to_csv(prov/"source_inventory.csv",index=False)
        pd.DataFrame(self.catalog,columns=["figure_name","purpose","relative_path","pdf_path","csv_path","inputs","split","training_seed","query_count","evaluation_region","creation","notes","metric_direction"]).to_csv(prov/"figure_catalog.csv",index=False)
        pd.DataFrame(self.skips,columns=["item","reason"]).to_csv(prov/"skipped_items.csv",index=False)
        pd.DataFrame(self.checks,columns=sorted(set().union(*(r.keys() for r in self.checks))) if self.checks else ["check","status"]).to_csv(prov/"validation_checks.csv",index=False)
        self.write_readme()
        content=['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
                 '<title>Embryoid body 保存成果物解析</title><style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;background:#f4f6f8;color:#223}h2{border-bottom:2px solid #abc;padding-top:24px}article{background:white;border:1px solid #ddd;border-radius:8px;margin:18px 0;padding:18px}img{width:100%;height:auto}p{line-height:1.65}small{overflow-wrap:anywhere}a{color:#176780}.notes{background:#eff5f8;padding:12px}</style>',
                 '<h1>Embryoid body：保存済み成果物の追加解析</h1><p><a href="README.md">日本語の結果・解釈</a> · <a href="99_provenance/figure_catalog.csv">図の索引</a> · <a href="99_provenance/skipped_items.csv">スキップ理由</a> · <a href="99_provenance/analysis_manifest.json">実行条件</a></p>',
                 '<p class="notes">Recallは%・差はpp。Recall/NDCG/cosineは高いほど良く、RMSE/density/MAEは低いほど良い。CIは固定モデル・referenceの細胞bootstrapで、生物学的反復のCIではありません。tは埋め込み最適化時間です。seed 2は除外。旧教師benchmarkと最終埋め込み・場評価は別の評価です。</p>']
        descriptions={"00_overview":"固定構成の主要図（実体コピー）","01_existing_split":"既存split・同じqueryでの最終埋め込み比較","02_temporal_holdout":"時間holdout・学習seed 0/1を区別","03_teacher_estimators":"旧教師推定器の保存された全設定","04_learned_field":"学習済み場・参照軌道近傍と空間格子を区別","05_ablations":"全query ablationと保存50-query exact診断を区別"}
        for section in SECTIONS[:-1]:
            content.append(f'<h2>{section} — {descriptions[section]}</h2>')
            found=[r for r in self.catalog if r["relative_path"].startswith(section+"/")]
            if not found:
                content.append('<p>生成済みの図はありません。スキップ理由を参照してください。</p>')
            for r in found:
                esc=lambda x:html.escape(str(x),quote=True)
                content.append(f'<article><h3>{esc(r["figure_name"])}</h3><p>{esc(r["purpose"])}</p><p>split: {esc(r["split"])} / 学習seed: {esc(r["training_seed"])} / query・評価点数: {esc(r["query_count"])} / 領域: {esc(r["evaluation_region"])} / {esc(r["creation"])}</p><a href="{esc(r["relative_path"])}"><img loading="lazy" src="{esc(r["relative_path"])}" alt="{esc(r["figure_name"])}"></a><p><a href="{esc(r["pdf_path"])}">PDF</a> · <a href="{esc(r["csv_path"])}">集計CSV</a></p><p>{esc(r["metric_direction"])}</p><p>{esc(r["notes"])}</p><details><summary>入力元</summary><small>{esc(r["inputs"])}</small></details></article>')
        content.append('</html>')
        gallery="\n".join(content)
        (self.out/"index.html").write_text(gallery,encoding="utf-8")
        links=re.findall(r'(?:href|src)="([^"]+)"',gallery)
        missing=[x for x in links if not (self.out/x).exists() and x!="99_provenance/analysis_manifest.json"]
        require(not missing,f"Broken gallery links: {missing}")
        manifest=dict(status=self.status,created_utc=datetime.now(timezone.utc).isoformat(),root=str(self.root),output=str(self.out),
            command=sys.argv,script_sha256=sha(Path(__file__)),python=sys.version,platform=platform.platform(),
            package_versions=dict(numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
            device="CPU",threads=1,time_limit_seconds=self.args.max_seconds,analysis_seconds=analysis_seconds,
            elapsed_seconds=time.monotonic()-self.start,bootstrap_iterations=self.args.bootstrap,bootstrap_seed=self.args.seed,
            bootstrap_memory_mb=self.args.bootstrap_memory_mb,bootstrap_method="paired percentile; fixed time-stratum counts for temporal",
            existing_bootstrap_reused=False,ci_scope=CI_NOTE,training_seeds=[0,1],excluded_training_seeds=[2],
            temporal_seed_evidence={k:dict(evaluation_seed=c["evaluation_seed"],training_seed=c["seed"],runs=c["seed_evidence"]) for k,c in self.contexts.items()},
            verified_temporal_training_seeds=sorted(c["seed"] for c in self.contexts.values() if c["temporal"]),
            missing_comparison_contexts=[k for k in ("existing","temporal0","temporal1") if k not in self.contexts],
            n_figures=len(self.catalog),n_generated=sum(r["creation"]=="generated" for r in self.catalog),n_copies=sum(r["creation"]!="generated" for r in self.catalog),
            n_png=len(list(self.out.rglob("*.png"))),n_pdf=len(list(self.out.rglob("*.pdf"))),n_csv=len(list(self.out.rglob("*.csv"))),
            n_skipped=len(self.skips),used_sources_unchanged=all(r["unchanged"] for r in records),html_links_checked=len(links),html_missing_links=missing,
            completed_tasks=self.completed_tasks,prohibited_operations_executed=False,
            allowed_inputs="CSV/JSON, saved labels and 2D coordinates, saved predictions/exact/weights only; no trajectory or model reads",
            visual_review="pending external inspection of representative PNGs")
        (prov/"analysis_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)+"\n")
        print(json.dumps({k:manifest[k] for k in ("status","output","elapsed_seconds","n_figures","n_png","n_pdf","n_csv","n_skipped","used_sources_unchanged")},ensure_ascii=False),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,help="New directory only; default ROOT/final_analysis")
    parser.add_argument("--max-seconds",type=float,default=600)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20260917)
    parser.add_argument("--bootstrap-memory-mb",type=float,default=32)
    parser.add_argument("--inventory-hint",type=Path)
    args=parser.parse_args(argv)
    require(args.max_seconds>0 and args.bootstrap>=2 and args.bootstrap_memory_mb>=1,"Invalid budget/bootstrap options")
    if args.output is None:
        args.output=args.root/"final_analysis"
    app=Analysis(args)
    def timeout(signum,frame):
        raise BudgetExceeded()
    signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,max(.001,args.max_seconds-min(10,args.max_seconds*.1)))
    tasks=[]
    if args.inventory_hint:
        tasks.append(("inventory_hint",lambda:app.track(args.inventory_hint)))
    for key,folder,u,f,seed,temp in [
        ("existing","fit_grid_vs_main","main","fit_grid",0,False),
        ("temporal0","temporal_holdout","temporal_holdout_uniform","temporal_holdout_fit_grid",0,True),
        ("temporal1","temporal_seed_1","temporal_seed_1_uniform","temporal_seed_1_fit_grid",1,True)]:
        tasks.append((f"load/{key}",lambda k=key,d=folder,u=u,f=f,s=seed,t=temp:app.load_context(k,d,u,f,s,t)))
    def ctx(key):
        require(key in app.contexts,f"Required saved context unavailable: {key}")
        return app.contexts[key]
    for key in ("existing","temporal0"):
        for name,fn in [("performance",app.performance),("paired",app.paired),
                        ("recall_distribution",app.distributions),("zero_recall_fraction",lambda c:app.distributions(c,True)),
                        ("recall_at_k",app.recall_k),("embedding_by_stage",app.embeddings)]:
            tasks.append((f"{key}/{name}",lambda k=key,fn=fn:fn(ctx(k))))
    tasks.extend([("temporal1/paired",lambda:app.paired(ctx("temporal1"))),
                  ("temporal0/embedding_holdout",lambda:app.embeddings(ctx("temporal0"),highlight=True)),
                  ("temporal1/embedding_holdout",lambda:app.embeddings(ctx("temporal1"),highlight=True)),
                  ("temporal/split_counts",app.temporal_counts),
                  ("temporal/recall_by_timepoint",lambda:app.temporal_metric("recall_at_15")),
                  ("temporal/density_by_timepoint",lambda:app.temporal_metric("density_log_distortion")),
                  ("temporal/neighbor_diagnostics",lambda:app.temporal_metric("diagnostics")),
                  ("temporal/seed_comparison",app.seed_comparison)])
    for kind in ("rmse","bias","runtime","ess"):
        tasks.append((f"teacher/{kind}",lambda k=kind:app.teacher_metrics(k)))
    tasks.append(("teacher/importance_weights",app.importance_weights))
    for kind in ("validation","loss","std"):
        tasks.append((f"benchmark_training/{kind}",lambda k=kind:app.benchmark_curves(k)))
    tasks.extend([("field/load",app.field_data)])
    for kind in ("near","near_time","grid_time","region","distribution"):
        tasks.append((f"field/{kind}",lambda k=kind:app.field_plot(k)))
    tasks.append(("training/current_runs",app.current_training_curves))
    for key in ("existing","temporal0"):
        tasks.extend([(f"ablation/{key}/full",lambda k=key:app.ablation(ctx(k))),
                      (f"ablation/{key}/exact50",lambda k=key:app.ablation(ctx(k),exact=True)),
                      (f"ablation/{key}/recall_map",lambda k=key:app.embeddings(ctx(k),difference=True))])
    app.skip("temporal_seed_2","Excluded by explicit selection: incomplete, no checkpoint loading or resumption")
    app.skip("temporal_seed_1/standard_umap_and_recall_at_k","Not saved for training seed 1; seed-0 results not substituted")
    try:
        for name,fn in tasks:
            app.task(name,fn)
        app.status=("partial_missing_inputs" if len(app.contexts)<3 else "complete_with_skips" if app.skips else "complete")
    except BudgetExceeded:
        app.status="time_budget_reached"
        for name,_ in tasks:
            if name not in app.executed:
                app.skip(name,"Not started: CLI analysis/drawing time budget reached")
    except BaseException as exc:
        app.status="failed"
        app.skip(app.current_task,f"Unexpected {type(exc).__name__}: {exc}")
        raise
    finally:
        app.finalize()


if __name__=="__main__":
    main()
