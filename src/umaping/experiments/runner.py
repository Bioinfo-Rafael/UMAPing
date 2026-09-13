"""Experiment runner: download -> prepare -> train -> evaluate -> analyze ->
analyze-advanced -> baselines, for one of the datasets registered in
`experiments/registry.py`.

Deliberately thin: every stage after "prepare" reuses `pipeline.py`'s
existing, already-tested stage functions directly (`run_training`,
`run_evaluation`, `run_analysis`, `run_advanced_analysis`) -- nothing here
reimplements checkpointing, resumability, or the base metric/figure suite.
The only genuinely new work this module adds is `run_experiment_baselines`,
which runs the additional Experiment-B/C/D-specific baselines
(`experiments/baselines.py`) against the same prepared dataset and reference
trajectory `run_training` already produced.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from umaping.config import Config
from umaping.data.preprocessing import load_prepared_dataset
from umaping.dynamics import ReferenceTrajectory
from umaping.evaluation.embedding import evaluate_embedding
from umaping.experiments.baselines import (
    BaselineOutcome,
    BaselineResult,
    BaselineUnavailable,
    run_exact_repulsion_diagnostic_baseline,
    run_numap_baseline,
    run_ours_baselines,
    run_param_repulsor_baseline,
    run_parametric_umap_baseline,
    run_reduced_repulsion_umap_baseline,
    run_standard_umap_baseline,
    run_weighted_knn_baseline,
)
from umaping.experiments.registry import ExperimentSpec, get_experiment
from umaping.pipeline import (
    _primary_label,
    init_run_dir,
    run_advanced_analysis,
    run_analysis,
    run_evaluation,
    run_layout,
    run_training,
    stage_preprocess,
)
from umaping.utils.io import guard_run_dir, save_json
from umaping.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_experiment(
    experiment_name: str,
    run_dir: str | Path,
    device: torch.device,
    seed: int | None = None,
    download_only: bool = False,
    prepare_only: bool = False,
    resume: bool = False,
) -> None:
    spec: ExperimentSpec = get_experiment(experiment_name)
    cfg = Config.load(spec.config_path)
    if seed is not None:
        cfg.seed = seed
    run_dir = Path(run_dir)

    logger.info("Downloading '%s' (source: %s; citation: %s)...", experiment_name, spec.source_url, spec.citation)
    spec.download_fn(cfg.dataset.raw_dir)
    if download_only:
        logger.info("--download-only: stopping after download. Raw data under %s", cfg.dataset.raw_dir)
        return

    guard_run_dir(run_dir, resume)
    set_seed(cfg.seed)

    if prepare_only:
        layout = init_run_dir(cfg, run_dir)
        stage_preprocess(cfg, layout)
        logger.info("--prepare-only: stopping after preprocessing. Prepared dataset cached under %s/cache", run_dir)
        return

    logger.info("Running full pipeline for experiment '%s' (dataset=%s)...", experiment_name, cfg.dataset.name)
    run_training(cfg, run_dir, device)
    run_evaluation(run_dir, device)
    run_analysis(run_dir, device)
    run_advanced_analysis(run_dir, device)
    run_experiment_baselines(run_dir, device, cfg, seed=cfg.seed)
    logger.info(
        "Experiment '%s' complete. See %s/metrics (incl. baseline_comparison.csv, timing.json) and %s/figures",
        experiment_name,
        run_dir,
        run_dir,
    )


def _subset(array: np.ndarray | None, idx: np.ndarray | None) -> np.ndarray | None:
    if array is None or idx is None:
        return array
    return array[idx]


def run_experiment_baselines(run_dir: str | Path, device: torch.device, cfg: Config, seed: int) -> None:
    """Runs every baseline in the Section-4 common registry against the
    already-prepared dataset and already-built reference trajectory (never
    retraining `ours`'s own models), and writes
    `metrics/baseline_comparison.csv` (one row per baseline: aggregate
    recall/label-accuracy/latency, or an "unavailable" row with the reason)
    plus `metrics/timing.json` (fit/query latency only, separated out for
    convenience)."""
    layout = run_layout(run_dir)
    prepared = load_prepared_dataset(layout["cache"] / "prepared_dataset.npz")
    trajectory = ReferenceTrajectory.load(layout["memory"] / "reference_trajectory.npz")
    reference_embedding_ours = trajectory.positions[-1]

    reference_labels = _primary_label(prepared.reference_labels, cfg.dataset.name)
    query_labels = _primary_label(prepared.query_labels, cfg.dataset.name)

    outcomes: list[BaselineOutcome] = []

    logger.info("Running baseline: standard_umap")
    standard_result = run_standard_umap_baseline(prepared, cfg)
    outcomes.append(standard_result)

    logger.info("Running baseline: reduced_repulsion_umap")
    outcomes.append(run_reduced_repulsion_umap_baseline(prepared, cfg))

    logger.info("Running baseline: weighted_knn")
    outcomes.append(run_weighted_knn_baseline(prepared, standard_result.reference_embedding, cfg))

    logger.info("Running baseline: parametric_umap")
    outcomes.append(run_parametric_umap_baseline(prepared, cfg))

    logger.info("Running baseline: numap_sep_spectralnet")
    outcomes.append(run_numap_baseline(prepared, cfg, device=device))

    logger.info("Running baseline: param_repulsor")
    outcomes.append(run_param_repulsor_baseline(prepared, cfg))

    logger.info("Running baselines: ours / ours_oracle_neighbors / no_repulsion")
    outcomes.extend(run_ours_baselines(run_dir, device, cfg, reference_embedding_ours, seed))

    logger.info("Running baseline: exact_repulsion_diagnostic")
    outcomes.append(
        run_exact_repulsion_diagnostic_baseline(
            run_dir, device, cfg, reference_embedding_ours, seed, query_subset_n=cfg.eval.repulsion_oracle_query_subset
        )
    )

    rows: list[dict] = []
    timing: dict[str, dict] = {}
    for outcome in outcomes:
        if isinstance(outcome, BaselineUnavailable):
            rows.append({"method": outcome.name, "available": False, "reason": outcome.reason})
            continue

        idx = np.asarray(outcome.extra["query_subset_indices"]) if "query_subset_indices" in outcome.extra else None
        q_high = _subset(prepared.query_features, idx) if idx is not None else prepared.query_features
        q_labels = _subset(query_labels, idx) if (idx is not None and query_labels is not None) else query_labels

        metrics, _ = evaluate_embedding(
            q_high,
            prepared.reference_features,
            outcome.query_embedding,
            outcome.reference_embedding,
            k=cfg.eval.k,
            reference_labels=reference_labels,
            query_labels=q_labels,
            trustworthiness_n_neighbors=cfg.eval.trustworthiness_n_neighbors,
            mean_query_latency_seconds=outcome.mean_query_latency_seconds,
        )
        row = {"method": outcome.name, "available": True, **metrics.to_dict()}
        rows.append(row)
        timing[outcome.name] = {
            "fit_time_seconds": outcome.fit_time_seconds,
            "mean_query_latency_seconds": outcome.mean_query_latency_seconds,
        }

    pd.DataFrame(rows).to_csv(layout["metrics"] / "baseline_comparison.csv", index=False)
    save_json(layout["metrics"] / "timing.json", timing)
