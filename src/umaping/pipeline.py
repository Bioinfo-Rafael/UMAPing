"""Stage orchestration + resumability for the `train` / `pipeline` /
`evaluate` / `analyze` CLI commands (docs/method.md; README "Outputs").

A deliberate, documented addition beyond the spec's literal file tree: this
is where all filesystem I/O and checkpoint/resume bookkeeping lives, so the
pure functions in training/*.py and evaluation/*.py never touch disk and
stay trivially unit-testable. Every expensive stage checks a `.done` marker
(utils/io.is_done/mark_done) before doing any work, so re-running `train` or
`pipeline` on a run-dir that already has completed stages just continues
from wherever it left off.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from umaping.baselines import (
    embed_all_queries,
    embed_all_queries_spectral_only,
    run_standard_umap,
)
from umaping.config import Config
from umaping.data.coil import prepare_coil_dataset
from umaping.data.mock import prepare_mock_dataset
from umaping.data.pancreas import prepare_pancreas_dataset
from umaping.data.preprocessing import PreparedDataset, load_prepared_dataset, save_prepared_dataset
from umaping.dynamics import ReferenceTrajectory
from umaping.evaluation.advanced import (
    compute_per_query_diagnostics,
    denoising_report_by_t,
    field_smoothness_report,
)
from umaping.evaluation.embedding import evaluate_embedding, procrustes_align
from umaping.evaluation.field import evaluate_repulsion_field
from umaping.evaluation.plotting import (
    plot_embedding,
    plot_field_denoising_error_by_t,
    plot_field_smoothness_by_t,
    plot_loss_curve,
    plot_query_fuzzy_error_distribution,
    plot_query_neighbor_recall_distribution,
    plot_query_periphery_score,
    plot_query_tail_failure_comparison,
    plot_query_trajectories,
    plot_repulsion_field,
    plot_retriever_recall,
    plot_retriever_training,
)
from umaping.evaluation.retrieval import evaluate_retrieval
from umaping.evaluation.spectral import evaluate_spectral
from umaping.graph import build_reference_graph, chunked_exact_knn, query_fuzzy_weights, row_degree
from umaping.inference import InferenceEngine
from umaping.models.mlp import batched_forward
from umaping.models.repulsion import RepulsionField
from umaping.models.retriever import DualEncoder
from umaping.models.spectral import SpectralCalibration, SpectralEncoderNet
from umaping.training.flow import build_reference_trajectory, train_repulsion_field
from umaping.training.retriever import train_retriever
from umaping.training.spectral import compute_calibration, train_spectral
from umaping.umap_forces import find_ab_params
from umaping.utils.io import (
    atomic_save_sparse,
    atomic_torch_save,
    collect_package_versions,
    ensure_dir,
    is_done,
    load_json,
    load_sparse,
    mark_done,
    save_json,
)
from umaping.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_layout(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "root": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "memory": run_dir / "memory",
        "metrics": run_dir / "metrics",
        "figures": run_dir / "figures",
        "cache": run_dir / "cache",
    }


def init_run_dir(cfg: Config, run_dir: str | Path) -> dict[str, Path]:
    layout = run_layout(run_dir)
    for path in layout.values():
        ensure_dir(path)

    config_path = layout["root"] / "config.yaml"
    if not config_path.exists():
        cfg.save(config_path)

    metadata_path = layout["root"] / "metadata.json"
    if not metadata_path.exists():
        save_json(
            metadata_path,
            {
                "dataset": cfg.dataset.name,
                "created": _utc_now(),
                "package_versions": collect_package_versions(),
                "stages_completed": [],
            },
        )
    else:
        metadata = load_json(metadata_path)
        metadata["package_versions"] = collect_package_versions()
        metadata["last_resumed"] = _utc_now()
        save_json(metadata_path, metadata)
    return layout


def _record_stage_complete(layout: dict[str, Path], stage: str) -> None:
    metadata_path = layout["root"] / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {"stages_completed": []}
    stages = metadata.setdefault("stages_completed", [])
    if stage not in stages:
        stages.append(stage)
    metadata["last_updated"] = _utc_now()
    save_json(metadata_path, metadata)


def _label_sets(prepared: PreparedDataset, dataset_name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if dataset_name in ("coil20", "coil100"):
        return {"object_id": prepared.reference_labels["object_id"]}, {"object_id": prepared.query_labels["object_id"]}
    if dataset_name == "pancreas":
        ref = {"celltype": prepared.reference_labels["celltype"], "tech": prepared.reference_labels["tech"]}
        query = {"celltype": prepared.query_labels["celltype"], "tech": prepared.query_labels["tech"]}
        return ref, query
    if dataset_name == "mock":
        return {"cluster": prepared.reference_labels["cluster"]}, {"cluster": prepared.query_labels["cluster"]}
    if dataset_name == "fashion_mnist":
        return {"class": prepared.reference_labels["class"]}, {"class": prepared.query_labels["class"]}
    if dataset_name == "twenty_newsgroups":
        return {"newsgroup": prepared.reference_labels["newsgroup"]}, {"newsgroup": prepared.query_labels["newsgroup"]}
    if dataset_name == "mnist_oos":
        return {"digit": prepared.reference_labels["digit"]}, {"digit": prepared.query_labels["digit"]}
    if dataset_name == "hong_ed":
        return {"admitted": prepared.reference_labels["admitted"]}, {"admitted": prepared.query_labels["admitted"]}
    if dataset_name in ("organoid", "embryoid_body"):
        # The grouping column's actual name is auto-detected per-dataset
        # (see data/_scrna_common.py) and is whatever key `PreparedDataset`
        # was built with; expose every label key present rather than one
        # hardcoded name.
        return dict(prepared.reference_labels), dict(prepared.query_labels)
    return {}, {}


def _primary_label(labels: dict[str, np.ndarray], dataset_name: str) -> np.ndarray | None:
    if dataset_name in ("coil20", "coil100"):
        return labels.get("object_id")
    if dataset_name == "pancreas":
        return labels.get("celltype")
    if dataset_name == "mock":
        return labels.get("cluster")
    if dataset_name == "fashion_mnist":
        return labels.get("class")
    if dataset_name == "twenty_newsgroups":
        return labels.get("newsgroup")
    if dataset_name == "mnist_oos":
        return labels.get("digit")
    if dataset_name == "hong_ed":
        return labels.get("admitted")
    if dataset_name in ("organoid", "embryoid_body"):
        return next(iter(labels.values())) if labels else None
    return None


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


def stage_preprocess(cfg: Config, layout: dict[str, Path]) -> PreparedDataset:
    cache_path = layout["cache"] / "prepared_dataset.npz"
    if is_done(layout["cache"], "preprocess") and cache_path.exists():
        logger.info("Preprocessing already done; loading cached prepared dataset.")
        return load_prepared_dataset(cache_path)

    logger.info("Preprocessing dataset '%s' (reference-only fitting throughout)...", cfg.dataset.name)
    name = cfg.dataset.name
    params = cfg.dataset.params
    if name in ("coil20", "coil100"):
        prepared, _ = prepare_coil_dataset(
            dataset=name,
            raw_dir=cfg.dataset.raw_dir,
            use_pca=params.get("use_pca", True),
            pca_dim=params.get("pca_dim", 256),
            seed=cfg.seed,
            holdout_period=params.get("holdout_period", 4),
        )
    elif name == "pancreas":
        prepared, _ = prepare_pancreas_dataset(
            raw_dir=cfg.dataset.raw_dir,
            n_hvg=params.get("n_hvg", 2000),
            n_pcs=params.get("n_pcs", 50),
            seed=cfg.seed,
            query_tech=tuple(params.get("query_tech", ["smartseq2", "celseq2"])),
        )
    elif name == "mock":
        # Synthetic, no download: exists purely to smoke-test the whole
        # pipeline (through evaluate/analyze) fully locally in seconds. See
        # configs/mock.yaml and data/mock.py.
        prepared = prepare_mock_dataset(
            n_reference=params.get("n_reference", 200),
            n_query=params.get("n_query", 50),
            input_dim=cfg.dataset.input_dim,
            n_clusters=params.get("n_clusters", 5),
            seed=cfg.seed,
        )
    elif name == "fashion_mnist":
        from umaping.data.fashion_mnist import prepare_fashion_mnist_dataset

        prepared, _ = prepare_fashion_mnist_dataset(
            raw_dir=cfg.dataset.raw_dir,
            use_pca=params.get("use_pca", True),
            pca_dim=params.get("pca_dim", 100),
            seed=cfg.seed,
            n_reference_subsample=params.get("n_reference_subsample"),
            n_query_subsample=params.get("n_query_subsample"),
        )
    elif name == "twenty_newsgroups":
        from umaping.data.twenty_newsgroups import prepare_twenty_newsgroups_dataset

        prepared, _ = prepare_twenty_newsgroups_dataset(
            raw_dir=cfg.dataset.raw_dir,
            n_components=params.get("n_components", 100),
            max_features=params.get("max_features", 20000),
            seed=cfg.seed,
            n_reference_subsample=params.get("n_reference_subsample"),
            n_query_subsample=params.get("n_query_subsample"),
        )
    elif name == "mnist_oos":
        from umaping.data.mnist_oos import prepare_mnist_dataset

        prepared, _ = prepare_mnist_dataset(
            raw_dir=cfg.dataset.raw_dir,
            use_pca=params.get("use_pca", True),
            pca_dim=params.get("pca_dim", 50),
            seed=cfg.seed,
            n_reference_subsample=params.get("n_reference_subsample"),
            n_query_subsample=params.get("n_query_subsample"),
        )
    elif name == "hong_ed":
        from umaping.data.hong_ed import prepare_hong_ed_dataset

        prepared, _ = prepare_hong_ed_dataset(
            raw_dir=cfg.dataset.raw_dir,
            query_fraction=params.get("query_fraction", 0.2),
            subset_mode=params.get("subset_mode", "full"),
            small_n=params.get("small_n", 500),
            medium_n=params.get("medium_n", 5000),
            seed=cfg.seed,
        )
    elif name == "organoid":
        from umaping.data.organoid import prepare_organoid_dataset

        query_groups = params.get("query_groups")
        prepared, _ = prepare_organoid_dataset(
            raw_dir=cfg.dataset.raw_dir,
            n_hvg=params.get("n_hvg", 2000),
            n_pcs=params.get("n_pcs", 50),
            seed=cfg.seed,
            query_groups=tuple(query_groups) if query_groups else None,
        )
    elif name == "embryoid_body":
        from umaping.data.embryoid_body import prepare_embryoid_body_dataset

        query_groups = params.get("query_groups")
        prepared, _ = prepare_embryoid_body_dataset(
            raw_dir=cfg.dataset.raw_dir,
            n_hvg=params.get("n_hvg", 2000),
            n_pcs=params.get("n_pcs", 50),
            seed=cfg.seed,
            query_groups=tuple(query_groups) if query_groups else None,
        )
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Expected one of coil20, coil100, pancreas, mock, fashion_mnist, "
            f"twenty_newsgroups, mnist_oos, hong_ed, organoid, embryoid_body. Did you mean to add a raw "
            f"dataset directory under {cfg.dataset.raw_dir}?"
        )

    if prepared.input_dim != cfg.dataset.input_dim:
        raise ValueError(
            f"Preprocessing produced input_dim={prepared.input_dim}, but config.dataset.input_dim="
            f"{cfg.dataset.input_dim}. Update the config so the two agree before training a model."
        )

    save_prepared_dataset(cache_path, prepared)
    np.save(layout["memory"] / "reference_features.npy", prepared.reference_features)
    mark_done(layout["cache"], "preprocess")
    return prepared


def stage_graph(cfg: Config, layout: dict[str, Path], prepared: PreparedDataset):
    mu_path = layout["memory"] / "graph_directed.npz"
    w_path = layout["memory"] / "graph_symmetric.npz"
    if is_done(layout["memory"], "graph") and mu_path.exists() and w_path.exists():
        logger.info("Reference graph already built; loading cached Mu/W.")
        return load_sparse(mu_path), load_sparse(w_path)

    logger.info("Building reference kNN graph (N=%d, n_neighbors=%d)...", prepared.reference_features.shape[0], cfg.umap.n_neighbors)
    graph = build_reference_graph(
        prepared.reference_features,
        n_neighbors=cfg.umap.n_neighbors,
        local_connectivity=cfg.umap.local_connectivity,
        n_iter=cfg.umap.smooth_knn_n_iter,
        bandwidth=cfg.umap.smooth_knn_bandwidth,
        min_k_dist_scale=cfg.umap.smooth_knn_min_k_dist_scale,
        set_op_mix_ratio=cfg.umap.set_op_mix_ratio,
        eps=cfg.umap.eps,
    )
    atomic_save_sparse(mu_path, graph.mu)
    atomic_save_sparse(w_path, graph.w)
    mark_done(layout["memory"], "graph")
    return graph.mu, graph.w


def stage_retriever(cfg: Config, layout: dict[str, Path], prepared: PreparedDataset, mu, device: torch.device):
    ckpt_path = layout["checkpoints"] / "retriever.pt"
    keys_path = layout["memory"] / "retriever_keys.npy"
    if is_done(layout["checkpoints"], "retriever") and ckpt_path.exists() and keys_path.exists():
        logger.info("Retriever already trained; loading checkpoint.")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DualEncoder(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        return model, np.load(keys_path)

    logger.info("Training retriever for %d steps on device=%s...", cfg.retriever.steps, device)
    model = DualEncoder(
        input_dim=cfg.dataset.input_dim,
        hidden_dims=cfg.retriever.hidden_dims,
        retrieval_dim=cfg.retriever.retrieval_dim,
        activation=cfg.retriever.activation,
        temperature=cfg.retriever.temperature,
    )
    state = train_retriever(model, prepared.reference_features, mu, cfg.retriever, device, seed=cfg.seed)
    save_json(
        layout["metrics"] / "retriever_training.json",
        {"losses": state.losses, "batch_top1_acc": state.batch_top1_acc},
    )

    hparams = {
        "input_dim": cfg.dataset.input_dim,
        "hidden_dims": cfg.retriever.hidden_dims,
        "retrieval_dim": cfg.retriever.retrieval_dim,
        "activation": cfg.retriever.activation,
        "temperature": cfg.retriever.temperature,
    }
    atomic_torch_save({"state_dict": model.state_dict(), "hparams": hparams}, ckpt_path)

    raw_keys = batched_forward(model.key_encoder, prepared.reference_features, device)
    keys = raw_keys / np.maximum(np.linalg.norm(raw_keys, axis=1, keepdims=True), 1e-12)
    np.save(keys_path, keys.astype(np.float32))

    mark_done(layout["checkpoints"], "retriever")
    return model, keys


def stage_spectral(cfg: Config, layout: dict[str, Path], prepared: PreparedDataset, w, device: torch.device):
    ckpt_path = layout["checkpoints"] / "spectral_encoder.pt"
    calib_path = layout["memory"] / "spectral_calibration.npz"
    embedding_cache_path = layout["cache"] / "reference_spectral_embedding.npy"
    if (
        is_done(layout["checkpoints"], "spectral")
        and ckpt_path.exists()
        and calib_path.exists()
        and embedding_cache_path.exists()
    ):
        logger.info("Spectral encoder already trained; loading checkpoint + calibration.")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = SpectralEncoderNet(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        calibration = SpectralCalibration.load(calib_path)
        return model, calibration, np.load(embedding_cache_path)

    logger.info("Training spectral encoder for %d steps on device=%s...", cfg.spectral.steps, device)
    model = SpectralEncoderNet(
        input_dim=cfg.dataset.input_dim,
        hidden_dims=cfg.spectral.hidden_dims,
        raw_output_dim=cfg.spectral.raw_output_dim,
        activation=cfg.spectral.activation,
    )
    state = train_spectral(model, prepared.reference_features, w, cfg.spectral, device, seed=cfg.seed)
    save_json(
        layout["metrics"] / "spectral_training.json",
        {
            "dirichlet_losses": state.dirichlet_losses,
            "ortho_losses": state.ortho_losses,
            "total_losses": state.total_losses,
        },
    )

    calibration, reference_embedding = compute_calibration(
        model,
        prepared.reference_features,
        w,
        embedding_dim=cfg.umap.embedding_dim,
        calibration_target_scale=cfg.spectral.calibration_target_scale,
        device=device,
    )

    hparams = {
        "input_dim": cfg.dataset.input_dim,
        "hidden_dims": cfg.spectral.hidden_dims,
        "raw_output_dim": cfg.spectral.raw_output_dim,
        "activation": cfg.spectral.activation,
    }
    atomic_torch_save({"state_dict": model.state_dict(), "hparams": hparams}, ckpt_path)
    calibration.save(calib_path)
    np.save(embedding_cache_path, reference_embedding)
    mark_done(layout["checkpoints"], "spectral")
    return model, calibration, reference_embedding


def stage_flow_dynamics(cfg: Config, layout: dict[str, Path], reference_embedding: np.ndarray, w, device: torch.device):
    traj_path = layout["memory"] / "reference_trajectory.npz"
    if is_done(layout["memory"], "flow_dynamics") and traj_path.exists():
        logger.info("Reference trajectory already built; loading.")
        return ReferenceTrajectory.load(traj_path)

    logger.info("Integrating reference mean UMAP dynamics for %d steps...", cfg.flow.n_steps)
    a, b = find_ab_params(cfg.umap.spread, cfg.umap.min_dist)
    trajectory = build_reference_trajectory(
        reference_embedding, w, a, b, cfg.umap.negative_sample_rate, cfg.flow, device, seed=cfg.seed
    )
    trajectory.save(traj_path)
    mark_done(layout["memory"], "flow_dynamics")
    return trajectory


def stage_repulsion(cfg: Config, layout: dict[str, Path], trajectory: ReferenceTrajectory, w, device: torch.device):
    ckpt_path = layout["checkpoints"] / "repulsion_field.pt"
    if is_done(layout["checkpoints"], "repulsion") and ckpt_path.exists():
        logger.info("Repulsion field already trained; loading checkpoint.")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = RepulsionField(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    logger.info("Training repulsion field for %d steps on device=%s...", cfg.repulsion.steps, device)
    a, b = find_ab_params(cfg.umap.spread, cfg.umap.min_dist)
    model = RepulsionField(
        embedding_dim=cfg.umap.embedding_dim,
        hidden_dim=cfg.repulsion.hidden_dim,
        n_residual_blocks=cfg.repulsion.n_residual_blocks,
        time_embed_dim=cfg.repulsion.time_embed_dim,
        activation=cfg.repulsion.activation,
    )
    row_mass = row_degree(w)
    state = train_repulsion_field(
        model, trajectory, a, b, cfg.repulsion, row_mass, device, seed=cfg.seed, grad_clip=cfg.flow.grad_clip
    )
    save_json(layout["metrics"] / "repulsion_training.json", {"losses": state.losses})

    hparams = {
        "embedding_dim": cfg.umap.embedding_dim,
        "hidden_dim": cfg.repulsion.hidden_dim,
        "n_residual_blocks": cfg.repulsion.n_residual_blocks,
        "time_embed_dim": cfg.repulsion.time_embed_dim,
        "activation": cfg.repulsion.activation,
    }
    atomic_torch_save({"state_dict": model.state_dict(), "hparams": hparams}, ckpt_path)
    mark_done(layout["checkpoints"], "repulsion")
    return model


def run_training(cfg: Config, run_dir: str | Path, device: torch.device) -> dict[str, Path]:
    set_seed(cfg.seed)
    layout = init_run_dir(cfg, run_dir)

    prepared = stage_preprocess(cfg, layout)
    _record_stage_complete(layout, "preprocess")

    mu, w = stage_graph(cfg, layout, prepared)
    _record_stage_complete(layout, "graph")

    _retriever, _retriever_keys = stage_retriever(cfg, layout, prepared, mu, device)
    _record_stage_complete(layout, "retriever")

    _spectral_model, _calibration, reference_embedding = stage_spectral(cfg, layout, prepared, w, device)
    _record_stage_complete(layout, "spectral")

    trajectory = stage_flow_dynamics(cfg, layout, reference_embedding, w, device)
    _record_stage_complete(layout, "flow_dynamics")

    stage_repulsion(cfg, layout, trajectory, w, device)
    _record_stage_complete(layout, "repulsion")

    return layout


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluation(run_dir: str | Path, device: torch.device) -> None:
    layout = run_layout(run_dir)
    cfg = Config.load(layout["root"] / "config.yaml")
    set_seed(cfg.seed)

    prepared = load_prepared_dataset(layout["cache"] / "prepared_dataset.npz")
    w = load_sparse(layout["memory"] / "graph_symmetric.npz")
    trajectory = ReferenceTrajectory.load(layout["memory"] / "reference_trajectory.npz")
    reference_embedding_ours = trajectory.positions[-1]
    a, b = find_ab_params(cfg.umap.spread, cfg.umap.min_dist)

    reference_labels = _primary_label(prepared.reference_labels, cfg.dataset.name)
    query_labels = _primary_label(prepared.query_labels, cfg.dataset.name)

    # -- Retriever metrics --------------------------------------------------
    engine_full = InferenceEngine.load(run_dir, device, seed=cfg.seed)
    retrieval_metrics, retrieval_df = evaluate_retrieval(
        engine_full, prepared.query_features, prepared.reference_features, k=cfg.eval.k
    )
    save_json(layout["metrics"] / "retriever.json", retrieval_metrics.to_dict())

    # -- Spectral metrics -----------------------------------------------------
    spectral_ckpt = torch.load(layout["checkpoints"] / "spectral_encoder.pt", map_location="cpu", weights_only=False)
    spectral_model = SpectralEncoderNet(**spectral_ckpt["hparams"])
    spectral_model.load_state_dict(spectral_ckpt["state_dict"])
    calibration = SpectralCalibration.load(layout["memory"] / "spectral_calibration.npz")
    raw = batched_forward(spectral_model, prepared.reference_features, device)
    z_whitened = raw @ calibration.whitening
    spectral_metrics = evaluate_spectral(z_whitened, w, calibration.projection, embedding_dim=cfg.umap.embedding_dim)
    save_json(layout["metrics"] / "spectral.json", spectral_metrics.to_dict())

    # -- Repulsion field metrics ----------------------------------------------
    repulsion_ckpt = torch.load(layout["checkpoints"] / "repulsion_field.pt", map_location="cpu", weights_only=False)
    repulsion_model = RepulsionField(**repulsion_ckpt["hparams"])
    repulsion_model.load_state_dict(repulsion_ckpt["state_dict"])
    field_metrics, _field_df = evaluate_repulsion_field(
        repulsion_model,
        trajectory,
        a,
        b,
        n_eval_points=2000,
        teacher_negative_samples=cfg.repulsion.teacher_negative_samples,
        device=device,
        seed=cfg.seed,
        jitter_sigma=cfg.repulsion.jitter_sigma,
        grad_clip=cfg.flow.grad_clip,
    )
    save_json(layout["metrics"] / "field.json", field_metrics.to_dict())

    # -- Embedding metrics: baselines + ablations -----------------------------
    per_method_metrics: dict[str, dict] = {}
    per_query_recall_by_method: dict[str, np.ndarray] = {}

    def _eval_method(
        name: str,
        query_embedding: np.ndarray,
        reference_embedding: np.ndarray,
        latency=None,
        extra=None,
        query_high_dim: np.ndarray | None = None,
        query_labels_override: np.ndarray | None = None,
    ):
        """`query_high_dim`/`query_labels_override` default to the full query
        set; pass them explicitly when `query_embedding` only covers a
        *subset* of queries (e.g. the repulsion-oracle diagnostic), so every
        array `evaluate_embedding` compares stays the same length."""
        m, per_query_recall = evaluate_embedding(
            query_high_dim if query_high_dim is not None else prepared.query_features,
            prepared.reference_features,
            query_embedding,
            reference_embedding,
            k=cfg.eval.k,
            reference_labels=reference_labels,
            query_labels=query_labels_override if query_labels_override is not None else query_labels,
            trustworthiness_n_neighbors=cfg.eval.trustworthiness_n_neighbors,
            mean_query_latency_seconds=latency,
        )
        d = m.to_dict()
        if extra:
            d.update(extra)
        per_method_metrics[name] = d
        per_query_recall_by_method[name] = per_query_recall

    # 4. Full method.
    emb, _neigh, _w, latencies, _ = embed_all_queries(engine_full, prepared.query_features)
    _eval_method("ours_full", emb, reference_embedding_ours, latency=float(np.mean(latencies)))

    # 3. Ours + ORACLE neighbors.
    engine_oracle_neighbors = InferenceEngine.load(run_dir, device, neighbor_source="oracle", seed=cfg.seed)
    emb, _, _, latencies, _ = embed_all_queries(engine_oracle_neighbors, prepared.query_features)
    _eval_method("oracle_neighbors", emb, reference_embedding_ours, latency=float(np.mean(latencies)))

    # 5. No-repulsion ablation.
    engine_no_repulsion = InferenceEngine.load(run_dir, device, use_repulsion=False, seed=cfg.seed)
    emb, _, _, latencies, _ = embed_all_queries(engine_no_repulsion, prepared.query_features)
    _eval_method("no_repulsion", emb, reference_embedding_ours, latency=float(np.mean(latencies)))

    # 6. Repulsion-oracle diagnostic, on a configurable query subset.
    subset_n = min(cfg.eval.repulsion_oracle_query_subset, prepared.query_features.shape[0])
    subset_idx = np.random.default_rng(cfg.seed).choice(prepared.query_features.shape[0], size=subset_n, replace=False)
    engine_repulsion_oracle = InferenceEngine.load(
        run_dir, device, repulsion_mode="oracle_mc", oracle_mc_samples=cfg.eval.oracle_repulsion_samples, seed=cfg.seed
    )
    emb_subset, _, _, latencies, _ = embed_all_queries(engine_repulsion_oracle, prepared.query_features[subset_idx])
    _eval_method(
        "repulsion_oracle_diagnostic",
        emb_subset,
        reference_embedding_ours,
        latency=float(np.mean(latencies)),
        extra={"query_subset_indices": subset_idx.tolist()},
        query_high_dim=prepared.query_features[subset_idx],
        query_labels_override=query_labels[subset_idx] if query_labels is not None else None,
    )

    # 2. Spectral-only.
    emb_spectral_query = embed_all_queries_spectral_only(engine_full, prepared.query_features)
    reference_embedding_spectral = engine_full.spectral_embedder.embed(prepared.reference_features)
    _eval_method("spectral_only", emb_spectral_query, reference_embedding_spectral)

    # 1. Standard umap-learn (fit reference, transform query).
    ref_emb_umap, query_emb_umap = run_standard_umap(
        prepared.reference_features,
        prepared.query_features,
        cfg.umap.n_neighbors,
        cfg.umap.min_dist,
        cfg.umap.spread,
        cfg.umap.embedding_dim,
        cfg.seed,
    )
    _, coord_rmse = procrustes_align(ref_emb_umap, reference_embedding_ours)
    _eval_method("standard_umap", query_emb_umap, ref_emb_umap, extra={"procrustes_rmse_vs_ours": coord_rmse})

    save_json(layout["metrics"] / "embedding.json", per_method_metrics)

    per_query_df = retrieval_df.copy()
    per_query_df["embedding_recall_ours_full"] = per_query_recall_by_method["ours_full"]
    if query_labels is not None:
        per_query_df["true_label"] = query_labels
    per_query_df.to_csv(layout["metrics"] / "per_query.csv", index=False)


# ---------------------------------------------------------------------------
# Analysis / figures
# ---------------------------------------------------------------------------


def run_analysis(run_dir: str | Path, device: torch.device) -> None:
    layout = run_layout(run_dir)
    cfg = Config.load(layout["root"] / "config.yaml")
    set_seed(cfg.seed)

    prepared = load_prepared_dataset(layout["cache"] / "prepared_dataset.npz")
    trajectory = ReferenceTrajectory.load(layout["memory"] / "reference_trajectory.npz")
    reference_embedding_ours = trajectory.positions[-1]
    label_sets_ref, label_sets_query = _label_sets(prepared, cfg.dataset.name)
    a, b = find_ab_params(cfg.umap.spread, cfg.umap.min_dist)

    engine_full = InferenceEngine.load(run_dir, device, seed=cfg.seed)

    emb_full, _, _, _, _ = embed_all_queries(engine_full, prepared.query_features)
    plot_embedding(
        reference_embedding_ours,
        label_sets_ref,
        layout["figures"] / "embedding_ours.png",
        "Ours (full method)",
        query_embedding=emb_full,
        query_label_sets=label_sets_query,
    )

    emb_spectral = embed_all_queries_spectral_only(engine_full, prepared.query_features)
    ref_spectral = engine_full.spectral_embedder.embed(prepared.reference_features)
    plot_embedding(
        ref_spectral,
        label_sets_ref,
        layout["figures"] / "embedding_spectral_only.png",
        "Spectral-only",
        query_embedding=emb_spectral,
        query_label_sets=label_sets_query,
    )

    engine_oracle = InferenceEngine.load(run_dir, device, neighbor_source="oracle", seed=cfg.seed)
    emb_oracle, _, _, _, _ = embed_all_queries(engine_oracle, prepared.query_features)
    plot_embedding(
        reference_embedding_ours,
        label_sets_ref,
        layout["figures"] / "embedding_oracle_neighbors.png",
        "Ours + oracle neighbors",
        query_embedding=emb_oracle,
        query_label_sets=label_sets_query,
    )

    ref_umap, query_umap = run_standard_umap(
        prepared.reference_features,
        prepared.query_features,
        cfg.umap.n_neighbors,
        cfg.umap.min_dist,
        cfg.umap.spread,
        cfg.umap.embedding_dim,
        cfg.seed,
    )
    plot_embedding(
        ref_umap,
        label_sets_ref,
        layout["figures"] / "embedding_umap_transform.png",
        "Standard umap-learn (fit reference, transform query)",
        query_embedding=query_umap,
        query_label_sets=label_sets_query,
    )

    retriever_hist_path = layout["metrics"] / "retriever_training.json"
    if retriever_hist_path.exists():
        hist = load_json(retriever_hist_path)
        plot_retriever_training(hist["losses"], hist["batch_top1_acc"], layout["figures"] / "loss_retriever.png")

    spectral_hist_path = layout["metrics"] / "spectral_training.json"
    if spectral_hist_path.exists():
        hist = load_json(spectral_hist_path)
        plot_loss_curve(
            {"dirichlet": hist["dirichlet_losses"], "orthogonality": hist["ortho_losses"], "total": hist["total_losses"]},
            layout["figures"] / "loss_spectral.png",
            "Spectral encoder training",
        )

    repulsion_hist_path = layout["metrics"] / "repulsion_training.json"
    if repulsion_hist_path.exists():
        hist = load_json(repulsion_hist_path)
        plot_loss_curve({"mse": hist["losses"]}, layout["figures"] / "loss_repulsion.png", "Repulsion field training")

    retrieval_metrics_path = layout["metrics"] / "retriever.json"
    if retrieval_metrics_path.exists():
        rm = load_json(retrieval_metrics_path)
        plot_retriever_recall(
            {
                "recall@k": rm["recall_at_k"],
                "candidate_recall@M": rm["candidate_recall_at_m"],
                "fuzzy_weighted_recall": rm["fuzzy_weighted_recall"],
                "ndcg@k": rm["ndcg_at_k"],
            },
            layout["figures"] / "retriever_recall.png",
        )

    subset = min(25, prepared.query_features.shape[0])
    idx = np.random.default_rng(cfg.seed).choice(prepared.query_features.shape[0], size=subset, replace=False)
    _, _, _, _, trajectories = embed_all_queries(engine_full, prepared.query_features[idx], return_trajectories=True)
    plot_query_trajectories(reference_embedding_ours, trajectories, layout["figures"] / "query_trajectories.png")

    repulsion_ckpt = torch.load(layout["checkpoints"] / "repulsion_field.pt", map_location="cpu", weights_only=False)
    repulsion_model = RepulsionField(**repulsion_ckpt["hparams"])
    repulsion_model.load_state_dict(repulsion_ckpt["state_dict"])

    span = reference_embedding_ours.max(axis=0) - reference_embedding_ours.min(axis=0)
    pad = 0.1 * float(np.max(span)) + 1e-6
    xmin, xmax = float(reference_embedding_ours[:, 0].min() - pad), float(reference_embedding_ours[:, 0].max() + pad)
    ymin, ymax = float(reference_embedding_ours[:, 1].min() - pad), float(reference_embedding_ours[:, 1].max() + pad)

    for t, suffix in [(0.0, "t000"), (0.5, "t050"), (1.0, "t100")]:
        plot_repulsion_field(
            repulsion_model,
            trajectory,
            a,
            b,
            t,
            ((xmin, xmax), (ymin, ymax)),
            device,
            layout["figures"] / f"repulsion_field_{suffix}.png",
            seed=cfg.seed,
        )


# ---------------------------------------------------------------------------
# Experiment A: analysis-only diagnostics for an already-trained run
# ---------------------------------------------------------------------------

_REQUIRED_ADVANCED_ANALYSIS_ARTIFACTS = [
    "config.yaml",
    "cache/prepared_dataset.npz",
    "memory/reference_features.npy",
    "memory/retriever_keys.npy",
    "memory/graph_symmetric.npz",
    "memory/spectral_calibration.npz",
    "memory/reference_trajectory.npz",
    "checkpoints/retriever.pt",
    "checkpoints/spectral_encoder.pt",
    "checkpoints/repulsion_field.pt",
]


def _require_trained_run(run_dir: Path) -> None:
    """`analyze-advanced` operates on an *already-trained* run and must never
    silently retrain a missing artifact: fail loudly, listing exactly what's
    missing, and tell the user to run `train`/`pipeline` first."""
    missing = [rel for rel in _REQUIRED_ADVANCED_ANALYSIS_ARTIFACTS if not (run_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(
            f"Run directory '{run_dir}' is missing required artifacts for advanced analysis: {missing}. "
            "Run `umaping train` or `umaping pipeline` against this run-dir first -- "
            "`analyze-advanced` never trains or retrains anything itself."
        )


def run_advanced_analysis(
    run_dir: str | Path,
    device: torch.device,
    ks: tuple[int, ...] = (5, 10, 15, 30),
    field_ts: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    grid_resolution: int = 24,
    low_m: int = 32,
    high_m: int = 2048,
    denoising_r_repeats: int = 16,
    denoising_n_eval_points: int = 300,
) -> None:
    """Experiment A: repulsion-field smoothness/denoising diagnostics
    (Section 2.1-2.2) plus per-query "did this land somewhere weird"
    diagnostics (Section 2.3), for a run that has *already* completed
    `train` (and ideally `evaluate`). Never trains anything -- see
    `_require_trained_run` above."""
    run_dir = Path(run_dir)
    _require_trained_run(run_dir)
    layout = run_layout(run_dir)
    cfg = Config.load(layout["root"] / "config.yaml")
    set_seed(cfg.seed)

    prepared = load_prepared_dataset(layout["cache"] / "prepared_dataset.npz")
    trajectory = ReferenceTrajectory.load(layout["memory"] / "reference_trajectory.npz")
    reference_embedding_ours = trajectory.positions[-1]
    a, b = find_ab_params(cfg.umap.spread, cfg.umap.min_dist)

    reference_labels = _primary_label(prepared.reference_labels, cfg.dataset.name)
    query_labels = _primary_label(prepared.query_labels, cfg.dataset.name)

    # -- 2.1 / 2.2: repulsion-field smoothness + Monte-Carlo denoising -------
    repulsion_ckpt = torch.load(layout["checkpoints"] / "repulsion_field.pt", map_location="cpu", weights_only=False)
    repulsion_model = RepulsionField(**repulsion_ckpt["hparams"])
    repulsion_model.load_state_dict(repulsion_ckpt["state_dict"])

    logger.info("Computing repulsion-field smoothness report (grid=%dx%d, ts=%s)...", grid_resolution, grid_resolution, field_ts)
    smoothness = field_smoothness_report(
        repulsion_model,
        trajectory,
        a,
        b,
        reference_embedding_ours,
        device,
        ts=field_ts,
        resolution=grid_resolution,
        low_m=low_m,
        high_m=high_m,
        seed=cfg.seed,
        clip=cfg.flow.grad_clip,
    )
    plot_field_smoothness_by_t(smoothness, layout["figures"] / "field_smoothness_by_t.png")

    logger.info("Running Monte-Carlo denoising test (R=%d, low_m=%d)...", denoising_r_repeats, low_m)
    denoising = denoising_report_by_t(
        repulsion_model,
        trajectory,
        a,
        b,
        reference_embedding_ours,
        device,
        ts=field_ts,
        n_eval_points=denoising_n_eval_points,
        low_m=low_m,
        r_repeats=denoising_r_repeats,
        seed=cfg.seed,
        clip=cfg.flow.grad_clip,
    )
    plot_field_denoising_error_by_t(denoising, layout["figures"] / "field_denoising_error_by_t.png")

    # -- 2.3: per-query diagnostics, across the same methods `evaluate` uses --
    n_query = prepared.query_features.shape[0]
    true_idx, true_dist = chunked_exact_knn(prepared.query_features, prepared.reference_features, k=cfg.eval.k)
    true_weights, _, _ = query_fuzzy_weights(
        true_dist,
        local_connectivity=cfg.umap.local_connectivity,
        n_iter=cfg.umap.smooth_knn_n_iter,
        bandwidth=cfg.umap.smooth_knn_bandwidth,
        min_k_dist_scale=cfg.umap.smooth_knn_min_k_dist_scale,
        eps=cfg.umap.eps,
    )
    neighbor_ids_common = [true_idx[i] for i in range(n_query)]
    neighbor_weights_common = [true_weights[i] for i in range(n_query)]

    per_query_by_method: dict[str, pd.DataFrame] = {}
    tail_by_method: dict[str, dict] = {}

    def _run_method(name: str, query_embedding: np.ndarray, reference_embedding: np.ndarray, idx_subset: np.ndarray | None = None):
        if idx_subset is None:
            q_high = prepared.query_features
            q_labels = query_labels
            n_ids = neighbor_ids_common
            n_w = neighbor_weights_common
        else:
            q_high = prepared.query_features[idx_subset]
            q_labels = query_labels[idx_subset] if query_labels is not None else None
            n_ids = [neighbor_ids_common[i] for i in idx_subset]
            n_w = [neighbor_weights_common[i] for i in idx_subset]
        df, tails = compute_per_query_diagnostics(
            name,
            q_high,
            prepared.reference_features,
            query_embedding,
            reference_embedding,
            n_ids,
            n_w,
            a,
            b,
            ks=ks,
            query_labels=q_labels,
            reference_labels=reference_labels,
        )
        per_query_by_method[name] = df
        tail_by_method[name] = tails

    engine_full = InferenceEngine.load(run_dir, device, seed=cfg.seed)
    emb, _, _, _, _ = embed_all_queries(engine_full, prepared.query_features)
    _run_method("ours_full", emb, reference_embedding_ours)

    engine_oracle_neighbors = InferenceEngine.load(run_dir, device, neighbor_source="oracle", seed=cfg.seed)
    emb, _, _, _, _ = embed_all_queries(engine_oracle_neighbors, prepared.query_features)
    _run_method("oracle_neighbors", emb, reference_embedding_ours)

    engine_no_repulsion = InferenceEngine.load(run_dir, device, use_repulsion=False, seed=cfg.seed)
    emb, _, _, _, _ = embed_all_queries(engine_no_repulsion, prepared.query_features)
    _run_method("no_repulsion", emb, reference_embedding_ours)

    subset_n = min(cfg.eval.repulsion_oracle_query_subset, n_query)
    subset_idx = np.random.default_rng(cfg.seed).choice(n_query, size=subset_n, replace=False)
    engine_repulsion_oracle = InferenceEngine.load(
        run_dir, device, repulsion_mode="oracle_mc", oracle_mc_samples=cfg.eval.oracle_repulsion_samples, seed=cfg.seed
    )
    emb_subset, _, _, _, _ = embed_all_queries(engine_repulsion_oracle, prepared.query_features[subset_idx])
    _run_method("repulsion_oracle_diagnostic", emb_subset, reference_embedding_ours, idx_subset=subset_idx)

    emb_spectral_query = embed_all_queries_spectral_only(engine_full, prepared.query_features)
    reference_embedding_spectral = engine_full.spectral_embedder.embed(prepared.reference_features)
    _run_method("spectral_only", emb_spectral_query, reference_embedding_spectral)

    ref_emb_umap, query_emb_umap = run_standard_umap(
        prepared.reference_features,
        prepared.query_features,
        cfg.umap.n_neighbors,
        cfg.umap.min_dist,
        cfg.umap.spread,
        cfg.umap.embedding_dim,
        cfg.seed,
    )
    _run_method("standard_umap", query_emb_umap, ref_emb_umap)

    # -- Figures + metrics outputs -------------------------------------------
    plot_query_neighbor_recall_distribution(per_query_by_method, k=cfg.eval.k, path=layout["figures"] / "query_neighbor_recall_distribution.png")
    plot_query_fuzzy_error_distribution(per_query_by_method, layout["figures"] / "query_fuzzy_error_distribution.png")
    plot_query_tail_failure_comparison(
        tail_by_method, metric_key=f"recall_at_{cfg.eval.k}_deficit", path=layout["figures"] / "query_tail_failure_comparison.png"
    )
    plot_query_periphery_score(per_query_by_method, layout["figures"] / "query_periphery_score.png")

    combined_df = pd.concat(per_query_by_method.values(), ignore_index=True)
    combined_df.to_csv(layout["metrics"] / "advanced_per_query.csv", index=False)

    save_json(
        layout["metrics"] / "advanced_analysis.json",
        {
            "field_smoothness": smoothness,
            "field_denoising": denoising,
            "per_query_tail_summaries": tail_by_method,
            "config": {
                "ks": list(ks),
                "field_ts": list(field_ts),
                "grid_resolution": grid_resolution,
                "low_m": low_m,
                "high_m": high_m,
                "denoising_r_repeats": denoising_r_repeats,
                "denoising_n_eval_points": denoising_n_eval_points,
            },
        },
    )
    logger.info("Advanced analysis complete. See %s/metrics/advanced_*.{json,csv} and %s/figures", run_dir, run_dir)
