"""All figure generation (docs/method.md; see OUTPUTS in README). Uses the
non-interactive Agg backend throughout since runs happen on headless remote
machines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from umaping.dynamics import ReferenceTrajectory, mean_negative_field  # noqa: E402
from umaping.models.repulsion import RepulsionField  # noqa: E402

_METHOD_COLORS = {
    "b_phi": "tab:blue",
    "low_m_teacher": "tab:orange",
    "high_m_teacher": "tab:green",
    "exact": "tab:red",
}


def plot_embedding(
    reference_embedding: np.ndarray,
    reference_label_sets: dict[str, np.ndarray],
    path: str | Path,
    title: str,
    query_embedding: np.ndarray | None = None,
    query_label_sets: dict[str, np.ndarray] | None = None,
) -> None:
    """One panel per label set (e.g. just "object_id" for COIL; both
    "celltype" and "tech" for pancreas)."""
    label_names = list(reference_label_sets.keys()) or ["_none"]
    n_panels = len(label_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.2), squeeze=False)
    axes = axes[0]

    for ax, label_name in zip(axes, label_names):
        if label_name == "_none":
            ax.scatter(reference_embedding[:, 0], reference_embedding[:, 1], s=6, alpha=0.6, c="tab:blue", label="reference")
        else:
            ref_labels = reference_label_sets[label_name]
            categories = pd.Categorical(ref_labels)
            ax.scatter(
                reference_embedding[:, 0],
                reference_embedding[:, 1],
                c=categories.codes,
                cmap="tab20",
                s=6,
                alpha=0.6,
                label="reference",
            )
            if query_embedding is not None and query_label_sets is not None and label_name in query_label_sets:
                q_codes = pd.Categorical(query_label_sets[label_name], categories=categories.categories).codes
                ax.scatter(
                    query_embedding[:, 0],
                    query_embedding[:, 1],
                    c=q_codes,
                    cmap="tab20",
                    s=18,
                    marker="x",
                    alpha=0.9,
                    label="query",
                )
        if query_embedding is not None and (query_label_sets is None or label_name not in query_label_sets):
            ax.scatter(query_embedding[:, 0], query_embedding[:, 1], c="black", s=10, marker="x", alpha=0.6, label="query")

        ax.set_title(f"{title}" + (f" -- {label_name}" if label_name != "_none" else ""))
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.legend(loc="best", fontsize=8, markerscale=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(curves: dict[str, list[float]], path: str | Path, title: str, ylabel: str = "loss") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, values in curves.items():
        if len(values) == 0:
            continue
        ax.plot(values, linewidth=1, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_retriever_training(losses: list[float], accuracies: list[float], path: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(losses, color="tab:blue", linewidth=1, label="InfoNCE loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(accuracies, color="tab:orange", linewidth=1, alpha=0.7, label="batch top-1 accuracy")
    ax2.set_ylabel("batch top-1 accuracy", color="tab:orange")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("Retriever training")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_retriever_recall(metrics: dict[str, float], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    ax.bar(names, values, color="tab:blue")
    ax.set_ylim(0, max(1.05, max(values) * 1.1 if values else 1.0))
    ax.set_ylabel("score")
    ax.set_title("Retriever recall metrics")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_query_trajectories(
    reference_embedding: np.ndarray,
    trajectories: list[np.ndarray],
    path: str | Path,
    max_trajectories: int = 25,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(reference_embedding[:, 0], reference_embedding[:, 1], c="lightgray", s=4, alpha=0.5, label="reference")
    for traj in trajectories[:max_trajectories]:
        ax.plot(traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.7, color="tab:blue")
        ax.scatter([traj[0, 0]], [traj[0, 1]], c="tab:green", s=18, marker="o", zorder=3)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], c="tab:red", s=24, marker="*", zorder=3)
    ax.set_title("Query trajectories (green = y*(0), red = final y*)")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_repulsion_field(
    model: RepulsionField,
    trajectory: ReferenceTrajectory,
    a: float,
    b: float,
    t: float,
    embedding_range: tuple[tuple[float, float], tuple[float, float]],
    device: torch.device,
    path: str | Path,
    resolution: int = 18,
    mc_samples: int = 256,
    seed: int = 0,
) -> None:
    (xmin, xmax), (ymin, ymax) = embedding_range
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

    rng = np.random.default_rng(seed)
    n = trajectory.positions.shape[1]
    idx = rng.integers(0, n, size=min(mc_samples, n))
    y_neg = torch.as_tensor(trajectory.positions_at(t, indices=idx), dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(grid_points, dtype=torch.float32, device=device)

    model = model.to(device).eval()
    with torch.no_grad():
        teacher = mean_negative_field(y_grid_t, y_neg, a, b).cpu().numpy()
        t_t = torch.full((grid_points.shape[0],), t, dtype=torch.float32, device=device)
        pred = model(y_grid_t, t_t).cpu().numpy()

    error = np.linalg.norm(pred - teacher, axis=1).reshape(resolution, resolution)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, field, name in zip(axes[:2], [pred, teacher], ["Predicted B_phi", "Monte-Carlo teacher"]):
        u = field[:, 0].reshape(resolution, resolution)
        v = field[:, 1].reshape(resolution, resolution)
        ax.quiver(grid_x, grid_y, u, v, angles="xy", scale_units="xy", pivot="mid")
        ax.set_title(f"{name} (t={t:.2f})")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    im = axes[2].imshow(error, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="magma", aspect="auto")
    axes[2].set_title("||predicted - teacher||")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment A: analysis-only advanced diagnostics (evaluation/advanced.py)
# ---------------------------------------------------------------------------


def plot_field_smoothness_by_t(report: dict, path: str | Path) -> None:
    """One panel per roughness metric; one line per field (b_phi vs. the
    low-M/high-M/exact teachers) across `report["ts"]`."""
    ts = report["ts"]
    metric_names = ["finite_diff_roughness", "second_order_roughness", "angular_variation_degrees", "magnitude_variation"]
    methods = list(report["overall"].keys())

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5.2 * len(metric_names), 4.6))
    for ax, metric in zip(axes, metric_names):
        for method in methods:
            ys = [report["per_t"][f"t={t:.2f}"].get(method, {}).get(metric, float("nan")) for t in ts]
            ax.plot(ts, ys, marker="o", label=method, color=_METHOD_COLORS.get(method))
        ax.set_xlabel("t")
        ax.set_title(metric)
    axes[0].set_ylabel("value")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Repulsion-field smoothness by t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_field_denoising_error_by_t(report: dict, path: str | Path) -> None:
    """MSE of b_phi vs. b_single (each against B_mean, the R-repeat-averaged
    low-M estimate), across t -- lower is better; b_phi below b_single at a
    given t means the learned field denoises better than one ordinary
    finite-sample estimate at that t."""
    ts = report["ts"]
    b_phi_mse = [report["per_t"][f"t={t:.2f}"]["vs_b_mean"]["b_phi"]["mse_mean"] for t in ts]
    single_mse = [report["per_t"][f"t={t:.2f}"]["vs_b_mean"]["single_mc_estimate"]["mse_mean"] for t in ts]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ts, b_phi_mse, marker="o", label="B_phi vs. B_mean", color="tab:blue")
    ax.plot(ts, single_mse, marker="o", label="single low-M MC vs. B_mean", color="tab:orange")
    if "vs_exact" in report["per_t"][f"t={ts[0]:.2f}"]:
        b_phi_exact = [report["per_t"][f"t={t:.2f}"]["vs_exact"]["b_phi"]["mse_mean"] for t in ts]
        single_exact = [report["per_t"][f"t={t:.2f}"]["vs_exact"]["single_mc_estimate"]["mse_mean"] for t in ts]
        ax.plot(ts, b_phi_exact, marker="s", linestyle="--", label="B_phi vs. exact", color="tab:blue")
        ax.plot(ts, single_exact, marker="s", linestyle="--", label="single low-M MC vs. exact", color="tab:orange")
    ax.set_xlabel("t")
    ax.set_ylabel("vector MSE")
    ax.set_title("Monte-Carlo denoising test: does B_phi beat one finite-sample estimate?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_query_neighbor_recall_distribution(per_query_by_method: dict[str, "pd.DataFrame"], k: int, path: str | Path) -> None:
    """Histogram of `recall_at_{k}` per method, one subplot each."""
    methods = list(per_query_by_method.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4), squeeze=False)
    axes = axes[0]
    col = f"recall_at_{k}"
    for ax, method in zip(axes, methods):
        df = per_query_by_method[method]
        ax.hist(df[col].to_numpy(), bins=20, range=(0, 1), color="tab:blue", alpha=0.8)
        ax.set_title(method, fontsize=9)
        ax.set_xlabel(col)
    axes[0].set_ylabel("count")
    fig.suptitle(f"Query-to-reference Recall@{k} distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_query_fuzzy_error_distribution(per_query_by_method: dict[str, "pd.DataFrame"], path: str | Path) -> None:
    """Overlaid histograms of `fuzzy_weighted_mse` per method."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, df in per_query_by_method.items():
        values = df["fuzzy_weighted_mse"].dropna().to_numpy()
        if values.size == 0:
            continue
        ax.hist(values, bins=30, alpha=0.5, label=method, density=True)
    ax.set_xlabel("fuzzy-weighted MSE (query vs. true reference neighbors)")
    ax.set_ylabel("density")
    ax.set_title("Fuzzy neighborhood consistency error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_query_tail_failure_comparison(tail_summaries_by_method: dict[str, dict], metric_key: str, path: str | Path) -> None:
    """Bar chart of mean / p95 / p99 / worst-5% / worst-1% for one tail
    metric, grouped by method -- the whole point being that a method can win
    on the mean while losing badly in the tail."""
    methods = list(tail_summaries_by_method.keys())
    fields = ["mean", "p95", "p99", "worst_5pct_mean", "worst_1pct_mean"]
    x = np.arange(len(fields))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, method in enumerate(methods):
        values = [tail_summaries_by_method[method].get(metric_key, {}).get(f, float("nan")) for f in fields]
        ax.bar(x + i * width, values, width=width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(fields, rotation=20, ha="right")
    ax.set_ylabel(metric_key)
    ax.set_title(f"Tail comparison: {metric_key}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_query_periphery_score(per_query_by_method: dict[str, "pd.DataFrame"], path: str | Path) -> None:
    """Overlaid histograms of `repulsion_accumulation_score` per method (the
    Islam & Fleischer-inspired periphery diagnostic; see
    evaluation/advanced.py::repulsion_accumulation_score for the caveat that
    this is not a verified reproduction of their exact published metric)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, df in per_query_by_method.items():
        if "repulsion_accumulation_score" not in df.columns:
            continue
        values = df["repulsion_accumulation_score"].dropna().to_numpy()
        if values.size == 0:
            continue
        ax.hist(values, bins=30, alpha=0.5, label=method, range=(0, 1), density=True)
    ax.set_xlabel("accumulation score (0 = surrounded, 1 = one-sided)")
    ax.set_ylabel("density")
    ax.set_title("Repulsion accumulation / periphery score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
